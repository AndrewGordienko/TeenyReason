"""Latent-space analysis helpers.

These helpers save env-level belief artifacts for the dashboard while keeping
the raw window coverage around for probe-distribution diagnostics.
"""

from pathlib import Path

import numpy as np
import torch

from ..models.belief_world_model import WorldEncoder
from ..models.belief_world_model import encode_probe_modes
from ..models.env_belief import (
    build_diverse_support_mask,
    EnvBeliefAggregator,
    EnvParamPredictorEnsemble,
    build_leave_one_group_out_masks,
    build_env_group_tensors,
    build_uncertainty_feature_tensor,
    compute_disjoint_support_splits,
)
from ..probe.probe_data import get_env_param_names


def encode_window_dataset(
    encoder: WorldEncoder,
    device: torch.device,
    windows: dict[str, np.ndarray],
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode every saved probe window into posterior mean and log-variance."""
    encoder.eval()
    states = windows["states"].astype(np.float32)
    actions = windows["actions"].astype(np.int64)
    rewards = windows["rewards"].astype(np.float32)

    means = []
    logvars = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            stop = start + batch_size
            state_t = torch.tensor(states[start:stop], dtype=torch.float32, device=device)
            action_t = torch.tensor(actions[start:stop], dtype=torch.long, device=device)
            reward_t = torch.tensor(rewards[start:stop], dtype=torch.float32, device=device)
            mean_t, logvar_t = encoder.encode_posterior(state_t, action_t, rewards=reward_t)
            means.append(mean_t.cpu().numpy().astype(np.float32))
            logvars.append(logvar_t.cpu().numpy().astype(np.float32))

    return (
        np.concatenate(means, axis=0).astype(np.float32),
        np.concatenate(logvars, axis=0).astype(np.float32),
    )


def project_latents_2d(latent_means: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project latents to 2D with a tiny PCA implementation."""
    centered = latent_means.astype(np.float32) - latent_means.mean(axis=0, keepdims=True).astype(np.float32)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    try:
        _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].astype(np.float32)
        projection = centered @ components.T
        explained = np.square(singular_values[:2]) / max(np.square(singular_values).sum(), 1e-6)
    except np.linalg.LinAlgError:
        components = np.zeros((2, centered.shape[1]), dtype=np.float32)
        if centered.shape[1] >= 1:
            components[0, 0] = 1.0
        if centered.shape[1] >= 2:
            components[1, 1] = 1.0
        projection = centered[:, :2] if centered.shape[1] >= 2 else np.pad(centered, ((0, 0), (0, max(0, 2 - centered.shape[1]))))
        projection = projection.astype(np.float32)
        explained = np.zeros((2,), dtype=np.float32)
    return projection.astype(np.float32), components, explained.astype(np.float32)


def build_env_belief_dataset(
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble,
    device: torch.device,
    window_mean: np.ndarray,
    window_logvar: np.ndarray,
    windows: dict[str, np.ndarray],
    support_size: int = 6,
    subset_count: int = 8,
) -> dict[str, np.ndarray]:
    """Aggregate window posteriors into one env-level belief per sampled world."""
    probe_mode_idx = encode_probe_modes(np.asarray(windows["probe_mode"], dtype="U"))
    grouped = build_env_group_tensors(
        window_mean=window_mean,
        window_logvar=window_logvar,
        env_instance_id=windows["env_instance_id"],
        env_params=windows["env_params"],
        view_group_id=probe_mode_idx,
    )

    belief_aggregator.eval()
    env_param_predictor.eval()
    with torch.no_grad():
        mean_t = torch.tensor(grouped["window_mean"], dtype=torch.float32, device=device)
        logvar_t = torch.tensor(grouped["window_logvar"], dtype=torch.float32, device=device)
        mask_t = torch.tensor(grouped["mask"], dtype=torch.float32, device=device)
        group_ids_t = torch.tensor(grouped["view_group_id"], dtype=torch.long, device=device)
        support_size = min(max(1, support_size), mean_t.shape[1])
        support_mask_t = build_diverse_support_mask(
            mask=mask_t,
            support_size=support_size,
            group_ids=group_ids_t,
        )
        support_group_count_t = torch.zeros((mean_t.shape[0],), dtype=torch.float32, device=device)
        support_group_ratio_t = torch.ones((mean_t.shape[0],), dtype=torch.float32, device=device)
        for env_idx in range(mean_t.shape[0]):
            support_idx = torch.nonzero(support_mask_t[env_idx] > 0, as_tuple=False).squeeze(-1)
            if support_idx.numel() == 0:
                support_group_count_t[env_idx] = 0.0
                support_group_ratio_t[env_idx] = 0.0
                continue
            support_groups = group_ids_t[env_idx, support_idx]
            unique_groups = torch.unique(support_groups[support_groups >= 0], sorted=True)
            support_group_count_t[env_idx] = float(unique_groups.numel())
            support_group_ratio_t[env_idx] = float(unique_groups.numel()) / max(float(support_idx.numel()), 1.0)
        env_mean_t, env_logvar_t, env_view_spread_t = belief_aggregator(
            mean_t,
            logvar_t,
            support_mask_t,
        )
        env_param_preds_t = env_param_predictor.predict_all(env_mean_t)
        env_param_mean_t = env_param_preds_t.mean(dim=0)
        env_param_std_t = env_param_preds_t.std(dim=0, unbiased=False)
        env_view_spread_mean_t = env_view_spread_t.mean(dim=1)
        split_payload = compute_disjoint_support_splits(
            aggregator=belief_aggregator,
            grouped_mean=mean_t,
            grouped_logvar=logvar_t,
            support_mask=support_mask_t,
            env_param_predictor=env_param_predictor,
        )
        subset_size = int(split_payload["split_count"].min(dim=1).values.float().mean().item())
        leave_masks_t, leave_valid_t = build_leave_one_group_out_masks(support_mask_t, group_ids_t)
        leaveout_latent_std_t = torch.zeros_like(env_mean_t)
        leaveout_param_std_t = torch.zeros_like(env_param_preds_t.mean(dim=0))
        leaveout_shift_t = torch.zeros((mean_t.shape[0],), dtype=torch.float32, device=device)
        leaveout_prediction_error_t = torch.zeros((mean_t.shape[0],), dtype=torch.float32, device=device)
        if leave_masks_t.shape[1] > 0 and torch.any(leave_valid_t > 0):
            batch_size, leave_count, max_views = leave_masks_t.shape
            latent_dim = mean_t.shape[-1]
            repeated_mean = mean_t[:, None, :, :].expand(-1, leave_count, -1, -1)
            repeated_logvar = logvar_t[:, None, :, :].expand(-1, leave_count, -1, -1)
            leave_mean_t, _leave_logvar_t, _leave_spread_t = belief_aggregator(
                repeated_mean.reshape(batch_size * leave_count, max_views, latent_dim),
                repeated_logvar.reshape(batch_size * leave_count, max_views, latent_dim),
                leave_masks_t.reshape(batch_size * leave_count, max_views),
            )
            leave_mean_t = leave_mean_t.reshape(batch_size, leave_count, latent_dim)
            leave_param_mean_t = env_param_predictor.predict_all(
                leave_mean_t.reshape(batch_size * leave_count, latent_dim)
            ).mean(dim=0).reshape(batch_size, leave_count, -1)
            for env_idx in range(batch_size):
                valid_idx = torch.nonzero(leave_valid_t[env_idx] > 0, as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue
                leaveout_latent_std_t[env_idx] = leave_mean_t[env_idx, valid_idx].std(dim=0, unbiased=False)
                leaveout_param_std_t[env_idx] = leave_param_mean_t[env_idx, valid_idx].std(dim=0, unbiased=False)
                leaveout_shift_t[env_idx] = torch.linalg.norm(
                    leave_mean_t[env_idx, valid_idx] - env_mean_t[env_idx].unsqueeze(0),
                    dim=-1,
                ).mean()
                leaveout_prediction_error_t[env_idx] = torch.mean(
                    torch.abs(
                        leave_param_mean_t[env_idx, valid_idx]
                        - torch.tensor(grouped["env_params"][env_idx], dtype=torch.float32, device=device).unsqueeze(0)
                    )
                )

    env_mean = np.nan_to_num(env_mean_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_logvar = np.nan_to_num(env_logvar_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_view_spread = np.nan_to_num(env_view_spread_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_param_prediction = np.nan_to_num(
        env_param_mean_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_param_std = np.nan_to_num(
        env_param_std_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_subset_mean = np.nan_to_num(
        split_payload["env_mean"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_subset_param_mean = np.nan_to_num(
        split_payload["env_param_mean"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_subset_latent_std = np.nan_to_num(env_subset_mean.std(axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_subset_param_std = np.nan_to_num(env_subset_param_mean.std(axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_split_latent_disagreement = np.nan_to_num(
        np.linalg.norm(env_subset_mean[:, 0, :] - env_subset_mean[:, 1, :], axis=-1).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_split_param_disagreement = np.nan_to_num(
        np.linalg.norm(env_subset_param_mean[:, 0, :] - env_subset_param_mean[:, 1, :], axis=-1).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_leaveout_latent_std = np.nan_to_num(
        leaveout_latent_std_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_leaveout_param_std = np.nan_to_num(
        leaveout_param_std_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_support_group_count = np.nan_to_num(
        support_group_count_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_support_group_ratio = np.nan_to_num(
        support_group_ratio_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_subset_shift = np.nan_to_num(np.linalg.norm(
        env_subset_mean - env_mean[:, None, :],
        axis=-1,
    ).mean(axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_between_distance = np.linalg.norm(
        env_mean[:, None, :] - env_mean[None, :, :],
        axis=-1,
    ).astype(np.float32)
    np.fill_diagonal(env_between_distance, np.inf)
    env_nearest_between_distance = np.nan_to_num(
        env_between_distance.min(axis=1).astype(np.float32),
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    env_gap_ratio = np.nan_to_num(
        env_split_latent_disagreement / np.clip(env_nearest_between_distance, 1e-6, None),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    norm_split_a = env_subset_mean[:, 0, :].astype(np.float32)
    norm_split_b = env_subset_mean[:, 1, :].astype(np.float32)
    norm_split_a = norm_split_a / np.clip(np.linalg.norm(norm_split_a, axis=1, keepdims=True), 1e-6, None)
    norm_split_b = norm_split_b / np.clip(np.linalg.norm(norm_split_b, axis=1, keepdims=True), 1e-6, None)
    split_similarity = norm_split_a @ norm_split_b.T
    positive_similarity = np.diag(split_similarity).astype(np.float32)
    np.fill_diagonal(split_similarity, -np.inf)
    hard_negative_a = np.nan_to_num(np.max(split_similarity, axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    hard_negative_b = np.nan_to_num(np.max(split_similarity.T, axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_split_retrieval_margin_deficit = np.clip(
        0.20 - 0.5 * ((positive_similarity - hard_negative_a) + (positive_similarity - hard_negative_b)),
        a_min=0.0,
        a_max=None,
    ).astype(np.float32)
    env_leaveout_shift = np.nan_to_num(leaveout_shift_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_param_abs_error = np.nan_to_num(np.abs(env_param_prediction - grouped["env_params"]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_param_error_mean = np.nan_to_num(env_param_abs_error.mean(axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_split_prediction_error = np.nan_to_num(
        np.mean(np.abs(env_subset_param_mean - grouped["env_params"][:, None, :]), axis=(1, 2)).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_leaveout_prediction_error = np.nan_to_num(
        leaveout_prediction_error_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_coverage_penalty = np.clip(1.0 - env_support_group_ratio, 0.0, 1.0).astype(np.float32)
    with torch.no_grad():
        uncertainty_features_t = build_uncertainty_feature_tensor(
            env_param_std_mean=torch.tensor(env_param_std.mean(axis=1), dtype=torch.float32, device=device),
            split_param_disagreement=torch.tensor(env_split_param_disagreement, dtype=torch.float32, device=device),
            split_latent_disagreement=torch.tensor(env_split_latent_disagreement, dtype=torch.float32, device=device),
            split_env_shift=torch.tensor(env_subset_shift, dtype=torch.float32, device=device),
            leaveout_param_std_mean=torch.tensor(env_leaveout_param_std.mean(axis=1), dtype=torch.float32, device=device),
            leaveout_shift=torch.tensor(env_leaveout_shift, dtype=torch.float32, device=device),
            env_view_spread_mean=torch.tensor(env_view_spread.mean(axis=1), dtype=torch.float32, device=device),
            support_group_ratio=torch.tensor(env_support_group_ratio, dtype=torch.float32, device=device),
        )
        env_uncertainty = np.nan_to_num(
            belief_aggregator.predict_uncertainty(uncertainty_features_t).cpu().numpy().astype(np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    uncertainty_feature_summary = belief_aggregator.uncertainty_feature_summary()
    projection, pca_components, pca_explained = project_latents_2d(env_mean)

    return {
        "env_instance_id": grouped["env_instance_id"].astype(np.int32),
        "env_window_count": grouped["window_count"].astype(np.int32),
        "env_support_count": support_mask_t.sum(dim=1).cpu().numpy().astype(np.int32),
        "env_support_group_count": env_support_group_count.astype(np.float32),
        "env_support_group_ratio": env_support_group_ratio.astype(np.float32),
        "env_subset_size_used": np.full((env_mean.shape[0],), subset_size, dtype=np.int32),
        "env_params": grouped["env_params"].astype(np.float32),
        "env_belief_mean": env_mean.astype(np.float32),
        "env_belief_logvar": env_logvar.astype(np.float32),
        "env_view_spread": env_view_spread.astype(np.float32),
        "env_subset_mean": env_subset_mean.astype(np.float32),
        "env_subset_latent_std": env_subset_latent_std.astype(np.float32),
        "env_subset_param_std": env_subset_param_std.astype(np.float32),
        "env_subset_shift": env_subset_shift.astype(np.float32),
        "env_nearest_between_distance": env_nearest_between_distance.astype(np.float32),
        "env_gap_ratio": env_gap_ratio.astype(np.float32),
        "env_split_mean_a": env_subset_mean[:, 0, :].astype(np.float32),
        "env_split_mean_b": env_subset_mean[:, 1, :].astype(np.float32),
        "env_split_param_mean_a": env_subset_param_mean[:, 0, :].astype(np.float32),
        "env_split_param_mean_b": env_subset_param_mean[:, 1, :].astype(np.float32),
        "env_split_latent_disagreement": env_split_latent_disagreement.astype(np.float32),
        "env_split_param_disagreement": env_split_param_disagreement.astype(np.float32),
        "env_split_retrieval_margin_deficit": env_split_retrieval_margin_deficit.astype(np.float32),
        "env_leaveout_latent_std": env_leaveout_latent_std.astype(np.float32),
        "env_leaveout_param_std": env_leaveout_param_std.astype(np.float32),
        "env_leaveout_shift": env_leaveout_shift.astype(np.float32),
        "env_param_prediction": env_param_prediction.astype(np.float32),
        "env_param_std": env_param_std.astype(np.float32),
        "env_param_abs_error": env_param_abs_error.astype(np.float32),
        "env_param_error_mean": env_param_error_mean.astype(np.float32),
        "env_split_prediction_error": env_split_prediction_error.astype(np.float32),
        "env_leaveout_prediction_error": env_leaveout_prediction_error.astype(np.float32),
        "env_coverage_penalty": env_coverage_penalty.astype(np.float32),
        "env_uncertainty": env_uncertainty.astype(np.float32),
        "env_uncertainty_feature_names": uncertainty_feature_summary["names"],
        "env_uncertainty_feature_weights": uncertainty_feature_summary["weights"].astype(np.float32),
        "projection_2d": projection.astype(np.float32),
        "pca_components": pca_components.astype(np.float32),
        "pca_explained": pca_explained.astype(np.float32),
    }


def build_latent_snapshot(
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble,
    device: torch.device,
    windows: dict[str, np.ndarray],
    env_name: str | None = None,
    benchmark_tag: str | None = None,
    support_size: int = 6,
    subset_count: int = 8,
) -> dict[str, np.ndarray]:
    """Build one dashboard-friendly env-belief snapshot from recorded probe windows."""
    window_mean, window_logvar = encode_window_dataset(encoder=encoder, device=device, windows=windows)
    env_dataset = build_env_belief_dataset(
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        device=device,
        window_mean=window_mean,
        window_logvar=window_logvar,
        windows=windows,
        support_size=support_size,
        subset_count=subset_count,
    )
    reward_sum = np.sum(windows["rewards"], axis=1, dtype=np.float32)

    snapshot = {
        "window_env_instance_id": windows["env_instance_id"].astype(np.int32),
        "window_episode_id": windows["episode_id"].astype(np.int32),
        "window_end_step_idx": windows["end_step_idx"].astype(np.int32),
        "window_probe_mode": windows["probe_mode"].astype("U"),
        "window_reward_sum": reward_sum.astype(np.float32),
        "window_terminated": windows["terminated"].astype(np.int8),
        "window_truncated": windows["truncated"].astype(np.int8),
        "window_latent_mean": window_mean.astype(np.float32),
        "window_latent_logvar": window_logvar.astype(np.float32),
        "env_param_names": np.asarray(
            get_env_param_names(env_name, env_dataset["env_params"].shape[1]),
            dtype="U",
        ),
        **env_dataset,
    }
    if env_name is not None:
        snapshot["env_name"] = np.asarray(env_name)
    if benchmark_tag is not None:
        snapshot["benchmark_tag"] = np.asarray(benchmark_tag)
    return snapshot


def save_latent_snapshot(path: str | Path, snapshot: dict[str, np.ndarray]) -> None:
    """Persist one latent snapshot artifact to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **snapshot)


def load_latent_snapshot(path: str | Path) -> dict[str, np.ndarray]:
    """Load a saved latent snapshot artifact."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def list_latent_snapshot_paths(artifact_dir: str | Path) -> list[Path]:
    """Find all saved latent snapshot artifacts in the artifact directory."""
    artifact_dir = Path(artifact_dir)
    return sorted(artifact_dir.glob("*_latent_snapshot.npz"))
