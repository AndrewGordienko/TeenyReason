"""Latent-space analysis helpers.

These helpers save env-level belief artifacts for the dashboard while keeping
the raw window coverage around for probe-distribution diagnostics.
"""

import numpy as np
import torch

from ..crawler.predictive import (
    group_window_targets_numpy,
    masked_group_average_numpy,
    masked_group_average_torch,
)
from ..models.belief_world_model import (
    ContrastiveProjector,
    build_future_summary_targets,
    encode_probe_modes,
)
from ..models.env_belief import (
    EnvBeliefAggregator,
    EnvParamPredictorEnsemble,
    build_env_group_tensors,
    build_leave_one_group_out_masks,
    build_split_source_mask,
    build_support_budget_mask,
    build_uncertainty_feature_tensor,
    compute_disjoint_support_splits,
)
from .metrics import project_latents_2d
from .snapshot import (
    build_latent_snapshot,
    compute_message_rate_distortion,
    list_latent_snapshot_paths,
    load_latent_snapshot,
    save_latent_snapshot,
)


def _split_retrieval_rank_and_margin(
    split_mean_a: np.ndarray,
    split_mean_b: np.ndarray,
    margin: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute retrieval rank and margin deficit for paired split beliefs."""
    norm_split_a = np.asarray(split_mean_a, dtype=np.float32)
    norm_split_b = np.asarray(split_mean_b, dtype=np.float32)
    norm_split_a = norm_split_a / np.clip(np.linalg.norm(norm_split_a, axis=1, keepdims=True), 1e-6, None)
    norm_split_b = norm_split_b / np.clip(np.linalg.norm(norm_split_b, axis=1, keepdims=True), 1e-6, None)
    split_similarity = norm_split_a @ norm_split_b.T
    positive_similarity = np.diag(split_similarity).astype(np.float32)
    split_rank_order = np.argsort(-split_similarity, axis=1)
    split_rank_position = (
        np.argmax(
            split_rank_order == np.arange(split_rank_order.shape[0], dtype=np.int32)[:, None],
            axis=1,
        )
        + 1
    ).astype(np.int32)
    np.fill_diagonal(split_similarity, -np.inf)
    hard_negative_a = np.nan_to_num(
        np.max(split_similarity, axis=1).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    hard_negative_b = np.nan_to_num(
        np.max(split_similarity.T, axis=1).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    margin_deficit = np.clip(
        margin - 0.5 * ((positive_similarity - hard_negative_a) + (positive_similarity - hard_negative_b)),
        a_min=0.0,
        a_max=None,
    ).astype(np.float32)
    return split_rank_position, margin_deficit


def build_env_belief_dataset(
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble,
    env_future_predictor,
    env_metric_projector: ContrastiveProjector | None,
    device: torch.device,
    window_mean: np.ndarray,
    window_logvar: np.ndarray,
    windows: dict[str, np.ndarray],
    env_name: str | None = None,
    support_size: int = 4,
    subset_count: int = 1,
) -> dict[str, np.ndarray]:
    """Aggregate window posteriors into one env-level belief per sampled world."""
    probe_mode_idx = encode_probe_modes(np.asarray(windows["probe_mode"], dtype="U"))
    probe_mode_names = np.unique(np.asarray(windows["probe_mode"], dtype="U"))
    grouped = build_env_group_tensors(
        window_mean=window_mean,
        window_logvar=window_logvar,
        env_instance_id=windows["env_instance_id"],
        env_params=windows["env_params"],
        view_group_id=probe_mode_idx,
    )
    param_normalizer_mean = np.asarray(windows["env_params"], dtype=np.float32).mean(
        axis=0,
        keepdims=True,
    )
    param_normalizer_std = np.asarray(windows["env_params"], dtype=np.float32).std(
        axis=0,
        keepdims=True,
    )
    param_normalizer_std = np.where(
        np.abs(param_normalizer_std) < 1e-6,
        1.0,
        param_normalizer_std,
    ).astype(np.float32)
    split_idx = max(2, int(windows["actions"].shape[1]) // 2)
    future_summary = build_future_summary_targets(
        states=windows["states"][:, split_idx:, :].astype(np.float32),
        actions=windows["actions"][:, split_idx:].astype(np.int64),
        rewards=windows["rewards"][:, split_idx:].astype(np.float32),
        terminated=windows["terminated"].astype(np.bool_),
        truncated=windows["truncated"].astype(np.bool_),
        action_vocab_size=int(np.max(windows["actions"])) + 1,
        probe_mode=np.asarray(windows["probe_mode"], dtype="U"),
        env_name=env_name,
    )
    grouped_future_summary = group_window_targets_numpy(
        future_summary,
        windows["env_instance_id"],
    )

    belief_aggregator.eval()
    env_param_predictor.eval()
    with torch.no_grad():
        mean_t = torch.tensor(grouped["window_mean"], dtype=torch.float32, device=device)
        logvar_t = torch.tensor(grouped["window_logvar"], dtype=torch.float32, device=device)
        mask_t = torch.tensor(grouped["mask"], dtype=torch.float32, device=device)
        group_ids_t = torch.tensor(grouped["view_group_id"], dtype=torch.long, device=device)
        param_normalizer_mean_t = torch.tensor(
            param_normalizer_mean,
            dtype=torch.float32,
            device=device,
        )
        param_normalizer_std_t = torch.tensor(
            param_normalizer_std,
            dtype=torch.float32,
            device=device,
        )
        grouped_env_params_t = torch.tensor(
            grouped["env_params"],
            dtype=torch.float32,
            device=device,
        )
        support_size = min(max(1, support_size), mean_t.shape[1])
        support_mask_t = build_support_budget_mask(
            mask=mask_t,
            support_size=support_size,
            subset_count=max(1, int(subset_count)),
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
        support_mask_np = support_mask_t.cpu().numpy().astype(np.float32)
        window_support_mask = np.zeros((windows["env_instance_id"].shape[0],), dtype=np.float32)
        env_row_by_id = {
            int(env_id): int(row_idx)
            for row_idx, env_id in enumerate(grouped["env_instance_id"].tolist())
        }
        env_position_by_id: dict[int, int] = {}
        for window_idx, env_id_value in enumerate(windows["env_instance_id"].tolist()):
            env_id = int(env_id_value)
            env_position = env_position_by_id.get(env_id, 0)
            env_position_by_id[env_id] = env_position + 1
            row_idx = env_row_by_id.get(env_id)
            if row_idx is None or env_position >= support_mask_np.shape[1]:
                continue
            window_support_mask[window_idx] = float(support_mask_np[row_idx, env_position] > 0)
        env_stats_t = belief_aggregator.aggregate_stats(
            mean_t,
            logvar_t,
            support_mask_t,
            group_ids_t,
        )
        env_mean_t = env_stats_t["env_mean_raw"]
        env_mean_unit_t = env_stats_t["env_mean"]
        if env_metric_projector is not None:
            env_metric_projector.eval()
            env_metric_mean_t = env_metric_projector.project_raw(env_mean_t)
            env_metric_mean_unit_t = env_metric_projector(env_mean_t)
        else:
            env_metric_mean_t = env_mean_t
            env_metric_mean_unit_t = env_mean_unit_t
        env_logvar_t = env_stats_t["env_logvar"]
        env_view_spread_t = env_stats_t["view_spread"]
        env_mechanics_posterior_mean_t = env_stats_t["mechanics_posterior_mean"]
        env_mechanics_posterior_std_t = env_stats_t["mechanics_posterior_std"]
        env_mechanics_posterior_entropy_t = env_stats_t["mechanics_posterior_entropy"]
        env_param_preds_t = env_param_predictor.predict_all(env_mean_t)
        env_param_mean_t = env_param_preds_t.mean(dim=0)
        env_param_std_t = env_param_preds_t.std(dim=0, unbiased=False)
        env_view_spread_mean_t = env_view_spread_t.mean(dim=1)
        split_payload = compute_disjoint_support_splits(
            aggregator=belief_aggregator,
            grouped_mean=mean_t,
            grouped_logvar=logvar_t,
            support_mask=build_split_source_mask(mask_t, support_mask_t),
            group_ids=group_ids_t,
            env_param_predictor=env_param_predictor,
        )
        cross_split_payload = compute_disjoint_support_splits(
            aggregator=belief_aggregator,
            grouped_mean=mean_t,
            grouped_logvar=logvar_t,
            support_mask=build_split_source_mask(mask_t, support_mask_t),
            group_ids=group_ids_t,
            env_param_predictor=env_param_predictor,
            split_mode="cross_family",
        )
        if env_metric_projector is not None:
            env_metric_subset_mean_t = env_metric_projector.project_raw(
                split_payload["env_mean"].reshape(-1, split_payload["env_mean"].shape[-1])
            ).reshape(split_payload["env_mean"].shape[0], split_payload["env_mean"].shape[1], -1)
            env_metric_subset_mean_unit_t = env_metric_projector(
                split_payload["env_mean"].reshape(-1, split_payload["env_mean"].shape[-1])
            ).reshape(split_payload["env_mean"].shape[0], split_payload["env_mean"].shape[1], -1)
            env_cross_metric_subset_mean_t = env_metric_projector.project_raw(
                cross_split_payload["env_mean"].reshape(-1, cross_split_payload["env_mean"].shape[-1])
            ).reshape(cross_split_payload["env_mean"].shape[0], cross_split_payload["env_mean"].shape[1], -1)
        else:
            env_metric_subset_mean_t = split_payload["env_mean"]
            env_metric_subset_mean_unit_t = split_payload["env_mean_unit"]
            env_cross_metric_subset_mean_t = cross_split_payload["env_mean"]
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
            leave_stats_t = belief_aggregator.aggregate_stats(
                repeated_mean.reshape(batch_size * leave_count, max_views, latent_dim),
                repeated_logvar.reshape(batch_size * leave_count, max_views, latent_dim),
                leave_masks_t.reshape(batch_size * leave_count, max_views),
                group_ids_t[:, None, :].expand(-1, leave_count, -1).reshape(
                    batch_size * leave_count,
                    max_views,
                ),
            )
            leave_mean_t = leave_stats_t["env_mean_raw"].reshape(batch_size, leave_count, latent_dim)
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
                leave_param_mean_raw = (
                    leave_param_mean_t[env_idx, valid_idx] * param_normalizer_std_t
                    + param_normalizer_mean_t
                )
                leaveout_prediction_error_t[env_idx] = torch.mean(
                    torch.abs(
                        leave_param_mean_raw
                        - grouped_env_params_t[env_idx].unsqueeze(0)
                    )
                )
        future_summary_t = torch.tensor(grouped_future_summary, dtype=torch.float32, device=device)
        heldout_mask_t = torch.clamp(mask_t - support_mask_t, min=0.0)
        env_future_target_t, _heldout_count_t = masked_group_average_torch(
            future_summary_t,
            heldout_mask_t,
            fallback_mask=mask_t,
        )
        split_future_target_a_t, _split_count_a_t = masked_group_average_torch(
            future_summary_t,
            split_payload["mask"][:, 0, :],
            fallback_mask=mask_t,
        )
        split_future_target_b_t, _split_count_b_t = masked_group_average_torch(
            future_summary_t,
            split_payload["mask"][:, 1, :],
            fallback_mask=mask_t,
        )
        env_future_prediction_error_t = torch.zeros((mean_t.shape[0],), dtype=torch.float32, device=device)
        env_split_future_prediction_error_t = torch.zeros((mean_t.shape[0],), dtype=torch.float32, device=device)
        if env_future_predictor is not None:
            env_future_predictor.eval()
            env_future_pred_t = env_future_predictor(env_mean_t)
            env_future_pred_a_t = env_future_predictor(split_payload["env_mean"][:, 0, :])
            env_future_pred_b_t = env_future_predictor(split_payload["env_mean"][:, 1, :])
            env_future_prediction_error_t = torch.mean(
                torch.abs(env_future_pred_t - env_future_target_t),
                dim=1,
            )
            env_split_future_prediction_error_t = 0.5 * (
                torch.mean(torch.abs(env_future_pred_a_t - split_future_target_b_t), dim=1)
                + torch.mean(torch.abs(env_future_pred_b_t - split_future_target_a_t), dim=1)
            )

    env_mean = np.nan_to_num(env_mean_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_mean_unit = np.nan_to_num(env_mean_unit_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_metric_mean = np.nan_to_num(env_metric_mean_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_metric_mean_unit = np.nan_to_num(env_metric_mean_unit_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_logvar = np.nan_to_num(env_logvar_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_view_spread = np.nan_to_num(env_view_spread_t.cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_mechanics_posterior_mean = np.nan_to_num(
        env_mechanics_posterior_mean_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_mechanics_posterior_std = np.nan_to_num(
        env_mechanics_posterior_std_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_mechanics_posterior_entropy = np.nan_to_num(
        env_mechanics_posterior_entropy_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_param_prediction_normalized = np.nan_to_num(
        env_param_mean_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_param_prediction = np.nan_to_num(
        (env_param_prediction_normalized * param_normalizer_std + param_normalizer_mean).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_param_std_normalized = np.nan_to_num(
        env_param_std_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_param_std = np.nan_to_num(
        (env_param_std_normalized * param_normalizer_std).astype(np.float32),
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
    env_metric_subset_mean = np.nan_to_num(
        env_metric_subset_mean_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_subset_param_mean_normalized = np.nan_to_num(
        split_payload["env_param_mean"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_subset_param_mean = np.nan_to_num(
        (
            env_subset_param_mean_normalized * param_normalizer_std[:, None, :]
            + param_normalizer_mean[:, None, :]
        ).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_subset_latent_std = np.nan_to_num(env_subset_mean.std(axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_subset_param_std_normalized = np.nan_to_num(
        env_subset_param_mean_normalized.std(axis=1).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_subset_param_std = np.nan_to_num(env_subset_param_mean.std(axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_split_latent_disagreement = np.nan_to_num(
        np.linalg.norm(env_metric_subset_mean[:, 0, :] - env_metric_subset_mean[:, 1, :], axis=-1).astype(np.float32),
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
    env_split_param_disagreement_normalized = np.nan_to_num(
        np.linalg.norm(
            env_subset_param_mean_normalized[:, 0, :] - env_subset_param_mean_normalized[:, 1, :],
            axis=-1,
        ).astype(np.float32),
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
    env_leaveout_param_std_normalized = np.nan_to_num(
        leaveout_param_std_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_leaveout_param_std = np.nan_to_num(
        (env_leaveout_param_std_normalized * param_normalizer_std).astype(np.float32),
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
    env_dominant_probe_mode_idx = np.zeros((env_mean.shape[0],), dtype=np.int32)
    for env_idx in range(env_mean.shape[0]):
        support_idx = np.flatnonzero(support_mask_t[env_idx].cpu().numpy() > 0)
        if support_idx.size == 0:
            continue
        support_groups = grouped["view_group_id"][env_idx, support_idx]
        support_groups = support_groups[support_groups >= 0]
        if support_groups.size == 0:
            continue
        unique_groups, group_counts = np.unique(support_groups.astype(np.int32), return_counts=True)
        env_dominant_probe_mode_idx[env_idx] = int(unique_groups[int(np.argmax(group_counts))])
    env_dominant_probe_mode = probe_mode_names[np.clip(env_dominant_probe_mode_idx, 0, max(len(probe_mode_names) - 1, 0))]
    env_subset_shift = np.nan_to_num(np.linalg.norm(
        env_metric_subset_mean - env_metric_mean[:, None, :],
        axis=-1,
    ).mean(axis=1).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    env_split_group_overlap = np.nan_to_num(
        split_payload["split_group_overlap"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_split_balanced_half = np.nan_to_num(
        split_payload["split_balanced_half"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_split_group_count_a = np.nan_to_num(
        split_payload["split_group_count_a"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_split_group_count_b = np.nan_to_num(
        split_payload["split_group_count_b"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_cross_family_subset_mean = np.nan_to_num(
        cross_split_payload["env_mean"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_cross_family_metric_subset_mean = np.nan_to_num(
        env_cross_metric_subset_mean_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_cross_family_split_group_overlap = np.nan_to_num(
        cross_split_payload["split_group_overlap"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_cross_family_split_group_count_a = np.nan_to_num(
        cross_split_payload["split_group_count_a"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_cross_family_split_group_count_b = np.nan_to_num(
        cross_split_payload["split_group_count_b"].cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_between_distance = np.linalg.norm(
        env_metric_mean[:, None, :] - env_metric_mean[None, :, :],
        axis=-1,
    ).astype(np.float32)
    np.fill_diagonal(env_between_distance, np.inf)
    env_nearest_between_distance = np.nan_to_num(
        env_between_distance.min(axis=1).astype(np.float32),
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    env_between_distance_unit = np.linalg.norm(
        env_metric_mean_unit[:, None, :] - env_metric_mean_unit[None, :, :],
        axis=-1,
    ).astype(np.float32)
    np.fill_diagonal(env_between_distance_unit, np.inf)
    env_nearest_between_distance_unit = np.nan_to_num(
        env_between_distance_unit.min(axis=1).astype(np.float32),
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
    split_rank_position, env_split_retrieval_margin_deficit = _split_retrieval_rank_and_margin(
        env_metric_subset_mean[:, 0, :],
        env_metric_subset_mean[:, 1, :],
    )
    cross_split_rank_position, env_cross_family_split_retrieval_margin_deficit = _split_retrieval_rank_and_margin(
        env_cross_family_metric_subset_mean[:, 0, :],
        env_cross_family_metric_subset_mean[:, 1, :],
    )
    env_cross_family_split_latent_disagreement = np.nan_to_num(
        np.linalg.norm(
            env_cross_family_metric_subset_mean[:, 0, :] - env_cross_family_metric_subset_mean[:, 1, :],
            axis=-1,
        ).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_cross_family_gap_ratio = np.nan_to_num(
        env_cross_family_split_latent_disagreement / np.clip(env_nearest_between_distance, 1e-6, None),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_pairwise_between_distance = np.nan_to_num(
        env_between_distance[np.triu_indices(env_between_distance.shape[0], k=1)].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_pairwise_between_distance_unit = np.nan_to_num(
        env_between_distance_unit[np.triu_indices(env_between_distance_unit.shape[0], k=1)].astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
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
    env_future_prediction_error = np.nan_to_num(
        env_future_prediction_error_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_split_future_prediction_error = np.nan_to_num(
        env_split_future_prediction_error_t.cpu().numpy().astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    env_coverage_penalty = np.clip(1.0 - env_support_group_ratio, 0.0, 1.0).astype(np.float32)
    with torch.no_grad():
        uncertainty_features_t = build_uncertainty_feature_tensor(
            mechanics_posterior_std_mean=torch.tensor(
                env_mechanics_posterior_std.mean(axis=1),
                dtype=torch.float32,
                device=device,
            ),
            mechanics_posterior_entropy=torch.tensor(
                env_mechanics_posterior_entropy,
                dtype=torch.float32,
                device=device,
            ),
            env_param_std_mean=torch.tensor(env_param_std_normalized.mean(axis=1), dtype=torch.float32, device=device),
            split_param_disagreement=torch.tensor(
                env_split_param_disagreement_normalized,
                dtype=torch.float32,
                device=device,
            ),
            split_latent_disagreement=torch.tensor(env_split_latent_disagreement, dtype=torch.float32, device=device),
            split_env_shift=torch.tensor(env_subset_shift, dtype=torch.float32, device=device),
            leaveout_param_std_mean=torch.tensor(
                env_leaveout_param_std_normalized.mean(axis=1),
                dtype=torch.float32,
                device=device,
            ),
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
    projection, pca_components, pca_explained = project_latents_2d(env_metric_mean)

    return {
        "env_instance_id": grouped["env_instance_id"].astype(np.int32),
        "env_window_count": grouped["window_count"].astype(np.int32),
        "env_support_available_count": mask_t.sum(dim=1).cpu().numpy().astype(np.int32),
        "env_support_count": support_mask_t.sum(dim=1).cpu().numpy().astype(np.int32),
        "env_support_used_count": support_mask_t.sum(dim=1).cpu().numpy().astype(np.int32),
        "env_heldout_count": _heldout_count_t.cpu().numpy().astype(np.int32),
        "env_support_group_count": env_support_group_count.astype(np.float32),
        "env_support_group_ratio": env_support_group_ratio.astype(np.float32),
        "env_dominant_probe_mode": env_dominant_probe_mode.astype("U"),
        "window_is_support": window_support_mask.astype(np.int8),
        "window_is_heldout": (1 - window_support_mask.astype(np.int8)).astype(np.int8),
        "env_subset_size_used": np.full((env_mean.shape[0],), subset_size, dtype=np.int32),
        "env_params": grouped["env_params"].astype(np.float32),
        "env_belief_mean": env_mean.astype(np.float32),
        "env_belief_mean_unit": env_mean_unit.astype(np.float32),
        "env_belief_norm": np.linalg.norm(env_mean, axis=1).astype(np.float32),
        "env_metric_mean": env_metric_mean.astype(np.float32),
        "env_metric_mean_unit": env_metric_mean_unit.astype(np.float32),
        "env_metric_norm": np.linalg.norm(env_metric_mean, axis=1).astype(np.float32),
        "env_belief_logvar": env_logvar.astype(np.float32),
        "env_view_spread": env_view_spread.astype(np.float32),
        "env_mechanics_posterior_mean": env_mechanics_posterior_mean.astype(np.float32),
        "env_mechanics_posterior_std": env_mechanics_posterior_std.astype(np.float32),
        "env_mechanics_posterior_entropy": env_mechanics_posterior_entropy.astype(np.float32),
        "env_subset_mean": env_subset_mean.astype(np.float32),
        "env_subset_mean_unit": split_payload["env_mean_unit"].cpu().numpy().astype(np.float32),
        "env_metric_subset_mean": env_metric_subset_mean.astype(np.float32),
        "env_metric_subset_mean_unit": env_metric_subset_mean_unit_t.cpu().numpy().astype(np.float32),
        "env_subset_latent_std": env_subset_latent_std.astype(np.float32),
        "env_subset_param_std": env_subset_param_std.astype(np.float32),
        "env_subset_param_std_normalized": env_subset_param_std_normalized.astype(np.float32),
        "env_subset_shift": env_subset_shift.astype(np.float32),
        "env_split_group_overlap": env_split_group_overlap.astype(np.float32),
        "env_split_balanced_half": env_split_balanced_half.astype(np.float32),
        "env_split_group_count_a": env_split_group_count_a.astype(np.float32),
        "env_split_group_count_b": env_split_group_count_b.astype(np.float32),
        "env_cross_family_split_group_overlap": env_cross_family_split_group_overlap.astype(np.float32),
        "env_cross_family_split_group_count_a": env_cross_family_split_group_count_a.astype(np.float32),
        "env_cross_family_split_group_count_b": env_cross_family_split_group_count_b.astype(np.float32),
        "env_nearest_between_distance": env_nearest_between_distance.astype(np.float32),
        "env_nearest_between_distance_unit": env_nearest_between_distance_unit.astype(np.float32),
        "env_gap_ratio": env_gap_ratio.astype(np.float32),
        "env_cross_family_gap_ratio": env_cross_family_gap_ratio.astype(np.float32),
        "env_split_mean_a": env_subset_mean[:, 0, :].astype(np.float32),
        "env_split_mean_b": env_subset_mean[:, 1, :].astype(np.float32),
        "env_split_mean_unit_a": split_payload["env_mean_unit"][:, 0, :].cpu().numpy().astype(np.float32),
        "env_split_mean_unit_b": split_payload["env_mean_unit"][:, 1, :].cpu().numpy().astype(np.float32),
        "env_metric_split_mean_a": env_metric_subset_mean[:, 0, :].astype(np.float32),
        "env_metric_split_mean_b": env_metric_subset_mean[:, 1, :].astype(np.float32),
        "env_metric_split_mean_unit_a": env_metric_subset_mean_unit_t[:, 0, :].cpu().numpy().astype(np.float32),
        "env_metric_split_mean_unit_b": env_metric_subset_mean_unit_t[:, 1, :].cpu().numpy().astype(np.float32),
        "env_cross_family_split_mean_a": env_cross_family_subset_mean[:, 0, :].astype(np.float32),
        "env_cross_family_split_mean_b": env_cross_family_subset_mean[:, 1, :].astype(np.float32),
        "env_cross_family_metric_split_mean_a": env_cross_family_metric_subset_mean[:, 0, :].astype(np.float32),
        "env_cross_family_metric_split_mean_b": env_cross_family_metric_subset_mean[:, 1, :].astype(np.float32),
        "env_split_param_mean_a": env_subset_param_mean[:, 0, :].astype(np.float32),
        "env_split_param_mean_b": env_subset_param_mean[:, 1, :].astype(np.float32),
        "env_split_latent_disagreement": env_split_latent_disagreement.astype(np.float32),
        "env_split_param_disagreement": env_split_param_disagreement.astype(np.float32),
        "env_split_param_disagreement_normalized": env_split_param_disagreement_normalized.astype(np.float32),
        "env_split_retrieval_margin_deficit": env_split_retrieval_margin_deficit.astype(np.float32),
        "env_split_retrieval_rank": split_rank_position.astype(np.int32),
        "env_cross_family_split_latent_disagreement": env_cross_family_split_latent_disagreement.astype(np.float32),
        "env_cross_family_split_retrieval_margin_deficit": env_cross_family_split_retrieval_margin_deficit.astype(np.float32),
        "env_cross_family_split_retrieval_rank": cross_split_rank_position.astype(np.int32),
        "env_pairwise_between_distance": env_pairwise_between_distance.astype(np.float32),
        "env_pairwise_between_distance_unit": env_pairwise_between_distance_unit.astype(np.float32),
        "env_leaveout_latent_std": env_leaveout_latent_std.astype(np.float32),
        "env_leaveout_param_std": env_leaveout_param_std.astype(np.float32),
        "env_leaveout_param_std_normalized": env_leaveout_param_std_normalized.astype(np.float32),
        "env_leaveout_shift": env_leaveout_shift.astype(np.float32),
        "env_param_prediction": env_param_prediction.astype(np.float32),
        "env_param_prediction_normalized": env_param_prediction_normalized.astype(np.float32),
        "env_param_std": env_param_std.astype(np.float32),
        "env_param_std_normalized": env_param_std_normalized.astype(np.float32),
        "env_param_abs_error": env_param_abs_error.astype(np.float32),
        "env_param_error_mean": env_param_error_mean.astype(np.float32),
        "env_split_prediction_error": env_split_prediction_error.astype(np.float32),
        "env_leaveout_prediction_error": env_leaveout_prediction_error.astype(np.float32),
        "env_future_prediction_error": env_future_prediction_error.astype(np.float32),
        "env_split_future_prediction_error": env_split_future_prediction_error.astype(np.float32),
        "env_coverage_penalty": env_coverage_penalty.astype(np.float32),
        "env_uncertainty": env_uncertainty.astype(np.float32),
        "env_uncertainty_feature_names": uncertainty_feature_summary["names"],
        "env_uncertainty_feature_weights": uncertainty_feature_summary["weights"].astype(np.float32),
        "projection_2d": projection.astype(np.float32),
        "pca_components": pca_components.astype(np.float32),
        "pca_explained": pca_explained.astype(np.float32),
    }
