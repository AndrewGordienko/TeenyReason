"""Latent-space analysis helpers.

These helpers save env-level belief artifacts for the dashboard while keeping
the raw window coverage around for probe-distribution diagnostics.
"""

from pathlib import Path

import numpy as np
import torch

from ..models.belief_world_model import WorldEncoder
from ..models.env_belief import (
    EnvBeliefAggregator,
    EnvParamPredictorEnsemble,
    build_env_group_tensors,
    sample_env_belief_subsets,
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
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].astype(np.float32)
    projection = centered @ components.T
    explained = np.square(singular_values[:2]) / max(np.square(singular_values).sum(), 1e-6)
    return projection.astype(np.float32), components, explained.astype(np.float32)


def build_env_belief_dataset(
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble,
    device: torch.device,
    window_mean: np.ndarray,
    window_logvar: np.ndarray,
    windows: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Aggregate window posteriors into one env-level belief per sampled world."""
    grouped = build_env_group_tensors(
        window_mean=window_mean,
        window_logvar=window_logvar,
        env_instance_id=windows["env_instance_id"],
        env_params=windows["env_params"],
    )

    belief_aggregator.eval()
    env_param_predictor.eval()
    with torch.no_grad():
        mean_t = torch.tensor(grouped["window_mean"], dtype=torch.float32, device=device)
        logvar_t = torch.tensor(grouped["window_logvar"], dtype=torch.float32, device=device)
        mask_t = torch.tensor(grouped["mask"], dtype=torch.float32, device=device)
        env_mean_t, env_logvar_t, env_view_spread_t = belief_aggregator(mean_t, logvar_t, mask_t)
        env_param_preds_t = env_param_predictor.predict_all(env_mean_t)
        subset_payload = sample_env_belief_subsets(
            aggregator=belief_aggregator,
            grouped_mean=mean_t,
            grouped_logvar=logvar_t,
            grouped_mask=mask_t,
            env_param_predictor=env_param_predictor,
            subset_count=8,
            subset_size=6,
        )

    env_mean = env_mean_t.cpu().numpy().astype(np.float32)
    env_logvar = env_logvar_t.cpu().numpy().astype(np.float32)
    env_view_spread = env_view_spread_t.cpu().numpy().astype(np.float32)
    env_param_prediction = env_param_preds_t.mean(dim=0).cpu().numpy().astype(np.float32)
    env_param_std = env_param_preds_t.std(dim=0).cpu().numpy().astype(np.float32)
    env_subset_mean = subset_payload["env_mean"].cpu().numpy().astype(np.float32)
    env_subset_param_mean = subset_payload["env_param_mean"].cpu().numpy().astype(np.float32)
    env_subset_latent_std = env_subset_mean.std(axis=1).astype(np.float32)
    env_subset_param_std = env_subset_param_mean.std(axis=1).astype(np.float32)
    env_uncertainty = (
        env_subset_param_std.mean(axis=1) + 0.25 * env_subset_latent_std.mean(axis=1)
    ).astype(np.float32)
    projection, pca_components, pca_explained = project_latents_2d(env_mean)

    return {
        "env_instance_id": grouped["env_instance_id"].astype(np.int32),
        "env_window_count": grouped["window_count"].astype(np.int32),
        "env_params": grouped["env_params"].astype(np.float32),
        "env_belief_mean": env_mean.astype(np.float32),
        "env_belief_logvar": env_logvar.astype(np.float32),
        "env_view_spread": env_view_spread.astype(np.float32),
        "env_subset_mean": env_subset_mean.astype(np.float32),
        "env_subset_latent_std": env_subset_latent_std.astype(np.float32),
        "env_subset_param_std": env_subset_param_std.astype(np.float32),
        "env_param_prediction": env_param_prediction.astype(np.float32),
        "env_param_std": env_param_std.astype(np.float32),
        "env_uncertainty": env_uncertainty.astype(np.float32),
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
