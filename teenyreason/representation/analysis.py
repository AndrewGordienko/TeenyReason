"""Latent-space analysis helpers.

These functions turn the trained encoder plus probe windows into compact
artifacts that are easy to inspect later in a dashboard.
"""

from pathlib import Path

import numpy as np
import torch

from ..models.belief_world_model import WorldEncoder


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


def build_latent_snapshot(
    encoder: WorldEncoder,
    device: torch.device,
    windows: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build one dashboard-friendly latent snapshot from recorded probe windows."""
    latent_mean, latent_logvar = encode_window_dataset(encoder=encoder, device=device, windows=windows)
    projection, pca_components, pca_explained = project_latents_2d(latent_mean)
    reward_sum = np.sum(windows["rewards"], axis=1, dtype=np.float32)
    posterior_std = np.exp(0.5 * latent_logvar).astype(np.float32)
    uncertainty = posterior_std.mean(axis=1).astype(np.float32)

    return {
        "episode_id": windows["episode_id"].astype(np.int32),
        "end_step_idx": windows["end_step_idx"].astype(np.int32),
        "probe_mode": windows["probe_mode"].astype("U"),
        "env_params": windows["env_params"].astype(np.float32),
        "reward_sum": reward_sum.astype(np.float32),
        "terminated": windows["terminated"].astype(np.int8),
        "truncated": windows["truncated"].astype(np.int8),
        "latent_mean": latent_mean.astype(np.float32),
        "latent_logvar": latent_logvar.astype(np.float32),
        "posterior_std": posterior_std.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
        "projection_2d": projection.astype(np.float32),
        "pca_components": pca_components.astype(np.float32),
        "pca_explained": pca_explained.astype(np.float32),
    }


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
