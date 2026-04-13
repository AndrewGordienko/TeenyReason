"""Environment-level belief aggregation on top of per-window latents.

This module is the split point between:

- window latents: what happened in one specific intervention
- env beliefs: what hidden mechanics seem shared across many interventions
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class EnvBeliefAggregator(nn.Module):
    """Aggregate many window posteriors into one env-level belief."""

    def __init__(self, window_z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.window_z_dim = window_z_dim
        self.hidden_dim = hidden_dim
        self.view_net = nn.Sequential(
            nn.Linear(window_z_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, window_z_dim)
        self.logvar_head = nn.Linear(hidden_dim, window_z_dim)

    def forward(
        self,
        window_mean: torch.Tensor,
        window_logvar: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate valid views for each env into an env posterior and spread."""
        normalized_window_mean = F.normalize(window_mean, dim=-1)
        view_std = torch.exp(0.5 * window_logvar)
        view_features = self.view_net(torch.cat([normalized_window_mean, view_std], dim=-1))
        key_padding_mask = ~mask.bool()
        query = self.query.expand(window_mean.shape[0], -1, -1)
        attended, _weights = self.attn(
            query=query,
            key=view_features,
            value=view_features,
            key_padding_mask=key_padding_mask,
        )

        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (view_features * mask_f).sum(dim=1) / denom
        context = self.fuse(torch.cat([attended.squeeze(1), pooled], dim=-1))
        env_mean = F.normalize(self.mean_head(context), dim=-1)
        env_logvar = torch.clamp(self.logvar_head(context), -5.0, 2.0)

        centered = normalized_window_mean - env_mean.unsqueeze(1)
        view_var = (mask_f * centered.pow(2)).sum(dim=1) / denom
        view_spread = torch.sqrt(torch.clamp(view_var, min=1e-6))
        return env_mean, env_logvar, view_spread


class EnvParamPredictor(nn.Module):
    """One ensemble head that predicts hidden env parameters from env belief."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, env_mean: torch.Tensor) -> torch.Tensor:
        return self.net(env_mean)


class EnvParamPredictorEnsemble(nn.Module):
    """Ensemble env-parameter decoder used for mechanics uncertainty."""

    def __init__(
        self,
        ensemble_size: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.heads = nn.ModuleList(
            [EnvParamPredictor(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim) for _ in range(ensemble_size)]
        )

    def predict_all(self, env_mean: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(env_mean) for head in self.heads], dim=0)

    def forward(self, env_mean: torch.Tensor) -> torch.Tensor:
        return self.predict_all(env_mean).mean(dim=0)


def build_env_group_tensors(
    window_mean: np.ndarray,
    window_logvar: np.ndarray,
    env_instance_id: np.ndarray,
    env_params: np.ndarray,
) -> dict[str, np.ndarray]:
    """Pad variable numbers of windows into env-instance batches."""
    env_ids = np.asarray(env_instance_id, dtype=np.int64)
    unique_env_ids = np.unique(env_ids)
    max_views = max(int(np.sum(env_ids == env_id)) for env_id in unique_env_ids)
    z_dim = int(window_mean.shape[1])
    param_dim = int(env_params.shape[1])

    grouped_mean = np.zeros((len(unique_env_ids), max_views, z_dim), dtype=np.float32)
    grouped_logvar = np.zeros((len(unique_env_ids), max_views, z_dim), dtype=np.float32)
    grouped_mask = np.zeros((len(unique_env_ids), max_views), dtype=np.float32)
    grouped_counts = np.zeros(len(unique_env_ids), dtype=np.int32)
    grouped_env_params = np.zeros((len(unique_env_ids), param_dim), dtype=np.float32)

    for env_row, env_id in enumerate(unique_env_ids):
        indices = np.flatnonzero(env_ids == env_id)
        count = len(indices)
        grouped_mean[env_row, :count] = np.asarray(window_mean[indices], dtype=np.float32)
        grouped_logvar[env_row, :count] = np.asarray(window_logvar[indices], dtype=np.float32)
        grouped_mask[env_row, :count] = 1.0
        grouped_counts[env_row] = count
        grouped_env_params[env_row] = np.asarray(env_params[indices[0]], dtype=np.float32)

    return {
        "env_instance_id": unique_env_ids.astype(np.int64),
        "window_mean": grouped_mean,
        "window_logvar": grouped_logvar,
        "mask": grouped_mask,
        "window_count": grouped_counts,
        "env_params": grouped_env_params,
    }


def group_window_latents_torch(
    window_mean: torch.Tensor,
    window_logvar: torch.Tensor,
    env_instance_id: torch.Tensor,
    env_params: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Group differentiable window latents by env instance without breaking grads."""
    unique_env_ids = torch.unique(env_instance_id, sorted=True)
    mean_groups = []
    logvar_groups = []
    mask_groups = []
    count_groups = []
    env_param_groups = []

    for env_id in unique_env_ids:
        indices = torch.nonzero(env_instance_id == env_id, as_tuple=False).squeeze(-1)
        mean_groups.append(window_mean[indices])
        logvar_groups.append(window_logvar[indices])
        mask_groups.append(torch.ones(indices.shape[0], dtype=torch.float32, device=window_mean.device))
        count_groups.append(indices.shape[0])
        env_param_groups.append(env_params[indices[0]])

    grouped_mean = pad_sequence(mean_groups, batch_first=True)
    grouped_logvar = pad_sequence(logvar_groups, batch_first=True)
    grouped_mask = pad_sequence(mask_groups, batch_first=True)
    return {
        "env_instance_id": unique_env_ids,
        "window_mean": grouped_mean,
        "window_logvar": grouped_logvar,
        "mask": grouped_mask,
        "window_count": torch.tensor(count_groups, dtype=torch.int32, device=window_mean.device),
        "env_params": torch.stack(env_param_groups, dim=0),
    }


def build_env_subset_masks(
    mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split valid views into two random non-empty subsets per env."""
    mask_bool = mask.bool()
    mask_a = torch.zeros_like(mask_bool)
    mask_b = torch.zeros_like(mask_bool)

    for row_idx in range(mask_bool.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() == 1:
            mask_a[row_idx, valid_idx] = True
            mask_b[row_idx, valid_idx] = True
            continue

        permutation = valid_idx[torch.randperm(valid_idx.numel(), generator=generator, device=valid_idx.device)]
        split_point = max(1, int(math.ceil(permutation.numel() / 2.0)))
        left = permutation[:split_point]
        right = permutation[split_point:]
        if right.numel() == 0:
            right = left[-1:].clone()
        mask_a[row_idx, left] = True
        mask_b[row_idx, right] = True

    return mask_a.float(), mask_b.float()


def build_random_subset_masks(
    mask: torch.Tensor,
    subset_count: int,
    subset_size: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample several small view subsets for each env instance."""
    if subset_count <= 0:
        raise ValueError("subset_count must be positive")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")

    mask_bool = mask.bool()
    subset_masks = torch.zeros(
        (mask.shape[0], subset_count, mask.shape[1]),
        dtype=torch.float32,
        device=mask.device,
    )

    for row_idx in range(mask_bool.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() == 1:
            take_count = 1
        else:
            take_count = min(int(valid_idx.numel()) - 1, subset_size)
            take_count = max(take_count, 1)
        for subset_idx in range(subset_count):
            if valid_idx.numel() <= take_count:
                chosen = valid_idx
            else:
                permutation = valid_idx[
                    torch.randperm(valid_idx.numel(), generator=generator, device=valid_idx.device)
                ]
                chosen = permutation[:take_count]
            subset_masks[row_idx, subset_idx, chosen] = 1.0

    return subset_masks


def sample_env_belief_subsets(
    aggregator: EnvBeliefAggregator,
    grouped_mean: torch.Tensor,
    grouped_logvar: torch.Tensor,
    grouped_mask: torch.Tensor,
    env_param_predictor: EnvParamPredictorEnsemble | None = None,
    subset_count: int = 4,
    subset_size: int = 6,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate several random few-view subsets for each env instance."""
    subset_masks = build_random_subset_masks(
        mask=grouped_mask,
        subset_count=subset_count,
        subset_size=subset_size,
        generator=generator,
    )
    batch_size, sampled_subset_count, max_views = subset_masks.shape
    latent_dim = grouped_mean.shape[-1]
    repeated_mean = grouped_mean[:, None, :, :].expand(-1, sampled_subset_count, -1, -1)
    repeated_logvar = grouped_logvar[:, None, :, :].expand(-1, sampled_subset_count, -1, -1)

    env_mean, env_logvar, env_view_spread = aggregator(
        repeated_mean.reshape(batch_size * sampled_subset_count, max_views, latent_dim),
        repeated_logvar.reshape(batch_size * sampled_subset_count, max_views, latent_dim),
        subset_masks.reshape(batch_size * sampled_subset_count, max_views),
    )

    payload = {
        "mask": subset_masks,
        "env_mean": env_mean.reshape(batch_size, sampled_subset_count, latent_dim),
        "env_logvar": env_logvar.reshape(batch_size, sampled_subset_count, latent_dim),
        "view_spread": env_view_spread.reshape(batch_size, sampled_subset_count, latent_dim),
    }
    if env_param_predictor is not None:
        env_param_preds = env_param_predictor.predict_all(env_mean)
        param_dim = env_param_preds.shape[-1]
        payload["env_param_mean"] = env_param_preds.mean(dim=0).reshape(batch_size, sampled_subset_count, param_dim)
        payload["env_param_std"] = env_param_preds.std(dim=0).reshape(batch_size, sampled_subset_count, param_dim)
    return payload


def build_uncertainty_vector(
    subset_latent_std: np.ndarray,
    subset_param_std: np.ndarray,
    view_spread: np.ndarray,
    latent_dim: int,
) -> np.ndarray:
    """Map subset disagreement into the policy uncertainty half."""
    latent_component = np.asarray(subset_latent_std, dtype=np.float32).reshape(-1)
    if latent_component.size == 0:
        latent_component = np.zeros(latent_dim, dtype=np.float32)
    if latent_component.size < latent_dim:
        latent_component = np.pad(latent_component, (0, latent_dim - latent_component.size))
    latent_component = latent_component[:latent_dim]

    param_component = np.asarray(subset_param_std, dtype=np.float32).reshape(-1)
    if param_component.size:
        repeats = int(math.ceil(latent_dim / param_component.size))
        tiled_param = np.tile(param_component, repeats)[:latent_dim].astype(np.float32)
    else:
        tiled_param = np.zeros(latent_dim, dtype=np.float32)

    spread_component = np.asarray(view_spread, dtype=np.float32).reshape(-1)
    if spread_component.size < latent_dim:
        spread_component = np.pad(spread_component, (0, latent_dim - spread_component.size))
    spread_component = spread_component[:latent_dim]

    uncertainty = latent_component + 0.35 * tiled_param + 0.10 * spread_component
    return uncertainty.astype(np.float32)


def aggregate_env_posteriors(
    aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    device: torch.device,
    window_means: np.ndarray,
    window_logvars: np.ndarray,
    subset_count: int = 4,
    subset_size: int = 6,
) -> dict[str, np.ndarray]:
    """Aggregate one env's window posteriors into an env belief plus uncertainty."""
    if window_means.ndim != 2 or window_logvars.ndim != 2:
        raise ValueError("Expected `[num_views, z_dim]` window posterior arrays")

    mean_t = torch.tensor(window_means[None, ...], dtype=torch.float32, device=device)
    logvar_t = torch.tensor(window_logvars[None, ...], dtype=torch.float32, device=device)
    mask_t = torch.ones((1, window_means.shape[0]), dtype=torch.float32, device=device)

    aggregator.eval()
    with torch.no_grad():
        env_mean_t, env_logvar_t, view_spread_t = aggregator(mean_t, logvar_t, mask_t)
        subset_payload = sample_env_belief_subsets(
            aggregator=aggregator,
            grouped_mean=mean_t,
            grouped_logvar=logvar_t,
            grouped_mask=mask_t,
            env_param_predictor=env_param_predictor,
            subset_count=subset_count,
            subset_size=min(subset_size, max(1, window_means.shape[0] - 1)),
        )
        subset_env_mean = subset_payload["env_mean"]
        subset_latent_std = subset_env_mean.std(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        if env_param_predictor is None:
            env_param_mean = np.zeros((1,), dtype=np.float32)
            env_param_std = np.zeros((1,), dtype=np.float32)
            subset_param_std = np.zeros((1,), dtype=np.float32)
        else:
            env_param_predictor.eval()
            env_param_preds = env_param_predictor.predict_all(env_mean_t)
            env_param_mean = env_param_preds.mean(dim=0).squeeze(0).cpu().numpy().astype(np.float32)
            env_param_std = env_param_preds.std(dim=0).squeeze(0).cpu().numpy().astype(np.float32)
            subset_param_std = (
                subset_payload["env_param_mean"].std(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            )

    env_mean = env_mean_t.squeeze(0).cpu().numpy().astype(np.float32)
    env_logvar = env_logvar_t.squeeze(0).cpu().numpy().astype(np.float32)
    view_spread = view_spread_t.squeeze(0).cpu().numpy().astype(np.float32)
    uncertainty_vec = build_uncertainty_vector(
        subset_latent_std=subset_latent_std,
        subset_param_std=subset_param_std,
        view_spread=view_spread,
        latent_dim=env_mean.shape[0],
    )
    belief = np.concatenate([env_mean, uncertainty_vec], axis=0).astype(np.float32)
    return {
        "belief": belief,
        "env_mean": env_mean,
        "env_logvar": env_logvar,
        "view_spread": view_spread,
        "env_param_mean": env_param_mean.astype(np.float32),
        "env_param_std": env_param_std.astype(np.float32),
        "subset_latent_std": subset_latent_std.astype(np.float32),
        "subset_param_std": subset_param_std.astype(np.float32),
    }
