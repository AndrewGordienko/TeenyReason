"""Grouping helpers for env-level belief inputs and uncertainty features."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .env_belief_common import sanitize_tensor


def build_uncertainty_feature_tensor(
    mechanics_posterior_std_mean: torch.Tensor,
    mechanics_posterior_entropy: torch.Tensor,
    env_param_std_mean: torch.Tensor,
    split_param_disagreement: torch.Tensor,
    split_latent_disagreement: torch.Tensor,
    split_env_shift: torch.Tensor,
    leaveout_param_std_mean: torch.Tensor,
    leaveout_shift: torch.Tensor,
    env_view_spread_mean: torch.Tensor,
    support_group_ratio: torch.Tensor,
) -> torch.Tensor:
    """Assemble support-ambiguity features for uncertainty calibration."""
    return torch.stack(
        [
            sanitize_tensor(mechanics_posterior_std_mean),
            sanitize_tensor(mechanics_posterior_entropy),
            sanitize_tensor(env_param_std_mean),
            sanitize_tensor(split_param_disagreement),
            sanitize_tensor(split_latent_disagreement),
            sanitize_tensor(split_env_shift),
            sanitize_tensor(leaveout_param_std_mean),
            sanitize_tensor(leaveout_shift),
            sanitize_tensor(env_view_spread_mean),
            sanitize_tensor(1.0 - support_group_ratio),
        ],
        dim=1,
    )


def build_env_group_tensors(
    window_mean: np.ndarray,
    window_logvar: np.ndarray,
    env_instance_id: np.ndarray,
    env_params: np.ndarray,
    view_group_id: np.ndarray | None = None,
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
    grouped_view_group = np.full((len(unique_env_ids), max_views), -1, dtype=np.int64)

    for env_row, env_id in enumerate(unique_env_ids):
        indices = np.flatnonzero(env_ids == env_id)
        count = len(indices)
        grouped_mean[env_row, :count] = np.asarray(window_mean[indices], dtype=np.float32)
        grouped_logvar[env_row, :count] = np.asarray(window_logvar[indices], dtype=np.float32)
        grouped_mask[env_row, :count] = 1.0
        grouped_counts[env_row] = count
        grouped_env_params[env_row] = np.asarray(env_params[indices[0]], dtype=np.float32)
        if view_group_id is not None:
            grouped_view_group[env_row, :count] = np.asarray(view_group_id[indices], dtype=np.int64)

    payload = {
        "env_instance_id": unique_env_ids.astype(np.int64),
        "window_mean": grouped_mean,
        "window_logvar": grouped_logvar,
        "mask": grouped_mask,
        "window_count": grouped_counts,
        "env_params": grouped_env_params,
    }
    if view_group_id is not None:
        payload["view_group_id"] = grouped_view_group
    return payload


def group_window_latents_torch(
    window_mean: torch.Tensor,
    window_logvar: torch.Tensor,
    env_instance_id: torch.Tensor,
    env_params: torch.Tensor,
    view_group_id: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Group differentiable window latents by env instance without breaking grads."""
    unique_env_ids = torch.unique(env_instance_id, sorted=True)
    mean_groups = []
    logvar_groups = []
    mask_groups = []
    count_groups = []
    env_param_groups = []
    group_id_groups = []

    for env_id in unique_env_ids:
        indices = torch.nonzero(env_instance_id == env_id, as_tuple=False).squeeze(-1)
        mean_groups.append(window_mean[indices])
        logvar_groups.append(window_logvar[indices])
        mask_groups.append(torch.ones(indices.shape[0], dtype=torch.float32, device=window_mean.device))
        count_groups.append(indices.shape[0])
        env_param_groups.append(env_params[indices[0]])
        if view_group_id is not None:
            group_id_groups.append(view_group_id[indices])

    grouped_mean = pad_sequence(mean_groups, batch_first=True)
    grouped_logvar = pad_sequence(logvar_groups, batch_first=True)
    grouped_mask = pad_sequence(mask_groups, batch_first=True)
    payload = {
        "env_instance_id": unique_env_ids,
        "window_mean": grouped_mean,
        "window_logvar": grouped_logvar,
        "mask": grouped_mask,
        "window_count": torch.tensor(count_groups, dtype=torch.int32, device=window_mean.device),
        "env_params": torch.stack(env_param_groups, dim=0),
    }
    if view_group_id is not None:
        payload["view_group_id"] = pad_sequence(group_id_groups, batch_first=True, padding_value=-1)
    return payload
