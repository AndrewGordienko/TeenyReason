"""Helpers for grouping probe targets and building held-out predictive targets."""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def group_window_targets_torch(
    values: torch.Tensor,
    env_instance_id: torch.Tensor,
) -> torch.Tensor:
    """Pad per-window targets into `[num_envs, max_views, target_dim]`."""
    unique_env_ids = torch.unique(env_instance_id, sorted=True)
    value_groups = []
    for env_id in unique_env_ids:
        indices = torch.nonzero(env_instance_id == env_id, as_tuple=False).squeeze(-1)
        value_groups.append(values[indices])
    return pad_sequence(value_groups, batch_first=True)


def group_window_targets_numpy(
    values: np.ndarray,
    env_instance_id: np.ndarray,
) -> np.ndarray:
    """Pad per-window targets into `[num_envs, max_views, target_dim]`."""
    env_ids = np.asarray(env_instance_id, dtype=np.int64)
    unique_env_ids = np.unique(env_ids)
    max_views = max(int(np.sum(env_ids == env_id)) for env_id in unique_env_ids)
    target_dim = int(values.shape[-1])
    grouped_values = np.zeros((len(unique_env_ids), max_views, target_dim), dtype=np.float32)

    for env_row, env_id in enumerate(unique_env_ids):
        indices = np.flatnonzero(env_ids == env_id)
        count = len(indices)
        grouped_values[env_row, :count] = np.asarray(values[indices], dtype=np.float32)
    return grouped_values


def masked_group_average_torch(
    grouped_values: torch.Tensor,
    mask: torch.Tensor,
    fallback_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average grouped values under a mask, falling back if the mask is empty."""
    mask = mask.float()
    if fallback_mask is None:
        fallback_mask = torch.ones_like(mask)
    else:
        fallback_mask = fallback_mask.float()

    use_mask = mask
    empty_rows = mask.sum(dim=1) <= 0
    if torch.any(empty_rows):
        use_mask = mask.clone()
        use_mask[empty_rows] = fallback_mask[empty_rows]

    denom = use_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    averaged = (grouped_values * use_mask.unsqueeze(-1)).sum(dim=1) / denom
    return averaged, use_mask.sum(dim=1)


def masked_group_average_numpy(
    grouped_values: np.ndarray,
    mask: np.ndarray,
    fallback_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average grouped values under a mask, falling back if the mask is empty."""
    mask = np.asarray(mask, dtype=np.float32)
    if fallback_mask is None:
        fallback_mask = np.ones_like(mask, dtype=np.float32)
    else:
        fallback_mask = np.asarray(fallback_mask, dtype=np.float32)

    use_mask = mask.copy()
    empty_rows = np.sum(mask, axis=1) <= 0
    if np.any(empty_rows):
        use_mask[empty_rows] = fallback_mask[empty_rows]

    denom = np.clip(np.sum(use_mask, axis=1, keepdims=True), 1.0, None)
    averaged = np.sum(grouped_values * use_mask[..., None], axis=1) / denom
    return averaged.astype(np.float32), np.sum(use_mask, axis=1).astype(np.float32)
