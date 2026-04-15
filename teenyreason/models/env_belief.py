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


def sanitize_tensor(values: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Replace non-finite tensor values with a finite fallback."""
    if torch.isfinite(values).all():
        return values
    return torch.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)


def safe_normalize(values: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Normalize vectors with explicit finite-value cleanup."""
    return sanitize_tensor(F.normalize(sanitize_tensor(values), dim=dim, eps=eps))


def rescale_positive_features(feature_tensor: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Rescale positive disagreement features to preserve useful variation."""
    positive = sanitize_tensor(feature_tensor).clamp_min(0.0)
    feature_mean = positive.mean(dim=0, keepdim=True).detach()
    feature_std = positive.std(dim=0, unbiased=False, keepdim=True).detach()
    feature_scale = torch.clamp(feature_mean + feature_std, min=eps)
    return positive / feature_scale


UNCERTAINTY_FEATURE_NAMES = (
    "decoder_ensemble_std",
    "split_param_disagreement",
    "split_latent_disagreement",
    "split_env_shift",
    "leaveout_param_std",
    "leaveout_shift",
    "view_spread",
    "coverage_penalty",
)

UNCERTAINTY_FEATURE_INIT_WEIGHTS = (
    2.0,
    1.5,
    0.35,
    0.50,
    1.50,
    1.00,
    0.50,
    0.25,
)


def inverse_softplus(value: float) -> float:
    """Map a positive scalar into the pre-softplus domain."""
    safe_value = max(float(value), 1e-6)
    return math.log(math.expm1(safe_value))


class MonotonicUncertaintyHead(nn.Module):
    """Monotone uncertainty map so larger disagreement cannot reduce uncertainty."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.log_feature_scale = nn.Parameter(torch.zeros(input_dim))
        init_weights = torch.tensor(
            [inverse_softplus(value) for value in UNCERTAINTY_FEATURE_INIT_WEIGHTS[:input_dim]],
            dtype=torch.float32,
        )
        self.log_feature_weight = nn.Parameter(init_weights)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        positive_features = rescale_positive_features(feature_tensor).clamp_max(25.0)
        feature_scale = F.softplus(self.log_feature_scale).unsqueeze(0) + 1e-4
        feature_weight = F.softplus(self.log_feature_weight).unsqueeze(0)
        transformed = torch.log1p(positive_features * feature_scale)
        return torch.sum(transformed * feature_weight, dim=-1) + F.softplus(self.bias)

    def normalized_weights(self) -> torch.Tensor:
        feature_weight = F.softplus(self.log_feature_weight)
        total = feature_weight.sum().clamp_min(1e-6)
        return feature_weight / total


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
        self.delta_gate_logit = nn.Parameter(torch.tensor(-1.0))
        self.uncertainty_head = MonotonicUncertaintyHead(len(UNCERTAINTY_FEATURE_NAMES))

    def aggregate_stats(
        self,
        window_mean: torch.Tensor,
        window_logvar: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Aggregate valid views for each env into raw and normalized beliefs."""
        window_mean = sanitize_tensor(window_mean)
        window_logvar = sanitize_tensor(window_logvar)
        normalized_window_mean = safe_normalize(window_mean, dim=-1)
        view_std = sanitize_tensor(torch.exp(0.5 * window_logvar), fill_value=1.0)
        view_features = sanitize_tensor(self.view_net(torch.cat([normalized_window_mean, view_std], dim=-1)))
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
        pooled_window_mean = (window_mean * mask_f).sum(dim=1) / denom
        context = sanitize_tensor(self.fuse(torch.cat([attended.squeeze(1), pooled], dim=-1)))
        residual_delta = sanitize_tensor(self.mean_head(context))
        delta_gate = torch.sigmoid(self.delta_gate_logit)
        env_mean_raw = sanitize_tensor(pooled_window_mean + delta_gate * residual_delta)
        env_mean = safe_normalize(env_mean_raw, dim=-1)
        env_logvar = torch.clamp(sanitize_tensor(self.logvar_head(context)), -5.0, 2.0)

        centered = normalized_window_mean - env_mean.unsqueeze(1)
        view_var = (mask_f * centered.pow(2)).sum(dim=1) / denom
        view_spread = torch.sqrt(torch.clamp(view_var, min=1e-6))
        return {
            "env_mean": env_mean,
            "env_mean_raw": env_mean_raw,
            "env_logvar": env_logvar,
            "view_spread": view_spread,
        }

    def forward(
        self,
        window_mean: torch.Tensor,
        window_logvar: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate valid views for each env into one normalized env belief."""
        stats = self.aggregate_stats(window_mean=window_mean, window_logvar=window_logvar, mask=mask)
        return stats["env_mean"], stats["env_logvar"], stats["view_spread"]

    def predict_uncertainty(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Map belief disagreement features to an env-level uncertainty estimate."""
        feature_tensor = sanitize_tensor(feature_tensor)
        return sanitize_tensor(self.uncertainty_head(feature_tensor))

    def uncertainty_feature_summary(self) -> dict[str, np.ndarray]:
        """Expose learned uncertainty feature importance for diagnostics."""
        weights = self.uncertainty_head.normalized_weights().detach().cpu().numpy().astype(np.float32)
        return {
            "names": np.asarray(UNCERTAINTY_FEATURE_NAMES, dtype="U"),
            "weights": weights,
        }


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


def build_uncertainty_feature_tensor(
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
        payload["view_group_id"] = pad_sequence(
            group_id_groups,
            batch_first=True,
            padding_value=-1,
        )
    return payload


def build_diverse_support_mask(
    mask: torch.Tensor,
    support_size: int,
    group_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose a small support set, preferring one view per group before repeats."""
    if support_size <= 0:
        raise ValueError("support_size must be positive")

    mask_bool = mask.bool()
    support_mask = torch.zeros_like(mask_bool, dtype=torch.float32)

    for row_idx in range(mask_bool.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() <= support_size:
            support_mask[row_idx, valid_idx] = 1.0
            continue

        chosen: list[int] = []
        if group_ids is not None:
            valid_groups = group_ids[row_idx, valid_idx]
            unique_groups = torch.unique(valid_groups[valid_groups >= 0], sorted=False)
            if unique_groups.numel() > 0:
                group_order = unique_groups[
                    torch.randperm(unique_groups.numel(), generator=generator, device=unique_groups.device)
                ]
                for group_value in group_order.tolist():
                    group_candidates = valid_idx[valid_groups == group_value]
                    if group_candidates.numel() == 0:
                        continue
                    selected = group_candidates[
                        torch.randint(
                            low=0,
                            high=group_candidates.numel(),
                            size=(1,),
                            generator=generator,
                            device=group_candidates.device,
                        )
                    ]
                    chosen.append(int(selected.item()))
                    if len(chosen) >= support_size:
                        break

        if len(chosen) < support_size:
            remaining_idx = torch.tensor(
                [idx for idx in valid_idx.tolist() if idx not in chosen],
                dtype=torch.long,
                device=valid_idx.device,
            )
            if remaining_idx.numel() > 0:
                take = min(int(remaining_idx.numel()), support_size - len(chosen))
                perm = remaining_idx[
                    torch.randperm(remaining_idx.numel(), generator=generator, device=remaining_idx.device)
                ]
                chosen.extend(int(item) for item in perm[:take].tolist())

        if not chosen:
            support_mask[row_idx, valid_idx[0]] = 1.0
            continue
        support_mask[row_idx, torch.tensor(chosen, dtype=torch.long, device=mask.device)] = 1.0

    return support_mask


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


def compute_disjoint_support_splits(
    aggregator: EnvBeliefAggregator,
    grouped_mean: torch.Tensor,
    grouped_logvar: torch.Tensor,
    support_mask: torch.Tensor,
    env_param_predictor: EnvParamPredictorEnsemble | None = None,
) -> dict[str, torch.Tensor]:
    """Build two non-overlapping support-set beliefs for each env instance."""
    split_mask_a, split_mask_b = build_env_subset_masks(support_mask)
    stats_a = aggregator.aggregate_stats(
        grouped_mean,
        grouped_logvar,
        split_mask_a,
    )
    stats_b = aggregator.aggregate_stats(
        grouped_mean,
        grouped_logvar,
        split_mask_b,
    )
    env_mean_a = stats_a["env_mean_raw"]
    env_mean_b = stats_b["env_mean_raw"]
    payload = {
        "mask": torch.stack([split_mask_a, split_mask_b], dim=1),
        "env_mean": torch.stack([env_mean_a, env_mean_b], dim=1),
        "env_mean_unit": torch.stack([stats_a["env_mean"], stats_b["env_mean"]], dim=1),
        "env_logvar": torch.stack([stats_a["env_logvar"], stats_b["env_logvar"]], dim=1),
        "view_spread": torch.stack([stats_a["view_spread"], stats_b["view_spread"]], dim=1),
        "split_count": torch.stack(
            [
                split_mask_a.sum(dim=1),
                split_mask_b.sum(dim=1),
            ],
            dim=1,
        ),
    }
    if env_param_predictor is not None:
        env_param_mean_a = env_param_predictor.predict_all(env_mean_a).mean(dim=0)
        env_param_mean_b = env_param_predictor.predict_all(env_mean_b).mean(dim=0)
        payload["env_param_mean"] = torch.stack([env_param_mean_a, env_param_mean_b], dim=1)
        payload["env_param_disagreement"] = torch.linalg.norm(
            env_param_mean_a - env_param_mean_b,
            dim=-1,
        )
    payload["latent_disagreement"] = torch.linalg.norm(env_mean_a - env_mean_b, dim=-1)
    return payload


def build_random_subset_masks(
    mask: torch.Tensor,
    subset_count: int,
    subset_size: int,
    group_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample several small view subsets for each env instance."""
    if subset_count <= 0:
        raise ValueError("subset_count must be positive")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")

    subset_masks = torch.zeros(
        (mask.shape[0], subset_count, mask.shape[1]),
        dtype=torch.float32,
        device=mask.device,
    )

    for subset_idx in range(subset_count):
        subset_masks[:, subset_idx, :] = build_diverse_support_mask(
            mask=mask,
            support_size=subset_size,
            group_ids=group_ids,
            generator=generator,
        )

    return subset_masks


def build_leave_one_group_out_masks(
    mask: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build leave-one-group-out masks for each env when multiple groups exist."""
    if group_ids is None:
        empty = torch.zeros((mask.shape[0], 0, mask.shape[1]), dtype=torch.float32, device=mask.device)
        valid = torch.zeros((mask.shape[0], 0), dtype=torch.float32, device=mask.device)
        return empty, valid

    mask_bool = mask.bool()
    max_group_count = 0
    grouped_unique: list[torch.Tensor] = []
    for row_idx in range(mask.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            groups = torch.empty((0,), dtype=torch.long, device=mask.device)
        else:
            valid_groups = group_ids[row_idx, valid_idx]
            groups = torch.unique(valid_groups[valid_groups >= 0], sorted=True)
        grouped_unique.append(groups)
        max_group_count = max(max_group_count, int(groups.numel()))

    leave_masks = torch.zeros((mask.shape[0], max_group_count, mask.shape[1]), dtype=torch.float32, device=mask.device)
    valid_leaveouts = torch.zeros((mask.shape[0], max_group_count), dtype=torch.float32, device=mask.device)
    if max_group_count == 0:
        return leave_masks, valid_leaveouts

    for row_idx, groups in enumerate(grouped_unique):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if groups.numel() <= 1:
            continue
        row_group_ids = group_ids[row_idx]
        for group_pos, group_value in enumerate(groups.tolist()):
            keep_mask = mask_bool[row_idx] & (row_group_ids != int(group_value))
            if torch.any(keep_mask):
                leave_masks[row_idx, group_pos, keep_mask] = 1.0
                valid_leaveouts[row_idx, group_pos] = 1.0

    return leave_masks, valid_leaveouts


def compute_support_group_stats(
    support_mask: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure how diverse the chosen support set is for each env row."""
    support_bool = support_mask.bool()
    support_count = support_bool.sum(dim=1).float().clamp_min(1.0)
    if group_ids is None:
        ones = torch.ones_like(support_count)
        return support_count.clone(), ones

    group_counts = []
    ratios = []
    for row_idx in range(support_bool.shape[0]):
        valid_idx = torch.nonzero(support_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            group_counts.append(0.0)
            ratios.append(0.0)
            continue
        row_groups = group_ids[row_idx, valid_idx]
        unique_groups = torch.unique(row_groups[row_groups >= 0], sorted=True)
        count = float(unique_groups.numel())
        group_counts.append(count)
        ratios.append(count / max(float(valid_idx.numel()), 1.0))

    return (
        torch.tensor(group_counts, dtype=torch.float32, device=support_mask.device),
        torch.tensor(ratios, dtype=torch.float32, device=support_mask.device),
    )


def sample_env_belief_subsets(
    aggregator: EnvBeliefAggregator,
    grouped_mean: torch.Tensor,
    grouped_logvar: torch.Tensor,
    grouped_mask: torch.Tensor,
    group_ids: torch.Tensor | None = None,
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
        group_ids=group_ids,
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
    env_param_std: np.ndarray,
    leaveout_latent_std: np.ndarray | None,
    leaveout_param_std: np.ndarray | None,
    subset_shift: float,
    leaveout_shift: float,
    support_diversity_ratio: float,
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

    env_param_component = np.asarray(env_param_std, dtype=np.float32).reshape(-1)
    if env_param_component.size:
        repeats = int(math.ceil(latent_dim / env_param_component.size))
        env_param_component = np.tile(env_param_component, repeats)[:latent_dim].astype(np.float32)
    else:
        env_param_component = np.zeros(latent_dim, dtype=np.float32)

    leaveout_latent = np.asarray(
        np.zeros(latent_dim, dtype=np.float32) if leaveout_latent_std is None else leaveout_latent_std,
        dtype=np.float32,
    ).reshape(-1)
    if leaveout_latent.size < latent_dim:
        leaveout_latent = np.pad(leaveout_latent, (0, latent_dim - leaveout_latent.size))
    leaveout_latent = leaveout_latent[:latent_dim]

    leaveout_param = np.asarray(
        np.zeros(latent_dim, dtype=np.float32) if leaveout_param_std is None else leaveout_param_std,
        dtype=np.float32,
    ).reshape(-1)
    if leaveout_param.size:
        repeats = int(math.ceil(latent_dim / leaveout_param.size))
        leaveout_param = np.tile(leaveout_param, repeats)[:latent_dim].astype(np.float32)
    else:
        leaveout_param = np.zeros(latent_dim, dtype=np.float32)

    subset_shift_component = np.full((latent_dim,), float(max(subset_shift, 0.0)), dtype=np.float32)
    leaveout_shift_component = np.full((latent_dim,), float(max(leaveout_shift, 0.0)), dtype=np.float32)
    coverage_penalty_component = np.full(
        (latent_dim,),
        float(max(0.0, 1.0 - support_diversity_ratio)),
        dtype=np.float32,
    )

    uncertainty = (
        0.25 * latent_component
        + 0.45 * tiled_param
        + 0.30 * env_param_component
        + 0.20 * leaveout_latent
        + 0.30 * leaveout_param
        + 0.25 * subset_shift_component
        + 0.20 * leaveout_shift_component
        + 0.10 * spread_component
        + 0.30 * coverage_penalty_component
    )
    return uncertainty.astype(np.float32)


def aggregate_env_posteriors(
    aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    device: torch.device,
    window_means: np.ndarray,
    window_logvars: np.ndarray,
    probe_group_ids: np.ndarray | None = None,
    subset_count: int = 4,
    subset_size: int = 6,
    support_size: int = 6,
) -> dict[str, np.ndarray]:
    """Aggregate one env's window posteriors into an env belief plus uncertainty."""
    if window_means.ndim != 2 or window_logvars.ndim != 2:
        raise ValueError("Expected `[num_views, z_dim]` window posterior arrays")

    mean_t = torch.tensor(window_means[None, ...], dtype=torch.float32, device=device)
    logvar_t = torch.tensor(window_logvars[None, ...], dtype=torch.float32, device=device)
    mask_t = torch.ones((1, window_means.shape[0]), dtype=torch.float32, device=device)
    group_ids_t = None
    if probe_group_ids is not None:
        group_ids_t = torch.tensor(probe_group_ids[None, ...], dtype=torch.long, device=device)

    aggregator.eval()
    with torch.no_grad():
        support_size = min(max(1, support_size), window_means.shape[0])
        support_mask_t = build_diverse_support_mask(
            mask=mask_t,
            support_size=support_size,
            group_ids=group_ids_t,
        )
        support_group_count_t, support_group_ratio_t = compute_support_group_stats(
            support_mask=support_mask_t,
            group_ids=group_ids_t,
        )
        env_stats_t = aggregator.aggregate_stats(mean_t, logvar_t, support_mask_t)
        env_mean_t = env_stats_t["env_mean"]
        env_mean_raw_t = env_stats_t["env_mean_raw"]
        env_logvar_t = env_stats_t["env_logvar"]
        view_spread_t = env_stats_t["view_spread"]
        split_payload = compute_disjoint_support_splits(
            aggregator=aggregator,
            grouped_mean=mean_t,
            grouped_logvar=logvar_t,
            support_mask=support_mask_t,
            env_param_predictor=env_param_predictor,
        )
        split_env_mean = split_payload["env_mean"]
        subset_latent_std = split_env_mean.std(dim=1, unbiased=False).squeeze(0).cpu().numpy().astype(np.float32)
        subset_shift = float(
            torch.linalg.norm(
                split_env_mean - env_mean_t.unsqueeze(1),
                dim=-1,
            ).mean().item()
        )
        split_latent_disagreement = float(split_payload["latent_disagreement"].mean().item())
        leaveout_latent_std = None
        leaveout_param_std = None
        leaveout_shift = 0.0
        leaveout_masks_t, leaveout_valid_t = build_leave_one_group_out_masks(support_mask_t, group_ids_t)
        if leaveout_masks_t.shape[1] > 0 and torch.any(leaveout_valid_t > 0):
            repeated_mean = mean_t[:, None, :, :].expand(-1, leaveout_masks_t.shape[1], -1, -1)
            repeated_logvar = logvar_t[:, None, :, :].expand(-1, leaveout_masks_t.shape[1], -1, -1)
            leave_stats_t = aggregator.aggregate_stats(
                repeated_mean.reshape(leaveout_masks_t.shape[1], mean_t.shape[1], mean_t.shape[2]),
                repeated_logvar.reshape(leaveout_masks_t.shape[1], logvar_t.shape[1], logvar_t.shape[2]),
                leaveout_masks_t.reshape(leaveout_masks_t.shape[1], leaveout_masks_t.shape[2]),
            )
            leave_mean_t = leave_stats_t["env_mean_raw"]
            valid_idx = torch.nonzero(leaveout_valid_t.reshape(-1) > 0, as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                leave_mean_t = leave_mean_t[valid_idx]
                leaveout_latent_std = leave_mean_t.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
                leaveout_shift = float(
                    torch.linalg.norm(
                        leave_mean_t - env_mean_t.expand_as(leave_mean_t),
                        dim=-1,
                    ).mean().item()
                )
                if env_param_predictor is not None:
                    leave_param_preds = env_param_predictor.predict_all(leave_mean_t)
                    leaveout_param_std = leave_param_preds.mean(dim=0).std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
        if env_param_predictor is None:
            env_param_mean = np.zeros((1,), dtype=np.float32)
            env_param_std = np.zeros((1,), dtype=np.float32)
            subset_param_std = np.zeros((1,), dtype=np.float32)
            split_param_disagreement = 0.0
        else:
            env_param_predictor.eval()
            env_param_preds = env_param_predictor.predict_all(env_mean_raw_t)
            env_param_mean = env_param_preds.mean(dim=0).squeeze(0).cpu().numpy().astype(np.float32)
            env_param_std = env_param_preds.std(dim=0, unbiased=False).squeeze(0).cpu().numpy().astype(np.float32)
            subset_param_std = (
                split_payload["env_param_mean"].std(dim=1, unbiased=False).squeeze(0).cpu().numpy().astype(np.float32)
            )
            split_param_disagreement = float(split_payload["env_param_disagreement"].mean().item())

    env_mean = env_mean_t.squeeze(0).cpu().numpy().astype(np.float32)
    env_mean_raw = env_mean_raw_t.squeeze(0).cpu().numpy().astype(np.float32)
    env_logvar = env_logvar_t.squeeze(0).cpu().numpy().astype(np.float32)
    view_spread = view_spread_t.squeeze(0).cpu().numpy().astype(np.float32)
    uncertainty_vec = build_uncertainty_vector(
        subset_latent_std=subset_latent_std,
        subset_param_std=subset_param_std,
        view_spread=view_spread,
        env_param_std=env_param_std,
        leaveout_latent_std=leaveout_latent_std,
        leaveout_param_std=leaveout_param_std,
        subset_shift=subset_shift + 0.5 * split_latent_disagreement + 0.5 * split_param_disagreement,
        leaveout_shift=leaveout_shift,
        support_diversity_ratio=float(support_group_ratio_t.squeeze(0).item()),
        latent_dim=env_mean.shape[0],
    )
    belief = np.concatenate([env_mean, uncertainty_vec], axis=0).astype(np.float32)
    return {
        "belief": belief,
        "env_mean": env_mean,
        "env_mean_raw": env_mean_raw,
        "env_logvar": env_logvar,
        "view_spread": view_spread,
        "env_param_mean": env_param_mean.astype(np.float32),
        "env_param_std": env_param_std.astype(np.float32),
        "subset_latent_std": subset_latent_std.astype(np.float32),
        "subset_param_std": subset_param_std.astype(np.float32),
        "leaveout_latent_std": (
            np.zeros_like(subset_latent_std, dtype=np.float32)
            if leaveout_latent_std is None
            else leaveout_latent_std.astype(np.float32)
        ),
        "leaveout_param_std": (
            np.zeros_like(subset_param_std, dtype=np.float32)
            if leaveout_param_std is None
            else leaveout_param_std.astype(np.float32)
        ),
        "subset_shift": np.asarray([subset_shift], dtype=np.float32),
        "split_latent_disagreement": np.asarray([split_latent_disagreement], dtype=np.float32),
        "split_param_disagreement": np.asarray([split_param_disagreement], dtype=np.float32),
        "leaveout_shift": np.asarray([leaveout_shift], dtype=np.float32),
        "support_group_count": support_group_count_t.cpu().numpy().astype(np.float32),
        "support_group_ratio": support_group_ratio_t.cpu().numpy().astype(np.float32),
        "subset_size_used": split_payload["split_count"].min(dim=1).values.cpu().numpy().astype(np.int32),
        "support_count": np.asarray([int(support_mask_t.sum().item())], dtype=np.int32),
    }
