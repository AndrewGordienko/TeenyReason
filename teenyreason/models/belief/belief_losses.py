"""Losses and geometry diagnostics for the belief world model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .belief_common import safe_normalize, sanitize_tensor


def retrieval_safe_normalize(
    values: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-2,
) -> torch.Tensor:
    """Normalize retrieval embeddings with a detached, well-clamped norm."""
    finite_values = sanitize_tensor(values)
    norm = torch.linalg.vector_norm(finite_values, dim=dim, keepdim=True)
    norm = sanitize_tensor(norm, fill_value=eps).clamp_min(eps)
    return sanitize_tensor(finite_values / norm.detach())


def safe_vector_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute a smooth L2 distance that stays well-behaved near zero."""
    diff = sanitize_tensor(left - right)
    squared = diff.pow(2).sum(dim=dim)
    return torch.sqrt(squared + eps)


def safe_pairwise_distance(
    values: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute pairwise L2 distances with a small epsilon for stable gradients."""
    values = sanitize_tensor(values)
    diff = values[:, None, :] - values[None, :, :]
    squared = diff.pow(2).sum(dim=-1)
    return torch.sqrt(squared + eps)


def supervised_same_env_contrastive_loss(
    embeddings: torch.Tensor,
    env_instance_id: torch.Tensor,
    probe_mode_idx: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pull together embeddings from the same env instance across probe modes."""
    embeddings = safe_normalize(embeddings, dim=-1)
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    logits = embeddings @ embeddings.T / max(temperature, 1e-4)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    same_env = env_instance_id[:, None] == env_instance_id[None, :]
    different_mode = probe_mode_idx[:, None] != probe_mode_idx[None, :]
    positive_mask = same_env & different_mode & (~self_mask)
    fallback_mask = same_env & (~self_mask)
    has_cross_mode_positive = positive_mask.any(dim=1, keepdim=True)
    positive_mask = torch.where(has_cross_mode_positive, positive_mask, fallback_mask)
    valid_rows = positive_mask.any(dim=1)
    if not torch.any(valid_rows):
        return embeddings.sum() * 0.0

    exp_logits = sanitize_tensor(torch.exp(logits) * (~self_mask))
    positive_mass = (exp_logits * positive_mask).sum(dim=1).clamp_min(1e-6)
    total_mass = exp_logits.sum(dim=1).clamp_min(1e-6)
    loss = -torch.log(positive_mass / total_mass)
    return loss[valid_rows].mean()


def pairwise_env_geometry_loss(
    latent_mean: torch.Tensor,
    normalized_env_params: torch.Tensor,
) -> torch.Tensor:
    """Encourage latent distances to reflect true env-parameter distances."""
    if latent_mean.shape[0] < 2:
        return latent_mean.sum() * 0.0

    normalized_latent = safe_normalize(latent_mean, dim=-1)
    normalized_env_params = sanitize_tensor(normalized_env_params)
    latent_distance = 1.0 - normalized_latent @ normalized_latent.T
    env_distance = safe_pairwise_distance(normalized_env_params)
    mask = ~torch.eye(latent_mean.shape[0], dtype=torch.bool, device=latent_mean.device)
    if not torch.any(mask):
        return latent_mean.sum() * 0.0

    latent_values = sanitize_tensor(latent_distance[mask])
    env_values = sanitize_tensor(env_distance[mask])
    latent_values = latent_values / latent_values.mean().clamp_min(1e-6)
    env_values = env_values / env_values.mean().clamp_min(1e-6)
    return F.mse_loss(latent_values, env_values)


def within_between_env_loss(
    env_mean: torch.Tensor,
    subset_env_mean: torch.Tensor,
    env_params: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Keep same-env subset beliefs tighter than neighboring different-env beliefs."""
    if env_mean.shape[0] < 2:
        return env_mean.sum() * 0.0

    env_mean = sanitize_tensor(env_mean)
    subset_env_mean = sanitize_tensor(subset_env_mean)
    env_params = sanitize_tensor(env_params)
    within = safe_vector_distance(subset_env_mean, env_mean.unsqueeze(1)).mean(dim=1)
    between = safe_pairwise_distance(env_mean)
    self_mask = torch.eye(env_mean.shape[0], dtype=torch.bool, device=env_mean.device)
    between = between.masked_fill(self_mask, float("inf"))

    param_distance = safe_pairwise_distance(env_params)
    non_self = ~self_mask
    valid_param = param_distance[non_self]
    if valid_param.numel() == 0:
        nearest_between = between.min(dim=1).values
    else:
        hard_negative_cutoff = torch.quantile(valid_param, 0.35)
        hard_negative_mask = (param_distance >= hard_negative_cutoff) & non_self
        masked_between = between.masked_fill(~hard_negative_mask, float("inf"))
        nearest_between = masked_between.min(dim=1).values
        fallback_between = between.min(dim=1).values
        nearest_between = torch.where(torch.isfinite(nearest_between), nearest_between, fallback_between)

    margin_loss = F.relu(within + margin - nearest_between).mean()
    ratio_loss = torch.clamp(within / nearest_between.clamp_min(1e-3), max=10.0).mean()
    return margin_loss + 0.10 * ratio_loss


def info_nce_loss(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between latent-prefix queries and future-summary keys."""
    query_embeddings = retrieval_safe_normalize(query_embeddings, dim=-1)
    key_embeddings = retrieval_safe_normalize(key_embeddings, dim=-1)
    logits = query_embeddings @ key_embeddings.T / max(temperature, 1e-4)
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    logits = torch.clamp(logits, min=-20.0, max=20.0)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def subset_retrieval_loss(
    anchor_env_mean: torch.Tensor,
    positive_env_mean: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    """Force one disjoint support half to retrieve the matching other half."""
    return info_nce_loss(
        retrieval_safe_normalize(anchor_env_mean, dim=-1),
        retrieval_safe_normalize(positive_env_mean, dim=-1),
        temperature=temperature,
    )


def hard_negative_retrieval_loss(
    anchor_env_mean: torch.Tensor,
    positive_env_mean: torch.Tensor,
    env_params: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Push matching worlds above semantically similar non-matches."""
    if anchor_env_mean.shape[0] < 2:
        return anchor_env_mean.sum() * 0.0

    anchor_env_mean = retrieval_safe_normalize(anchor_env_mean, dim=-1)
    positive_env_mean = retrieval_safe_normalize(positive_env_mean, dim=-1)
    env_params = sanitize_tensor(env_params)
    similarity = torch.clamp(anchor_env_mean @ positive_env_mean.T, min=-20.0, max=20.0)
    positive_similarity = torch.diagonal(similarity)

    param_distance = torch.cdist(env_params, env_params, p=2)
    self_mask = torch.eye(anchor_env_mean.shape[0], dtype=torch.bool, device=anchor_env_mean.device)
    valid_param = param_distance[~self_mask]
    if valid_param.numel() == 0:
        return anchor_env_mean.sum() * 0.0

    param_scale = valid_param.mean().clamp_min(1e-3)
    negative_weight = torch.exp(-param_distance / param_scale)
    negative_weight = negative_weight.masked_fill(self_mask, 0.0)
    ranking_loss = F.relu(margin + similarity - positive_similarity.unsqueeze(1))
    weighted_loss = ranking_loss * negative_weight
    return weighted_loss.sum() / negative_weight.sum().clamp_min(1.0)


def split_gap_ratio_loss(
    env_mean: torch.Tensor,
    split_mean_a: torch.Tensor,
    split_mean_b: torch.Tensor,
    margin: float = 0.10,
    target_ratio: float = 0.35,
) -> torch.Tensor:
    """Make same-world split halves much closer than nearby different worlds."""
    if env_mean.shape[0] < 2:
        return env_mean.sum() * 0.0
    env_mean = sanitize_tensor(env_mean)
    split_mean_a = sanitize_tensor(split_mean_a)
    split_mean_b = sanitize_tensor(split_mean_b)
    same_gap = safe_vector_distance(split_mean_a, split_mean_b)
    between = safe_pairwise_distance(env_mean)
    self_mask = torch.eye(env_mean.shape[0], dtype=torch.bool, device=env_mean.device)
    between = between.masked_fill(self_mask, float("inf"))
    nearest_between = between.min(dim=1).values.clamp_min(1e-3)
    margin_loss = F.relu(same_gap + margin - nearest_between).mean()
    ratio_loss = F.relu(same_gap / nearest_between - target_ratio).mean()
    return margin_loss + 0.25 * ratio_loss


def split_geometry_stats(
    env_mean: torch.Tensor,
    split_mean_a: torch.Tensor,
    split_mean_b: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Summarize same-world split geometry against nearby different worlds."""
    if env_mean.shape[0] == 0:
        empty = env_mean.new_zeros((0,))
        return {
            "same_gap": empty,
            "nearest_between": empty,
            "gap_ratio": empty,
        }

    env_mean = sanitize_tensor(env_mean)
    split_mean_a = sanitize_tensor(split_mean_a)
    split_mean_b = sanitize_tensor(split_mean_b)
    same_gap = safe_vector_distance(split_mean_a, split_mean_b)
    if env_mean.shape[0] < 2:
        nearest_between = torch.ones_like(same_gap)
    else:
        between = safe_pairwise_distance(env_mean)
        self_mask = torch.eye(env_mean.shape[0], dtype=torch.bool, device=env_mean.device)
        between = between.masked_fill(self_mask, float("inf"))
        nearest_between = between.min(dim=1).values.clamp_min(1e-3)
    return {
        "same_gap": same_gap,
        "nearest_between": nearest_between,
        "gap_ratio": same_gap / nearest_between.clamp_min(1e-3),
    }


def split_retrieval_margin_deficit(
    split_mean_a: torch.Tensor,
    split_mean_b: torch.Tensor,
    margin: float = 0.20,
) -> torch.Tensor:
    """Return how far each split-half pair is from a safe retrieval margin."""
    if split_mean_a.shape[0] < 2:
        return split_mean_a.new_zeros((split_mean_a.shape[0],))

    norm_a = retrieval_safe_normalize(split_mean_a, dim=-1)
    norm_b = retrieval_safe_normalize(split_mean_b, dim=-1)
    similarity = norm_a @ norm_b.T
    positive = torch.diagonal(similarity)
    self_mask = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
    hard_negative_a = similarity.masked_fill(self_mask, float("-inf")).max(dim=1).values
    hard_negative_b = similarity.T.masked_fill(self_mask, float("-inf")).max(dim=1).values
    margin_a = positive - hard_negative_a
    margin_b = positive - hard_negative_b
    deficit_a = F.relu(margin - margin_a)
    deficit_b = F.relu(margin - margin_b)
    return 0.5 * (deficit_a + deficit_b)


def uncertainty_ranking_loss(
    uncertainty_signal: torch.Tensor,
    target_error: torch.Tensor,
    margin: float = 0.02,
) -> torch.Tensor:
    """Encourage higher uncertainty for envs with larger mechanics error."""
    if uncertainty_signal.shape[0] < 2:
        return uncertainty_signal.sum() * 0.0
    error_diff = target_error[:, None] - target_error[None, :]
    uncert_diff = uncertainty_signal[:, None] - uncertainty_signal[None, :]
    pair_mask = torch.abs(error_diff) > 1e-4
    if not torch.any(pair_mask):
        return uncertainty_signal.sum() * 0.0
    direction = torch.sign(error_diff[pair_mask])
    signed_gap = direction * uncert_diff[pair_mask]
    return F.relu(margin - signed_gap).mean()


def correlation_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Encourage uncertainty scores to preserve the ordering structure of error."""
    if predicted.shape[0] < 2:
        return predicted.sum() * 0.0
    predicted_centered = sanitize_tensor(predicted - predicted.mean())
    target_centered = sanitize_tensor(target - target.mean())
    numerator = torch.sum(predicted_centered * target_centered)
    denominator = torch.linalg.norm(predicted_centered) * torch.linalg.norm(target_centered)
    if not torch.isfinite(denominator) or float(denominator.item()) <= 1e-6:
        return predicted.sum() * 0.0
    correlation = torch.clamp(numerator / denominator, -1.0, 1.0)
    return 1.0 - correlation


def standardize_1d(values: torch.Tensor) -> torch.Tensor:
    """Center and scale a 1D signal while staying finite."""
    values = sanitize_tensor(values)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def uncertainty_separation_loss(
    uncertainty_signal: torch.Tensor,
    target_error: torch.Tensor,
    margin: float = 0.12,
) -> torch.Tensor:
    """Force high-error worlds to carry more uncertainty than low-error worlds."""
    if uncertainty_signal.shape[0] < 4:
        return uncertainty_signal.sum() * 0.0
    low_cut = torch.quantile(target_error, 0.25)
    high_cut = torch.quantile(target_error, 0.75)
    low_mask = target_error <= low_cut
    high_mask = target_error >= high_cut
    if not torch.any(low_mask) or not torch.any(high_mask):
        return uncertainty_signal.sum() * 0.0
    low_mean = uncertainty_signal[low_mask].mean()
    high_mean = uncertainty_signal[high_mask].mean()
    return F.relu(margin - (high_mean - low_mean))


def uncertainty_spread_floor_loss(
    uncertainty_signal: torch.Tensor,
    min_std: float = 0.10,
) -> torch.Tensor:
    """Keep uncertainty from collapsing to a near-constant scalar."""
    if uncertainty_signal.numel() < 2:
        return uncertainty_signal.sum() * 0.0
    return F.relu(min_std - torch.std(uncertainty_signal, unbiased=False))


def env_belief_spread_loss(
    env_mean: torch.Tensor,
    min_nearest_distance: float = 0.18,
    min_dim_std: float = 0.08,
) -> torch.Tensor:
    """Prevent env beliefs from collapsing into a microscopic cluster."""
    if env_mean.shape[0] < 2:
        return env_mean.sum() * 0.0
    env_mean = retrieval_safe_normalize(env_mean, dim=-1)
    between = safe_pairwise_distance(env_mean)
    self_mask = torch.eye(env_mean.shape[0], dtype=torch.bool, device=env_mean.device)
    between = between.masked_fill(self_mask, float("inf"))
    nearest = between.min(dim=1).values
    nearest_floor_loss = F.relu(min_nearest_distance - nearest).mean()
    dim_std = torch.std(env_mean, dim=0, unbiased=False)
    dim_floor_loss = F.relu(min_dim_std - dim_std).mean()
    return nearest_floor_loss + 0.5 * dim_floor_loss


def env_uniformity_loss(
    env_mean: torch.Tensor,
    temperature: float = 3.0,
    min_mean_distance: float = 0.45,
) -> torch.Tensor:
    """Encourage a globally spread env-belief cloud instead of a tiny codebook."""
    if env_mean.shape[0] < 2:
        return env_mean.sum() * 0.0
    env_mean = retrieval_safe_normalize(env_mean, dim=-1)
    pairwise = safe_pairwise_distance(env_mean)
    self_mask = torch.eye(env_mean.shape[0], dtype=torch.bool, device=env_mean.device)
    valid_distance = pairwise[~self_mask]
    if valid_distance.numel() == 0:
        return env_mean.sum() * 0.0
    repulsion = torch.exp(-temperature * valid_distance.pow(2)).mean()
    mean_distance_floor = F.relu(min_mean_distance - valid_distance.mean())
    return repulsion + 0.5 * mean_distance_floor


def vicreg_variance_covariance_loss(
    values: torch.Tensor,
    min_std: float = 0.35,
) -> torch.Tensor:
    """Encourage dimension usage and reduce redundant collapsed coordinates."""
    if values.shape[0] < 2:
        return values.sum() * 0.0
    values = sanitize_tensor(values)
    centered = values - values.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = F.relu(min_std - std).mean()
    normalized = centered / std.clamp_min(1e-4)
    covariance = (normalized.T @ normalized) / max(normalized.shape[0] - 1, 1)
    off_diag = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = off_diag.pow(2).mean()
    return variance_loss + 0.05 * covariance_loss


def gaussian_moment_regularizer(
    values: torch.Tensor,
    *,
    target_std: float = 1.0,
    covariance_weight: float = 0.05,
) -> torch.Tensor:
    """Encourage one batch of embeddings to look like a centered Gaussian cloud."""
    if values.shape[0] < 2:
        return values.sum() * 0.0
    values = sanitize_tensor(values)
    batch_mean = values.mean(dim=0)
    centered = values - batch_mean.unsqueeze(0)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    normalized = centered / std.clamp_min(1e-4)
    covariance = (normalized.T @ normalized) / max(normalized.shape[0] - 1, 1)
    off_diag = covariance - torch.diag(torch.diag(covariance))
    mean_loss = batch_mean.pow(2).mean()
    std_loss = (std - float(target_std)).pow(2).mean()
    covariance_loss = off_diag.pow(2).mean()
    return mean_loss + std_loss + float(covariance_weight) * covariance_loss
