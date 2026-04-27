"""Env-belief aggregation pass for belief-model training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ...crawler.predictive import group_window_targets_torch, masked_group_average_torch
from .belief_common import sanitize_tensor
from .belief_losses import (
    correlation_alignment_loss,
    env_belief_spread_loss,
    env_param_anchor_loss,
    env_uniformity_loss,
    gaussian_moment_regularizer,
    hard_negative_retrieval_loss,
    info_nce_loss,
    pairwise_env_geometry_loss,
    retrieval_safe_normalize,
    split_gap_ratio_loss,
    split_geometry_stats,
    split_retrieval_margin_deficit,
    standardize_1d,
    subset_retrieval_loss,
    uncertainty_ranking_loss,
    uncertainty_separation_loss,
    uncertainty_spread_floor_loss,
    vicreg_variance_covariance_loss,
    within_between_env_loss,
)
from .belief_training_env_config import (
    build_env_belief_phase_config,
    build_primary_env_loss_terms,
    cap_primary_env_loss_terms,
)
from .belief_training_common import modules_are_finite, sanitize_modules_
from ..env_belief import (
    build_leave_one_group_out_masks,
    build_split_source_mask,
    build_support_budget_mask,
    build_uncertainty_feature_tensor,
    compute_disjoint_support_splits,
    compute_support_group_stats,
    group_window_latents_torch,
)


def dominant_family_index(group_ids: torch.Tensor, mask: torch.Tensor, num_families: int) -> torch.Tensor:
    """Return the dominant probe family per env row under one support mask."""
    family_counts = []
    for family_idx in range(num_families):
        family_counts.append(((group_ids == family_idx).float() * mask).sum(dim=1))
    return torch.stack(family_counts, dim=1).argmax(dim=1)


def build_family_value_context(
    env_mean: torch.Tensor,
    env_param_std: torch.Tensor,
    env_future_prediction_error: torch.Tensor,
    env_mechanics_posterior_entropy: torch.Tensor,
    support_group_ratio: torch.Tensor,
) -> torch.Tensor:
    """Build one family-scoring context vector from the current env belief."""
    return torch.cat(
        [
            env_mean,
            env_param_std.mean(dim=1, keepdim=True),
            env_future_prediction_error.unsqueeze(1),
            env_mechanics_posterior_entropy.unsqueeze(1),
            support_group_ratio.unsqueeze(1),
        ],
        dim=1,
    )


def build_family_mask_tensor(
    grouped_probe_mode_idx: torch.Tensor,
    grouped_mask: torch.Tensor,
    num_families: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build family-specific masks and validity flags for grouped env windows."""
    family_masks = []
    family_valid = []
    for family_idx in range(num_families):
        family_mask = grouped_mask * (grouped_probe_mode_idx == family_idx).float()
        family_masks.append(family_mask)
        family_valid.append((family_mask.sum(dim=1) > 0).float())
    return torch.stack(family_masks, dim=1), torch.stack(family_valid, dim=1)


def describe_dominant_loss_term(loss_terms: dict[str, torch.Tensor]) -> tuple[str, float]:
    """Return the largest finite weighted loss term for quick debugging."""
    dominant_name = "unknown"
    dominant_value = 0.0
    for name, value in loss_terms.items():
        scalar = float(value.detach().item())
        if not torch.isfinite(value):
            return name, scalar
        if abs(scalar) >= abs(dominant_value):
            dominant_name = name
            dominant_value = scalar
    return dominant_name, dominant_value


def split_retrieval_stats(
    split_mean_a: torch.Tensor,
    split_mean_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-1 and MRR retrieval stats for two matched env split views."""
    if split_mean_a.shape[0] == 0:
        zero = split_mean_a.new_tensor(0.0)
        return zero, zero
    norm_a = sanitize_tensor(retrieval_safe_normalize(split_mean_a, dim=-1))
    norm_b = sanitize_tensor(retrieval_safe_normalize(split_mean_b, dim=-1))
    similarities = torch.matmul(norm_a, norm_b.T)
    ranking = torch.argsort(similarities, dim=1, descending=True)
    target_index = torch.arange(similarities.shape[0], device=similarities.device)
    top1 = (ranking[:, 0] == target_index).float().mean()
    match_positions = (ranking == target_index.unsqueeze(1)).float().argmax(dim=1) + 1
    mrr = torch.reciprocal(match_positions.float()).mean()
    return sanitize_tensor(top1), sanitize_tensor(mrr)


def control_score(
    return_prediction: torch.Tensor,
    risk_prediction: torch.Tensor,
) -> torch.Tensor:
    """Shared cheap-control score used to supervise ranking-oriented belief heads."""
    return sanitize_tensor(return_prediction - 2.0 * risk_prediction)


def count_nonfinite_grads(train_modules: list[nn.Module]) -> int:
    """Count parameter gradients that contain NaN/Inf values."""
    count = 0
    for module in train_modules:
        for param in module.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                count += 1
    return count


def count_nonfinite_params(train_modules: list[nn.Module]) -> int:
    """Count parameter tensors that contain NaN/Inf values."""
    count = 0
    for module in train_modules:
        for param in module.parameters():
            if not torch.isfinite(param).all():
                count += 1
    return count


def run_env_belief_update(
    *,
    epoch_index: int,
    total_epochs: int,
    encoder,
    belief_aggregator,
    env_param_predictor,
    env_future_predictor,
    env_family_future_predictor,
    family_value_predictor,
    belief_message_projector,
    controller_context_projector,
    oracle_context_projector,
    controller_trust_predictor,
    env_metric_projector,
    env_mode_adversary,
    affordance_predictor,
    decision_predictor,
    return_predictor,
    risk_predictor,
    optimizer: torch.optim.Optimizer,
    train_modules: list[nn.Module],
    window_states: torch.Tensor,
    window_actions: torch.Tensor,
    window_rewards: torch.Tensor,
    env_instance_id: torch.Tensor,
    probe_mode_idx: torch.Tensor,
    target_env_params: torch.Tensor,
    target_affordances: torch.Tensor,
    target_decision: torch.Tensor,
    target_return: torch.Tensor,
    target_risk: torch.Tensor,
    target_future_summary: torch.Tensor,
    belief_subset_count: int,
    belief_subset_size: int,
    max_grad_norm: float,
    physics_loss_weight: float,
    env_consistency_loss_weight: float,
    contrastive_loss_weight: float,
    env_geometry_loss_weight: float,
    env_within_between_loss_weight: float,
    env_retrieval_loss_weight: float,
) -> dict[str, float]:
    """Train the env-level belief heads using aggregated window posteriors."""
    device = window_states.device
    mse_loss = nn.MSELoss()

    env_mean_full, env_logvar_full = encoder.encode_posterior(window_states, window_actions, rewards=window_rewards)
    env_mean_full = sanitize_tensor(env_mean_full)
    env_logvar_full = sanitize_tensor(env_logvar_full)
    env_group_tensors_torch = group_window_latents_torch(
        window_mean=env_mean_full,
        window_logvar=env_logvar_full,
        env_instance_id=env_instance_id,
        env_params=target_env_params,
        view_group_id=probe_mode_idx,
    )
    grouped_mean = env_group_tensors_torch["window_mean"]
    grouped_logvar = env_group_tensors_torch["window_logvar"]
    grouped_mask = env_group_tensors_torch["mask"]
    env_target_env_params = env_group_tensors_torch["env_params"]
    grouped_probe_mode_idx = env_group_tensors_torch["view_group_id"]
    grouped_future_summary = group_window_targets_torch(target_future_summary, env_instance_id)
    grouped_affordances = group_window_targets_torch(target_affordances, env_instance_id)
    grouped_decision = group_window_targets_torch(target_decision, env_instance_id)
    grouped_return = group_window_targets_torch(target_return, env_instance_id)
    grouped_risk = group_window_targets_torch(target_risk, env_instance_id)

    support_mask = build_support_budget_mask(
        mask=grouped_mask,
        support_size=max(1, belief_subset_size),
        subset_count=max(1, belief_subset_count),
        group_ids=grouped_probe_mode_idx,
    )
    _support_group_count, support_group_ratio = compute_support_group_stats(support_mask, grouped_probe_mode_idx)

    env_stats = belief_aggregator.aggregate_stats(grouped_mean, grouped_logvar, support_mask, grouped_probe_mode_idx)
    env_mean = sanitize_tensor(env_stats["env_mean_raw"])
    env_logvar = sanitize_tensor(env_stats["env_logvar"])
    env_view_spread = sanitize_tensor(env_stats["view_spread"])
    env_mechanics_posterior_mean = sanitize_tensor(env_stats["mechanics_posterior_mean"])
    env_mechanics_posterior_std = sanitize_tensor(env_stats["mechanics_posterior_std"])
    env_mechanics_posterior_entropy = sanitize_tensor(env_stats["mechanics_posterior_entropy"])
    env_param_preds = sanitize_tensor(env_param_predictor.predict_all(env_mean))
    env_param_mean = env_param_preds.mean(dim=0)
    env_param_std = env_param_preds.std(dim=0, unbiased=False)
    env_view_spread_mean = env_view_spread.mean(dim=1)
    env_param_loss = torch.stack(
        [mse_loss(env_param_preds[member_idx], env_target_env_params) for member_idx in range(env_param_predictor.ensemble_size)],
        dim=0,
    ).mean()
    mechanics_posterior_var = env_mechanics_posterior_std.pow(2).clamp_min(1e-4)
    mechanics_posterior_loss = torch.mean(
        0.5
        * (
            (env_target_env_params - env_mechanics_posterior_mean).pow(2) / mechanics_posterior_var
            + torch.log(mechanics_posterior_var)
        )
    )

    split_payload = compute_disjoint_support_splits(
        aggregator=belief_aggregator,
        grouped_mean=grouped_mean,
        grouped_logvar=grouped_logvar,
        support_mask=build_split_source_mask(grouped_mask, support_mask),
        group_ids=grouped_probe_mode_idx,
        env_param_predictor=env_param_predictor,
    )
    env_mean_a = sanitize_tensor(split_payload["env_mean"][:, 0, :])
    env_mean_b = sanitize_tensor(split_payload["env_mean"][:, 1, :])
    env_logvar_a = sanitize_tensor(split_payload["env_logvar"][:, 0, :])
    env_logvar_b = sanitize_tensor(split_payload["env_logvar"][:, 1, :])
    split_mechanics_mean = sanitize_tensor(split_payload["mechanics_posterior_mean"])
    split_mechanics_std = sanitize_tensor(split_payload["mechanics_posterior_std"])
    env_split_loss = (
        mse_loss(env_mean_a, env_mean_b)
        + 0.25 * mse_loss(torch.exp(0.5 * env_logvar_a), torch.exp(0.5 * env_logvar_b))
        + 0.35 * mse_loss(split_mechanics_mean[:, 0, :], split_mechanics_mean[:, 1, :])
        + 0.15 * mse_loss(split_mechanics_std[:, 0, :], split_mechanics_std[:, 1, :])
    )
    env_metric_mean = sanitize_tensor(env_metric_projector.project_raw(env_mean))
    env_metric_mean_a = sanitize_tensor(env_metric_projector.project_raw(env_mean_a))
    env_metric_mean_b = sanitize_tensor(env_metric_projector.project_raw(env_mean_b))
    env_metric_mean_unit = sanitize_tensor(retrieval_safe_normalize(env_metric_mean, dim=-1))
    env_metric_mean_unit_a = sanitize_tensor(retrieval_safe_normalize(env_metric_mean_a, dim=-1))
    env_metric_mean_unit_b = sanitize_tensor(retrieval_safe_normalize(env_metric_mean_b, dim=-1))
    env_split_contrastive_loss = info_nce_loss(env_metric_projector(env_mean_a), env_metric_projector(env_mean_b))
    env_retrieval_loss = 0.5 * (
        subset_retrieval_loss(env_metric_mean_a, env_metric_mean_b)
        + subset_retrieval_loss(env_metric_mean_b, env_metric_mean_a)
    ) + 0.5 * (
        hard_negative_retrieval_loss(env_metric_mean_a, env_metric_mean_b, env_target_env_params)
        + hard_negative_retrieval_loss(env_metric_mean_b, env_metric_mean_a, env_target_env_params)
    )
    raw_env_retrieval_loss = 0.5 * (
        subset_retrieval_loss(env_mean_a, env_mean_b, temperature=0.10)
        + subset_retrieval_loss(env_mean_b, env_mean_a, temperature=0.10)
    ) + 0.5 * (
        hard_negative_retrieval_loss(env_mean_a, env_mean_b, env_target_env_params, margin=0.20)
        + hard_negative_retrieval_loss(env_mean_b, env_mean_a, env_target_env_params, margin=0.20)
    )
    env_gap_stats = split_geometry_stats(env_metric_mean, env_metric_mean_a, env_metric_mean_b)
    env_gap_ratio_loss = split_gap_ratio_loss(env_metric_mean, env_metric_mean_a, env_metric_mean_b)
    env_unit_gap_ratio_loss = split_gap_ratio_loss(
        env_metric_mean_unit,
        env_metric_mean_unit_a,
        env_metric_mean_unit_b,
    )
    retrieval_margin_deficit = split_retrieval_margin_deficit(env_metric_mean_a, env_metric_mean_b)
    env_retrieval_margin_loss = retrieval_margin_deficit.mean()
    env_unit_retrieval_margin_loss = split_retrieval_margin_deficit(
        env_metric_mean_unit_a,
        env_metric_mean_unit_b,
    ).mean()
    raw_env_retrieval_margin_loss = split_retrieval_margin_deficit(env_mean_a, env_mean_b, margin=0.24).mean()
    heldout_mask = torch.clamp(grouped_mask - support_mask, min=0.0)
    env_future_target, _heldout_count = masked_group_average_torch(
        grouped_future_summary,
        heldout_mask,
        fallback_mask=grouped_mask,
    )
    env_affordance_target, _heldout_affordance_count = masked_group_average_torch(
        grouped_affordances,
        heldout_mask,
        fallback_mask=grouped_mask,
    )
    env_decision_target, _heldout_decision_count = masked_group_average_torch(
        grouped_decision,
        heldout_mask,
        fallback_mask=grouped_mask,
    )
    env_return_target, _heldout_return_count = masked_group_average_torch(
        grouped_return,
        heldout_mask,
        fallback_mask=grouped_mask,
    )
    env_risk_target, _heldout_risk_count = masked_group_average_torch(
        grouped_risk,
        heldout_mask,
        fallback_mask=grouped_mask,
    )
    split_mask_a = split_payload["mask"][:, 0, :]
    split_mask_b = split_payload["mask"][:, 1, :]
    split_target_a, _split_count_a = masked_group_average_torch(grouped_future_summary, split_mask_a, fallback_mask=grouped_mask)
    split_target_b, _split_count_b = masked_group_average_torch(grouped_future_summary, split_mask_b, fallback_mask=grouped_mask)
    split_return_target_a, _split_return_count_a = masked_group_average_torch(
        grouped_return,
        split_mask_a,
        fallback_mask=grouped_mask,
    )
    split_return_target_b, _split_return_count_b = masked_group_average_torch(
        grouped_return,
        split_mask_b,
        fallback_mask=grouped_mask,
    )
    split_risk_target_a, _split_risk_count_a = masked_group_average_torch(
        grouped_risk,
        split_mask_a,
        fallback_mask=grouped_mask,
    )
    split_risk_target_b, _split_risk_count_b = masked_group_average_torch(
        grouped_risk,
        split_mask_b,
        fallback_mask=grouped_mask,
    )
    env_future_pred = sanitize_tensor(env_future_predictor(env_mean))
    env_future_pred_a = sanitize_tensor(env_future_predictor(env_mean_a))
    env_future_pred_b = sanitize_tensor(env_future_predictor(env_mean_b))
    env_future_loss = (
        mse_loss(env_future_pred, env_future_target)
        + 0.5 * mse_loss(env_future_pred_a, split_target_b)
        + 0.5 * mse_loss(env_future_pred_b, split_target_a)
    )
    env_future_prediction_error = sanitize_tensor(torch.mean(torch.abs(env_future_pred - env_future_target), dim=1))
    split_future_prediction_error = sanitize_tensor(
        0.5
        * (
            torch.mean(torch.abs(env_future_pred_a - split_target_b), dim=1)
            + torch.mean(torch.abs(env_future_pred_b - split_target_a), dim=1)
        )
    )
    num_families = int(env_family_future_predictor.num_families)
    family_masks, family_valid = build_family_mask_tensor(grouped_probe_mode_idx, grouped_mask, num_families)
    family_future_shift_target = torch.zeros(
        (grouped_mean.shape[0], num_families),
        dtype=torch.float32,
        device=device,
    )
    family_param_error_target = torch.zeros_like(family_future_shift_target)
    family_belief_shift_target = torch.zeros_like(family_future_shift_target)
    for family_idx in range(num_families):
        family_mask = family_masks[:, family_idx, :]
        family_valid_mask = family_valid[:, family_idx] > 0
        family_future_target, _family_count = masked_group_average_torch(
            grouped_future_summary,
            family_mask,
            fallback_mask=family_mask,
        )
        family_future_shift_target[:, family_idx] = torch.mean(
            torch.abs(family_future_target - env_future_target),
            dim=1,
        )
        family_stats = belief_aggregator.aggregate_stats(
            grouped_mean,
            grouped_logvar,
            family_mask,
            grouped_probe_mode_idx,
        )
        family_env_mean = sanitize_tensor(family_stats["env_mean_raw"])
        family_belief_shift_target[:, family_idx] = torch.linalg.norm(family_env_mean - env_mean, dim=1)
        family_param_mean = env_param_predictor.predict_all(family_env_mean).mean(dim=0)
        family_param_error_target[:, family_idx] = torch.mean(
            torch.abs(family_param_mean - env_target_env_params),
            dim=1,
        )
        family_future_shift_target[:, family_idx] = torch.where(
            family_valid_mask,
            family_future_shift_target[:, family_idx],
            torch.zeros_like(family_future_shift_target[:, family_idx]),
        )
        family_param_error_target[:, family_idx] = torch.where(
            family_valid_mask,
            family_param_error_target[:, family_idx],
            torch.zeros_like(family_param_error_target[:, family_idx]),
        )
        family_belief_shift_target[:, family_idx] = torch.where(
            family_valid_mask,
            family_belief_shift_target[:, family_idx],
            torch.zeros_like(family_belief_shift_target[:, family_idx]),
        )

    flat_family_idx = grouped_probe_mode_idx.reshape(-1)
    flat_family_valid = (grouped_mask.reshape(-1) > 0) & (flat_family_idx >= 0)
    env_family_future_loss = env_future_loss.new_tensor(0.0)
    if torch.any(flat_family_valid):
        repeated_env_mean = env_mean.unsqueeze(1).expand(-1, grouped_mean.shape[1], -1).reshape(-1, env_mean.shape[-1])
        flat_future_summary = grouped_future_summary.reshape(-1, grouped_future_summary.shape[-1])
        valid_family_idx = flat_family_idx[flat_family_valid]
        family_target = flat_future_summary[flat_family_valid]
        family_pred = sanitize_tensor(env_family_future_predictor(repeated_env_mean[flat_family_valid], valid_family_idx))
        env_family_future_loss = mse_loss(family_pred, family_target)

    subset_env_mean = sanitize_tensor(split_payload["env_mean"])
    subset_env_param_mean = sanitize_tensor(split_payload["env_param_mean"])
    subset_param_disagreement = sanitize_tensor(split_payload["env_param_disagreement"])
    subset_latent_disagreement = sanitize_tensor(split_payload["latent_disagreement"])
    subset_metric_env_mean = sanitize_tensor(
        env_metric_projector.project_raw(subset_env_mean.reshape(-1, subset_env_mean.shape[-1])).reshape(
            subset_env_mean.shape[0],
            subset_env_mean.shape[1],
            -1,
        )
    )
    subset_env_shift = torch.linalg.norm(subset_env_mean - env_mean.unsqueeze(1), dim=-1).mean(dim=1)
    split_prediction_error = sanitize_tensor(
        torch.mean(torch.abs(subset_env_param_mean - env_target_env_params.unsqueeze(1)), dim=(1, 2))
    )
    split_env_param_loss = mse_loss(
        subset_env_param_mean,
        env_target_env_params.unsqueeze(1).expand_as(subset_env_param_mean),
    )

    leave_masks, leave_valid = build_leave_one_group_out_masks(support_mask, grouped_probe_mode_idx)
    leaveout_param_std = torch.zeros_like(env_param_mean)
    leaveout_shift = torch.zeros((grouped_mean.shape[0],), dtype=torch.float32, device=device)
    leaveout_prediction_error = torch.zeros((grouped_mean.shape[0],), dtype=torch.float32, device=device)
    leaveout_future_prediction_error = torch.zeros(
        (grouped_mean.shape[0],),
        dtype=torch.float32,
        device=device,
    )
    env_leaveout_future_loss = env_future_loss.new_tensor(0.0)
    if leave_masks.shape[1] > 0 and torch.any(leave_valid > 0):
        batch_size_env, leave_count, max_views = leave_masks.shape
        latent_dim = grouped_mean.shape[-1]
        repeated_mean = grouped_mean[:, None, :, :].expand(-1, leave_count, -1, -1)
        repeated_logvar = grouped_logvar[:, None, :, :].expand(-1, leave_count, -1, -1)
        leave_stats = belief_aggregator.aggregate_stats(
            repeated_mean.reshape(batch_size_env * leave_count, max_views, latent_dim),
            repeated_logvar.reshape(batch_size_env * leave_count, max_views, latent_dim),
            leave_masks.reshape(batch_size_env * leave_count, max_views),
            grouped_probe_mode_idx[:, None, :].expand(-1, leave_count, -1).reshape(batch_size_env * leave_count, max_views),
        )
        leave_env_mean = leave_stats["env_mean_raw"].reshape(batch_size_env, leave_count, latent_dim)
        leave_param_mean = env_param_predictor.predict_all(
            leave_env_mean.reshape(batch_size_env * leave_count, latent_dim)
        ).mean(dim=0).reshape(batch_size_env, leave_count, -1)
        leave_future_pred = sanitize_tensor(
            env_future_predictor(
                leave_env_mean.reshape(batch_size_env * leave_count, latent_dim)
            ).reshape(batch_size_env, leave_count, -1)
        )
        leave_valid_expanded = leave_valid.unsqueeze(-1)
        env_leaveout_future_loss = (
            torch.sum(
                (leave_future_pred - env_future_target.unsqueeze(1)).pow(2)
                * leave_valid_expanded
            )
            / leave_valid_expanded.sum().clamp_min(1.0)
        )
        for env_idx in range(batch_size_env):
            valid_idx = torch.nonzero(leave_valid[env_idx] > 0, as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            leaveout_param_std[env_idx] = leave_param_mean[env_idx, valid_idx].std(dim=0, unbiased=False)
            leaveout_shift[env_idx] = torch.linalg.norm(
                leave_env_mean[env_idx, valid_idx] - env_mean[env_idx].unsqueeze(0),
                dim=-1,
            ).mean()
            leaveout_prediction_error[env_idx] = torch.mean(
                torch.abs(leave_param_mean[env_idx, valid_idx] - env_target_env_params[env_idx].unsqueeze(0))
            )
            leaveout_future_prediction_error[env_idx] = torch.mean(
                torch.abs(leave_future_pred[env_idx, valid_idx] - env_future_target[env_idx].unsqueeze(0))
            )
    env_leaveout_future_loss_raw = sanitize_tensor(env_leaveout_future_loss)
    env_leaveout_future_scale = torch.clamp(
        env_leaveout_future_loss_raw.detach(),
        min=1.0,
        max=6.0,
    )
    env_leaveout_future_loss_objective = torch.clamp(
        env_leaveout_future_loss_raw / env_leaveout_future_scale,
        max=3.0,
    )

    env_subset_consistency_loss = mse_loss(
        subset_metric_env_mean,
        env_metric_mean.unsqueeze(1).expand_as(subset_metric_env_mean),
    )
    env_within_between_loss = within_between_env_loss(env_metric_mean, subset_metric_env_mean, env_target_env_params)
    raw_env_subset_consistency_loss = mse_loss(
        subset_env_mean,
        env_mean.unsqueeze(1).expand_as(subset_env_mean),
    )
    raw_env_within_between_loss = within_between_env_loss(env_mean, subset_env_mean, env_target_env_params)
    env_geometry_belief_loss = pairwise_env_geometry_loss(env_metric_mean, env_target_env_params)
    raw_env_geometry_belief_loss = pairwise_env_geometry_loss(env_mean, env_target_env_params)
    env_anchor_loss = (
        1.25 * env_param_anchor_loss(env_mean, env_target_env_params)
        + env_param_anchor_loss(env_metric_mean, env_target_env_params)
        + 0.75 * env_param_anchor_loss(env_metric_mean_a, env_target_env_params)
        + 0.75 * env_param_anchor_loss(env_metric_mean_b, env_target_env_params)
    )
    env_spread_loss = env_belief_spread_loss(env_metric_mean)
    raw_env_spread_loss = env_belief_spread_loss(env_mean)
    env_uniformity = env_uniformity_loss(env_metric_mean)
    raw_env_uniformity = env_uniformity_loss(env_mean)
    env_vicreg_loss = vicreg_variance_covariance_loss(torch.cat([env_metric_mean, env_metric_mean_a, env_metric_mean_b], dim=0))
    raw_env_vicreg_loss = vicreg_variance_covariance_loss(torch.cat([env_mean, env_mean_a, env_mean_b], dim=0))
    raw_env_gap_ratio_loss = split_gap_ratio_loss(env_mean, env_mean_a, env_mean_b, margin=0.16, target_ratio=0.20)
    env_prediction_error = torch.mean(torch.abs(env_param_mean - env_target_env_params), dim=1)
    leaveout_consistency_loss = leaveout_shift.mean() + 0.35 * leaveout_param_std.mean()
    dominant_family_idx = dominant_family_index(
        grouped_probe_mode_idx,
        support_mask,
        num_families=int(env_mode_adversary.net[-1].out_features),
    )
    env_mode_logits = env_mode_adversary(env_mean, reverse_scale=1.0)
    env_mode_adversary_loss = nn.CrossEntropyLoss()(env_mode_logits, dominant_family_idx)
    env_probe_leakage = (
        (env_mode_logits.argmax(dim=1) == dominant_family_idx).float().mean()
    )
    env_split_retrieval_top1, env_split_retrieval_mrr = split_retrieval_stats(
        env_metric_mean_a,
        env_metric_mean_b,
    )
    phase_config = build_env_belief_phase_config(
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        probe_leakage=float(env_probe_leakage.detach().item()),
        split_retrieval_top1=float(env_split_retrieval_top1.detach().item()),
    )
    uncertainty_features = build_uncertainty_feature_tensor(
        mechanics_posterior_std_mean=env_mechanics_posterior_std.mean(dim=1),
        mechanics_posterior_entropy=env_mechanics_posterior_entropy,
        env_param_std_mean=env_param_std.mean(dim=1),
        split_param_disagreement=subset_param_disagreement,
        split_latent_disagreement=subset_latent_disagreement,
        split_env_shift=subset_env_shift,
        leaveout_param_std_mean=leaveout_param_std.mean(dim=1),
        leaveout_shift=leaveout_shift,
        env_view_spread_mean=env_view_spread_mean,
        support_group_ratio=support_group_ratio,
    )
    uncertainty_signal = sanitize_tensor(belief_aggregator.predict_uncertainty(uncertainty_features))
    uncertainty_target = sanitize_tensor(
        0.20 * env_prediction_error
        + 0.10 * split_prediction_error
        + 0.10 * leaveout_prediction_error
        + 0.25 * env_future_prediction_error
        + 0.20 * split_future_prediction_error
        + 0.10 * leaveout_future_prediction_error
        + 0.05 * env_mechanics_posterior_std.mean(dim=1)
    )
    standardized_uncertainty_signal = standardize_1d(uncertainty_signal)
    standardized_uncertainty_target = standardize_1d(uncertainty_target.detach())
    family_value_context = build_family_value_context(
        env_mean=env_mean.detach(),
        env_param_std=env_param_std.detach(),
        env_future_prediction_error=env_future_prediction_error.detach(),
        env_mechanics_posterior_entropy=env_mechanics_posterior_entropy.detach(),
        support_group_ratio=support_group_ratio.detach(),
    )
    repeated_family_context = family_value_context[:, None, :].expand(-1, num_families, -1).reshape(
        grouped_mean.shape[0] * num_families,
        family_value_context.shape[-1],
    )
    repeated_family_idx = torch.arange(num_families, device=device).unsqueeze(0).expand(grouped_mean.shape[0], -1).reshape(-1)
    family_value_pred = family_value_predictor(repeated_family_context, repeated_family_idx).reshape(
        grouped_mean.shape[0],
        num_families,
        3,
    )
    family_value_target = torch.stack(
        [
            family_param_error_target,
            family_future_shift_target,
            family_belief_shift_target,
        ],
        dim=-1,
    )
    family_valid_expanded = family_valid.unsqueeze(-1)
    family_valid_denom = family_valid.sum().clamp_min(1.0)
    family_value_loss = (
        torch.sum((family_value_pred - family_value_target.detach()).pow(2) * family_valid_expanded)
        / family_valid_expanded.sum().clamp_min(1.0)
    )
    family_belief_consistency_loss = torch.sum(family_belief_shift_target * family_valid) / family_valid_denom
    family_future_consistency_loss = torch.sum(family_future_shift_target * family_valid) / family_valid_denom
    family_param_consistency_loss = torch.sum(family_param_error_target * family_valid) / family_valid_denom
    env_param_support_loss = sanitize_tensor(
        env_param_loss
        + 0.35 * split_env_param_loss
        + 0.20 * family_param_consistency_loss
        + 0.15 * leaveout_prediction_error.mean()
        + 1.20 * env_anchor_loss
    )
    uncertainty_calibration_loss = (
        F.smooth_l1_loss(standardized_uncertainty_signal, standardized_uncertainty_target)
        + 0.70 * uncertainty_ranking_loss(uncertainty_signal, uncertainty_target.detach(), margin=0.04)
        + 0.50 * correlation_alignment_loss(uncertainty_signal, uncertainty_target.detach())
        + 0.35 * uncertainty_separation_loss(uncertainty_signal, uncertainty_target.detach())
        + 0.15 * uncertainty_spread_floor_loss(uncertainty_signal, min_std=0.20)
    )
    belief_message = sanitize_tensor(belief_message_projector(env_mean, uncertainty_signal.unsqueeze(-1)))
    belief_message_env_param = sanitize_tensor(env_param_predictor.predict_all(belief_message))
    belief_message_env_future = sanitize_tensor(env_future_predictor(belief_message))
    env_expression_loss = (
        0.05 * mse_loss(belief_message, env_mean.detach())
        + 0.60 * torch.stack(
            [
                mse_loss(belief_message_env_param[member_idx], env_target_env_params)
                for member_idx in range(env_param_predictor.ensemble_size)
            ],
            dim=0,
        ).mean()
        + 0.35 * mse_loss(belief_message_env_future, env_future_target)
    )
    controller_mechanics_code, controller_affordance_code = controller_context_projector(
        env_mean,
        uncertainty_signal.unsqueeze(-1),
    )
    controller_mechanics_code_a, controller_affordance_code_a = controller_context_projector(
        env_mean_a,
        uncertainty_signal.unsqueeze(-1),
    )
    controller_mechanics_code_b, controller_affordance_code_b = controller_context_projector(
        env_mean_b,
        uncertainty_signal.unsqueeze(-1),
    )
    controller_mechanics_param_preds = sanitize_tensor(
        env_param_predictor.predict_all(controller_mechanics_code)
    )
    controller_mechanics_loss = torch.stack(
        [
            mse_loss(
                controller_mechanics_param_preds[member_idx],
                env_target_env_params,
            )
            for member_idx in range(env_param_predictor.ensemble_size)
        ],
        dim=0,
    ).mean() + 0.10 * mse_loss(controller_mechanics_code, env_mean.detach())
    controller_affordance_future = sanitize_tensor(
        env_future_predictor(controller_affordance_code)
    )
    controller_affordance_future_a = sanitize_tensor(
        env_future_predictor(controller_affordance_code_a)
    )
    controller_affordance_future_b = sanitize_tensor(
        env_future_predictor(controller_affordance_code_b)
    )
    controller_return_pred = sanitize_tensor(return_predictor(controller_affordance_code))
    controller_return_pred_a = sanitize_tensor(return_predictor(controller_affordance_code_a))
    controller_return_pred_b = sanitize_tensor(return_predictor(controller_affordance_code_b))
    controller_risk_pred = sanitize_tensor(risk_predictor(controller_affordance_code))
    controller_risk_pred_a = sanitize_tensor(risk_predictor(controller_affordance_code_a))
    controller_risk_pred_b = sanitize_tensor(risk_predictor(controller_affordance_code_b))
    oracle_mechanics_code, oracle_affordance_code = oracle_context_projector(
        env_target_env_params
    )
    oracle_mechanics_param_preds = sanitize_tensor(
        env_param_predictor.predict_all(oracle_mechanics_code)
    )
    oracle_mechanics_loss = torch.stack(
        [
            mse_loss(
                oracle_mechanics_param_preds[member_idx],
                env_target_env_params,
            )
            for member_idx in range(env_param_predictor.ensemble_size)
        ],
        dim=0,
    ).mean()
    oracle_affordance_future = sanitize_tensor(
        env_future_predictor(oracle_affordance_code)
    )
    oracle_return_pred = sanitize_tensor(return_predictor(oracle_affordance_code))
    oracle_risk_pred = sanitize_tensor(risk_predictor(oracle_affordance_code))
    oracle_affordance_loss = (
        0.35 * mse_loss(oracle_affordance_future, env_future_target)
        + 0.25 * mse_loss(affordance_predictor(oracle_affordance_code), env_affordance_target)
        + 0.15 * mse_loss(decision_predictor(oracle_affordance_code), env_decision_target)
        + 0.15 * mse_loss(oracle_return_pred, env_return_target)
        + 0.10 * mse_loss(oracle_risk_pred, env_risk_target)
    )
    mechanics_distill_mse = mse_loss(controller_mechanics_code, oracle_mechanics_code.detach())
    affordance_distill_mse = mse_loss(controller_affordance_code, oracle_affordance_code.detach())
    mechanics_distill_cosine = (
        1.0
        - F.cosine_similarity(
            controller_mechanics_code,
            oracle_mechanics_code.detach(),
            dim=-1,
            eps=1e-6,
        ).mean()
    )
    affordance_distill_cosine = (
        1.0
        - F.cosine_similarity(
            controller_affordance_code,
            oracle_affordance_code.detach(),
            dim=-1,
            eps=1e-6,
        ).mean()
    )
    controller_oracle_distill_loss = (
        mechanics_distill_mse
        + affordance_distill_mse
        + mechanics_distill_cosine
        + affordance_distill_cosine
    )
    mechanics_return_pred = sanitize_tensor(return_predictor(controller_mechanics_code))
    mechanics_risk_pred = sanitize_tensor(risk_predictor(controller_mechanics_code))
    controller_score_target = control_score(env_return_target, env_risk_target)
    controller_split_score_target_a = control_score(split_return_target_a, split_risk_target_a)
    controller_split_score_target_b = control_score(split_return_target_b, split_risk_target_b)
    controller_score_pred = control_score(controller_return_pred, controller_risk_pred)
    controller_score_pred_a = control_score(controller_return_pred_a, controller_risk_pred_a)
    controller_score_pred_b = control_score(controller_return_pred_b, controller_risk_pred_b)
    oracle_score_pred = control_score(oracle_return_pred, oracle_risk_pred)
    mechanics_score_pred = control_score(mechanics_return_pred, mechanics_risk_pred)
    controller_score_loss = (
        mse_loss(controller_score_pred, controller_score_target)
        + 0.50 * mse_loss(controller_score_pred_a, controller_split_score_target_b)
        + 0.50 * mse_loss(controller_score_pred_b, controller_split_score_target_a)
    )
    controller_score_consistency_loss = (
        0.50 * mse_loss(controller_score_pred_a, controller_score_pred_b)
        + 0.30 * mse_loss(controller_score_pred, oracle_score_pred.detach())
        + 0.20
        * (
            1.0
            - F.cosine_similarity(
                controller_affordance_code_a,
                controller_affordance_code_b,
                dim=-1,
                eps=1e-6,
            ).mean()
        )
    )
    controller_teacher_mismatch = (
        torch.mean(torch.abs(controller_mechanics_code - oracle_mechanics_code.detach()), dim=1)
        + torch.mean(torch.abs(controller_affordance_code - oracle_affordance_code.detach()), dim=1)
    )
    mechanics_score_error = torch.abs(
        mechanics_score_pred.detach() - oracle_score_pred.detach()
    ).squeeze(-1)
    controller_score_error = torch.abs(
        controller_score_pred.detach() - oracle_score_pred.detach()
    ).squeeze(-1)
    teacher_gain = torch.clamp(mechanics_score_error - controller_score_error, min=0.0)
    teacher_margin = torch.abs(
        oracle_score_pred.detach() - mechanics_score_pred.detach()
    ).squeeze(-1)
    trust_gate = torch.sigmoid(4.0 * (teacher_margin - 0.10))
    mismatch_gate = torch.sigmoid(2.0 * (0.75 - controller_teacher_mismatch.detach()))
    controller_trust_target = sanitize_tensor(
        torch.clamp(
            (teacher_gain / teacher_margin.clamp_min(1e-4)) * trust_gate * mismatch_gate,
            0.0,
            1.0,
        )
    )
    controller_trust_pred = sanitize_tensor(
        controller_trust_predictor(env_mean, uncertainty_signal.unsqueeze(-1)).squeeze(-1)
    )
    controller_trust_loss = F.smooth_l1_loss(controller_trust_pred, controller_trust_target)
    controller_affordance_loss = (
        0.30
        * (
            mse_loss(controller_affordance_future, env_future_target)
            + 0.50 * mse_loss(controller_affordance_future_a, split_target_b)
            + 0.50 * mse_loss(controller_affordance_future_b, split_target_a)
        )
        + 0.25 * mse_loss(affordance_predictor(controller_affordance_code), env_affordance_target)
        + 0.20 * mse_loss(decision_predictor(controller_affordance_code), env_decision_target)
        + 0.15 * mse_loss(controller_return_pred, env_return_target)
        + 0.10 * mse_loss(controller_risk_pred, env_risk_target)
        + 0.20 * controller_score_loss
        + 0.05 * mse_loss(controller_affordance_code, env_mean.detach())
    )
    controller_successor_loss = (
        0.50 * mse_loss(controller_affordance_future_a, split_target_b)
        + 0.50 * mse_loss(controller_affordance_future_b, split_target_a)
        + 0.10 * mse_loss(controller_affordance_code_a, controller_affordance_code_b)
        + 0.20 * controller_score_consistency_loss
    )
    env_gaussian_loss = gaussian_moment_regularizer(
        torch.cat(
            [
                env_mean,
                controller_mechanics_code,
                controller_affordance_code,
            ],
            dim=0,
        )
    )

    env_loss_terms = build_primary_env_loss_terms(
        phase=phase_config,
        physics_loss_weight=physics_loss_weight,
        env_consistency_loss_weight=env_consistency_loss_weight,
        contrastive_loss_weight=contrastive_loss_weight,
        env_retrieval_loss_weight=env_retrieval_loss_weight,
        env_param_loss=env_param_support_loss,
        mechanics_posterior_loss=mechanics_posterior_loss,
        env_split_loss=env_split_loss + 0.35 * env_subset_consistency_loss,
        env_future_loss=env_future_loss,
        env_family_future_loss=env_family_future_loss,
        env_leaveout_future_loss=env_leaveout_future_loss_objective,
        env_leaveout_consistency_loss=leaveout_consistency_loss,
        env_split_contrastive_loss=env_split_contrastive_loss,
        env_retrieval_loss=env_retrieval_loss,
        raw_env_retrieval_loss=raw_env_retrieval_loss,
        env_retrieval_margin_loss=env_retrieval_margin_loss,
        raw_env_retrieval_margin_loss=raw_env_retrieval_margin_loss,
        env_gap_ratio_loss=env_gap_ratio_loss,
        raw_env_gap_ratio_loss=raw_env_gap_ratio_loss,
        env_unit_gap_ratio_loss=env_unit_gap_ratio_loss,
        env_unit_retrieval_margin_loss=env_unit_retrieval_margin_loss,
        env_metric_geometry_loss=env_geometry_belief_loss + 0.80 * env_anchor_loss,
        env_spread_loss=env_spread_loss,
        raw_env_spread_loss=raw_env_spread_loss,
        env_uniformity_loss=env_uniformity,
        raw_env_uniformity_loss=raw_env_uniformity,
        env_vicreg_loss=env_vicreg_loss,
        raw_env_vicreg_loss=raw_env_vicreg_loss,
        env_mode_adversary_loss=env_mode_adversary_loss,
        uncertainty_calibration_loss=uncertainty_calibration_loss,
        env_expression_loss=env_expression_loss,
        controller_mechanics_loss=controller_mechanics_loss,
        controller_affordance_loss=controller_affordance_loss,
        controller_successor_loss=controller_successor_loss,
        oracle_mechanics_loss=oracle_mechanics_loss,
        oracle_affordance_loss=oracle_affordance_loss,
        controller_oracle_distill_loss=controller_oracle_distill_loss,
        controller_trust_loss=controller_trust_loss,
        env_gaussian_loss=env_gaussian_loss,
    )
    env_loss_terms = cap_primary_env_loss_terms(
        env_loss_terms,
        max_term_fraction=0.15,
    )
    env_loss = sanitize_tensor(sum(env_loss_terms.values()))
    conservative_env_loss = sanitize_tensor(
        env_loss_terms["physics"]
        + env_loss_terms["env_consistency"]
        + env_loss_terms["env_future"]
        + env_loss_terms["env_family_future"]
        + env_loss_terms["env_leaveout_future"]
        + env_loss_terms["env_leaveout_consistency"]
        + env_loss_terms["raw_env_retrieval"]
        + env_loss_terms["env_retrieval_margin"]
        + env_loss_terms["raw_env_retrieval_margin"]
        + env_loss_terms["env_gap_ratio"]
        + env_loss_terms["raw_env_gap_ratio"]
        + env_loss_terms["env_unit_gap_ratio"]
        + env_loss_terms["env_unit_retrieval_margin"]
        + env_loss_terms["env_metric_geometry"]
        + env_loss_terms["env_spread"]
        + env_loss_terms["raw_env_spread"]
        + env_loss_terms["env_uniformity"]
        + env_loss_terms["raw_env_uniformity"]
        + env_loss_terms["env_vicreg"]
        + env_loss_terms["raw_env_vicreg"]
        + env_loss_terms["env_leakage_control"]
        + env_loss_terms["env_expression"]
        + env_loss_terms["controller_mechanics"]
        + env_loss_terms["controller_affordance"]
        + env_loss_terms["controller_successor"]
        + env_loss_terms["oracle_mechanics"]
        + env_loss_terms["oracle_affordance"]
        + env_loss_terms["controller_oracle_distill"]
        + env_loss_terms["controller_trust"]
        + env_loss_terms["env_gaussian"]
        + env_loss_terms["uncertainty_calibration"]
    )
    core_env_loss = sanitize_tensor(
        physics_loss_weight * (env_param_support_loss + 0.75 * mechanics_posterior_loss)
        + env_consistency_loss_weight * env_split_loss
        + phase_config.predictive_scale * 0.26 * env_future_loss
        + phase_config.predictive_scale * 0.16 * env_leaveout_future_loss_objective
        + phase_config.predictive_scale * 0.08 * leaveout_consistency_loss
        + phase_config.env_expression_scale * 0.16 * env_expression_loss
        + phase_config.controller_scale * 0.05 * controller_mechanics_loss
        + phase_config.controller_scale * 0.06 * controller_affordance_loss
        + phase_config.controller_scale * 0.05 * controller_successor_loss
        + phase_config.controller_scale * 0.03 * oracle_mechanics_loss
        + phase_config.controller_scale * 0.04 * oracle_affordance_loss
        + phase_config.controller_scale * 0.04 * controller_oracle_distill_loss
        + phase_config.controller_scale * 0.03 * controller_trust_loss
        + phase_config.controller_scale * 0.02 * env_gaussian_loss
        + phase_config.metric_scale * 0.06 * env_gap_ratio_loss
        + phase_config.metric_scale * 0.08 * env_unit_gap_ratio_loss
        + phase_config.metric_scale * 0.07 * env_unit_retrieval_margin_loss
        + phase_config.metric_scale * 0.04 * env_retrieval_margin_loss
        + phase_config.metric_scale * 0.04 * raw_env_retrieval_margin_loss
        + phase_config.metric_scale * 0.04 * raw_env_gap_ratio_loss
        + phase_config.metric_scale * 0.04 * (env_geometry_belief_loss + 0.80 * env_anchor_loss)
        + phase_config.metric_scale * 0.04 * env_spread_loss
        + phase_config.metric_scale * 0.03 * raw_env_spread_loss
        + phase_config.metric_scale * 0.04 * env_vicreg_loss
        + phase_config.metric_scale * 0.03 * raw_env_vicreg_loss
        + phase_config.metric_scale * 0.04 * env_mode_adversary_loss
    )
    dominant_env_term_name, dominant_env_term_value = describe_dominant_loss_term(env_loss_terms)
    nonfinite_env_terms = ",".join(
        name for name, value in env_loss_terms.items() if not torch.isfinite(value)
    )

    skipped_env_updates = 0
    env_skip_reason = "ok"
    env_skip_detail = ""
    env_step_mode = "full"
    env_grad_norm_value = 0.0
    nonfinite_env_grad_count = 0
    nonfinite_env_param_count = 0
    if (
        not torch.isfinite(env_loss)
        and not torch.isfinite(conservative_env_loss)
        and not torch.isfinite(core_env_loss)
    ):
        skipped_env_updates = 1
        env_skip_reason = "loss_nonfinite"
        env_skip_detail = nonfinite_env_terms or dominant_env_term_name
        optimizer.zero_grad(set_to_none=True)
        sanitize_modules_(train_modules)
    else:
        optimizer.zero_grad(set_to_none=True)
        backward_loss = env_loss if torch.isfinite(env_loss) else conservative_env_loss
        backward_loss.backward(retain_graph=True)
        grad_norm = clip_grad_norm_(
            [param for module in train_modules for param in module.parameters()],
            min(float(max_grad_norm), float(phase_config.grad_clip_norm)),
        )
        env_grad_norm_value = float(torch.as_tensor(grad_norm).item())
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            fallback_loss = conservative_env_loss if torch.isfinite(conservative_env_loss) else core_env_loss
            fallback_mode = "fallback_conservative"
            if fallback_loss is core_env_loss:
                fallback_mode = "fallback_core"
            fallback_loss.backward(retain_graph=torch.isfinite(conservative_env_loss) and fallback_loss is conservative_env_loss)
            grad_norm = clip_grad_norm_(
                [param for module in train_modules for param in module.parameters()],
                min(float(max_grad_norm), float(phase_config.grad_clip_norm)),
            )
            env_grad_norm_value = float(torch.as_tensor(grad_norm).item())
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                core_env_loss.backward()
                grad_norm = clip_grad_norm_(
                    [param for module in train_modules for param in module.parameters()],
                    min(float(max_grad_norm), float(phase_config.grad_clip_norm)),
                )
                env_grad_norm_value = float(torch.as_tensor(grad_norm).item())
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    skipped_env_updates = 1
                    env_skip_reason = "grad_norm_nonfinite"
                    nonfinite_env_grad_count = count_nonfinite_grads(train_modules)
                    env_skip_detail = f"dominant={dominant_env_term_name};mode={env_step_mode}"
                    sanitize_modules_(train_modules)
                    optimizer.zero_grad(set_to_none=True)
                else:
                    env_step_mode = "fallback_core"
                    env_skip_reason = "fallback_core"
                    env_skip_detail = f"dominant={dominant_env_term_name}"
                    optimizer.step()
            else:
                env_step_mode = fallback_mode
                env_skip_reason = fallback_mode
                env_skip_detail = f"dominant={dominant_env_term_name}"
                optimizer.step()
        else:
            optimizer.step()
            if not modules_are_finite(train_modules):
                skipped_env_updates = 1
                env_skip_reason = "params_nonfinite_after_step"
                nonfinite_env_param_count = count_nonfinite_params(train_modules)
                env_skip_detail = f"dominant={dominant_env_term_name};mode={env_step_mode}"
                sanitize_modules_(train_modules)
                optimizer.zero_grad(set_to_none=True)

    return {
        "env_loss_total": float(env_loss.item()),
        "env_conservative_loss_total": float(conservative_env_loss.item()),
        "env_core_loss_total": float(core_env_loss.item()),
        "env_grad_norm": float(env_grad_norm_value),
        "env_phase_name": phase_config.name,
        "env_predictive_phase_scale": float(phase_config.predictive_scale),
        "env_metric_phase_scale": float(phase_config.metric_scale),
        "env_expression_phase_scale": float(phase_config.env_expression_scale),
        "env_controller_phase_scale": float(phase_config.controller_scale),
        "env_metric_gate_active": float(1.0 if phase_config.metric_gate_active else 0.0),
        "env_metric_gate_reason": phase_config.metric_gate_reason,
        "env_metric_warmup": float(phase_config.metric_scale),
        "env_retrieval_warmup": float(phase_config.metric_scale),
        "env_retrieval_pressure": float(phase_config.metric_scale),
        "env_step_mode": env_step_mode,
        "env_param_loss": float(env_param_loss.item()),
        "env_param_support_loss": float(env_param_support_loss.item()),
        "env_split_loss": float(env_split_loss.item()),
        "env_split_contrastive_loss": float(env_split_contrastive_loss.item()),
        "env_retrieval_loss": float(env_retrieval_loss.item()),
        "raw_env_retrieval_loss": float(raw_env_retrieval_loss.item()),
        "env_gap_ratio_loss": float(env_gap_ratio_loss.item()),
        "env_unit_gap_ratio_loss": float(env_unit_gap_ratio_loss.item()),
        "env_split_retrieval_top1": float(env_split_retrieval_top1.item()),
        "env_split_retrieval_mrr": float(env_split_retrieval_mrr.item()),
        "env_metric_geometry_loss": float(env_geometry_belief_loss.item()),
        "env_param_anchor_loss": float(env_anchor_loss.item()),
        "env_probe_leakage": float(env_probe_leakage.item()),
        "env_leakage_control_term": float(env_loss_terms["env_leakage_control"].item()),
        "env_spread_loss": float(env_spread_loss.item()),
        "env_uniformity_loss": float(env_uniformity.item()),
        "env_vicreg_loss": float(env_vicreg_loss.item()),
        "env_retrieval_margin_loss": float(env_retrieval_margin_loss.item()),
        "env_unit_retrieval_margin_loss": float(env_unit_retrieval_margin_loss.item()),
        "raw_env_retrieval_margin_loss": float(raw_env_retrieval_margin_loss.item()),
        "env_future_loss": float(env_future_loss.item()),
        "env_family_future_loss": float(env_family_future_loss.item()),
        "env_leaveout_future_loss": float(env_leaveout_future_loss_raw.item()),
        "env_leaveout_future_objective_loss": float(env_leaveout_future_loss_objective.item()),
        "family_value_loss": float(family_value_loss.item()),
        "family_belief_consistency_loss": float(family_belief_consistency_loss.item()),
        "family_future_consistency_loss": float(family_future_consistency_loss.item()),
        "family_param_consistency_loss": float(family_param_consistency_loss.item()),
        "belief_message_loss": float(env_expression_loss.item()),
        "env_expression_loss": float(env_expression_loss.item()),
        "controller_mechanics_loss": float(controller_mechanics_loss.item()),
        "controller_affordance_loss": float(controller_affordance_loss.item()),
        "controller_successor_loss": float(controller_successor_loss.item()),
        "controller_score_loss": float(controller_score_loss.item()),
        "controller_score_consistency_loss": float(controller_score_consistency_loss.item()),
        "oracle_mechanics_loss": float(oracle_mechanics_loss.item()),
        "oracle_affordance_loss": float(oracle_affordance_loss.item()),
        "controller_oracle_distill_loss": float(controller_oracle_distill_loss.item()),
        "controller_trust_loss": float(controller_trust_loss.item()),
        "controller_trust_target_mean": float(controller_trust_target.mean().item()),
        "env_gaussian_loss": float(env_gaussian_loss.item()),
        "uncertainty_calibration_loss": float(uncertainty_calibration_loss.item()),
        "env_within_between_loss": float(env_within_between_loss.item()),
        "env_mode_adversary_loss": float(env_mode_adversary_loss.item()),
        "skipped_env_updates": float(skipped_env_updates),
        "env_skip_reason": env_skip_reason,
        "env_skip_detail": env_skip_detail,
        "env_skip_nonfinite_terms": nonfinite_env_terms,
        "env_dominant_term_name": dominant_env_term_name,
        "env_dominant_term_value": float(dominant_env_term_value),
        "env_nonfinite_grad_count": float(nonfinite_env_grad_count),
        "env_nonfinite_param_count": float(nonfinite_env_param_count),
        "env_gap_within": float(env_gap_stats["same_gap"].mean().item()),
        "env_gap_between": float(env_gap_stats["nearest_between"].mean().item()),
    }
