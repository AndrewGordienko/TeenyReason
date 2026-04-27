"""Runtime aggregation helpers for env-level posteriors."""

from __future__ import annotations

import math

import numpy as np
import torch

from .env_belief_models import EnvBeliefAggregator, EnvParamPredictorEnsemble
from .env_belief_subsets import (
    build_leave_one_group_out_masks,
    build_split_source_mask,
    build_support_budget_mask,
    compute_disjoint_support_splits,
    compute_support_group_stats,
)


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
    coverage_penalty_component = np.full((latent_dim,), float(max(0.0, 1.0 - support_diversity_ratio)), dtype=np.float32)

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
    subset_count: int = 1,
    subset_size: int = 6,
    support_size: int = 4,
) -> dict[str, np.ndarray]:
    """Aggregate one env's window posteriors into an env belief plus uncertainty."""
    del subset_size
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
        support_mask_t = build_support_budget_mask(
            mask=mask_t,
            support_size=support_size,
            subset_count=max(1, subset_count),
            group_ids=group_ids_t,
        )
        support_group_count_t, support_group_ratio_t = compute_support_group_stats(support_mask_t, group_ids_t)
        env_stats_t = aggregator.aggregate_stats(mean_t, logvar_t, support_mask_t, group_ids_t)
        env_mean_t = env_stats_t["env_mean"]
        env_mean_raw_t = env_stats_t["env_mean_raw"]
        env_logvar_t = env_stats_t["env_logvar"]
        view_spread_t = env_stats_t["view_spread"]
        factor_mean_t = env_stats_t["factor_mean"]
        factor_std_t = env_stats_t["factor_std"]
        mechanics_posterior_mean_t = env_stats_t["mechanics_posterior_mean"]
        mechanics_posterior_std_t = env_stats_t["mechanics_posterior_std"]
        mechanics_posterior_logvar_t = env_stats_t["mechanics_posterior_logvar"]
        mechanics_posterior_entropy_t = env_stats_t["mechanics_posterior_entropy"]
        split_payload = compute_disjoint_support_splits(
            aggregator=aggregator,
            grouped_mean=mean_t,
            grouped_logvar=logvar_t,
            support_mask=build_split_source_mask(mask_t, support_mask_t),
            group_ids=group_ids_t,
            env_param_predictor=env_param_predictor,
        )
        split_env_mean = split_payload["env_mean"]
        subset_latent_std = split_env_mean.std(dim=1, unbiased=False).squeeze(0).cpu().numpy().astype(np.float32)
        subset_shift = float(torch.linalg.norm(split_env_mean - env_mean_t.unsqueeze(1), dim=-1).mean().item())
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
                None if group_ids_t is None else group_ids_t[:, None, :].expand(-1, leaveout_masks_t.shape[1], -1).reshape(
                    leaveout_masks_t.shape[1],
                    leaveout_masks_t.shape[2],
                ),
            )
            leave_mean_t = leave_stats_t["env_mean_raw"]
            valid_idx = torch.nonzero(leaveout_valid_t.reshape(-1) > 0, as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                leave_mean_t = leave_mean_t[valid_idx]
                leaveout_latent_std = leave_mean_t.std(dim=0, unbiased=False).cpu().numpy().astype(np.float32)
                leaveout_shift = float(torch.linalg.norm(leave_mean_t - env_mean_t.expand_as(leave_mean_t), dim=-1).mean().item())
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
            subset_param_std = split_payload["env_param_mean"].std(dim=1, unbiased=False).squeeze(0).cpu().numpy().astype(np.float32)
            split_param_disagreement = float(split_payload["env_param_disagreement"].mean().item())

    env_mean = env_mean_t.squeeze(0).cpu().numpy().astype(np.float32)
    env_mean_raw = env_mean_raw_t.squeeze(0).cpu().numpy().astype(np.float32)
    env_logvar = env_logvar_t.squeeze(0).cpu().numpy().astype(np.float32)
    view_spread = view_spread_t.squeeze(0).cpu().numpy().astype(np.float32)
    factor_mean = factor_mean_t.squeeze(0).cpu().numpy().astype(np.float32)
    factor_std = factor_std_t.squeeze(0).cpu().numpy().astype(np.float32)
    mechanics_posterior_mean = mechanics_posterior_mean_t.squeeze(0).cpu().numpy().astype(np.float32)
    mechanics_posterior_std = mechanics_posterior_std_t.squeeze(0).cpu().numpy().astype(np.float32)
    mechanics_posterior_logvar = mechanics_posterior_logvar_t.squeeze(0).cpu().numpy().astype(np.float32)
    mechanics_posterior_entropy = mechanics_posterior_entropy_t.cpu().numpy().astype(np.float32)
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
        "factor_mean": factor_mean.astype(np.float32),
        "factor_std": factor_std.astype(np.float32),
        "mechanics_posterior_mean": mechanics_posterior_mean.astype(np.float32),
        "mechanics_posterior_std": mechanics_posterior_std.astype(np.float32),
        "mechanics_posterior_logvar": mechanics_posterior_logvar.astype(np.float32),
        "mechanics_posterior_entropy": mechanics_posterior_entropy.astype(np.float32),
        "subset_latent_std": subset_latent_std.astype(np.float32),
        "subset_param_std": subset_param_std.astype(np.float32),
        "leaveout_latent_std": np.zeros_like(subset_latent_std, dtype=np.float32) if leaveout_latent_std is None else leaveout_latent_std.astype(np.float32),
        "leaveout_param_std": np.zeros_like(subset_param_std, dtype=np.float32) if leaveout_param_std is None else leaveout_param_std.astype(np.float32),
        "subset_shift": np.asarray([subset_shift], dtype=np.float32),
        "split_latent_disagreement": np.asarray([split_latent_disagreement], dtype=np.float32),
        "split_param_disagreement": np.asarray([split_param_disagreement], dtype=np.float32),
        "leaveout_shift": np.asarray([leaveout_shift], dtype=np.float32),
        "support_group_count": support_group_count_t.cpu().numpy().astype(np.float32),
        "support_group_ratio": support_group_ratio_t.cpu().numpy().astype(np.float32),
        "subset_size_used": split_payload["split_count"].min(dim=1).values.cpu().numpy().astype(np.int32),
        "split_group_overlap": split_payload["split_group_overlap"].cpu().numpy().astype(np.float32),
        "split_balanced_half": split_payload["split_balanced_half"].cpu().numpy().astype(np.float32),
        "split_group_count_a": split_payload["split_group_count_a"].cpu().numpy().astype(np.float32),
        "split_group_count_b": split_payload["split_group_count_b"].cpu().numpy().astype(np.float32),
        "support_count": np.asarray([int(support_mask_t.sum().item())], dtype=np.int32),
    }
