"""Latent snapshot artifact payload builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...envs import get_env_display_name
from ...cognition.representation import load_latent_snapshot
from ..diagnostics import (
    compute_failure_lift,
    compute_linear_env_fit,
    compute_mode_leakage,
    compute_neighbor_env_alignment,
    compute_per_param_env_fit,
    compute_same_env_gap_ratio,
    compute_same_env_spread,
    compute_split_retrieval_stats,
    compute_uncertainty_error_alignment,
    downsample_indices,
    mode_payload_rows,
)
from .common import (
    build_support_validity_payload,
    load_optional_string,
    normalize_projection_2d,
)


def build_latent_payload(path: Path) -> dict:
    """Convert one latent snapshot artifact into a JSON-friendly payload."""
    snapshot = load_latent_snapshot(path)
    env_name = load_optional_string(snapshot, "env_name")
    benchmark_tag = load_optional_string(snapshot, "benchmark_tag")
    indices = downsample_indices(int(snapshot["env_belief_mean"].shape[0]))
    projection = normalize_projection_2d(snapshot["projection_2d"])[indices]
    uncertainty = snapshot["env_uncertainty"][indices]
    env_params = snapshot["env_params"][indices]
    env_window_count = snapshot["env_window_count"][indices]
    env_support_count = snapshot.get("env_support_count", snapshot["env_window_count"])[indices]
    env_support_available_count = snapshot.get(
        "env_support_available_count",
        snapshot["env_window_count"],
    )[indices]
    env_heldout_count = snapshot.get(
        "env_heldout_count",
        np.clip(env_support_available_count - env_support_count, 0, None),
    )[indices]
    env_support_group_ratio = snapshot.get(
        "env_support_group_ratio",
        np.ones_like(snapshot.get("env_support_count", snapshot["env_window_count"]), dtype=np.float32),
    )[indices]
    env_split_group_overlap = snapshot.get(
        "env_split_group_overlap",
        np.zeros_like(snapshot.get("env_support_count", snapshot["env_window_count"]), dtype=np.float32),
    )[indices]
    env_split_balanced_half = snapshot.get(
        "env_split_balanced_half",
        np.zeros_like(snapshot.get("env_support_count", snapshot["env_window_count"]), dtype=np.float32),
    )[indices]
    full_predictive_env_mean = snapshot["env_belief_mean"].astype(np.float32)
    full_env_mean = snapshot.get("env_metric_mean", full_predictive_env_mean).astype(np.float32)
    full_env_mean_unit = snapshot.get(
        "env_metric_mean_unit",
        snapshot.get("env_belief_mean_unit", full_env_mean),
    ).astype(np.float32)
    full_env_params = snapshot["env_params"].astype(np.float32)
    full_uncertainty = snapshot["env_uncertainty"].astype(np.float32)
    full_env_belief_norm = snapshot.get(
        "env_belief_norm",
        np.linalg.norm(full_predictive_env_mean, axis=1).astype(np.float32),
    ).astype(np.float32)
    full_env_param_error = snapshot.get(
        "env_param_error_mean",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_mechanics_posterior_std = snapshot.get(
        "env_mechanics_posterior_std",
        np.zeros((full_env_mean.shape[0], 0), dtype=np.float32),
    ).astype(np.float32)
    full_mechanics_posterior_entropy = snapshot.get(
        "env_mechanics_posterior_entropy",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_future_probe_error = snapshot.get(
        "env_future_prediction_error",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_split_mean_a = snapshot.get("env_metric_split_mean_a", snapshot.get("env_split_mean_a", full_env_mean)).astype(np.float32)
    full_split_mean_b = snapshot.get("env_metric_split_mean_b", snapshot.get("env_split_mean_b", full_env_mean)).astype(np.float32)
    full_cross_split_mean_a = snapshot.get(
        "env_cross_family_metric_split_mean_a",
        full_split_mean_a,
    ).astype(np.float32)
    full_cross_split_mean_b = snapshot.get(
        "env_cross_family_metric_split_mean_b",
        full_split_mean_b,
    ).astype(np.float32)
    full_split_latent_disagreement = snapshot.get(
        "env_split_latent_disagreement",
        np.linalg.norm(full_split_mean_a - full_split_mean_b, axis=1).astype(np.float32),
    ).astype(np.float32)
    full_nearest_between_distance = snapshot.get(
        "env_nearest_between_distance",
        np.ones((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_gap_ratio = snapshot.get(
        "env_gap_ratio",
        full_split_latent_disagreement / np.clip(full_nearest_between_distance, 1e-6, None),
    ).astype(np.float32)
    full_split_retrieval_margin_deficit = snapshot.get(
        "env_split_retrieval_margin_deficit",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_pairwise_between_distance = snapshot.get("env_pairwise_between_distance")
    if full_pairwise_between_distance is None:
        pairwise_between = np.linalg.norm(
            full_env_mean[:, None, :] - full_env_mean[None, :, :],
            axis=-1,
        ).astype(np.float32)
        full_pairwise_between_distance = pairwise_between[np.triu_indices(pairwise_between.shape[0], k=1)]
    full_pairwise_between_distance = np.asarray(full_pairwise_between_distance, dtype=np.float32)
    full_pairwise_between_distance_unit = snapshot.get("env_pairwise_between_distance_unit")
    if full_pairwise_between_distance_unit is None:
        pairwise_between_unit = np.linalg.norm(
            full_env_mean_unit[:, None, :] - full_env_mean_unit[None, :, :],
            axis=-1,
        ).astype(np.float32)
        full_pairwise_between_distance_unit = pairwise_between_unit[np.triu_indices(pairwise_between_unit.shape[0], k=1)]
    full_pairwise_between_distance_unit = np.asarray(full_pairwise_between_distance_unit, dtype=np.float32)
    full_env_view_spread = snapshot.get("env_subset_latent_std", snapshot["env_view_spread"]).astype(np.float32)
    full_env_leaveout_latent_std = snapshot.get(
        "env_leaveout_latent_std",
        np.zeros_like(full_env_view_spread, dtype=np.float32),
    ).astype(np.float32)
    full_env_leaveout_shift = snapshot.get(
        "env_leaveout_shift",
        np.zeros((full_env_view_spread.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_window_probe_mode = snapshot["window_probe_mode"].astype("U")
    full_split_group_overlap = snapshot.get(
        "env_split_group_overlap",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_split_balanced_half = snapshot.get(
        "env_split_balanced_half",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_split_group_count_a = snapshot.get(
        "env_split_group_count_a",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_split_group_count_b = snapshot.get(
        "env_split_group_count_b",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_cross_split_group_overlap = snapshot.get(
        "env_cross_family_split_group_overlap",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_cross_split_group_count_a = snapshot.get(
        "env_cross_family_split_group_count_a",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_cross_split_group_count_b = snapshot.get(
        "env_cross_family_split_group_count_b",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_cross_gap_ratio = snapshot.get(
        "env_cross_family_gap_ratio",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_env_dominant_probe_mode = snapshot.get(
        "env_dominant_probe_mode",
        np.asarray(["unknown"] * full_env_mean.shape[0], dtype="U"),
    ).astype("U")
    full_support_top_family_share = snapshot.get(
        "env_support_top_family_share",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_support_effective_family_count = snapshot.get(
        "env_support_effective_family_count",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_support_family_entropy = snapshot.get(
        "env_support_family_entropy",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_support_tied_top_family_count = snapshot.get(
        "env_support_tied_top_family_count",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_window_terminated = snapshot["window_terminated"].astype(np.float32)
    full_window_latent_mean = snapshot["window_latent_mean"].astype(np.float32)
    full_param_names = snapshot["env_param_names"].astype("U")

    env_ids = snapshot["env_instance_id"].astype(np.int32)
    window_env_ids = snapshot["window_env_instance_id"].astype(np.int32)
    window_reward_sum = snapshot["window_reward_sum"].astype(np.float32)
    env_uncertainty_by_id = {
        int(env_id): float(full_uncertainty[idx])
        for idx, env_id in enumerate(env_ids.tolist())
    }
    full_window_uncertainty = np.asarray(
        [env_uncertainty_by_id.get(int(env_id), 0.0) for env_id in window_env_ids.tolist()],
        dtype=np.float32,
    )
    env_terminated_rate = np.zeros(env_ids.shape[0], dtype=np.float32)
    env_reward_mean = np.zeros(env_ids.shape[0], dtype=np.float32)
    for idx, env_id in enumerate(env_ids.tolist()):
        env_mask = window_env_ids == env_id
        env_terminated_rate[idx] = float(np.mean(full_window_terminated[env_mask])) if np.any(env_mask) else 0.0
        env_reward_mean[idx] = float(np.mean(window_reward_sum[env_mask])) if np.any(env_mask) else 0.0
    terminated = env_terminated_rate[indices]
    reward_sum = env_reward_mean[indices]

    mode_counts = mode_payload_rows(
        probe_mode=full_window_probe_mode,
        uncertainty=full_window_uncertainty,
        terminated=full_window_terminated,
    )
    split_retrieval = compute_split_retrieval_stats(full_split_mean_a, full_split_mean_b)
    cross_split_retrieval = compute_split_retrieval_stats(full_cross_split_mean_a, full_cross_split_mean_b)
    full_split_retrieval_rank = snapshot.get(
        "env_split_retrieval_rank",
        split_retrieval["ranks"],
    )
    full_split_retrieval_rank = np.asarray(full_split_retrieval_rank, dtype=np.int32)
    full_cross_split_retrieval_rank = snapshot.get(
        "env_cross_family_split_retrieval_rank",
        cross_split_retrieval["ranks"],
    )
    full_cross_split_retrieval_rank = np.asarray(full_cross_split_retrieval_rank, dtype=np.int32)
    diagnostics = {
        "linear_env_fit_r2": compute_linear_env_fit(full_predictive_env_mean, full_env_params),
        "per_param_env_fit_r2": compute_per_param_env_fit(full_predictive_env_mean, full_env_params, full_param_names),
        "neighbor_env_alignment": compute_neighbor_env_alignment(full_env_mean, full_env_params),
        "neighbor_env_alignment_unit": compute_neighbor_env_alignment(full_env_mean_unit, full_env_params),
        "split_retrieval_top1": split_retrieval["top1"],
        "split_retrieval_top5": split_retrieval["top5"],
        "split_retrieval_mrr": split_retrieval["mrr"],
        "split_retrieval_median_rank": split_retrieval["median_rank"],
        "cross_family_split_retrieval_top1": cross_split_retrieval["top1"],
        "cross_family_split_retrieval_top5": cross_split_retrieval["top5"],
        "cross_family_split_retrieval_mrr": cross_split_retrieval["mrr"],
        "cross_family_split_retrieval_median_rank": cross_split_retrieval["median_rank"],
        "window_mode_leakage": compute_mode_leakage(full_window_latent_mean, full_window_probe_mode),
        "env_mode_leakage": compute_mode_leakage(full_predictive_env_mean, full_env_dominant_probe_mode),
        "same_env_spread": compute_same_env_spread(
            full_split_latent_disagreement,
            full_env_leaveout_shift,
        ),
        "same_env_gap_ratio": compute_same_env_gap_ratio(full_env_mean, full_split_mean_a, full_split_mean_b),
        "failure_lift": compute_failure_lift(full_uncertainty, env_terminated_rate.astype(np.float32)),
        "uncertainty_error_alignment": compute_uncertainty_error_alignment(
            full_uncertainty,
            full_env_param_error,
        ),
        "env_param_uncertainty_mean": float(
            np.mean(snapshot.get("env_split_param_disagreement", np.zeros((full_env_mean.shape[0],), dtype=np.float32)).astype(np.float32))
            + 0.5 * np.mean(
                snapshot.get(
                    "env_leaveout_param_std",
                    np.zeros_like(snapshot.get("env_param_std", np.zeros((full_env_mean.shape[0], 1), dtype=np.float32)), dtype=np.float32),
                ).astype(np.float32)
            )
        ),
        "support_group_ratio_mean": float(np.mean(
            snapshot.get(
                "env_support_group_ratio",
                np.ones((full_env_mean.shape[0],), dtype=np.float32),
            ).astype(np.float32)
        )),
        "split_group_overlap_mean": float(np.mean(full_split_group_overlap)),
        "cross_family_split_group_overlap_mean": float(np.mean(full_cross_split_group_overlap)),
        "cross_family_gap_ratio_mean": float(np.mean(full_cross_gap_ratio)),
        "split_balanced_half_fraction": float(np.mean(full_split_balanced_half)),
        "split_group_count_a_mean": float(np.mean(full_split_group_count_a)),
        "split_group_count_b_mean": float(np.mean(full_split_group_count_b)),
        "cross_family_split_group_count_a_mean": float(np.mean(full_cross_split_group_count_a)),
        "cross_family_split_group_count_b_mean": float(np.mean(full_cross_split_group_count_b)),
        "support_group_count_mean": float(np.mean(
            snapshot.get(
                "env_support_group_count",
                np.ones((full_env_mean.shape[0],), dtype=np.float32),
            ).astype(np.float32)
        )),
        "support_top_family_share_mean": float(np.mean(full_support_top_family_share)),
        "support_effective_family_count_mean": float(np.mean(full_support_effective_family_count)),
        "support_family_entropy_mean": float(np.mean(full_support_family_entropy)),
        "support_tied_top_family_count_mean": float(np.mean(full_support_tied_top_family_count)),
        "support_available_count_mean": float(np.mean(
            snapshot.get(
                "env_support_available_count",
                snapshot["env_window_count"],
            ).astype(np.float32)
        )),
        "heldout_count_mean": float(np.mean(
            snapshot.get(
                "env_heldout_count",
                np.clip(
                    snapshot["env_window_count"]
                    - snapshot.get("env_support_count", snapshot["env_window_count"]),
                    0,
                    None,
                ),
            ).astype(np.float32)
        )),
        "env_param_error_mean": float(np.mean(full_env_param_error)),
        "future_probe_error_mean": float(np.mean(full_future_probe_error)),
        "mechanics_posterior_std_mean": float(np.mean(full_mechanics_posterior_std)) if full_mechanics_posterior_std.size else 0.0,
        "mechanics_posterior_entropy_mean": float(np.mean(full_mechanics_posterior_entropy)) if full_mechanics_posterior_entropy.size else 0.0,
        "pairwise_between_mean": float(np.mean(full_pairwise_between_distance)) if full_pairwise_between_distance.size else 0.0,
        "pairwise_between_p10": float(np.quantile(full_pairwise_between_distance, 0.10)) if full_pairwise_between_distance.size else 0.0,
        "pairwise_between_mean_unit": float(np.mean(full_pairwise_between_distance_unit)) if full_pairwise_between_distance_unit.size else 0.0,
        "belief_norm_mean": float(np.mean(full_env_belief_norm)),
        "belief_norm_std": float(np.std(full_env_belief_norm)),
        "uncertainty_feature_importance": [
            {
                "name": str(name),
                "weight": float(weight),
            }
            for name, weight in zip(
                snapshot.get("env_uncertainty_feature_names", np.asarray([], dtype="U")).tolist(),
                snapshot.get("env_uncertainty_feature_weights", np.asarray([], dtype=np.float32)).tolist(),
            )
        ],
    }

    points = []
    for idx in range(len(indices)):
        points.append(
            {
                "x": float(projection[idx, 0]),
                "y": float(projection[idx, 1]),
                "reward_sum": float(reward_sum[idx]),
                "uncertainty": float(uncertainty[idx]),
                "window_count": int(env_window_count[idx]),
                "support_count": int(env_support_count[idx]),
                "support_available_count": int(env_support_available_count[idx]),
                "heldout_count": int(env_heldout_count[idx]),
                "support_group_ratio": float(env_support_group_ratio[idx]),
                "split_group_overlap": float(env_split_group_overlap[idx]),
                "split_balanced_half": float(env_split_balanced_half[idx]),
                "terminated": bool(int(terminated[idx])),
                "terminated_numeric": float(terminated[idx]),
                "env_param_mean": float(np.mean(env_params[idx])),
                "env_error": float(full_env_param_error[indices[idx]]),
                "future_probe_error": float(full_future_probe_error[indices[idx]]),
                "mechanics_posterior_entropy": float(full_mechanics_posterior_entropy[indices[idx]]) if full_mechanics_posterior_entropy.size else 0.0,
                "gap_ratio": float(full_gap_ratio[indices[idx]]),
                "nearest_between_distance": float(full_nearest_between_distance[indices[idx]]),
                "belief_norm": float(full_env_belief_norm[indices[idx]]),
                "split_retrieval_margin_deficit": float(full_split_retrieval_margin_deficit[indices[idx]]),
                "same_env_spread": float(
                    full_split_latent_disagreement[indices[idx]]
                    + 0.5 * full_env_leaveout_shift[indices[idx]]
                ),
            }
        )

    window_count_mean = float(snapshot["env_window_count"].mean())
    support_count_mean = float(snapshot.get("env_support_count", snapshot["env_window_count"]).mean())
    support_available_count_mean = float(
        snapshot.get("env_support_available_count", snapshot["env_window_count"]).mean()
    )
    heldout_count_mean = float(
        snapshot.get(
            "env_heldout_count",
            np.clip(
                snapshot["env_window_count"]
                - snapshot.get("env_support_count", snapshot["env_window_count"]),
                0,
                None,
            ),
        ).mean()
    )
    support_group_count_mean = float(np.mean(
        snapshot.get(
            "env_support_group_count",
            np.ones_like(snapshot.get("env_support_count", snapshot["env_window_count"]), dtype=np.float32),
        )
    ))
    support_group_ratio_mean = float(np.mean(
        snapshot.get(
            "env_support_group_ratio",
            np.ones_like(snapshot.get("env_support_count", snapshot["env_window_count"]), dtype=np.float32),
        )
    ))
    support_validity = build_support_validity_payload(
        num_envs=int(snapshot["env_belief_mean"].shape[0]),
        num_windows=int(snapshot["window_latent_mean"].shape[0]),
        window_count_mean=window_count_mean,
        support_count_mean=support_count_mean,
        support_group_count_mean=support_group_count_mean,
        support_group_ratio_mean=support_group_ratio_mean,
        split_group_overlap_mean=float(np.mean(full_split_group_overlap)),
    )
    sysid_payload = {
        "available": bool("sysid_validation_top1" in snapshot),
        "validation_top1": float(np.asarray(snapshot.get("sysid_validation_top1", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]),
        "validation_margin": float(np.asarray(snapshot.get("sysid_validation_margin", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]),
        "validation_nll": float(np.asarray(snapshot.get("sysid_validation_nll", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]),
        "trusted": bool(np.asarray(snapshot.get("sysid_trusted", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0] > 0.5),
        "particle_entropy_mean": float(np.mean(snapshot.get("particle_entropy", np.asarray([], dtype=np.float32)))) if "particle_entropy" in snapshot else 0.0,
        "particle_ess_ratio_mean": float(np.mean(snapshot.get("particle_ess_ratio", np.asarray([], dtype=np.float32)))) if "particle_ess_ratio" in snapshot else 0.0,
        "particle_leaveout_shift_mean": float(np.mean(snapshot.get("particle_leaveout_shift", np.asarray([], dtype=np.float32)))) if "particle_leaveout_shift" in snapshot else 0.0,
    }

    return {
        "name": path.name,
        "artifact_mtime": float(path.stat().st_mtime),
        "env_name": env_name,
        "env_display_name": None if env_name is None else get_env_display_name(env_name),
        "benchmark_tag": benchmark_tag,
        "summary": {
            "num_envs": int(snapshot["env_belief_mean"].shape[0]),
            "num_windows": int(snapshot["window_latent_mean"].shape[0]),
            "latent_dim": int(snapshot["env_belief_mean"].shape[1]),
            "env_param_dim": int(snapshot["env_params"].shape[1]),
            "uncertainty_mean": float(snapshot["env_uncertainty"].mean()),
            "terminated_rate": float(np.mean(env_terminated_rate)),
            "window_count_mean": window_count_mean,
            "support_count_mean": support_count_mean,
            "support_available_count_mean": support_available_count_mean,
            "heldout_count_mean": heldout_count_mean,
            "support_group_count_mean": support_group_count_mean,
            "support_group_ratio_mean": support_group_ratio_mean,
            "support_top_family_share_mean": float(np.mean(full_support_top_family_share)),
            "support_effective_family_count_mean": float(np.mean(full_support_effective_family_count)),
            "support_family_entropy_mean": float(np.mean(full_support_family_entropy)),
            "support_tied_top_family_count_mean": float(np.mean(full_support_tied_top_family_count)),
            "split_group_overlap_mean": float(np.mean(full_split_group_overlap)),
            "cross_family_split_group_overlap_mean": float(np.mean(full_cross_split_group_overlap)),
            "cross_family_split_group_count_a_mean": float(np.mean(full_cross_split_group_count_a)),
            "cross_family_split_group_count_b_mean": float(np.mean(full_cross_split_group_count_b)),
            "cross_family_gap_ratio_mean": float(np.mean(full_cross_gap_ratio)),
            "split_balanced_half_fraction": float(np.mean(full_split_balanced_half)),
            "pca_explained": [float(value) for value in snapshot["pca_explained"].tolist()],
            "sampled_points": int(len(points)),
        },
        "support_validity": support_validity,
        "system_id": sysid_payload,
        "diagnostics": diagnostics,
        "series": {
            "pairwise_between_distance": full_pairwise_between_distance.astype(np.float32),
            "pairwise_between_distance_unit": full_pairwise_between_distance_unit.astype(np.float32),
            "split_retrieval_rank": full_split_retrieval_rank.astype(np.int32),
            "cross_family_split_retrieval_rank": full_cross_split_retrieval_rank.astype(np.int32),
        },
        "compression": {
            "bits": snapshot.get("compression_bits", np.asarray([], dtype=np.int32)).astype(np.int32),
            "mechanics_fit_r2": snapshot.get("compression_mechanics_fit_r2", np.asarray([], dtype=np.float32)).astype(np.float32),
            "split_retrieval_top1": snapshot.get("compression_split_retrieval_top1", np.asarray([], dtype=np.float32)).astype(np.float32),
            "split_retrieval_mrr": snapshot.get("compression_split_retrieval_mrr", np.asarray([], dtype=np.float32)).astype(np.float32),
            "message_norm_mean": snapshot.get("compression_message_norm_mean", np.asarray([], dtype=np.float32)).astype(np.float32),
        },
        "mode_counts": mode_counts,
        "points": points,
    }
