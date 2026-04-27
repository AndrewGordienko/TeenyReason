"""Artifact loading and payload building for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..app.config import build_experiment_config
from ..envs import get_env_display_name
from ..representation import list_latent_snapshot_paths, load_latent_snapshot
from .diagnostics import (
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
    summarize_solve_array,
)


def list_benchmark_paths(artifact_dir: Path) -> list[Path]:
    """Find all saved benchmark summary artifacts."""
    return sorted(artifact_dir.glob("*_solve_benchmark.npz"))


def load_benchmark_summary(path: Path) -> dict[str, np.ndarray]:
    """Load one saved benchmark summary artifact."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_optional_json_rows(values: np.ndarray | None) -> list[dict]:
    """Parse an optional string array of JSON rows."""
    if values is None:
        return []
    rows = []
    for item in np.asarray(values).tolist():
        try:
            rows.append(json.loads(str(item)))
        except json.JSONDecodeError:
            rows.append({})
    return rows


def normalize_matched_eval_summary(row: dict | None) -> dict:
    """Normalize one stored matched-eval summary row into a stable shape."""
    row = row if isinstance(row, dict) else {}
    returns = [
        float(value)
        for value in row.get("returns", [])
        if value is not None
    ]
    episode_total_env_steps = [
        int(value)
        for value in row.get("episode_total_env_steps", [])
        if value is not None
    ]
    fixture_count = int(row.get("fixture_count", min(len(returns), len(episode_total_env_steps))))
    solved_count = int(row.get("solved_count", 0))
    mean_return = float(
        row.get(
            "mean_return",
            float(np.mean(np.asarray(returns, dtype=np.float32))) if returns else 0.0,
        )
    )
    mean_total_env_steps = float(
        row.get(
            "mean_total_env_steps",
            float(np.mean(np.asarray(episode_total_env_steps, dtype=np.float32)))
            if episode_total_env_steps
            else 0.0,
        )
    )
    return {
        "returns": returns,
        "episode_total_env_steps": episode_total_env_steps,
        "mean_return": mean_return,
        "mean_total_env_steps": mean_total_env_steps,
        "solved_count": solved_count,
        "fixture_count": fixture_count,
        "available": fixture_count > 0,
    }


def summarize_matched_eval_rows(rows: list[dict]) -> dict:
    """Aggregate matched controller eval rows across seeds."""
    normalized_rows = [normalize_matched_eval_summary(row) for row in rows]
    valid_rows = [row for row in normalized_rows if row["available"]]
    if not valid_rows:
        return {
            "not_run": True,
            "count": 0,
            "mean_return": {"median": 0.0, "mean": 0.0, "count": 0},
            "mean_total_env_steps": {"median": 0.0, "mean": 0.0, "count": 0},
            "solved": 0,
            "fixtures": 0,
            "success_rate": 0.0,
        }
    mean_returns = np.asarray([row["mean_return"] for row in valid_rows], dtype=np.float32)
    mean_steps = np.asarray(
        [row["mean_total_env_steps"] for row in valid_rows],
        dtype=np.float32,
    )
    solved = int(sum(int(row["solved_count"]) for row in valid_rows))
    fixtures = int(sum(int(row["fixture_count"]) for row in valid_rows))
    return {
        "not_run": False,
        "count": len(valid_rows),
        "mean_return": {
            "median": float(np.median(mean_returns)),
            "mean": float(np.mean(mean_returns)),
            "count": int(mean_returns.size),
        },
        "mean_total_env_steps": {
            "median": float(np.median(mean_steps)),
            "mean": float(np.mean(mean_steps)),
            "count": int(mean_steps.size),
        },
        "solved": solved,
        "fixtures": fixtures,
        "success_rate": float(solved) / float(max(fixtures, 1)),
    }


def aggregate_json_counter_rows(rows: list[dict]) -> dict[str, float]:
    """Sum numeric JSON-counter rows into one stable dashboard summary."""
    totals: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            try:
                totals[str(key)] = float(totals.get(str(key), 0.0) + float(value))
            except (TypeError, ValueError):
                continue
    return totals


def average_json_metric_rows(rows: list[dict]) -> dict[str, float]:
    """Average metric-style JSON rows while ignoring missing keys."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            try:
                totals[str(key)] = float(totals.get(str(key), 0.0) + float(value))
            except (TypeError, ValueError):
                continue
            counts[str(key)] = counts.get(str(key), 0) + 1
    return {
        key: float(total) / float(max(counts.get(key, 0), 1))
        for key, total in totals.items()
    }


def aggregate_json_list_rows(rows: list[object]) -> dict[str, float]:
    """Count list-shaped JSON rows after collapsing them into stable labels."""
    totals: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, list):
            continue
        label = " / ".join(str(item) for item in row if str(item))
        if not label:
            continue
        totals[label] = float(totals.get(label, 0.0) + 1.0)
    return totals


def load_benchmark_profile_name(path: Path) -> str | None:
    """Read the benchmark profile stored in one summary artifact."""
    try:
        summary = load_benchmark_summary(path)
    except OSError:
        return None
    return load_optional_string(summary, "benchmark_profile")


def order_benchmark_paths(benchmark_paths: list[Path]) -> list[Path]:
    """Keep non-archived benchmark summaries ahead of archived planner runs."""
    profile_cache = {
        path.name: (load_benchmark_profile_name(path) or "")
        for path in benchmark_paths
    }
    return sorted(
        benchmark_paths,
        key=lambda path: (
            profile_cache.get(path.name) == "archived_planner",
            -float(path.stat().st_mtime),
            path.name,
        ),
    )


def preferred_benchmark_summary_name(
    benchmark_paths: list[Path],
    context: dict | None,
) -> str | None:
    """Prefer the newest non-archived benchmark for the default dashboard headline."""
    if not benchmark_paths:
        return None

    ordered_paths = order_benchmark_paths(benchmark_paths)
    name_to_path = {path.name: path for path in ordered_paths}
    preferred_name = None if context is None else context.get("default_benchmark_summary")
    if isinstance(preferred_name, str) and preferred_name in name_to_path:
        preferred_profile = load_benchmark_profile_name(name_to_path[preferred_name])
        if preferred_profile != "archived_planner":
            return preferred_name

    for path in ordered_paths:
        if load_benchmark_profile_name(path) != "archived_planner":
            return path.name

    if isinstance(preferred_name, str) and preferred_name in name_to_path:
        return preferred_name
    return ordered_paths[0].name


def build_index_payload(artifact_dir: Path) -> dict:
    """List the available dashboard artifacts."""
    context = load_dashboard_context(artifact_dir)
    latent_paths = list_latent_snapshot_paths(artifact_dir)
    benchmark_paths = order_benchmark_paths(list_benchmark_paths(artifact_dir))
    preferred_benchmark = preferred_benchmark_summary_name(benchmark_paths, context)
    if context is not None and preferred_benchmark:
        context = {
            **context,
            "default_benchmark_summary": preferred_benchmark,
        }
    return {
        "artifact_dir": str(artifact_dir),
        "latent_snapshots": [path.name for path in latent_paths],
        "benchmark_summaries": [path.name for path in benchmark_paths],
        "latent_snapshot_mtimes": {
            path.name: float(path.stat().st_mtime)
            for path in latent_paths
        },
        "benchmark_summary_mtimes": {
            path.name: float(path.stat().st_mtime)
            for path in benchmark_paths
        },
        "run_context": context,
    }


def load_dashboard_context(artifact_dir: Path) -> dict | None:
    """Load the most recent training selection written by the benchmark entrypoint."""
    context_path = artifact_dir / "dashboard_context.json"
    if not context_path.exists():
        return load_main_module_context()
    try:
        return json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return load_main_module_context()


def load_main_module_context() -> dict | None:
    """Fallback to the root main.py selection when no saved dashboard context exists yet."""
    try:
        import main as root_main
    except Exception:
        return None

    env_name = getattr(root_main, "ENV_NAME", None)
    seeds = getattr(root_main, "SEEDS", None)
    if not isinstance(env_name, str):
        return None

    try:
        benchmark_tag = build_experiment_config(env_name).benchmark_tag
    except Exception:
        benchmark_tag = env_name.replace("-", "_").replace("/", "_").lower()

    if not isinstance(seeds, list) or not seeds:
        seeds = [0, 1, 2, 3, 4]

    return {
        "env_name": env_name,
        "env_display_name": get_env_display_name(env_name),
        "benchmark_tag": benchmark_tag,
        "default_benchmark_summary": f"{benchmark_tag}_solve_benchmark.npz",
        "default_latent_snapshot": f"{benchmark_tag}_seed_{seeds[-1]}_latent_snapshot.npz",
        "seeds": seeds,
    }


def load_optional_string(data: dict[str, np.ndarray], key: str) -> str | None:
    """Read an optional string-like field stored in an NPZ artifact."""
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def load_array_with_fallback(
    data: dict[str, np.ndarray],
    primary_key: str,
    fallback_key: str,
) -> np.ndarray:
    """Read one required array while staying compatible with older artifact keys."""
    value = data.get(primary_key)
    if value is not None:
        return value
    return data[fallback_key]


def build_support_validity_payload(
    *,
    num_envs: int,
    num_windows: int,
    window_count_mean: float,
    support_count_mean: float,
    support_group_count_mean: float,
    support_group_ratio_mean: float,
    split_group_overlap_mean: float,
) -> dict:
    """Describe whether a latent snapshot has enough support to trust split metrics."""
    reasons: list[str] = []
    affected_metrics = [
        "mechanics fit",
        "split retrieval",
        "same-env spread",
        "gap ratio",
    ]

    if num_envs > 0 and num_windows <= num_envs:
        reasons.append("the artifact has at most one saved window per env belief")
    if window_count_mean < 2.0:
        reasons.append("window coverage per env is below two views")
    if support_count_mean < 2.0:
        reasons.append("support coverage per env is below two windows")
    if support_count_mean > 6.0:
        reasons.append("canonical support budget is being exceeded")
    # Named mechanics probes deliberately put different families on opposite
    # split halves. That makes retrieval harder, but it is a stronger test of
    # whether the belief encodes the world instead of the probe style.
    strict_cross_family_split = (
        support_group_count_mean >= 4.0
        and support_group_ratio_mean >= 0.95
        and split_group_overlap_mean <= 0.25
    )
    paired_support = support_group_count_mean >= 4.0 and split_group_overlap_mean >= 0.75
    if support_group_ratio_mean < 0.60:
        reasons.append("support diversity across probe families is narrow")

    if reasons:
        return {
            "status": "invalid",
            "is_valid": False,
            "headline": "Snapshot structurally undercovered",
            "detail": (
                f"This artifact averages {window_count_mean:.1f} windows and "
                f"{support_count_mean:.1f} support windows per env, so several "
                "representation metrics can look artificially strong."
            ),
            "reasons": reasons,
            "affected_metrics": affected_metrics,
        }

    fragile_reasons: list[str] = []
    if window_count_mean < 4.0:
        fragile_reasons.append("window coverage is still thin")
    if support_count_mean < 4.0:
        fragile_reasons.append("support subsets are still small")
    if support_group_ratio_mean < 0.85 and not paired_support:
        fragile_reasons.append("support families are only partly diverse")
    if split_group_overlap_mean < 0.75 and not strict_cross_family_split:
        fragile_reasons.append("split halves only partly overlap by probe family")

    if fragile_reasons:
        return {
            "status": "fragile",
            "is_valid": True,
            "headline": "Snapshot coverage still thin",
            "detail": (
                f"This artifact averages {window_count_mean:.1f} windows and "
                f"{support_count_mean:.1f} support windows per env. The geometry "
                "is usable, but split-based diagnostics should still be treated cautiously."
            ),
            "reasons": fragile_reasons,
            "affected_metrics": affected_metrics,
        }

    return {
        "status": "ok",
        "is_valid": True,
        "headline": "Snapshot coverage looks healthy",
        "detail": (
            f"This artifact averages {window_count_mean:.1f} windows and "
            f"{support_count_mean:.1f} support windows per env, with broad probe-family coverage."
        ),
        "reasons": [],
        "affected_metrics": affected_metrics,
    }


def build_latent_payload(path: Path) -> dict:
    """Convert one latent snapshot artifact into a JSON-friendly payload."""
    snapshot = load_latent_snapshot(path)
    env_name = load_optional_string(snapshot, "env_name")
    benchmark_tag = load_optional_string(snapshot, "benchmark_tag")
    indices = downsample_indices(int(snapshot["env_belief_mean"].shape[0]))
    projection = snapshot["projection_2d"][indices]
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
    full_cross_gap_ratio = snapshot.get(
        "env_cross_family_gap_ratio",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_env_dominant_probe_mode = snapshot.get(
        "env_dominant_probe_mode",
        np.asarray(["unknown"] * full_env_mean.shape[0], dtype="U"),
    ).astype("U")
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
        "support_group_count_mean": float(np.mean(
            snapshot.get(
                "env_support_group_count",
                np.ones((full_env_mean.shape[0],), dtype=np.float32),
            ).astype(np.float32)
        )),
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
            "split_group_overlap_mean": float(np.mean(full_split_group_overlap)),
            "cross_family_split_group_overlap_mean": float(np.mean(full_cross_split_group_overlap)),
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


def build_benchmark_payload(path: Path) -> dict:
    """Convert one benchmark summary artifact into a JSON-friendly payload."""
    summary = load_benchmark_summary(path)
    env_name = load_optional_string(summary, "env_name")
    benchmark_profile = load_optional_string(summary, "benchmark_profile")
    seeds = summary["seeds"].astype(np.int64).tolist()
    baseline_episode_solves = load_array_with_fallback(
        summary,
        "baseline_episode_solves",
        "baseline_solves",
    ).astype(np.int64)
    probe_episode_solves = load_array_with_fallback(
        summary,
        "probe_episode_solves",
        "probe_solves",
    ).astype(np.int64)
    probe_shadow_episode_solves = summary.get(
        "probe_shadow_episode_solves",
        summary.get("probe_shadow_solves", probe_episode_solves),
    ).astype(np.int64)
    probe_no_expression_episode_solves = summary.get(
        "probe_no_expression_episode_solves",
        summary.get(
            "probe_no_expression_solves",
            np.full_like(probe_episode_solves, -1),
        ),
    ).astype(np.int64)
    full_system_episode_solves = summary.get(
        "full_system_episode_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    full_system_state_only_episode_solves = summary.get(
        "full_system_state_only_episode_solves",
        np.full_like(full_system_episode_solves, -1),
    ).astype(np.int64)
    full_system_oracle_episode_solves = summary.get(
        "full_system_oracle_episode_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    sim_fanout_episode_solves = summary.get(
        "sim_fanout_episode_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    baseline_step_solves = summary.get(
        "baseline_step_solves",
        np.full_like(baseline_episode_solves, -1),
    ).astype(np.int64)
    probe_step_solves = summary.get(
        "probe_step_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    probe_shadow_step_solves = summary.get(
        "probe_shadow_step_solves",
        np.full_like(probe_shadow_episode_solves, -1),
    ).astype(np.int64)
    probe_no_expression_step_solves = summary.get(
        "probe_no_expression_step_solves",
        np.full_like(probe_no_expression_episode_solves, -1),
    ).astype(np.int64)
    full_system_step_solves = summary.get(
        "full_system_step_solves",
        np.full_like(full_system_episode_solves, -1),
    ).astype(np.int64)
    full_system_state_only_step_solves = summary.get(
        "full_system_state_only_step_solves",
        np.full_like(full_system_state_only_episode_solves, -1),
    ).astype(np.int64)
    full_system_oracle_step_solves = summary.get(
        "full_system_oracle_step_solves",
        np.full_like(full_system_oracle_episode_solves, -1),
    ).astype(np.int64)
    sim_fanout_step_solves = summary.get(
        "sim_fanout_step_solves",
        np.full_like(sim_fanout_episode_solves, -1),
    ).astype(np.int64)
    baseline_total_env_steps = summary.get(
        "baseline_total_env_steps",
        np.full_like(baseline_episode_solves, 0),
    ).astype(np.int64)
    probe_total_env_steps = summary.get(
        "probe_total_env_steps",
        np.full_like(probe_episode_solves, 0),
    ).astype(np.int64)
    probe_shadow_total_env_steps = summary.get(
        "probe_shadow_total_env_steps",
        np.full_like(probe_shadow_episode_solves, 0),
    ).astype(np.int64)
    probe_no_expression_total_env_steps = summary.get(
        "probe_no_expression_total_env_steps",
        np.full_like(probe_no_expression_episode_solves, 0),
    ).astype(np.int64)
    full_system_total_env_steps = summary.get(
        "full_system_total_env_steps",
        np.full_like(full_system_episode_solves, 0),
    ).astype(np.int64)
    full_system_state_only_total_env_steps = summary.get(
        "full_system_state_only_total_env_steps",
        np.full_like(full_system_state_only_episode_solves, 0),
    ).astype(np.int64)
    full_system_oracle_total_env_steps = summary.get(
        "full_system_oracle_total_env_steps",
        np.full_like(full_system_oracle_episode_solves, 0),
    ).astype(np.int64)
    sim_fanout_total_env_steps = summary.get(
        "sim_fanout_total_env_steps",
        np.full_like(sim_fanout_episode_solves, 0),
    ).astype(np.int64)
    baseline_completed_episodes = summary.get(
        "baseline_completed_episodes",
        np.full_like(baseline_episode_solves, 0),
    ).astype(np.int64)
    probe_completed_episodes = summary.get(
        "probe_completed_episodes",
        np.full_like(probe_episode_solves, 0),
    ).astype(np.int64)
    probe_shadow_completed_episodes = summary.get(
        "probe_shadow_completed_episodes",
        np.full_like(probe_shadow_episode_solves, 0),
    ).astype(np.int64)
    probe_no_expression_completed_episodes = summary.get(
        "probe_no_expression_completed_episodes",
        np.full_like(probe_no_expression_episode_solves, 0),
    ).astype(np.int64)
    probe_no_expression_available = summary.get(
        "probe_no_expression_available",
        (probe_no_expression_completed_episodes > 0).astype(np.int8),
    ).astype(np.int8)
    latent_claim_valid = summary.get(
        "latent_claim_valid",
        np.zeros_like(probe_episode_solves, dtype=np.int8),
    ).astype(np.int8)
    full_system_completed_episodes = summary.get(
        "full_system_completed_episodes",
        np.full_like(full_system_episode_solves, 0),
    ).astype(np.int64)
    full_system_state_only_completed_episodes = summary.get(
        "full_system_state_only_completed_episodes",
        np.full_like(full_system_state_only_episode_solves, 0),
    ).astype(np.int64)
    full_system_oracle_completed_episodes = summary.get(
        "full_system_oracle_completed_episodes",
        np.full_like(full_system_oracle_episode_solves, 0),
    ).astype(np.int64)
    sim_fanout_completed_episodes = summary.get(
        "sim_fanout_completed_episodes",
        np.full_like(sim_fanout_episode_solves, 0),
    ).astype(np.int64)
    full_system_controller_style = summary.get(
        "full_system_controller_style",
        np.asarray([""] * len(full_system_episode_solves), dtype="U"),
    ).astype("U")
    full_system_oracle_controller_style = summary.get(
        "full_system_oracle_controller_style",
        np.asarray([""] * len(full_system_oracle_episode_solves), dtype="U"),
    ).astype("U")
    sim_fanout_controller_style = summary.get(
        "sim_fanout_controller_style",
        np.asarray([""] * len(sim_fanout_episode_solves), dtype="U"),
    ).astype("U")
    probe_encoder_steps = summary.get(
        "probe_encoder_steps",
        np.zeros_like(probe_episode_solves),
    ).astype(np.int64)
    baseline_control_env_steps = summary.get(
        "baseline_control_env_steps",
        baseline_total_env_steps,
    ).astype(np.int64)
    probe_probe_env_steps = summary.get(
        "probe_probe_env_steps",
        probe_total_env_steps,
    ).astype(np.int64)
    probe_control_env_steps = summary.get(
        "probe_control_env_steps",
        probe_total_env_steps,
    ).astype(np.int64)
    probe_shadow_probe_env_steps = summary.get(
        "probe_shadow_probe_env_steps",
        probe_shadow_total_env_steps,
    ).astype(np.int64)
    probe_shadow_control_env_steps = summary.get(
        "probe_shadow_control_env_steps",
        probe_shadow_total_env_steps,
    ).astype(np.int64)
    probe_post_expression_env_steps = summary.get(
        "probe_post_expression_env_steps",
        probe_control_env_steps,
    ).astype(np.int64)
    probe_shadow_post_expression_env_steps = summary.get(
        "probe_shadow_post_expression_env_steps",
        probe_shadow_control_env_steps,
    ).astype(np.int64)
    probe_post_expression_episodes = summary.get(
        "probe_post_expression_episodes",
        probe_episode_solves,
    ).astype(np.int64)
    probe_shadow_post_expression_episodes = summary.get(
        "probe_shadow_post_expression_episodes",
        probe_shadow_episode_solves,
    ).astype(np.int64)
    probe_no_expression_probe_env_steps = summary.get(
        "probe_no_expression_probe_env_steps",
        probe_no_expression_total_env_steps,
    ).astype(np.int64)
    probe_no_expression_control_env_steps = summary.get(
        "probe_no_expression_control_env_steps",
        probe_no_expression_total_env_steps,
    ).astype(np.int64)
    probe_no_expression_post_expression_env_steps = summary.get(
        "probe_no_expression_post_expression_env_steps",
        probe_no_expression_control_env_steps,
    ).astype(np.int64)
    probe_no_expression_post_expression_episodes = summary.get(
        "probe_no_expression_post_expression_episodes",
        probe_no_expression_episode_solves,
    ).astype(np.int64)
    full_system_probe_env_steps = summary.get(
        "full_system_probe_env_steps",
        full_system_total_env_steps,
    ).astype(np.int64)
    full_system_control_env_steps = summary.get(
        "full_system_control_env_steps",
        full_system_total_env_steps,
    ).astype(np.int64)
    full_system_post_context_env_steps = summary.get(
        "full_system_post_context_env_steps",
        full_system_control_env_steps,
    ).astype(np.int64)
    full_system_post_context_episodes = summary.get(
        "full_system_post_context_episodes",
        full_system_episode_solves,
    ).astype(np.int64)
    full_system_oracle_probe_env_steps = summary.get(
        "full_system_oracle_probe_env_steps",
        full_system_oracle_total_env_steps,
    ).astype(np.int64)
    full_system_oracle_control_env_steps = summary.get(
        "full_system_oracle_control_env_steps",
        full_system_oracle_total_env_steps,
    ).astype(np.int64)
    full_system_oracle_post_context_env_steps = summary.get(
        "full_system_oracle_post_context_env_steps",
        full_system_oracle_control_env_steps,
    ).astype(np.int64)
    full_system_oracle_post_context_episodes = summary.get(
        "full_system_oracle_post_context_episodes",
        full_system_oracle_episode_solves,
    ).astype(np.int64)
    sim_fanout_probe_env_steps = summary.get(
        "sim_fanout_probe_env_steps",
        sim_fanout_total_env_steps,
    ).astype(np.int64)
    sim_fanout_control_env_steps = summary.get(
        "sim_fanout_control_env_steps",
        sim_fanout_total_env_steps,
    ).astype(np.int64)
    sim_fanout_post_context_env_steps = summary.get(
        "sim_fanout_post_context_env_steps",
        sim_fanout_control_env_steps,
    ).astype(np.int64)
    sim_fanout_post_context_episodes = summary.get(
        "sim_fanout_post_context_episodes",
        sim_fanout_episode_solves,
    ).astype(np.int64)
    probe_windows_total = summary.get(
        "probe_windows_total",
        np.zeros_like(probe_episode_solves),
    ).astype(np.int64)
    probe_expression_scale_median = summary.get(
        "probe_expression_scale_median",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_expression_scale_active_fraction = summary.get(
        "probe_expression_scale_active_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_ready_handoff_fraction = summary.get(
        "probe_fair_ready_handoff_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_expression_enabled_fraction = summary.get(
        "probe_fair_expression_enabled_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_expression_force_muted_fraction = summary.get(
        "probe_fair_expression_force_muted_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_ready_confidence_median = summary.get(
        "probe_fair_ready_confidence_median",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_muted_confidence_median = summary.get(
        "probe_fair_muted_confidence_median",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_expression_ready_but_muted_fraction = summary.get(
        "probe_expression_ready_but_muted_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_shadow_expression_enabled_fraction = summary.get(
        "probe_shadow_expression_enabled_fraction",
        np.zeros_like(probe_shadow_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_shadow_expression_scale_median = summary.get(
        "probe_shadow_expression_scale_median",
        np.zeros_like(probe_shadow_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_shadow_confidence_median = summary.get(
        "probe_shadow_confidence_median",
        np.zeros_like(probe_shadow_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_shadow_strict_miss_fraction = summary.get(
        "probe_shadow_strict_miss_fraction",
        np.zeros_like(probe_shadow_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_second_probe_raw_future_gain_mean = summary.get(
        "probe_second_probe_raw_future_gain_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_second_probe_future_estimate_mean = summary.get(
        "probe_second_probe_future_estimate_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_second_probe_choice_future_gain_mean = summary.get(
        "probe_second_probe_choice_future_gain_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_family_coverage_satisfied_fraction = summary.get(
        "probe_family_coverage_satisfied_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_second_probe_value_driven_fraction = summary.get(
        "probe_second_probe_value_driven_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_uniformity_pressure_active_fraction = summary.get(
        "probe_uniformity_pressure_active_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_offline_gap_mean = summary.get(
        "probe_online_offline_gap_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_subset_stability_mean = summary.get(
        "probe_online_subset_stability_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_geometry_complete_fraction = summary.get(
        "probe_online_geometry_complete_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_split_latent_disagreement_mean = summary.get(
        "probe_online_split_latent_disagreement_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_split_retrieval_margin_deficit_mean = summary.get(
        "probe_online_split_retrieval_margin_deficit_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_leaveout_shift_mean = summary.get(
        "probe_online_leaveout_shift_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_teacher_action_agreement = summary.get(
        "probe_teacher_action_agreement",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_env_expression_delta = summary.get(
        "probe_env_expression_delta",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_forced_env_expression_delta = summary.get(
        "probe_forced_env_expression_delta",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_forced_env_expression_scale = summary.get(
        "probe_forced_env_expression_scale",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_strict_usage_status = summary.get(
        "probe_strict_usage_status",
        np.asarray(["unused"] * len(probe_episode_solves), dtype="U"),
    ).astype("U")
    full_system_zero_context_ablation_delta = summary.get(
        "full_system_zero_context_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_shuffled_context_ablation_delta = summary.get(
        "full_system_shuffled_context_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_stale_context_ablation_delta = summary.get(
        "full_system_stale_context_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_online_refinement_ablation_delta = summary.get(
        "full_system_online_refinement_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_frozen_context_ablation_delta = summary.get(
        "full_system_frozen_context_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_actor_only_ablation_delta = summary.get(
        "full_system_actor_only_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_state_only_ablation_delta = summary.get(
        "full_system_state_only_ablation_delta",
        np.zeros_like(full_system_state_only_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_zero_context_ablation_delta = summary.get(
        "full_system_oracle_zero_context_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_shuffled_context_ablation_delta = summary.get(
        "full_system_oracle_shuffled_context_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_stale_context_ablation_delta = summary.get(
        "full_system_oracle_stale_context_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_online_refinement_ablation_delta = summary.get(
        "full_system_oracle_online_refinement_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_frozen_context_ablation_delta = summary.get(
        "full_system_oracle_frozen_context_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_actor_only_ablation_delta = summary.get(
        "full_system_oracle_actor_only_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_run_classification = summary.get(
        "probe_run_classification",
        np.asarray(["protocol_win"] * len(probe_episode_solves), dtype="U"),
    ).astype("U")
    belief_mode = summary.get(
        "belief_mode",
        np.asarray(["latent_pool"] * len(probe_episode_solves), dtype="U"),
    ).astype("U")
    belief_progress_index = summary.get(
        "belief_progress_index",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    system_id_progress_index = summary.get(
        "system_id_progress_index",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    sysid_trusted = summary.get(
        "sysid_trusted",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    sysid_validation_top1 = summary.get(
        "sysid_validation_top1",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    sysid_validation_margin = summary.get(
        "sysid_validation_margin",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    sysid_validation_nll = summary.get(
        "sysid_validation_nll",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_entropy_mean = summary.get(
        "particle_entropy_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_entropy_norm_mean = summary.get(
        "particle_entropy_norm_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_ess_ratio_mean = summary.get(
        "particle_ess_ratio_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_leaveout_shift_mean = summary.get(
        "particle_leaveout_shift_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_subset_stability_mean = summary.get(
        "particle_subset_stability_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    derived_particle_subset_stability = np.clip(
        1.0 - particle_leaveout_shift_mean / 0.35,
        0.0,
        1.0,
    ).astype(np.float32)
    particle_subset_stability_mean = np.maximum(
        particle_subset_stability_mean,
        derived_particle_subset_stability,
    ).astype(np.float32)
    latent_mechanics_fit = summary.get(
        "latent_mechanics_fit",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_split_top1 = summary.get(
        "latent_split_top1",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_neighbor_alignment = summary.get(
        "latent_neighbor_alignment",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_gap_ratio = summary.get(
        "latent_gap_ratio",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_heldout_probe_error = summary.get(
        "latent_heldout_probe_error",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_probe_leakage = summary.get(
        "latent_probe_leakage",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_uncert_error_corr = summary.get(
        "latent_uncert_error_corr",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_win_gate = {}
    latent_win_gate_raw = summary.get("latent_win_gate_json")
    if latent_win_gate_raw is not None:
        try:
            latent_win_gate = json.loads(
                str(
                    latent_win_gate_raw.item()
                    if getattr(latent_win_gate_raw, "shape", None) == ()
                    else latent_win_gate_raw
                )
            )
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            latent_win_gate = {}
    latent_win_gate_failure_reasons = []
    latent_win_gate_failure_reasons_raw = summary.get("latent_win_gate_failure_reasons_json")
    if latent_win_gate_failure_reasons_raw is not None:
        try:
            latent_win_gate_failure_reasons = json.loads(
                str(
                    latent_win_gate_failure_reasons_raw.item()
                    if latent_win_gate_failure_reasons_raw.shape == ()
                    else latent_win_gate_failure_reasons_raw
                )
            )
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            latent_win_gate_failure_reasons = []
    probe_stop_reasons_rows = load_optional_json_rows(summary.get("probe_stop_reasons_json"))
    probe_final_stop_reason = summary.get(
        "probe_final_stop_reason",
        np.asarray([""] * len(probe_episode_solves), dtype="U"),
    ).astype("U")
    probe_family_expected_gain_rows = load_optional_json_rows(summary.get("probe_family_expected_gain_json"))
    probe_family_realized_gain_rows = load_optional_json_rows(summary.get("probe_family_realized_gain_json"))
    probe_family_future_error_rows = load_optional_json_rows(summary.get("probe_family_future_error_json"))
    probe_family_selection_count_rows = load_optional_json_rows(summary.get("probe_family_selection_count_json"))
    probe_readiness_reason_rows = load_optional_json_rows(summary.get("probe_readiness_reason_counts_json"))
    probe_readiness_component_rows = load_optional_json_rows(summary.get("probe_readiness_component_means_json"))
    probe_fair_stop_blocker_rows = load_optional_json_rows(summary.get("probe_fair_stop_blocker_counts_json"))
    probe_shadow_blocker_rows = load_optional_json_rows(summary.get("probe_shadow_blocker_counts_json"))
    probe_second_probe_selection_rows = load_optional_json_rows(summary.get("probe_second_probe_selection_count_json"))
    probe_fair_handoff_probe_families_rows = load_optional_json_rows(
        summary.get("probe_fair_handoff_probe_families_json")
    )
    probe_readiness_component_timeline_rows = load_optional_json_rows(
        summary.get("probe_readiness_component_timeline_json")
    )
    probe_online_future_quality_trace_rows = load_optional_json_rows(
        summary.get("probe_online_future_quality_trace_json")
    )
    probe_online_subset_stability_trace_rows = load_optional_json_rows(
        summary.get("probe_online_subset_stability_trace_json")
    )
    probe_online_offline_gap_trace_rows = load_optional_json_rows(
        summary.get("probe_online_offline_gap_trace_json")
    )
    latent_support_diagnostic_rows = load_optional_json_rows(
        summary.get("latent_support_diagnostics_json")
    )
    latent_claim_rejection_rows = load_optional_json_rows(
        summary.get("latent_claim_rejection_reasons_json")
    )
    full_system_state_only_eval_returns_rows = load_optional_json_rows(
        summary.get("full_system_state_only_eval_returns_json")
    )
    full_system_learned_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_learned_eval_summary_json"))
    ]
    full_system_state_only_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_state_only_eval_summary_json"))
    ]
    full_system_zero_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_zero_context_eval_summary_json"))
    ]
    full_system_shuffled_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_shuffled_context_eval_summary_json"))
    ]
    full_system_stale_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_stale_context_eval_summary_json"))
    ]
    full_system_online_refinement_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_online_refinement_eval_summary_json"))
    ]
    full_system_frozen_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_frozen_context_eval_summary_json"))
    ]
    full_system_actor_only_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_actor_only_eval_summary_json"))
    ]
    full_system_oracle_learned_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_learned_eval_summary_json"))
    ]
    full_system_oracle_zero_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_zero_context_eval_summary_json"))
    ]
    full_system_oracle_shuffled_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_shuffled_context_eval_summary_json"))
    ]
    full_system_oracle_stale_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_stale_context_eval_summary_json"))
    ]
    full_system_oracle_online_refinement_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_online_refinement_eval_summary_json"))
    ]
    full_system_oracle_frozen_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_frozen_context_eval_summary_json"))
    ]
    full_system_oracle_actor_only_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_actor_only_eval_summary_json"))
    ]
    while len(full_system_state_only_eval_summary_rows) < len(seeds):
        idx = len(full_system_state_only_eval_summary_rows)
        legacy_returns = (
            full_system_state_only_eval_returns_rows[idx]
            if idx < len(full_system_state_only_eval_returns_rows)
            and isinstance(full_system_state_only_eval_returns_rows[idx], list)
            else []
        )
        completed = int(full_system_state_only_completed_episodes[idx])
        total_steps = int(full_system_state_only_total_env_steps[idx])
        mean_total_steps = (
            float(total_steps) / float(max(completed, 1))
            if completed > 0
            else 0.0
        )
        full_system_state_only_eval_summary_rows.append(
            normalize_matched_eval_summary(
                {
                    "returns": legacy_returns,
                    "episode_total_env_steps": [mean_total_steps] * max(len(legacy_returns), completed),
                    "mean_return": float(np.mean(np.asarray(legacy_returns, dtype=np.float32))) if legacy_returns else 0.0,
                    "mean_total_env_steps": mean_total_steps,
                    "solved_count": 0,
                    "fixture_count": max(len(legacy_returns), completed),
                }
            )
        )

    rows = []
    for idx, seed in enumerate(seeds):
        learned_eval_summary = (
            full_system_learned_eval_summary_rows[idx]
            if idx < len(full_system_learned_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        state_only_eval_summary = (
            full_system_state_only_eval_summary_rows[idx]
            if idx < len(full_system_state_only_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        zero_context_eval_summary = (
            full_system_zero_context_eval_summary_rows[idx]
            if idx < len(full_system_zero_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        shuffled_context_eval_summary = (
            full_system_shuffled_context_eval_summary_rows[idx]
            if idx < len(full_system_shuffled_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        stale_context_eval_summary = (
            full_system_stale_context_eval_summary_rows[idx]
            if idx < len(full_system_stale_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        online_refinement_eval_summary = (
            full_system_online_refinement_eval_summary_rows[idx]
            if idx < len(full_system_online_refinement_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        frozen_context_eval_summary = (
            full_system_frozen_context_eval_summary_rows[idx]
            if idx < len(full_system_frozen_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        actor_only_eval_summary = (
            full_system_actor_only_eval_summary_rows[idx]
            if idx < len(full_system_actor_only_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_learned_eval_summary = (
            full_system_oracle_learned_eval_summary_rows[idx]
            if idx < len(full_system_oracle_learned_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_zero_context_eval_summary = (
            full_system_oracle_zero_context_eval_summary_rows[idx]
            if idx < len(full_system_oracle_zero_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_shuffled_context_eval_summary = (
            full_system_oracle_shuffled_context_eval_summary_rows[idx]
            if idx < len(full_system_oracle_shuffled_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_stale_context_eval_summary = (
            full_system_oracle_stale_context_eval_summary_rows[idx]
            if idx < len(full_system_oracle_stale_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_online_refinement_eval_summary = (
            full_system_oracle_online_refinement_eval_summary_rows[idx]
            if idx < len(full_system_oracle_online_refinement_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_frozen_context_eval_summary = (
            full_system_oracle_frozen_context_eval_summary_rows[idx]
            if idx < len(full_system_oracle_frozen_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_actor_only_eval_summary = (
            full_system_oracle_actor_only_eval_summary_rows[idx]
            if idx < len(full_system_oracle_actor_only_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        support_diag = (
            latent_support_diagnostic_rows[idx]
            if idx < len(latent_support_diagnostic_rows)
            else {}
        )
        latent_claim_rejections = (
            latent_claim_rejection_rows[idx]
            if idx < len(latent_claim_rejection_rows)
            else []
        )
        if not isinstance(latent_claim_rejections, list):
            latent_claim_rejections = []
        rows.append(
            {
                "seed": int(seed),
                "baseline_episode_solve": int(baseline_episode_solves[idx]),
                "probe_episode_solve": int(probe_episode_solves[idx]),
                "probe_shadow_episode_solve": int(probe_shadow_episode_solves[idx]),
                "probe_no_expression_episode_solve": int(probe_no_expression_episode_solves[idx]),
                "full_system_episode_solve": int(full_system_episode_solves[idx]),
                "full_system_state_only_episode_solve": int(full_system_state_only_episode_solves[idx]),
                "full_system_oracle_episode_solve": int(full_system_oracle_episode_solves[idx]),
                "sim_fanout_episode_solve": int(sim_fanout_episode_solves[idx]),
                "baseline_step_solve": int(baseline_step_solves[idx]),
                "probe_step_solve": int(probe_step_solves[idx]),
                "probe_shadow_step_solve": int(probe_shadow_step_solves[idx]),
                "probe_no_expression_step_solve": int(probe_no_expression_step_solves[idx]),
                "full_system_step_solve": int(full_system_step_solves[idx]),
                "full_system_state_only_step_solve": int(full_system_state_only_step_solves[idx]),
                "full_system_oracle_step_solve": int(full_system_oracle_step_solves[idx]),
                "sim_fanout_step_solve": int(sim_fanout_step_solves[idx]),
                "baseline_total_env_steps": int(baseline_total_env_steps[idx]),
                "probe_total_env_steps": int(probe_total_env_steps[idx]),
                "probe_shadow_total_env_steps": int(probe_shadow_total_env_steps[idx]),
                "probe_no_expression_total_env_steps": int(probe_no_expression_total_env_steps[idx]),
                "full_system_total_env_steps": int(full_system_total_env_steps[idx]),
                "full_system_state_only_total_env_steps": int(full_system_state_only_total_env_steps[idx]),
                "full_system_oracle_total_env_steps": int(full_system_oracle_total_env_steps[idx]),
                "sim_fanout_total_env_steps": int(sim_fanout_total_env_steps[idx]),
                "baseline_control_env_steps": int(baseline_control_env_steps[idx]),
                "probe_probe_env_steps": int(probe_probe_env_steps[idx]),
                "probe_control_env_steps": int(probe_control_env_steps[idx]),
                "probe_post_expression_env_steps": int(probe_post_expression_env_steps[idx]),
                "probe_post_expression_episodes": int(probe_post_expression_episodes[idx]),
                "probe_shadow_probe_env_steps": int(probe_shadow_probe_env_steps[idx]),
                "probe_shadow_control_env_steps": int(probe_shadow_control_env_steps[idx]),
                "probe_shadow_post_expression_env_steps": int(probe_shadow_post_expression_env_steps[idx]),
                "probe_shadow_post_expression_episodes": int(probe_shadow_post_expression_episodes[idx]),
                "probe_no_expression_probe_env_steps": int(probe_no_expression_probe_env_steps[idx]),
                "probe_no_expression_control_env_steps": int(probe_no_expression_control_env_steps[idx]),
                "probe_no_expression_post_expression_env_steps": int(probe_no_expression_post_expression_env_steps[idx]),
                "probe_no_expression_post_expression_episodes": int(probe_no_expression_post_expression_episodes[idx]),
                "full_system_probe_env_steps": int(full_system_probe_env_steps[idx]),
                "full_system_control_env_steps": int(full_system_control_env_steps[idx]),
                "full_system_post_context_env_steps": int(full_system_post_context_env_steps[idx]),
                "full_system_post_context_episodes": int(full_system_post_context_episodes[idx]),
                "full_system_oracle_probe_env_steps": int(full_system_oracle_probe_env_steps[idx]),
                "full_system_oracle_control_env_steps": int(full_system_oracle_control_env_steps[idx]),
                "full_system_oracle_post_context_env_steps": int(full_system_oracle_post_context_env_steps[idx]),
                "full_system_oracle_post_context_episodes": int(full_system_oracle_post_context_episodes[idx]),
                "sim_fanout_probe_env_steps": int(sim_fanout_probe_env_steps[idx]),
                "sim_fanout_control_env_steps": int(sim_fanout_control_env_steps[idx]),
                "sim_fanout_post_context_env_steps": int(sim_fanout_post_context_env_steps[idx]),
                "sim_fanout_post_context_episodes": int(sim_fanout_post_context_episodes[idx]),
                "probe_encoder_steps": int(probe_encoder_steps[idx]),
                "probe_windows_total": int(probe_windows_total[idx]),
                "full_system_completed_episodes": int(full_system_completed_episodes[idx]),
                "full_system_state_only_completed_episodes": int(full_system_state_only_completed_episodes[idx]),
                "full_system_oracle_completed_episodes": int(full_system_oracle_completed_episodes[idx]),
                "sim_fanout_completed_episodes": int(sim_fanout_completed_episodes[idx]),
                "probe_no_expression_available": bool(probe_no_expression_available[idx]),
                "latent_claim_valid": bool(latent_claim_valid[idx]),
                "latent_claim_rejection_reasons": [
                    str(reason) for reason in latent_claim_rejections
                ],
                "full_system_controller_style": str(full_system_controller_style[idx]),
                "full_system_oracle_controller_style": str(full_system_oracle_controller_style[idx]),
                "sim_fanout_controller_style": str(sim_fanout_controller_style[idx]),
                "probe_expression_scale_median": float(probe_expression_scale_median[idx]),
                "probe_expression_scale_active_fraction": float(probe_expression_scale_active_fraction[idx]),
                "probe_fair_ready_handoff_fraction": float(probe_fair_ready_handoff_fraction[idx]),
                "probe_fair_expression_enabled_fraction": float(probe_fair_expression_enabled_fraction[idx]),
                "probe_fair_expression_force_muted_fraction": float(probe_fair_expression_force_muted_fraction[idx]),
                "probe_fair_ready_confidence_median": float(probe_fair_ready_confidence_median[idx]),
                "probe_fair_muted_confidence_median": float(probe_fair_muted_confidence_median[idx]),
                "probe_expression_ready_but_muted_fraction": float(probe_expression_ready_but_muted_fraction[idx]),
                "probe_shadow_expression_enabled_fraction": float(probe_shadow_expression_enabled_fraction[idx]),
                "probe_shadow_expression_scale_median": float(probe_shadow_expression_scale_median[idx]),
                "probe_shadow_confidence_median": float(probe_shadow_confidence_median[idx]),
                "probe_shadow_strict_miss_fraction": float(probe_shadow_strict_miss_fraction[idx]),
                "probe_run_classification": str(probe_run_classification[idx]),
                "belief_mode": str(belief_mode[idx]),
                "belief_progress_index": float(belief_progress_index[idx]),
                "system_id_progress_index": float(system_id_progress_index[idx]),
                "sysid_trusted": bool(sysid_trusted[idx] > 0.5),
                "sysid_validation_top1": float(sysid_validation_top1[idx]),
                "sysid_validation_margin": float(sysid_validation_margin[idx]),
                "sysid_validation_nll": float(sysid_validation_nll[idx]),
                "particle_entropy_mean": float(particle_entropy_mean[idx]),
                "particle_entropy_norm_mean": float(particle_entropy_norm_mean[idx]),
                "particle_ess_ratio_mean": float(particle_ess_ratio_mean[idx]),
                "particle_leaveout_shift_mean": float(particle_leaveout_shift_mean[idx]),
                "particle_subset_stability_mean": float(particle_subset_stability_mean[idx]),
                "latent_mechanics_fit": float(latent_mechanics_fit[idx]),
                "latent_split_top1": float(latent_split_top1[idx]),
                "latent_neighbor_alignment": float(latent_neighbor_alignment[idx]),
                "latent_gap_ratio": float(latent_gap_ratio[idx]),
                "latent_heldout_probe_error": float(latent_heldout_probe_error[idx]),
                "latent_probe_leakage": float(latent_probe_leakage[idx]),
                "latent_uncert_error_corr": float(latent_uncert_error_corr[idx]),
                "latent_support_diagnostics": support_diag,
                "latent_center_window_share": float(support_diag.get("center_window_share", 0.0)),
                "latent_directional_window_share": float(support_diag.get("directional_window_share", 0.0)),
                "latent_mechanics_window_share": float(support_diag.get("mechanics_window_share", 0.0)),
                "latent_passive_window_share": float(support_diag.get("passive_window_share", 0.0)),
                "latent_stress_window_share": float(support_diag.get("stress_window_share", 0.0)),
                "latent_window_mode_leakage": float(support_diag.get("window_mode_leakage", 0.0)),
                "latent_env_mode_leakage": float(support_diag.get("env_mode_leakage", 0.0)),
                "probe_stop_reasons": probe_stop_reasons_rows[idx] if idx < len(probe_stop_reasons_rows) else {},
                "probe_final_stop_reason": str(probe_final_stop_reason[idx]),
                "probe_family_expected_gain": probe_family_expected_gain_rows[idx] if idx < len(probe_family_expected_gain_rows) else {},
                "probe_family_realized_gain": probe_family_realized_gain_rows[idx] if idx < len(probe_family_realized_gain_rows) else {},
                "probe_family_future_error": probe_family_future_error_rows[idx] if idx < len(probe_family_future_error_rows) else {},
                "probe_family_selection_count": probe_family_selection_count_rows[idx] if idx < len(probe_family_selection_count_rows) else {},
                "probe_readiness_reason_counts": probe_readiness_reason_rows[idx] if idx < len(probe_readiness_reason_rows) else {},
                "probe_readiness_component_means": probe_readiness_component_rows[idx] if idx < len(probe_readiness_component_rows) else {},
                "probe_fair_stop_blocker_counts": probe_fair_stop_blocker_rows[idx] if idx < len(probe_fair_stop_blocker_rows) else {},
                "probe_shadow_blocker_counts": probe_shadow_blocker_rows[idx] if idx < len(probe_shadow_blocker_rows) else {},
                "probe_second_probe_selection_count": probe_second_probe_selection_rows[idx] if idx < len(probe_second_probe_selection_rows) else {},
                "probe_second_probe_raw_future_gain_mean": float(probe_second_probe_raw_future_gain_mean[idx]),
                "probe_second_probe_future_estimate_mean": float(probe_second_probe_future_estimate_mean[idx]),
                "probe_second_probe_choice_future_gain_mean": float(probe_second_probe_choice_future_gain_mean[idx]),
                "probe_family_coverage_satisfied_fraction": float(probe_family_coverage_satisfied_fraction[idx]),
                "probe_second_probe_value_driven_fraction": float(probe_second_probe_value_driven_fraction[idx]),
                "probe_uniformity_pressure_active_fraction": float(probe_uniformity_pressure_active_fraction[idx]),
                "probe_env_expression_delta": float(probe_env_expression_delta[idx]),
                "probe_forced_env_expression_delta": float(probe_forced_env_expression_delta[idx]),
                "probe_forced_env_expression_scale": float(probe_forced_env_expression_scale[idx]),
                "probe_strict_usage_status": str(probe_strict_usage_status[idx]),
                "probe_fair_handoff_probe_families": (
                    probe_fair_handoff_probe_families_rows[idx]
                    if idx < len(probe_fair_handoff_probe_families_rows)
                    else []
                ),
                "probe_readiness_component_timeline": (
                    probe_readiness_component_timeline_rows[idx]
                    if idx < len(probe_readiness_component_timeline_rows)
                    else []
                ),
                "probe_online_future_quality_trace": (
                    probe_online_future_quality_trace_rows[idx]
                    if idx < len(probe_online_future_quality_trace_rows)
                    else []
                ),
                "probe_online_subset_stability_trace": (
                    probe_online_subset_stability_trace_rows[idx]
                    if idx < len(probe_online_subset_stability_trace_rows)
                    else []
                ),
                "probe_online_offline_gap_trace": (
                    probe_online_offline_gap_trace_rows[idx]
                    if idx < len(probe_online_offline_gap_trace_rows)
                    else []
                ),
                "probe_online_subset_stability_mean": float(probe_online_subset_stability_mean[idx]),
                "probe_online_offline_gap_mean": float(probe_online_offline_gap_mean[idx]),
                "probe_online_geometry_complete_fraction": float(probe_online_geometry_complete_fraction[idx]),
                "probe_online_split_latent_disagreement_mean": float(probe_online_split_latent_disagreement_mean[idx]),
                "probe_online_split_retrieval_margin_deficit_mean": float(probe_online_split_retrieval_margin_deficit_mean[idx]),
                "probe_online_leaveout_shift_mean": float(probe_online_leaveout_shift_mean[idx]),
                "probe_teacher_action_agreement": float(probe_teacher_action_agreement[idx]),
                "full_system_learned_eval_summary": learned_eval_summary,
                "full_system_state_only_eval_summary": state_only_eval_summary,
                "full_system_zero_context_eval_summary": zero_context_eval_summary,
                "full_system_shuffled_context_eval_summary": shuffled_context_eval_summary,
                "full_system_stale_context_eval_summary": stale_context_eval_summary,
                "full_system_online_refinement_eval_summary": online_refinement_eval_summary,
                "full_system_frozen_context_eval_summary": frozen_context_eval_summary,
                "full_system_actor_only_eval_summary": actor_only_eval_summary,
                "full_system_state_only_eval_returns": (
                    full_system_state_only_eval_returns_rows[idx]
                    if idx < len(full_system_state_only_eval_returns_rows)
                    else []
                ),
                "full_system_state_only_ablation_delta": float(full_system_state_only_ablation_delta[idx]),
                "full_system_zero_context_ablation_delta": float(full_system_zero_context_ablation_delta[idx]),
                "full_system_shuffled_context_ablation_delta": float(full_system_shuffled_context_ablation_delta[idx]),
                "full_system_stale_context_ablation_delta": float(full_system_stale_context_ablation_delta[idx]),
                "full_system_online_refinement_ablation_delta": float(full_system_online_refinement_ablation_delta[idx]),
                "full_system_frozen_context_ablation_delta": float(full_system_frozen_context_ablation_delta[idx]),
                "full_system_actor_only_ablation_delta": float(full_system_actor_only_ablation_delta[idx]),
                "full_system_oracle_zero_context_ablation_delta": float(full_system_oracle_zero_context_ablation_delta[idx]),
                "full_system_oracle_shuffled_context_ablation_delta": float(full_system_oracle_shuffled_context_ablation_delta[idx]),
                "full_system_oracle_stale_context_ablation_delta": float(full_system_oracle_stale_context_ablation_delta[idx]),
                "full_system_oracle_online_refinement_ablation_delta": float(full_system_oracle_online_refinement_ablation_delta[idx]),
                "full_system_oracle_frozen_context_ablation_delta": float(full_system_oracle_frozen_context_ablation_delta[idx]),
                "full_system_oracle_actor_only_ablation_delta": float(full_system_oracle_actor_only_ablation_delta[idx]),
                "full_system_oracle_learned_eval_summary": oracle_learned_eval_summary,
                "full_system_oracle_zero_context_eval_summary": oracle_zero_context_eval_summary,
                "full_system_oracle_shuffled_context_eval_summary": oracle_shuffled_context_eval_summary,
                "full_system_oracle_stale_context_eval_summary": oracle_stale_context_eval_summary,
                "full_system_oracle_online_refinement_eval_summary": oracle_online_refinement_eval_summary,
                "full_system_oracle_frozen_context_eval_summary": oracle_frozen_context_eval_summary,
                "full_system_oracle_actor_only_eval_summary": oracle_actor_only_eval_summary,
                "probe_strictly_muted_but_shadow_eligible": bool(
                    float(probe_fair_expression_enabled_fraction[idx]) <= 0.0
                    and float(probe_shadow_expression_enabled_fraction[idx]) > 0.0
                ),
                "probe_shadow_available": bool(
                    int(probe_shadow_completed_episodes[idx]) > 0
                    or int(probe_shadow_episode_solves[idx]) >= 0
                ),
                "full_system_available": bool(
                    int(full_system_completed_episodes[idx]) > 0
                    or int(full_system_episode_solves[idx]) >= 0
                ),
                "full_system_state_only_available": bool(state_only_eval_summary["available"]),
                "full_system_zero_context_available": bool(zero_context_eval_summary["available"]),
                "full_system_shuffled_context_available": bool(shuffled_context_eval_summary["available"]),
                "full_system_stale_context_available": bool(stale_context_eval_summary["available"]),
                "full_system_frozen_context_available": bool(frozen_context_eval_summary["available"]),
                "full_system_oracle_available": bool(
                    int(full_system_oracle_completed_episodes[idx]) > 0
                    or int(full_system_oracle_episode_solves[idx]) >= 0
                ),
                "full_system_oracle_frozen_context_available": bool(
                    oracle_frozen_context_eval_summary["available"]
                ),
                "sim_fanout_available": bool(
                    int(sim_fanout_completed_episodes[idx]) > 0
                    or int(sim_fanout_episode_solves[idx]) >= 0
                ),
            }
        )

    classification_counts: dict[str, int] = {}
    for label in np.asarray(probe_run_classification, dtype="U").tolist():
        key = str(label)
        classification_counts[key] = classification_counts.get(key, 0) + 1
    dominant_classification = max(
        classification_counts.items(),
        key=lambda item: (item[1], item[0]),
        default=("protocol_win", 0),
    )[0]
    readiness_reason_totals = aggregate_json_counter_rows(probe_readiness_reason_rows)
    readiness_component_means = average_json_metric_rows(probe_readiness_component_rows)
    fair_stop_blocker_totals = aggregate_json_counter_rows(probe_fair_stop_blocker_rows)
    shadow_blocker_totals = aggregate_json_counter_rows(probe_shadow_blocker_rows)
    second_probe_selection_totals = aggregate_json_counter_rows(probe_second_probe_selection_rows)
    fair_handoff_pair_totals = aggregate_json_list_rows(probe_fair_handoff_probe_families_rows)
    strict_usage_counts: dict[str, int] = {}
    for label in np.asarray(probe_strict_usage_status, dtype="U").tolist():
        key = str(label)
        strict_usage_counts[key] = strict_usage_counts.get(key, 0) + 1
    dominant_strict_usage_status = max(
        strict_usage_counts.items(),
        key=lambda item: (item[1], item[0]),
        default=("unused", 0),
    )[0]
    probe_env_expression_delta_summary = summarize_solve_array(
        probe_episode_solves,
        probe_completed_episodes,
    )
    baseline_episode_summary = summarize_solve_array(
        baseline_episode_solves,
        baseline_completed_episodes,
    )
    probe_env_expression_delta_mean = (
        float(np.mean(probe_env_expression_delta))
        if probe_env_expression_delta.size
        else 0.0
    )
    honesty_headline = ""
    if (
        dominant_strict_usage_status == "unused"
        and probe_env_expression_delta_summary["median"] < baseline_episode_summary["median"]
    ):
        honesty_headline = "Episode win without strict latent usage"
    elif probe_env_expression_delta_mean <= 0.0:
        honesty_headline = "Env expression harmful under matched eval"

    return {
        "name": path.name,
        "artifact_mtime": float(path.stat().st_mtime),
        "env_name": env_name,
        "env_display_name": None if env_name is None else get_env_display_name(env_name),
        "benchmark_profile": benchmark_profile,
        "benchmark_mode": load_optional_string(summary, "benchmark_mode"),
        "probe_budget_mode": load_optional_string(summary, "probe_budget_mode"),
        "full_system_controller_style": next(
            (str(value) for value in full_system_controller_style.tolist() if str(value)),
            "",
        ),
        "full_system_oracle_controller_style": next(
            (str(value) for value in full_system_oracle_controller_style.tolist() if str(value)),
            "",
        ),
        "sim_fanout_controller_style": next(
            (str(value) for value in sim_fanout_controller_style.tolist() if str(value)),
            "",
        ),
        "run_classification": dominant_classification,
        "probe_strict_usage_status": dominant_strict_usage_status,
        "probe_honesty_headline": honesty_headline,
        "latent_win_gate": latent_win_gate,
        "latent_win_gate_failure_reasons": latent_win_gate_failure_reasons,
        "probe_no_expression_available": bool(
            probe_no_expression_available.size > 0
            and np.all(probe_no_expression_available > 0)
        ),
        "latent_claim_valid": bool(
            latent_claim_valid.size > 0
            and np.all(latent_claim_valid > 0)
        ),
        "probe_shadow_available": bool(
            np.any(probe_shadow_completed_episodes > 0)
            or np.any(probe_shadow_episode_solves >= 0)
        ),
        "full_system_available": bool(
            np.any(full_system_completed_episodes > 0)
            or np.any(full_system_episode_solves >= 0)
        ),
        "full_system_state_only_available": bool(
            any(row["available"] for row in full_system_state_only_eval_summary_rows)
        ),
        "full_system_zero_context_available": bool(
            any(row["available"] for row in full_system_zero_context_eval_summary_rows)
        ),
        "full_system_shuffled_context_available": bool(
            any(row["available"] for row in full_system_shuffled_context_eval_summary_rows)
        ),
        "full_system_stale_context_available": bool(
            any(row["available"] for row in full_system_stale_context_eval_summary_rows)
        ),
        "full_system_frozen_context_available": bool(
            any(row["available"] for row in full_system_frozen_context_eval_summary_rows)
        ),
        "full_system_oracle_available": bool(
            np.any(full_system_oracle_completed_episodes > 0)
            or np.any(full_system_oracle_episode_solves >= 0)
        ),
        "full_system_oracle_frozen_context_available": bool(
            any(row["available"] for row in full_system_oracle_frozen_context_eval_summary_rows)
        ),
        "sim_fanout_available": bool(
            np.any(sim_fanout_completed_episodes > 0)
            or np.any(sim_fanout_episode_solves >= 0)
        ),
        "rows": rows,
        "summaries": {
            "baseline_episode": summarize_solve_array(
                baseline_episode_solves,
                baseline_completed_episodes,
            ),
            "probe_episode": summarize_solve_array(
                probe_episode_solves,
                probe_completed_episodes,
            ),
            "belief_progress_index": {
                "median": float(np.median(belief_progress_index)) if belief_progress_index.size else 0.0,
                "mean": float(np.mean(belief_progress_index)) if belief_progress_index.size else 0.0,
                "count": int(belief_progress_index.size),
            },
            "system_id": {
                "available": bool(
                    belief_mode.size
                    and (
                        np.any(belief_mode == "particle_sysid")
                        or np.any(sysid_validation_top1 > 0.0)
                    )
                ),
                "mode": (
                    "particle_sysid"
                    if bool(np.any(belief_mode == "particle_sysid"))
                    else "latent_pool"
                ),
                "progress_median": float(np.median(system_id_progress_index)) if system_id_progress_index.size else 0.0,
                "progress_mean": float(np.mean(system_id_progress_index)) if system_id_progress_index.size else 0.0,
                "trusted_fraction": float(np.mean(sysid_trusted)) if sysid_trusted.size else 0.0,
                "validation_top1_median": float(np.median(sysid_validation_top1)) if sysid_validation_top1.size else 0.0,
                "validation_margin_median": float(np.median(sysid_validation_margin)) if sysid_validation_margin.size else 0.0,
                "validation_nll_median": float(np.median(sysid_validation_nll)) if sysid_validation_nll.size else 0.0,
                "particle_entropy_median": float(np.median(particle_entropy_mean)) if particle_entropy_mean.size else 0.0,
                "particle_ess_ratio_median": float(np.median(particle_ess_ratio_mean)) if particle_ess_ratio_mean.size else 0.0,
                "particle_leaveout_shift_median": float(np.median(particle_leaveout_shift_mean)) if particle_leaveout_shift_mean.size else 0.0,
                "particle_subset_stability_median": float(np.median(particle_subset_stability_mean)) if particle_subset_stability_mean.size else 0.0,
            },
            "latent_mechanics_fit": {
                "median": float(np.median(latent_mechanics_fit)) if latent_mechanics_fit.size else 0.0,
                "mean": float(np.mean(latent_mechanics_fit)) if latent_mechanics_fit.size else 0.0,
                "count": int(latent_mechanics_fit.size),
            },
            "latent_split_top1": {
                "median": float(np.median(latent_split_top1)) if latent_split_top1.size else 0.0,
                "mean": float(np.mean(latent_split_top1)) if latent_split_top1.size else 0.0,
                "count": int(latent_split_top1.size),
            },
            "latent_neighbor_alignment": {
                "median": float(np.median(latent_neighbor_alignment)) if latent_neighbor_alignment.size else 0.0,
                "mean": float(np.mean(latent_neighbor_alignment)) if latent_neighbor_alignment.size else 0.0,
                "count": int(latent_neighbor_alignment.size),
            },
            "latent_gap_ratio": {
                "median": float(np.median(latent_gap_ratio)) if latent_gap_ratio.size else 0.0,
                "mean": float(np.mean(latent_gap_ratio)) if latent_gap_ratio.size else 0.0,
                "count": int(latent_gap_ratio.size),
            },
            "latent_heldout_probe_error": {
                "median": float(np.median(latent_heldout_probe_error)) if latent_heldout_probe_error.size else 0.0,
                "mean": float(np.mean(latent_heldout_probe_error)) if latent_heldout_probe_error.size else 0.0,
                "count": int(latent_heldout_probe_error.size),
            },
            "latent_probe_leakage": {
                "median": float(np.median(latent_probe_leakage)) if latent_probe_leakage.size else 0.0,
                "mean": float(np.mean(latent_probe_leakage)) if latent_probe_leakage.size else 0.0,
                "count": int(latent_probe_leakage.size),
            },
            "latent_uncert_error_corr": {
                "median": float(np.median(latent_uncert_error_corr)) if latent_uncert_error_corr.size else 0.0,
                "mean": float(np.mean(latent_uncert_error_corr)) if latent_uncert_error_corr.size else 0.0,
                "count": int(latent_uncert_error_corr.size),
            },
            "latent_support_diagnostics": average_json_metric_rows(latent_support_diagnostic_rows),
            "probe_shadow_episode": summarize_solve_array(
                probe_shadow_episode_solves,
                probe_shadow_completed_episodes,
            ),
            "probe_no_expression_episode": summarize_solve_array(
                probe_no_expression_episode_solves,
                probe_no_expression_completed_episodes,
            ),
            "full_system_episode": summarize_solve_array(
                full_system_episode_solves,
                full_system_completed_episodes,
            ),
            "full_system_oracle_episode": summarize_solve_array(
                full_system_oracle_episode_solves,
                full_system_oracle_completed_episodes,
            ),
            "sim_fanout_episode": summarize_solve_array(
                sim_fanout_episode_solves,
                sim_fanout_completed_episodes,
            ),
            "baseline_steps": summarize_solve_array(
                baseline_step_solves,
                baseline_total_env_steps,
            ),
            "probe_steps": summarize_solve_array(
                probe_step_solves,
                probe_total_env_steps,
            ),
            "probe_shadow_steps": summarize_solve_array(
                probe_shadow_step_solves,
                probe_shadow_total_env_steps,
            ),
            "probe_no_expression_steps": summarize_solve_array(
                probe_no_expression_step_solves,
                probe_no_expression_total_env_steps,
            ),
            "full_system_steps": summarize_solve_array(
                full_system_step_solves,
                full_system_total_env_steps,
            ),
            "full_system_oracle_steps": summarize_solve_array(
                full_system_oracle_step_solves,
                full_system_oracle_total_env_steps,
            ),
            "sim_fanout_steps": summarize_solve_array(
                sim_fanout_step_solves,
                sim_fanout_total_env_steps,
            ),
            "probe_post_expression_steps": summarize_solve_array(
                probe_post_expression_env_steps,
                probe_total_env_steps,
            ),
            "probe_shadow_post_expression_steps": summarize_solve_array(
                probe_shadow_post_expression_env_steps,
                probe_shadow_total_env_steps,
            ),
            "full_system_post_context_steps": summarize_solve_array(
                full_system_post_context_env_steps,
                full_system_total_env_steps,
            ),
            "full_system_oracle_post_context_steps": summarize_solve_array(
                full_system_oracle_post_context_env_steps,
                full_system_oracle_total_env_steps,
            ),
            "sim_fanout_post_context_steps": summarize_solve_array(
                sim_fanout_post_context_env_steps,
                sim_fanout_total_env_steps,
            ),
            "probe_post_expression_episodes": summarize_solve_array(
                probe_post_expression_episodes,
                probe_completed_episodes,
            ),
            "probe_shadow_post_expression_episodes": summarize_solve_array(
                probe_shadow_post_expression_episodes,
                probe_shadow_completed_episodes,
            ),
            "full_system_post_context_episodes": summarize_solve_array(
                full_system_post_context_episodes,
                full_system_completed_episodes,
            ),
            "full_system_oracle_post_context_episodes": summarize_solve_array(
                full_system_oracle_post_context_episodes,
                full_system_oracle_completed_episodes,
            ),
            "sim_fanout_post_context_episodes": summarize_solve_array(
                sim_fanout_post_context_episodes,
                sim_fanout_completed_episodes,
            ),
            "probe_expression_scale_median": {
                "median": float(np.median(probe_expression_scale_median)) if probe_expression_scale_median.size else 0.0,
                "mean": float(np.mean(probe_expression_scale_median)) if probe_expression_scale_median.size else 0.0,
                "count": int(probe_expression_scale_median.size),
            },
            "probe_expression_scale_active_fraction": {
                "median": float(np.median(probe_expression_scale_active_fraction)) if probe_expression_scale_active_fraction.size else 0.0,
                "mean": float(np.mean(probe_expression_scale_active_fraction)) if probe_expression_scale_active_fraction.size else 0.0,
                "count": int(probe_expression_scale_active_fraction.size),
            },
            "probe_fair_ready_handoff_fraction": {
                "median": float(np.median(probe_fair_ready_handoff_fraction)) if probe_fair_ready_handoff_fraction.size else 0.0,
                "mean": float(np.mean(probe_fair_ready_handoff_fraction)) if probe_fair_ready_handoff_fraction.size else 0.0,
                "count": int(probe_fair_ready_handoff_fraction.size),
            },
            "probe_fair_expression_enabled_fraction": {
                "median": float(np.median(probe_fair_expression_enabled_fraction)) if probe_fair_expression_enabled_fraction.size else 0.0,
                "mean": float(np.mean(probe_fair_expression_enabled_fraction)) if probe_fair_expression_enabled_fraction.size else 0.0,
                "count": int(probe_fair_expression_enabled_fraction.size),
            },
            "probe_fair_expression_force_muted_fraction": {
                "median": float(np.median(probe_fair_expression_force_muted_fraction)) if probe_fair_expression_force_muted_fraction.size else 0.0,
                "mean": float(np.mean(probe_fair_expression_force_muted_fraction)) if probe_fair_expression_force_muted_fraction.size else 0.0,
                "count": int(probe_fair_expression_force_muted_fraction.size),
            },
            "probe_fair_ready_confidence_median": {
                "median": float(np.median(probe_fair_ready_confidence_median)) if probe_fair_ready_confidence_median.size else 0.0,
                "mean": float(np.mean(probe_fair_ready_confidence_median)) if probe_fair_ready_confidence_median.size else 0.0,
                "count": int(probe_fair_ready_confidence_median.size),
            },
            "probe_fair_muted_confidence_median": {
                "median": float(np.median(probe_fair_muted_confidence_median)) if probe_fair_muted_confidence_median.size else 0.0,
                "mean": float(np.mean(probe_fair_muted_confidence_median)) if probe_fair_muted_confidence_median.size else 0.0,
                "count": int(probe_fair_muted_confidence_median.size),
            },
            "probe_expression_ready_but_muted_fraction": {
                "median": float(np.median(probe_expression_ready_but_muted_fraction)) if probe_expression_ready_but_muted_fraction.size else 0.0,
                "mean": float(np.mean(probe_expression_ready_but_muted_fraction)) if probe_expression_ready_but_muted_fraction.size else 0.0,
                "count": int(probe_expression_ready_but_muted_fraction.size),
            },
            "probe_shadow_expression_enabled_fraction": {
                "median": float(np.median(probe_shadow_expression_enabled_fraction)) if probe_shadow_expression_enabled_fraction.size else 0.0,
                "mean": float(np.mean(probe_shadow_expression_enabled_fraction)) if probe_shadow_expression_enabled_fraction.size else 0.0,
                "count": int(probe_shadow_expression_enabled_fraction.size),
            },
            "probe_shadow_expression_scale_median": {
                "median": float(np.median(probe_shadow_expression_scale_median)) if probe_shadow_expression_scale_median.size else 0.0,
                "mean": float(np.mean(probe_shadow_expression_scale_median)) if probe_shadow_expression_scale_median.size else 0.0,
                "count": int(probe_shadow_expression_scale_median.size),
            },
            "probe_shadow_confidence_median": {
                "median": float(np.median(probe_shadow_confidence_median)) if probe_shadow_confidence_median.size else 0.0,
                "mean": float(np.mean(probe_shadow_confidence_median)) if probe_shadow_confidence_median.size else 0.0,
                "count": int(probe_shadow_confidence_median.size),
            },
            "probe_shadow_strict_miss_fraction": {
                "median": float(np.median(probe_shadow_strict_miss_fraction)) if probe_shadow_strict_miss_fraction.size else 0.0,
                "mean": float(np.mean(probe_shadow_strict_miss_fraction)) if probe_shadow_strict_miss_fraction.size else 0.0,
                "count": int(probe_shadow_strict_miss_fraction.size),
            },
            "probe_second_probe_raw_future_gain_mean": {
                "median": float(np.median(probe_second_probe_raw_future_gain_mean)) if probe_second_probe_raw_future_gain_mean.size else 0.0,
                "mean": float(np.mean(probe_second_probe_raw_future_gain_mean)) if probe_second_probe_raw_future_gain_mean.size else 0.0,
                "count": int(probe_second_probe_raw_future_gain_mean.size),
            },
            "probe_second_probe_future_estimate_mean": {
                "median": float(np.median(probe_second_probe_future_estimate_mean)) if probe_second_probe_future_estimate_mean.size else 0.0,
                "mean": float(np.mean(probe_second_probe_future_estimate_mean)) if probe_second_probe_future_estimate_mean.size else 0.0,
                "count": int(probe_second_probe_future_estimate_mean.size),
            },
            "probe_second_probe_choice_future_gain_mean": {
                "median": float(np.median(probe_second_probe_choice_future_gain_mean)) if probe_second_probe_choice_future_gain_mean.size else 0.0,
                "mean": float(np.mean(probe_second_probe_choice_future_gain_mean)) if probe_second_probe_choice_future_gain_mean.size else 0.0,
                "count": int(probe_second_probe_choice_future_gain_mean.size),
            },
            "probe_family_coverage_satisfied_fraction": {
                "median": float(np.median(probe_family_coverage_satisfied_fraction)) if probe_family_coverage_satisfied_fraction.size else 0.0,
                "mean": float(np.mean(probe_family_coverage_satisfied_fraction)) if probe_family_coverage_satisfied_fraction.size else 0.0,
                "count": int(probe_family_coverage_satisfied_fraction.size),
            },
            "probe_second_probe_value_driven_fraction": {
                "median": float(np.median(probe_second_probe_value_driven_fraction)) if probe_second_probe_value_driven_fraction.size else 0.0,
                "mean": float(np.mean(probe_second_probe_value_driven_fraction)) if probe_second_probe_value_driven_fraction.size else 0.0,
                "count": int(probe_second_probe_value_driven_fraction.size),
            },
            "probe_uniformity_pressure_active_fraction": {
                "median": float(np.median(probe_uniformity_pressure_active_fraction)) if probe_uniformity_pressure_active_fraction.size else 0.0,
                "mean": float(np.mean(probe_uniformity_pressure_active_fraction)) if probe_uniformity_pressure_active_fraction.size else 0.0,
                "count": int(probe_uniformity_pressure_active_fraction.size),
            },
            "probe_env_expression_delta": {
                "median": float(np.median(probe_env_expression_delta)) if probe_env_expression_delta.size else 0.0,
                "mean": float(np.mean(probe_env_expression_delta)) if probe_env_expression_delta.size else 0.0,
                "count": int(probe_env_expression_delta.size),
            },
            "probe_forced_env_expression_delta": {
                "median": float(np.median(probe_forced_env_expression_delta)) if probe_forced_env_expression_delta.size else 0.0,
                "mean": float(np.mean(probe_forced_env_expression_delta)) if probe_forced_env_expression_delta.size else 0.0,
                "count": int(probe_forced_env_expression_delta.size),
            },
            "probe_forced_env_expression_scale": {
                "median": float(np.median(probe_forced_env_expression_scale)) if probe_forced_env_expression_scale.size else 0.0,
                "mean": float(np.mean(probe_forced_env_expression_scale)) if probe_forced_env_expression_scale.size else 0.0,
                "count": int(probe_forced_env_expression_scale.size),
            },
            "probe_online_offline_gap_mean": {
                "median": float(np.median(probe_online_offline_gap_mean)) if probe_online_offline_gap_mean.size else 0.0,
                "mean": float(np.mean(probe_online_offline_gap_mean)) if probe_online_offline_gap_mean.size else 0.0,
                "count": int(probe_online_offline_gap_mean.size),
            },
            "probe_online_subset_stability_mean": {
                "median": float(np.median(probe_online_subset_stability_mean)) if probe_online_subset_stability_mean.size else 0.0,
                "mean": float(np.mean(probe_online_subset_stability_mean)) if probe_online_subset_stability_mean.size else 0.0,
                "count": int(probe_online_subset_stability_mean.size),
            },
            "probe_online_geometry_complete_fraction": {
                "median": float(np.median(probe_online_geometry_complete_fraction)) if probe_online_geometry_complete_fraction.size else 0.0,
                "mean": float(np.mean(probe_online_geometry_complete_fraction)) if probe_online_geometry_complete_fraction.size else 0.0,
                "count": int(probe_online_geometry_complete_fraction.size),
            },
            "probe_online_split_latent_disagreement_mean": {
                "median": float(np.median(probe_online_split_latent_disagreement_mean)) if probe_online_split_latent_disagreement_mean.size else 0.0,
                "mean": float(np.mean(probe_online_split_latent_disagreement_mean)) if probe_online_split_latent_disagreement_mean.size else 0.0,
                "count": int(probe_online_split_latent_disagreement_mean.size),
            },
            "probe_online_split_retrieval_margin_deficit_mean": {
                "median": float(np.median(probe_online_split_retrieval_margin_deficit_mean)) if probe_online_split_retrieval_margin_deficit_mean.size else 0.0,
                "mean": float(np.mean(probe_online_split_retrieval_margin_deficit_mean)) if probe_online_split_retrieval_margin_deficit_mean.size else 0.0,
                "count": int(probe_online_split_retrieval_margin_deficit_mean.size),
            },
            "probe_online_leaveout_shift_mean": {
                "median": float(np.median(probe_online_leaveout_shift_mean)) if probe_online_leaveout_shift_mean.size else 0.0,
                "mean": float(np.mean(probe_online_leaveout_shift_mean)) if probe_online_leaveout_shift_mean.size else 0.0,
                "count": int(probe_online_leaveout_shift_mean.size),
            },
            "probe_teacher_action_agreement": {
                "median": float(np.median(probe_teacher_action_agreement)) if probe_teacher_action_agreement.size else 0.0,
                "mean": float(np.mean(probe_teacher_action_agreement)) if probe_teacher_action_agreement.size else 0.0,
                "count": int(probe_teacher_action_agreement.size),
            },
            "full_system_zero_context_ablation_delta": {
                "median": float(np.median(full_system_zero_context_ablation_delta)) if full_system_zero_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_zero_context_ablation_delta)) if full_system_zero_context_ablation_delta.size else 0.0,
                "count": int(full_system_zero_context_ablation_delta.size),
            },
            "full_system_shuffled_context_ablation_delta": {
                "median": float(np.median(full_system_shuffled_context_ablation_delta)) if full_system_shuffled_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_shuffled_context_ablation_delta)) if full_system_shuffled_context_ablation_delta.size else 0.0,
                "count": int(full_system_shuffled_context_ablation_delta.size),
            },
            "full_system_stale_context_ablation_delta": {
                "median": float(np.median(full_system_stale_context_ablation_delta)) if full_system_stale_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_stale_context_ablation_delta)) if full_system_stale_context_ablation_delta.size else 0.0,
                "count": int(full_system_stale_context_ablation_delta.size),
            },
            "full_system_online_refinement_ablation_delta": {
                "median": float(np.median(full_system_online_refinement_ablation_delta)) if full_system_online_refinement_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_online_refinement_ablation_delta)) if full_system_online_refinement_ablation_delta.size else 0.0,
                "count": int(full_system_online_refinement_ablation_delta.size),
            },
            "full_system_frozen_context_ablation_delta": {
                "median": float(np.median(full_system_frozen_context_ablation_delta)) if full_system_frozen_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_frozen_context_ablation_delta)) if full_system_frozen_context_ablation_delta.size else 0.0,
                "count": int(full_system_frozen_context_ablation_delta.size),
            },
            "full_system_actor_only_ablation_delta": {
                "median": float(np.median(full_system_actor_only_ablation_delta)) if full_system_actor_only_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_actor_only_ablation_delta)) if full_system_actor_only_ablation_delta.size else 0.0,
                "count": int(full_system_actor_only_ablation_delta.size),
            },
            "full_system_learned_eval": summarize_matched_eval_rows(
                full_system_learned_eval_summary_rows
            ),
            "full_system_state_only_eval": summarize_matched_eval_rows(
                full_system_state_only_eval_summary_rows
            ),
            "full_system_zero_context_eval": summarize_matched_eval_rows(
                full_system_zero_context_eval_summary_rows
            ),
            "full_system_shuffled_context_eval": summarize_matched_eval_rows(
                full_system_shuffled_context_eval_summary_rows
            ),
            "full_system_stale_context_eval": summarize_matched_eval_rows(
                full_system_stale_context_eval_summary_rows
            ),
            "full_system_online_refinement_eval": summarize_matched_eval_rows(
                full_system_online_refinement_eval_summary_rows
            ),
            "full_system_frozen_context_eval": summarize_matched_eval_rows(
                full_system_frozen_context_eval_summary_rows
            ),
            "full_system_actor_only_eval": summarize_matched_eval_rows(
                full_system_actor_only_eval_summary_rows
            ),
            "full_system_state_only_ablation_delta": {
                "median": float(np.median(full_system_state_only_ablation_delta)) if full_system_state_only_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_state_only_ablation_delta)) if full_system_state_only_ablation_delta.size else 0.0,
                "count": int(full_system_state_only_ablation_delta.size),
            },
            "full_system_oracle_zero_context_ablation_delta": {
                "median": float(np.median(full_system_oracle_zero_context_ablation_delta)) if full_system_oracle_zero_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_zero_context_ablation_delta)) if full_system_oracle_zero_context_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_zero_context_ablation_delta.size),
            },
            "full_system_oracle_shuffled_context_ablation_delta": {
                "median": float(np.median(full_system_oracle_shuffled_context_ablation_delta)) if full_system_oracle_shuffled_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_shuffled_context_ablation_delta)) if full_system_oracle_shuffled_context_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_shuffled_context_ablation_delta.size),
            },
            "full_system_oracle_stale_context_ablation_delta": {
                "median": float(np.median(full_system_oracle_stale_context_ablation_delta)) if full_system_oracle_stale_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_stale_context_ablation_delta)) if full_system_oracle_stale_context_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_stale_context_ablation_delta.size),
            },
            "full_system_oracle_online_refinement_ablation_delta": {
                "median": float(np.median(full_system_oracle_online_refinement_ablation_delta)) if full_system_oracle_online_refinement_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_online_refinement_ablation_delta)) if full_system_oracle_online_refinement_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_online_refinement_ablation_delta.size),
            },
            "full_system_oracle_frozen_context_ablation_delta": {
                "median": float(np.median(full_system_oracle_frozen_context_ablation_delta)) if full_system_oracle_frozen_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_frozen_context_ablation_delta)) if full_system_oracle_frozen_context_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_frozen_context_ablation_delta.size),
            },
            "full_system_oracle_actor_only_ablation_delta": {
                "median": float(np.median(full_system_oracle_actor_only_ablation_delta)) if full_system_oracle_actor_only_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_actor_only_ablation_delta)) if full_system_oracle_actor_only_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_actor_only_ablation_delta.size),
            },
            "full_system_oracle_learned_eval": summarize_matched_eval_rows(
                full_system_oracle_learned_eval_summary_rows
            ),
            "full_system_oracle_zero_context_eval": summarize_matched_eval_rows(
                full_system_oracle_zero_context_eval_summary_rows
            ),
            "full_system_oracle_shuffled_context_eval": summarize_matched_eval_rows(
                full_system_oracle_shuffled_context_eval_summary_rows
            ),
            "full_system_oracle_stale_context_eval": summarize_matched_eval_rows(
                full_system_oracle_stale_context_eval_summary_rows
            ),
            "full_system_oracle_online_refinement_eval": summarize_matched_eval_rows(
                full_system_oracle_online_refinement_eval_summary_rows
            ),
            "full_system_oracle_frozen_context_eval": summarize_matched_eval_rows(
                full_system_oracle_frozen_context_eval_summary_rows
            ),
            "full_system_oracle_actor_only_eval": summarize_matched_eval_rows(
                full_system_oracle_actor_only_eval_summary_rows
            ),
            "probe_readiness_reason_counts": readiness_reason_totals,
            "probe_readiness_component_means": readiness_component_means,
            "probe_fair_stop_blocker_counts": fair_stop_blocker_totals,
            "probe_shadow_blocker_counts": shadow_blocker_totals,
            "probe_second_probe_selection_count": second_probe_selection_totals,
            "probe_fair_handoff_pair_count": fair_handoff_pair_totals,
            "probe_strict_usage_counts": strict_usage_counts,
        },
    }
