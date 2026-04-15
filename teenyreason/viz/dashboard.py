"""Small localhost dashboard for latent snapshots and benchmark summaries."""

import json
import math
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template

from ..app.benchmark import build_experiment_config
from ..envs import get_env_display_name
from ..representation import list_latent_snapshot_paths, load_latent_snapshot


def sanitize_json_value(value):
    """Convert nested payload values into JSON-safe finite primitives."""
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else 0.0
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def list_benchmark_paths(artifact_dir: Path) -> list[Path]:
    """Find all saved benchmark summary artifacts."""
    return sorted(artifact_dir.glob("*_solve_benchmark.npz"))


def load_benchmark_summary(path: Path) -> dict[str, np.ndarray]:
    """Load one saved benchmark summary artifact."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def downsample_indices(count: int, max_points: int = 1500) -> np.ndarray:
    """Keep the browser payload small enough to stay responsive."""
    if count <= max_points:
        return np.arange(count, dtype=np.int32)
    return np.linspace(0, count - 1, num=max_points, dtype=np.int32)


def sampled_rows(values: np.ndarray, max_rows: int = 640) -> np.ndarray:
    """Keep expensive pairwise diagnostics bounded for large artifacts."""
    if values.shape[0] <= max_rows:
        return values
    return values[downsample_indices(values.shape[0], max_points=max_rows)]


def safe_pairwise_mean_distance(values: np.ndarray) -> float:
    """Average pairwise Euclidean distance across rows."""
    if values.shape[0] < 2:
        return 0.0
    deltas = values[:, None, :] - values[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    mask = ~np.eye(values.shape[0], dtype=bool)
    return float(distances[mask].mean()) if np.any(mask) else 0.0


def compute_linear_env_fit(latent_mean: np.ndarray, env_params: np.ndarray) -> float:
    """Fit env params from latent coordinates as a quick mechanics-alignment sanity check."""
    if latent_mean.shape[0] < 4 or env_params.size == 0:
        return 0.0
    design = np.concatenate(
        [
            latent_mean.astype(np.float32),
            np.ones((latent_mean.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    coef, _residuals, _rank, _singular_values = np.linalg.lstsq(design, env_params, rcond=None)
    prediction = design @ coef
    ss_res = np.sum(np.square(env_params - prediction), axis=0)
    ss_tot = np.sum(
        np.square(env_params - env_params.mean(axis=0, keepdims=True)),
        axis=0,
    )
    valid = ss_tot > 1e-6
    if not np.any(valid):
        return 0.0
    per_dim_r2 = 1.0 - ss_res[valid] / ss_tot[valid]
    return float(np.mean(per_dim_r2))


def compute_per_param_env_fit(
    latent_mean: np.ndarray,
    env_params: np.ndarray,
    param_names: np.ndarray,
) -> list[dict[str, float | str]]:
    """Report one linear env-fit R² per hidden parameter."""
    if latent_mean.shape[0] < 4 or env_params.size == 0:
        return [{"name": str(name), "r2": 0.0} for name in param_names.tolist()]

    design = np.concatenate(
        [
            latent_mean.astype(np.float32),
            np.ones((latent_mean.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    coef, _residuals, _rank, _singular_values = np.linalg.lstsq(design, env_params, rcond=None)
    prediction = design @ coef
    ss_res = np.sum(np.square(env_params - prediction), axis=0)
    ss_tot = np.sum(
        np.square(env_params - env_params.mean(axis=0, keepdims=True)),
        axis=0,
    )
    rows = []
    for idx, name in enumerate(param_names.tolist()):
        if idx >= env_params.shape[1] or ss_tot[idx] <= 1e-6:
            rows.append({"name": str(name), "r2": 0.0})
            continue
        rows.append({"name": str(name), "r2": float(1.0 - ss_res[idx] / ss_tot[idx])})
    return rows


def compute_neighbor_env_alignment(latent_mean: np.ndarray, env_params: np.ndarray) -> float:
    """Measure whether nearby latents tend to correspond to nearby env parameters."""
    latent_rows = sampled_rows(latent_mean.astype(np.float32))
    env_rows = sampled_rows(env_params.astype(np.float32), max_rows=latent_rows.shape[0])
    if latent_rows.shape[0] < 3:
        return 0.0

    latent_deltas = latent_rows[:, None, :] - latent_rows[None, :, :]
    latent_dist = np.sum(np.square(latent_deltas), axis=-1)
    np.fill_diagonal(latent_dist, np.inf)
    nearest_idx = np.argmin(latent_dist, axis=1)

    nn_param_distance = np.linalg.norm(env_rows - env_rows[nearest_idx], axis=1).mean()
    baseline_param_distance = safe_pairwise_mean_distance(env_rows)
    if baseline_param_distance <= 1e-6:
        return 0.0
    return float(1.0 - nn_param_distance / baseline_param_distance)


def compute_split_retrieval_stats(
    split_mean_a: np.ndarray,
    split_mean_b: np.ndarray,
) -> dict[str, float]:
    """Check whether one disjoint support half retrieves the matching other half."""
    if split_mean_a.shape[0] < 2 or split_mean_b.shape[0] != split_mean_a.shape[0]:
        return {
            "top1": 0.0,
            "top5": 0.0,
            "mrr": 0.0,
            "median_rank": 0.0,
            "ranks": np.zeros((0,), dtype=np.int32),
        }
    a = split_mean_a.astype(np.float32)
    b = split_mean_b.astype(np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-6, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-6, None)
    similarity = a @ b.T
    labels = np.arange(a.shape[0], dtype=np.int32)
    ranked_idx = np.argsort(-similarity, axis=1)
    rank_positions = np.argmax(ranked_idx == labels[:, None], axis=1) + 1
    nearest_idx = ranked_idx[:, 0]
    return {
        "top1": float(np.mean(nearest_idx == labels)),
        "top5": float(np.mean(rank_positions <= min(5, a.shape[0]))),
        "mrr": float(np.mean(1.0 / rank_positions)),
        "median_rank": float(np.median(rank_positions)),
        "ranks": rank_positions.astype(np.int32),
    }


def compute_mode_leakage(latent_mean: np.ndarray, probe_modes: np.ndarray) -> float:
    """Estimate how much latent variance is explained by probe mode instead of mechanics."""
    if latent_mean.shape[0] < 4:
        return 0.0
    global_mean = latent_mean.mean(axis=0, keepdims=True)
    total_var = float(np.mean(np.var(latent_mean, axis=0)))
    if total_var <= 1e-6:
        return 0.0

    between_var = 0.0
    for mode in np.unique(probe_modes):
        mode_rows = latent_mean[probe_modes == mode]
        if mode_rows.shape[0] == 0:
            continue
        center_shift = mode_rows.mean(axis=0, keepdims=True) - global_mean
        between_var += mode_rows.shape[0] * float(np.mean(np.square(center_shift)))
    between_var /= float(latent_mean.shape[0])
    return float(between_var / total_var)


def compute_same_env_spread(
    split_latent_disagreement: np.ndarray,
    leaveout_shift: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize disagreement between small support-set beliefs from one world."""
    if split_latent_disagreement.size == 0:
        return {"mean": 0.0, "p90": 0.0, "max": 0.0}
    spread_norm = np.asarray(split_latent_disagreement, dtype=np.float32)
    if leaveout_shift is not None and leaveout_shift.size != 0:
        spread_norm = spread_norm + 0.5 * np.asarray(leaveout_shift, dtype=np.float32)
    return {
        "mean": float(np.mean(spread_norm)),
        "p90": float(np.quantile(spread_norm, 0.90)),
        "max": float(np.max(spread_norm)),
    }


def compute_same_env_gap_ratio(
    env_mean: np.ndarray,
    split_mean_a: np.ndarray,
    split_mean_b: np.ndarray,
) -> dict[str, float]:
    """Compare same-world split disagreement to nearest different-world distance."""
    if env_mean.shape[0] < 2 or split_mean_a.shape[0] != env_mean.shape[0]:
        return {"mean": 0.0, "p90": 0.0}
    same_gap = np.linalg.norm(split_mean_a.astype(np.float32) - split_mean_b.astype(np.float32), axis=1)
    between = np.linalg.norm(
        env_mean.astype(np.float32)[:, None, :] - env_mean.astype(np.float32)[None, :, :],
        axis=-1,
    )
    np.fill_diagonal(between, np.inf)
    nearest_between = np.min(between, axis=1)
    ratio = same_gap / np.clip(nearest_between, 1e-6, None)
    return {
        "mean": float(np.mean(ratio)),
        "p90": float(np.quantile(ratio, 0.90)),
    }


def compute_failure_lift(uncertainty: np.ndarray, terminated: np.ndarray) -> dict[str, float]:
    """Compare termination rates in low- vs high-uncertainty windows."""
    if uncertainty.shape[0] < 8:
        rate = float(np.mean(terminated)) if terminated.size else 0.0
        return {"low_rate": rate, "high_rate": rate, "gap": 0.0, "lift": 1.0}

    low_cut = float(np.quantile(uncertainty, 0.25))
    high_cut = float(np.quantile(uncertainty, 0.75))
    low_mask = uncertainty <= low_cut
    high_mask = uncertainty >= high_cut
    low_rate = float(np.mean(terminated[low_mask])) if np.any(low_mask) else 0.0
    high_rate = float(np.mean(terminated[high_mask])) if np.any(high_mask) else 0.0
    lift = high_rate / max(low_rate, 1e-6)
    return {
        "low_rate": low_rate,
        "high_rate": high_rate,
        "gap": high_rate - low_rate,
        "lift": float(lift),
    }


def compute_uncertainty_error_alignment(
    uncertainty: np.ndarray,
    error: np.ndarray,
) -> dict[str, float]:
    """Check whether high-uncertainty beliefs also have larger mechanics error."""
    if uncertainty.shape[0] < 4 or error.shape[0] != uncertainty.shape[0]:
        mean_error = float(np.mean(error)) if error.size else 0.0
        return {"correlation": 0.0, "low_error": mean_error, "high_error": mean_error, "gap": 0.0}

    centered_u = uncertainty - np.mean(uncertainty)
    centered_e = error - np.mean(error)
    denom = float(np.sqrt(np.sum(centered_u ** 2) * np.sum(centered_e ** 2)))
    correlation = 0.0 if denom <= 1e-6 else float(np.sum(centered_u * centered_e) / denom)

    low_cut = float(np.quantile(uncertainty, 0.25))
    high_cut = float(np.quantile(uncertainty, 0.75))
    low_mask = uncertainty <= low_cut
    high_mask = uncertainty >= high_cut
    low_error = float(np.mean(error[low_mask])) if np.any(low_mask) else float(np.mean(error))
    high_error = float(np.mean(error[high_mask])) if np.any(high_mask) else float(np.mean(error))
    return {
        "correlation": correlation,
        "low_error": low_error,
        "high_error": high_error,
        "gap": high_error - low_error,
    }


def mode_payload_rows(
    probe_mode: np.ndarray,
    uncertainty: np.ndarray,
    terminated: np.ndarray,
) -> list[dict[str, float | int | str]]:
    """Summarize each probe mode with count, uncertainty, and failure pressure."""
    rows = []
    total_count = max(int(probe_mode.shape[0]), 1)
    for mode in np.unique(probe_mode):
        mask = probe_mode == mode
        rows.append(
            {
                "probe_mode": str(mode),
                "count": int(np.sum(mask)),
                "share": float(np.sum(mask) / total_count),
                "uncertainty_mean": float(np.mean(uncertainty[mask])),
                "terminated_rate": float(np.mean(terminated[mask])),
            }
        )
    return sorted(rows, key=lambda row: (-row["count"], row["probe_mode"]))


def build_index_payload(artifact_dir: Path) -> dict:
    """List the available dashboard artifacts."""
    context = load_dashboard_context(artifact_dir)
    return {
        "artifact_dir": str(artifact_dir),
        "latent_snapshots": [path.name for path in list_latent_snapshot_paths(artifact_dir)],
        "benchmark_summaries": [path.name for path in list_benchmark_paths(artifact_dir)],
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
    env_support_group_ratio = snapshot.get(
        "env_support_group_ratio",
        np.ones_like(snapshot.get("env_support_count", snapshot["env_window_count"]), dtype=np.float32),
    )[indices]
    env_view_spread = snapshot.get("env_subset_latent_std", snapshot["env_view_spread"])[indices]
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
    full_future_probe_error = snapshot.get(
        "env_future_prediction_error",
        np.zeros((full_env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_split_mean_a = snapshot.get("env_metric_split_mean_a", snapshot.get("env_split_mean_a", full_env_mean)).astype(np.float32)
    full_split_mean_b = snapshot.get("env_metric_split_mean_b", snapshot.get("env_split_mean_b", full_env_mean)).astype(np.float32)
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
    full_env_subset_shift = snapshot.get(
        "env_subset_shift",
        np.zeros((full_env_view_spread.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_env_leaveout_shift = snapshot.get(
        "env_leaveout_shift",
        np.zeros((full_env_view_spread.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    full_window_probe_mode = snapshot["window_probe_mode"].astype("U")
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
    full_split_retrieval_rank = snapshot.get(
        "env_split_retrieval_rank",
        split_retrieval["ranks"],
    )
    full_split_retrieval_rank = np.asarray(full_split_retrieval_rank, dtype=np.int32)
    diagnostics = {
        "linear_env_fit_r2": compute_linear_env_fit(full_predictive_env_mean, full_env_params),
        "per_param_env_fit_r2": compute_per_param_env_fit(full_predictive_env_mean, full_env_params, full_param_names),
        "neighbor_env_alignment": compute_neighbor_env_alignment(full_env_mean, full_env_params),
        "neighbor_env_alignment_unit": compute_neighbor_env_alignment(full_env_mean_unit, full_env_params),
        "split_retrieval_top1": split_retrieval["top1"],
        "split_retrieval_top5": split_retrieval["top5"],
        "split_retrieval_mrr": split_retrieval["mrr"],
        "split_retrieval_median_rank": split_retrieval["median_rank"],
        "window_mode_leakage": compute_mode_leakage(full_window_latent_mean, full_window_probe_mode),
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
        "env_param_error_mean": float(np.mean(full_env_param_error)),
        "future_probe_error_mean": float(np.mean(full_future_probe_error)),
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
                "support_group_ratio": float(env_support_group_ratio[idx]),
                "terminated": bool(int(terminated[idx])),
                "terminated_numeric": float(terminated[idx]),
                "env_param_mean": float(np.mean(env_params[idx])),
                "env_error": float(full_env_param_error[indices[idx]]),
                "future_probe_error": float(full_future_probe_error[indices[idx]]),
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
            "window_count_mean": float(snapshot["env_window_count"].mean()),
            "support_count_mean": float(snapshot.get("env_support_count", snapshot["env_window_count"]).mean()),
            "support_group_ratio_mean": float(np.mean(
                snapshot.get(
                    "env_support_group_ratio",
                    np.ones_like(snapshot.get("env_support_count", snapshot["env_window_count"]), dtype=np.float32),
                )
            )),
            "pca_explained": [float(value) for value in snapshot["pca_explained"].tolist()],
            "sampled_points": int(len(points)),
        },
        "diagnostics": diagnostics,
        "series": {
            "pairwise_between_distance": full_pairwise_between_distance.astype(np.float32),
            "pairwise_between_distance_unit": full_pairwise_between_distance_unit.astype(np.float32),
            "split_retrieval_rank": full_split_retrieval_rank.astype(np.int32),
        },
        "mode_counts": mode_counts,
        "points": points,
    }


def summarize_solve_array(values: np.ndarray, caps: np.ndarray) -> dict:
    """Mirror the console benchmark summary inside the dashboard."""
    values_list = values.astype(np.int64).tolist()
    caps_list = caps.astype(np.int64).tolist()
    solved_values = [value if value > 0 else None for value in values_list]
    capped = [value if value is not None else cap for value, cap in zip(solved_values, caps_list)]
    return {
        "success_rate": int(sum(1 for value in solved_values if value is not None)),
        "count": int(len(values_list)),
        "median": float(np.median(np.asarray(capped, dtype=np.float32))),
        "mean": float(np.mean(np.asarray(capped, dtype=np.float32))),
        "values": values_list,
    }


def build_benchmark_payload(path: Path) -> dict:
    """Convert one benchmark summary artifact into a JSON-friendly payload."""
    summary = load_benchmark_summary(path)
    env_name = load_optional_string(summary, "env_name")
    seeds = summary["seeds"].astype(np.int64).tolist()
    baseline_episode_solves = summary.get("baseline_episode_solves", summary["baseline_solves"]).astype(np.int64)
    probe_episode_solves = summary.get("probe_episode_solves", summary["probe_solves"]).astype(np.int64)
    baseline_step_solves = summary.get(
        "baseline_step_solves",
        np.full_like(baseline_episode_solves, -1),
    ).astype(np.int64)
    probe_step_solves = summary.get(
        "probe_step_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    baseline_total_env_steps = summary.get(
        "baseline_total_env_steps",
        np.full_like(baseline_episode_solves, 0),
    ).astype(np.int64)
    probe_total_env_steps = summary.get(
        "probe_total_env_steps",
        np.full_like(probe_episode_solves, 0),
    ).astype(np.int64)
    baseline_completed_episodes = summary.get(
        "baseline_completed_episodes",
        np.full_like(baseline_episode_solves, 0),
    ).astype(np.int64)
    probe_completed_episodes = summary.get(
        "probe_completed_episodes",
        np.full_like(probe_episode_solves, 0),
    ).astype(np.int64)
    probe_encoder_steps = summary.get(
        "probe_encoder_steps",
        np.zeros_like(probe_episode_solves),
    ).astype(np.int64)

    rows = []
    for idx, seed in enumerate(seeds):
        rows.append(
            {
                "seed": int(seed),
                "baseline_episode_solve": int(baseline_episode_solves[idx]),
                "probe_episode_solve": int(probe_episode_solves[idx]),
                "baseline_step_solve": int(baseline_step_solves[idx]),
                "probe_step_solve": int(probe_step_solves[idx]),
                "baseline_total_env_steps": int(baseline_total_env_steps[idx]),
                "probe_total_env_steps": int(probe_total_env_steps[idx]),
                "probe_encoder_steps": int(probe_encoder_steps[idx]),
            }
        )

    return {
        "name": path.name,
        "artifact_mtime": float(path.stat().st_mtime),
        "env_name": env_name,
        "env_display_name": None if env_name is None else get_env_display_name(env_name),
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
            "baseline_steps": summarize_solve_array(
                baseline_step_solves,
                baseline_total_env_steps,
            ),
            "probe_steps": summarize_solve_array(
                probe_step_solves,
                probe_total_env_steps,
            ),
        },
    }


def create_dashboard_app(artifact_dir: str | Path = "artifacts") -> Flask:
    """Build the Flask app used for local latent-space inspection."""
    artifact_root = Path(artifact_dir).resolve()
    template_dir = Path(__file__).with_name("templates")
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["ARTIFACT_DIR"] = str(artifact_root)

    @app.after_request
    def add_no_store_headers(response):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.get("/api/index")
    def api_index():
        return jsonify(sanitize_json_value(build_index_payload(artifact_root)))

    @app.get("/api/latent/<path:name>")
    def api_latent(name: str):
        path = artifact_root / name
        if not path.exists() or path.suffix != ".npz":
            return jsonify({"error": f"Unknown latent snapshot: {name}"}), 404
        return jsonify(sanitize_json_value(build_latent_payload(path)))

    @app.get("/api/benchmark/<path:name>")
    def api_benchmark(name: str):
        path = artifact_root / name
        if not path.exists() or path.suffix != ".npz":
            return jsonify({"error": f"Unknown benchmark summary: {name}"}), 404
        return jsonify(sanitize_json_value(build_benchmark_payload(path)))

    return app
