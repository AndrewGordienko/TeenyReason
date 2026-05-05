"""Dashboard-side diagnostics and summary helpers."""

from __future__ import annotations

import numpy as np

from ..cognition.representation.metrics import compute_linear_env_fit


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
) -> dict[str, float | np.ndarray]:
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


def summarize_solve_array(values: np.ndarray, caps: np.ndarray) -> dict[str, float | int | list[int]]:
    """Mirror the console benchmark summary inside the dashboard."""
    values_list = values.astype(np.int64).tolist()
    caps_list = caps.astype(np.int64).tolist()
    if not any(value > 0 for value in values_list) and not any(cap > 0 for cap in caps_list):
        return {
            "success_rate": 0,
            "count": 0,
            "median": 0.0,
            "mean": 0.0,
            "values": [],
            "not_run": True,
        }
    solved_values = [value if value > 0 else None for value in values_list]
    capped = [value if value is not None else cap for value, cap in zip(solved_values, caps_list)]
    return {
        "success_rate": int(sum(1 for value in solved_values if value is not None)),
        "count": int(len(values_list)),
        "median": float(np.median(np.asarray(capped, dtype=np.float32))),
        "mean": float(np.mean(np.asarray(capped, dtype=np.float32))),
        "values": values_list,
        "not_run": False,
    }
