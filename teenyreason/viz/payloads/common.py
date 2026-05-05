"""Shared artifact parsing and small dashboard payload helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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


def normalize_projection_2d(values: np.ndarray) -> np.ndarray:
    """Return a two-column projection array for dashboard scatter plots."""
    projection = np.asarray(values, dtype=np.float32)
    if projection.ndim == 1:
        projection = projection.reshape(-1, 1)
    if projection.ndim != 2:
        projection = np.zeros((0, 2), dtype=np.float32)
    if projection.shape[1] < 2:
        padding = np.zeros((projection.shape[0], 2 - projection.shape[1]), dtype=np.float32)
        projection = np.concatenate([projection, padding], axis=1)
    return projection[:, :2]


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
