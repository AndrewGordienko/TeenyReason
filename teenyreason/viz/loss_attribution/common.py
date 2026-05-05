"""Shared numeric helpers for loss attribution reports."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np


def _finite_values(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for item in values:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            result.append(value)
    return result

def _median(values: Iterable[object], default: float = 0.0) -> float:
    finite = _finite_values(values)
    if not finite:
        return float(default)
    return float(np.median(np.asarray(finite, dtype=np.float32)))

def _mean(values: Iterable[object], default: float = 0.0) -> float:
    finite = _finite_values(values)
    if not finite:
        return float(default)
    return float(np.mean(np.asarray(finite, dtype=np.float32)))

def _row_values(rows: list[dict], key: str, *, nonnegative: bool = False) -> list[float]:
    values = _finite_values(row.get(key) for row in rows)
    if nonnegative:
        values = [value for value in values if value >= 0.0]
    return values

def _capped_solve_values(rows: list[dict], solve_key: str, cap_key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        solve = _median([row.get(solve_key)], default=math.nan)
        cap = _median([row.get(cap_key)], default=math.nan)
        if np.isfinite(solve) and solve >= 0.0:
            values.append(float(solve))
        elif np.isfinite(cap) and cap >= 0.0:
            values.append(float(cap))
    return values

def _ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return float(numerator / denominator)

def _threshold_row(
    *,
    name: str,
    observed: float,
    target: float,
    direction: str,
    unit: str = "",
) -> dict[str, object]:
    if direction == ">=":
        margin = float(observed - target)
        passed = observed >= target
    elif direction == "<=":
        margin = float(target - observed)
        passed = observed <= target
    elif direction == ">":
        margin = float(observed - target)
        passed = observed > target
    elif direction == "<":
        margin = float(target - observed)
        passed = observed < target
    else:
        raise ValueError(f"Unsupported threshold direction: {direction}")
    ratio = _ratio(observed, target) if target != 0.0 else None
    return {
        "name": name,
        "observed": float(observed),
        "target": float(target),
        "direction": direction,
        "margin": margin,
        "ratio_to_target": ratio,
        "passed": bool(passed),
        "unit": unit,
    }

def _counter_rows(rows: list[dict], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, dict):
            for item, count in value.items():
                try:
                    counts[str(item)] += int(float(count))
                except (TypeError, ValueError):
                    continue
        elif isinstance(value, list):
            for item in value:
                counts[str(item)] += 1
        elif value is not None and str(value):
            counts[str(value)] += 1
    return dict(counts)

def _arm_summary(rows: list[dict], prefix: str) -> dict[str, object]:
    episode_values = _row_values(rows, f"{prefix}_episode_solve", nonnegative=True)
    step_values = _row_values(rows, f"{prefix}_step_solve", nonnegative=True)
    total_values = _row_values(rows, f"{prefix}_total_env_steps", nonnegative=True)
    completed_values = _row_values(rows, f"{prefix}_completed_episodes", nonnegative=True)
    capped_episode_values = _capped_solve_values(
        rows,
        f"{prefix}_episode_solve",
        f"{prefix}_completed_episodes",
    )
    capped_step_values = _capped_solve_values(
        rows,
        f"{prefix}_step_solve",
        f"{prefix}_total_env_steps",
    )
    run_count = len(rows)
    solve_count = max(len(episode_values), len(step_values))
    return {
        "solve_rate": float(solve_count) / float(max(run_count, 1)),
        "solve_episode_median": _median(episode_values, default=-1.0),
        "solve_step_median": _median(step_values, default=-1.0),
        "capped_episode_median": _median(capped_episode_values, default=0.0),
        "capped_step_median": _median(capped_step_values, default=0.0),
        "total_env_steps_median": _median(total_values, default=0.0),
        "completed_episodes_median": _median(completed_values, default=0.0),
        "run_count": int(run_count),
        "solve_count": int(solve_count),
    }

