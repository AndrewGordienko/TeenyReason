"""Live dashboard trace file loading and JSON normalization."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


LIVE_TRACE_FILENAME = "live_training_trace.json"
LIVE_TRACE_HISTORY_FILENAME = "live_training_history.json"


def clear_live_trace_history(artifact_dir: str | Path = "artifacts") -> None:
    """Remove archived live-trace sessions while leaving the current trace alone."""
    history_path = Path(artifact_dir) / LIVE_TRACE_HISTORY_FILENAME
    try:
        history_path.unlink()
    except FileNotFoundError:
        return


def load_live_trace_payload(artifact_dir: str | Path = "artifacts") -> dict[str, Any]:
    """Load the current live-trace payload or return an empty default."""
    artifact_root = Path(artifact_dir)
    trace_path = artifact_root / LIVE_TRACE_FILENAME
    history_runs = load_live_trace_history(artifact_root)
    if not trace_path.exists():
        return {
            "active": False,
            "finished": False,
            "available": False,
            "env_name": None,
            "env_display_name": None,
            "stage": {"id": "idle", "title": "Awaiting Run", "detail": ""},
            "run": {},
            "focus": {},
            "histories": {},
            "family_scores": [],
            "recent_windows": [],
            "recent_events": [],
            "cartpole_history": [],
            "history_runs": history_runs,
        }
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        payload["history_runs"] = history_runs
        return payload
    except (OSError, json.JSONDecodeError):
        return {
            "active": False,
            "finished": False,
            "available": False,
            "env_name": None,
            "env_display_name": None,
            "stage": {"id": "error", "title": "Live Trace Unavailable", "detail": ""},
            "run": {},
            "focus": {},
            "histories": {},
            "family_scores": [],
            "recent_windows": [],
            "recent_events": [],
            "cartpole_history": [],
            "history_runs": history_runs,
        }


def load_live_trace_history(artifact_dir: str | Path = "artifacts") -> list[dict[str, Any]]:
    """Load archived live-trace sessions for debugging after a run finishes."""
    artifact_root = Path(artifact_dir)
    history_path = artifact_root / LIVE_TRACE_HISTORY_FILENAME
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _summary_solve_values(summary: dict[str, Any], key: str) -> list[int]:
    values = summary.get(key, [])
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for item in values:
        if item is None:
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            result.append(value)
    return result


def _first_numeric_summary_value(summary: dict[str, Any], key: str) -> float | None:
    values = summary.get(key, [])
    if isinstance(values, list) and values:
        try:
            value = float(values[0])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    try:
        value = float(values)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _comparison_encoder_probe_steps(summary: dict[str, Any]) -> int:
    direct = _first_numeric_summary_value(summary, "encoder_probe_steps")
    if direct is not None:
        return max(0, int(direct))
    rows = summary.get("seed_results", [])
    if isinstance(rows, list) and rows:
        row = rows[0]
        if isinstance(row, dict):
            try:
                return max(0, int(row.get("encoder_probe_steps", 0)))
            except (TypeError, ValueError):
                return 0
    return 0


def _archive_solve_display(run_variant: str | None, summary: dict[str, Any]) -> tuple[str, list[int]]:
    """Choose one exact archived solve list for the finished variant."""
    variant = "" if run_variant is None else str(run_variant).lower()
    candidates: list[tuple[str, str]] = []
    if "state-only" in variant:
        candidates.append(("state-only eval means", "full_system_state_only_eval_mean_returns"))
    elif "sim-fanout" in variant:
        candidates.append(("sim-fanout solves", "sim_fanout_episode_solves"))
    elif "belief-controller-oracle" in variant or "belief-planner-oracle" in variant:
        candidates.append(("controller oracle solves", "full_system_oracle_episode_solves"))
    elif "belief-controller" in variant or "belief-native" in variant or "belief-planner" in variant:
        candidates.append(("controller solves", "full_system_episode_solves"))
    elif "probe-noexpr" in variant:
        candidates.append(("probe no-expr solves", "probe_no_expression_episode_solves"))
    elif "probe-shadow" in variant:
        candidates.append(("probe shadow solves", "probe_shadow_episode_solves"))
    candidates.append(("probe solves", "probe_episode_solves"))
    for label, key in candidates:
        values = _summary_solve_values(summary, key)
        if values:
            return label, values
    return "saved for debugging", []




def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, deque):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize_json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else 0.0
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


__all__ = [
    "LIVE_TRACE_FILENAME",
    "LIVE_TRACE_HISTORY_FILENAME",
    "_archive_solve_display",
    "_comparison_encoder_probe_steps",
    "_first_numeric_summary_value",
    "_sanitize_json_value",
    "clear_live_trace_history",
    "load_live_trace_history",
    "load_live_trace_payload",
]
