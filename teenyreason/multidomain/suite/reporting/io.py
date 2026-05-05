"""Small JSON and summary helpers for suite artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def summarize_rows(
    rows: list[dict[str, float | int]],
    *,
    baseline_key: str,
    probe_key: str,
) -> dict[str, float]:
    """Compute a compact mean summary from a small benchmark curve."""
    if not rows:
        return {
            "baseline_mean": 0.0,
            "probe_mean": 0.0,
            "probe_minus_baseline": 0.0,
        }
    baseline_mean = sum(float(row[baseline_key]) for row in rows) / float(len(rows))
    probe_mean = sum(float(row[probe_key]) for row in rows) / float(len(rows))
    return {
        "baseline_mean": baseline_mean,
        "probe_mean": probe_mean,
        "probe_minus_baseline": probe_mean - baseline_mean,
    }


def json_safe(value: Any):
    """Convert benchmark payloads into finite JSON primitives."""
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return json_safe(value.item())
    if torch is not None and isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    return value


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    return path
