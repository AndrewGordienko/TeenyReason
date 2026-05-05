"""Small helper functions for the RL-facing crawler library."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...cognition.representation.metrics import deterministic_rotation
from ..types import EvidenceBatch, EvidenceWindow

def sanitize_array(values: np.ndarray | Sequence[float], fill_value: float = 0.0) -> np.ndarray:
    """Convert arrays into finite float32 vectors."""
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    ).astype(np.float32)


def estimate_probe_family_cost(family: str) -> float:
    """Assign one simple relative interaction cost to each probe family."""
    family_name = str(family)
    if "passive" in family_name:
        return 0.70
    if "chirp" in family_name:
        return 0.90
    if "impulse" in family_name:
        return 1.00
    if "boundary" in family_name:
        return 1.15
    if "brake" in family_name:
        return 1.10
    return 1.00


def quantize_vector(
    values: np.ndarray,
    *,
    bits_per_dim: int,
    use_residual_sketch: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Quantize one message vector with optional residual sketch."""
    vector = sanitize_array(values).reshape(-1)
    if bits_per_dim <= 0 or vector.size == 0:
        return vector, None

    rotation = deterministic_rotation(vector.size, seed=bits_per_dim)
    rotated = rotation @ vector
    levels = max(2, 2 ** int(bits_per_dim))
    scale = float(np.max(np.abs(rotated)))
    if not np.isfinite(scale) or scale <= 1e-6:
        return vector, None

    normalized = np.clip(rotated / scale, -1.0, 1.0)
    grid = np.linspace(-1.0, 1.0, num=levels, dtype=np.float32)
    quantized_idx = np.abs(normalized[:, None] - grid[None, :]).argmin(axis=1)
    quantized_rotated = grid[quantized_idx] * scale
    coarse = rotation.T @ quantized_rotated
    coarse = sanitize_array(coarse)
    if not use_residual_sketch:
        return coarse, None

    residual = sanitize_array(vector - coarse)
    residual_scale = float(np.max(np.abs(residual)))
    if not np.isfinite(residual_scale) or residual_scale <= 1e-6:
        return coarse, np.zeros_like(vector, dtype=np.float32)
    residual_bits = max(2, bits_per_dim // 2)
    residual_levels = max(2, 2 ** residual_bits)
    residual_grid = np.linspace(-1.0, 1.0, num=residual_levels, dtype=np.float32)
    residual_norm = np.clip(residual / residual_scale, -1.0, 1.0)
    residual_idx = np.abs(residual_norm[:, None] - residual_grid[None, :]).argmin(axis=1)
    residual_quantized = residual_grid[residual_idx] * residual_scale
    return coarse, sanitize_array(residual_quantized)


def mean_pairwise_distance(values: np.ndarray) -> float:
    """Return the mean off-diagonal distance across a small set of vectors."""
    rows = sanitize_array(values)
    if rows.ndim != 2 or rows.shape[0] < 2:
        return 0.0
    deltas = rows[:, None, :] - rows[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    mask = ~np.eye(rows.shape[0], dtype=bool)
    if not np.any(mask):
        return 0.0
    return float(np.mean(distances[mask]))


def belief_source_from_mode(belief_mode: str, *, trusted: bool | None = None) -> str:
    """Return the coarse source label used by dashboards and ablations."""
    if str(belief_mode) == "particle_sysid":
        return "sysid"
    if str(belief_mode) == "oracle":
        return "oracle"
    if str(belief_mode) in {"zero", "shuffled", "stale"}:
        return str(belief_mode)
    del trusted
    return "learned"


def build_evidence_batch(
    *,
    windows: dict[str, np.ndarray],
    env_name: str,
    window_size: int,
    action_vocab_size: int,
) -> EvidenceBatch:
    """Convert stored NumPy windows into library-facing evidence objects."""
    evidence_windows: list[EvidenceWindow] = []
    for idx in range(int(windows["states"].shape[0])):
        evidence_windows.append(
            EvidenceWindow(
                states=sanitize_array(windows["states"][idx]),
                actions=np.asarray(windows["actions"][idx], dtype=np.int64),
                rewards=sanitize_array(windows["rewards"][idx]),
                terminated=bool(windows["terminated"][idx]),
                truncated=bool(windows["truncated"][idx]),
                probe_family=str(np.asarray(windows["probe_mode"][idx]).item()),
                env_instance_id=int(windows["env_instance_id"][idx]),
            )
        )
    return EvidenceBatch(
        windows=tuple(evidence_windows),
        env_name=env_name,
        window_size=int(window_size),
        action_vocab_size=int(action_vocab_size),
    )
