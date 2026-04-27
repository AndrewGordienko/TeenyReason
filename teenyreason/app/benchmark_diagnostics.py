"""Small derived diagnostics for benchmark artifacts."""

from __future__ import annotations

import numpy as np

from ..viz.diagnostics import compute_mode_leakage


MECHANICS_PROBE_FAMILIES = frozenset(
    (
        "passive_decay",
        "impulse_left",
        "impulse_right",
        "chirp",
        "boundary_push",
        "cart_brake",
    )
)
STRESS_PROBE_FAMILIES = frozenset(("boundary_push", "cart_brake"))
PASSIVE_PROBE_FAMILIES = frozenset(("passive_decay",))


def _mode_shares(modes: np.ndarray) -> dict[str, float | str]:
    """Summarize categorical probe-family coverage with a few stable scalars."""
    mode_names = np.asarray(modes, dtype="U")
    if mode_names.size == 0:
        return {
            "dominant_window_mode": "none",
            "dominant_window_share": 0.0,
            "center_window_share": 0.0,
            "directional_window_share": 0.0,
            "mechanics_window_share": 0.0,
            "passive_window_share": 0.0,
            "stress_window_share": 0.0,
            "noncenter_window_share": 0.0,
            "effective_window_families": 0.0,
        }
    unique, counts = np.unique(mode_names, return_counts=True)
    shares = counts.astype(np.float32) / float(max(mode_names.size, 1))
    dominant_idx = int(np.argmax(counts))
    directional_mask = np.asarray(
        [name.startswith("neg_") or name.startswith("pos_") for name in mode_names],
        dtype=np.bool_,
    )
    mechanics_mask = np.asarray([name in MECHANICS_PROBE_FAMILIES for name in mode_names], dtype=np.bool_)
    passive_mask = np.asarray([name in PASSIVE_PROBE_FAMILIES for name in mode_names], dtype=np.bool_)
    stress_mask = np.asarray([name in STRESS_PROBE_FAMILIES for name in mode_names], dtype=np.bool_)
    return {
        "dominant_window_mode": str(unique[dominant_idx]),
        "dominant_window_share": float(shares[dominant_idx]),
        "center_window_share": float(np.mean(mode_names == "center")),
        "directional_window_share": float(np.mean(directional_mask)),
        "mechanics_window_share": float(np.mean(mechanics_mask)),
        "passive_window_share": float(np.mean(passive_mask)),
        "stress_window_share": float(np.mean(stress_mask)),
        "noncenter_window_share": float(np.mean(mode_names != "center")),
        "effective_window_families": float(1.0 / max(float(np.sum(np.square(shares))), 1e-6)),
    }


def _pairwise_mean(values: np.ndarray) -> float:
    """Return mean off-diagonal Euclidean distance for one small latent table."""
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(rows[:, None, :] - rows[None, :, :], axis=-1)
    mask = ~np.eye(rows.shape[0], dtype=bool)
    return float(np.mean(distances[mask])) if np.any(mask) else 0.0


def _dominant_env_mode_share(modes: np.ndarray) -> float:
    mode_names = np.asarray(modes, dtype="U")
    if mode_names.size == 0:
        return 0.0
    _unique, counts = np.unique(mode_names, return_counts=True)
    return float(np.max(counts) / max(mode_names.size, 1))


def build_latent_support_diagnostics(snapshot: dict[str, np.ndarray]) -> dict[str, float | str]:
    """Build the support/geometry row that explains why a latent is stuck."""
    env_mean = snapshot["env_belief_mean"].astype(np.float32)
    window_mean = snapshot["window_latent_mean"].astype(np.float32)
    window_modes = snapshot["window_probe_mode"].astype("U")
    support_window_mask = snapshot.get(
        "window_is_support",
        np.ones((window_modes.shape[0],), dtype=np.int8),
    ).astype(np.int8)
    support_window_modes = window_modes[support_window_mask > 0]
    env_modes = snapshot.get(
        "env_dominant_probe_mode",
        np.asarray([], dtype="U"),
    ).astype("U")
    support_count = snapshot.get(
        "env_support_count",
        snapshot["env_window_count"],
    ).astype(np.float32)
    support_available_count = snapshot.get(
        "env_support_available_count",
        snapshot["env_window_count"],
    ).astype(np.float32)
    heldout_count = snapshot.get(
        "env_heldout_count",
        np.clip(support_available_count - support_count, 0.0, None),
    ).astype(np.float32)
    support_group_ratio = snapshot.get(
        "env_support_group_ratio",
        np.ones_like(support_count, dtype=np.float32),
    ).astype(np.float32)
    split_group_overlap = snapshot.get(
        "env_split_group_overlap",
        np.zeros_like(support_count, dtype=np.float32),
    ).astype(np.float32)
    cross_family_split_group_overlap = snapshot.get(
        "env_cross_family_split_group_overlap",
        np.zeros_like(support_count, dtype=np.float32),
    ).astype(np.float32)
    cross_family_gap_ratio = snapshot.get(
        "env_cross_family_gap_ratio",
        np.zeros_like(support_count, dtype=np.float32),
    ).astype(np.float32)
    nearest_between = snapshot.get(
        "env_nearest_between_distance",
        np.zeros((env_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    belief_norm = snapshot.get(
        "env_belief_norm",
        np.linalg.norm(env_mean, axis=1).astype(np.float32),
    ).astype(np.float32)

    diagnostics = {
        f"available_{name}": value
        for name, value in _mode_shares(window_modes).items()
    }
    diagnostics.update(_mode_shares(support_window_modes))
    diagnostics.update(
        {
            "support_count_mean": float(np.mean(support_count)) if support_count.size else 0.0,
            "support_available_count_mean": float(np.mean(support_available_count)) if support_available_count.size else 0.0,
            "heldout_count_mean": float(np.mean(heldout_count)) if heldout_count.size else 0.0,
            "support_group_ratio_mean": float(np.mean(support_group_ratio)) if support_group_ratio.size else 0.0,
            "split_group_overlap_mean": float(np.mean(split_group_overlap)) if split_group_overlap.size else 0.0,
            "cross_family_split_group_overlap_mean": (
                float(np.mean(cross_family_split_group_overlap))
                if cross_family_split_group_overlap.size
                else 0.0
            ),
            "cross_family_gap_ratio_mean": (
                float(np.mean(cross_family_gap_ratio))
                if cross_family_gap_ratio.size
                else 0.0
            ),
            "env_dominant_mode_share": _dominant_env_mode_share(env_modes),
            "window_mode_leakage": compute_mode_leakage(window_mean, window_modes),
            "env_mode_leakage": compute_mode_leakage(env_mean, env_modes) if env_modes.size else 0.0,
            "nearest_between_median": float(np.median(nearest_between)) if nearest_between.size else 0.0,
            "pairwise_between_mean": _pairwise_mean(env_mean),
            "belief_norm_std": float(np.std(belief_norm)) if belief_norm.size else 0.0,
        }
    )
    return diagnostics
