"""Probe-run reporting and checkpoint helpers.

These helpers are intentionally boring. They keep score aggregation, policy
snapshotting, and log formatting out of the PPO training loop so the main loop
stays focused on control flow instead of bookkeeping.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..core import RunningNormalizer


def default_family_metric_counter(family_names: tuple[str, ...]) -> dict[str, float]:
    """Create one simple zero-filled scalar counter per probe family."""
    return {family: 0.0 for family in family_names}


def default_family_score_counter(
    family_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Create stable per-family bookkeeping for expected-gain summaries."""
    metric_names = (
        "predicted_mechanics_reduction",
        "raw_predicted_future_error_reduction",
        "predicted_future_error_reduction",
        "future_gain_for_choice",
        "predicted_split_reduction",
        "predicted_entropy_reduction",
        "predicted_hypothesis_separation",
        "diversity_bonus",
        "coverage_bonus",
        "quota_bonus",
        "repeat_penalty",
        "global_repeat_penalty",
        "realized_gain_calibration",
        "realized_gain_bonus",
        "raw_future_error_estimate",
        "future_error_estimate",
        "signature_norm",
        "estimated_probe_cost",
        "predicted_marginal_value",
        "value_per_probe_step",
        "score",
        "selection_score",
    )
    return {
        family: {name: 0.0 for name in metric_names}
        for family in family_names
    }


def update_family_score_counter(
    totals: dict[str, dict[str, float]],
    counts: dict[str, int],
    rows: dict[str, dict[str, float]],
):
    """Accumulate expected-gain rows so the benchmark can report probe logic."""
    for family, metrics in rows.items():
        if family not in totals:
            totals[family] = {name: 0.0 for name in metrics}
        counts[family] = counts.get(family, 0) + 1
        for name, value in metrics.items():
            totals[family][name] = float(totals[family].get(name, 0.0) + float(value))


def update_family_scalar_counter(
    totals: dict[str, float],
    counts: dict[str, int],
    rows: dict[str, float],
):
    """Accumulate scalar family diagnostics such as realized gain or family error."""
    for family, value in rows.items():
        totals[family] = float(totals.get(family, 0.0) + float(value))
        counts[family] = counts.get(family, 0) + 1


def average_family_score_counter(
    totals: dict[str, dict[str, float]],
    counts: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Turn accumulated expected-gain totals into stable mean summaries."""
    averaged: dict[str, dict[str, float]] = {}
    for family, metrics in totals.items():
        denom = max(int(counts.get(family, 0)), 1)
        averaged[family] = {
            name: float(value) / float(denom)
            for name, value in metrics.items()
        }
    return averaged


def average_family_scalar_counter(
    totals: dict[str, float],
    counts: dict[str, int],
) -> dict[str, float]:
    """Turn accumulated scalar family diagnostics into stable mean summaries."""
    averaged: dict[str, float] = {}
    for family, value in totals.items():
        denom = max(int(counts.get(family, 0)), 1)
        averaged[family] = float(value) / float(denom)
    return averaged


def snapshot_policy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a policy state dict so later PPO updates do not overwrite it."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def snapshot_normalizer_state(normalizer: RunningNormalizer) -> dict[str, np.ndarray | float]:
    """Clone the running-normalizer state for later checkpoint saving."""
    return {
        "mean": np.asarray(normalizer.mean, dtype=np.float64).copy(),
        "var": np.asarray(normalizer.var, dtype=np.float64).copy(),
        "count": float(normalizer.count),
        "clip": float(normalizer.clip),
    }


def restore_normalizer_state(
    shape: int,
    state: dict[str, np.ndarray | float],
) -> RunningNormalizer:
    """Rebuild one running normalizer from a saved snapshot."""
    normalizer = RunningNormalizer(shape)
    normalizer.mean = np.asarray(state["mean"], dtype=np.float64).copy()
    normalizer.var = np.asarray(state["var"], dtype=np.float64).copy()
    normalizer.count = float(state["count"])
    normalizer.clip = float(state["clip"])
    return normalizer


def format_solve_status(solved_episode: int | None) -> str:
    """Render a compact solved/not-solved status for the episode logs."""
    if solved_episode is None:
        return "no"
    return f"yes@{solved_episode:04d}"


def format_peer_solve_status(peer_label: str, peer_solved_episode: int | None) -> str:
    """Render the sibling run's solve status in the same log-friendly format."""
    if peer_solved_episode is None:
        return f"{peer_label}=pending"
    return f"{peer_label}={peer_solved_episode:04d}"


def format_solve_steps_status(solved_env_steps: int | None) -> str:
    """Render solve-via-env-steps in the same compact style as solve episodes."""
    if solved_env_steps is None:
        return "pending"
    return str(solved_env_steps)
