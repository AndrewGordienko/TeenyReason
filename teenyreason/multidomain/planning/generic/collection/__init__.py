"""Training-data collectors for generic world-model control."""

from __future__ import annotations

import numpy as np

from ...gym_mpc import TransitionBatch, collect_probe_transitions
from ..config import AdvancedGymMPCConfig
from .frontier import collect_frontier_transitions
from .options import collect_option_archive_transitions
from .curriculum import collect_curriculum_repair_archive_transitions
from .reflex import collect_reflex_archive_transitions
from .replay import collect_replay_frontier_transitions
from .success import collect_success_archive_transitions


def collect_training_transitions(
    config: AdvancedGymMPCConfig,
) -> tuple[TransitionBatch, np.ndarray, np.ndarray, dict[str, object]]:
    """Collect model-training transitions using the configured generic policy."""
    collector = str(config.collector).strip().lower()
    if collector == "frontier":
        return collect_frontier_transitions(config)
    if collector == "replay_frontier":
        return collect_replay_frontier_transitions(config)
    if collector == "success_archive":
        return collect_success_archive_transitions(config)
    if collector == "reflex_archive":
        return collect_reflex_archive_transitions(config)
    if collector == "option_archive":
        return collect_option_archive_transitions(config)
    if collector == "curriculum_repair_archive":
        return collect_curriculum_repair_archive_transitions(config)
    if collector not in ("", "random"):
        raise ValueError(f"unknown generic collector: {config.collector!r}")
    batch, action_low, action_high = collect_probe_transitions(config)
    return batch, action_low, action_high, collector_diagnostics("random", batch, config)


def collector_diagnostics(
    collector: str,
    batch: TransitionBatch,
    config: AdvancedGymMPCConfig,
) -> dict[str, object]:
    returns = np.asarray(batch.episode_returns, dtype=np.float32).reshape(-1)
    best = float(np.max(returns)) if returns.size else 0.0
    return {
        "collector": collector,
        "collector_samples": int(batch.observations.shape[0]),
        "collector_episode_count": int(returns.size),
        "collector_best_return": best,
        "collector_return_mean": float(np.mean(returns)) if returns.size else 0.0,
        "collector_solve_gap": float(float(config.solve_return) - best),
        "collector_interaction_steps": int(batch.observations.shape[0]),
    }


__all__ = ["collect_training_transitions"]
