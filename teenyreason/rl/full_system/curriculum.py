"""Shared curriculum helpers for probe-policy and full-system training."""

from __future__ import annotations

import numpy as np

DEFAULT_FULL_SYSTEM_CURRICULUM_SCHEDULE = (
    (300, 1.00),
    (600, 0.75),
    (850, 0.50),
    (1000, 0.25),
    (10**9, 0.00),
)

DEFAULT_PROBE_CURRICULUM_SCHEDULE = (
    (250, 1.00),
    (500, 0.75),
    (750, 0.50),
    (900, 0.25),
    (10**9, 0.00),
)


def normalize_oracle_curriculum_schedule(
    schedule: list[tuple[int, float]] | tuple[tuple[int, float], ...] | None,
    *,
    default_schedule: tuple[tuple[int, float], ...],
) -> tuple[tuple[int, float], ...]:
    """Return one explicit oracle-weight schedule with clipped weights."""
    if schedule is None:
        return tuple(default_schedule)
    normalized: list[tuple[int, float]] = []
    for end_episode, oracle_weight in schedule:
        normalized.append((int(end_episode), float(np.clip(oracle_weight, 0.0, 1.0))))
    normalized.sort(key=lambda item: item[0])
    if not normalized:
        return ((10**9, 0.0),)
    if normalized[-1][0] < 10**9:
        normalized.append((10**9, normalized[-1][1]))
    return tuple(normalized)


def full_system_oracle_weight_for_episode(
    *,
    context_source: str,
    current_episode: int,
    curriculum_schedule: tuple[tuple[int, float], ...],
) -> float:
    """Resolve oracle-context blend weight for the current episode."""
    source = str(context_source)
    if source == "oracle":
        return 1.0
    if source == "learned":
        return 0.0
    if source != "curriculum":
        raise ValueError(f"Unsupported full-system context source: {context_source}")
    episode_idx = max(1, int(current_episode))
    for end_episode, oracle_weight in curriculum_schedule:
        if episode_idx <= int(end_episode):
            return float(np.clip(oracle_weight, 0.0, 1.0))
    return 0.0


def should_stop_belief_planner_plateau(
    *,
    current_episode: int,
    warmup_episodes: int,
    patience: int,
    last_meaningful_progress_episode: int | None,
) -> bool:
    """Return whether a plateau has lasted long enough to stop early."""
    if int(patience) <= 0:
        return False
    if int(current_episode) < max(1, int(warmup_episodes)):
        return False
    if last_meaningful_progress_episode is None:
        return False
    return (int(current_episode) - int(last_meaningful_progress_episode)) >= int(patience)

