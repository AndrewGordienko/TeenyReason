"""Shared progress, solve, and curriculum helpers for probe-policy training."""

import numpy as np
import torch
import torch.nn as nn

from .....crawler.types import ControllerBeliefContext
from ....core import sanitize_numpy


DEFAULT_PROBE_CURRICULUM_SCHEDULE = (
    (250, 1.00),
    (500, 0.75),
    (750, 0.50),
    (900, 0.25),
    (10**9, 0.00),
)


def normalize_full_system_curriculum_schedule(
    schedule: list[tuple[int, float]] | tuple[tuple[int, float], ...] | None,
) -> tuple[tuple[int, float], ...]:
    """Return one boring, explicit oracle-weight schedule for curriculum mode."""
    if schedule is None:
        return tuple(DEFAULT_PROBE_CURRICULUM_SCHEDULE)
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
    """Resolve how much oracle context should be mixed in this episode."""
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


def mix_controller_contexts(
    learned_context,
    oracle_context,
    *,
    oracle_weight: float,
):
    """Blend learned and oracle controller contexts for curriculum training."""
    oracle_weight = float(np.clip(oracle_weight, 0.0, 1.0))
    if oracle_weight >= 1.0:
        return oracle_context
    if oracle_weight <= 0.0:
        return learned_context
    learned_weight = 1.0 - oracle_weight
    return ControllerBeliefContext(
        mechanics_code=sanitize_numpy(
            learned_weight * learned_context.mechanics_code
            + oracle_weight * oracle_context.mechanics_code
        ),
        affordance_code=sanitize_numpy(
            learned_weight * learned_context.affordance_code
            + oracle_weight * oracle_context.affordance_code
        ),
        confidence=float(
            learned_weight * float(learned_context.confidence)
            + oracle_weight * float(oracle_context.confidence)
        ),
        uncertainty_scalar=float(
            learned_weight * float(learned_context.uncertainty_scalar)
            + oracle_weight * float(oracle_context.uncertainty_scalar)
        ),
        metadata={
            "source_kind": "mixed",
            "belief_source": "mixed",
            "solver_message_source": "mixed",
            "oracle_weight": oracle_weight,
            "learned_weight": learned_weight,
            "learned_source_kind": str(learned_context.metadata.get("source_kind", "learned")),
            "oracle_source_kind": str(oracle_context.metadata.get("source_kind", "oracle")),
            "learned_belief_source": str(learned_context.metadata.get("belief_source", "learned")),
            "oracle_belief_source": str(oracle_context.metadata.get("belief_source", "oracle")),
        },
    )


def rolling_solve_average(returns: list[float], window: int) -> float | None:
    """Return the current solve average once the requested window is full."""
    active_window = max(1, int(window))
    if len(returns) < active_window:
        return None
    return float(np.mean(np.asarray(returns[-active_window:], dtype=np.float32)))


def maybe_extend_solve_episode_limit(
    *,
    episode_return: float,
    episode: int,
    episode_limit: int,
    max_episode_limit: int,
    solved_return: float,
    solve_avg_window: int,
) -> int:
    """Give a late threshold hit enough room to become a rolling-average solve."""
    if int(solve_avg_window) <= 1:
        return int(episode_limit)
    if float(episode_return) < float(solved_return):
        return int(episode_limit)

    needed_limit = int(episode) + max(1, int(solve_avg_window)) - 1
    return min(max(int(episode_limit), needed_limit), int(max_episode_limit))


def late_exploitation_entropy_coef(
    *,
    base_entropy_coef: float,
    returns: list[float],
    best_return_so_far: float,
    solved_return: float,
) -> float:
    """Back off exploration once a continuous-control policy is near solved."""
    base_entropy_coef = float(base_entropy_coef)
    if not returns or solved_return <= 0.0:
        return base_entropy_coef

    recent_window = returns[-20:]
    recent_return = float(np.mean(np.asarray(recent_window, dtype=np.float32)))
    best_ratio = float(best_return_so_far) / max(float(solved_return), 1e-6)
    recent_ratio = recent_return / max(float(solved_return), 1e-6)
    progress_ratio = max(best_ratio, recent_ratio)
    if progress_ratio >= 0.95 and recent_ratio >= 0.60:
        return max(1e-5, 0.15 * base_entropy_coef)
    if progress_ratio >= 0.90 and recent_ratio >= 0.50:
        return max(1e-5, 0.30 * base_entropy_coef)
    if progress_ratio >= 0.80 and recent_ratio >= 0.30:
        return max(1e-5, 0.60 * base_entropy_coef)
    return base_entropy_coef


def cap_late_exploitation_action_std(
    *,
    policy: nn.Module,
    returns: list[float],
    best_return_so_far: float,
    solved_return: float,
) -> None:
    """Trim sampling noise after the policy has shown a near-solve gait."""
    if not returns or solved_return <= 0.0 or not hasattr(policy, "log_std"):
        return

    recent_window = returns[-20:]
    recent_return = float(np.mean(np.asarray(recent_window, dtype=np.float32)))
    best_ratio = float(best_return_so_far) / max(float(solved_return), 1e-6)
    recent_ratio = recent_return / max(float(solved_return), 1e-6)
    max_log_std = None
    if best_ratio >= 0.95 and recent_ratio >= 0.60:
        max_log_std = -1.25
    elif best_ratio >= 0.90 and recent_ratio >= 0.50:
        max_log_std = -1.00
    elif best_ratio >= 0.80 and recent_ratio >= 0.30:
        max_log_std = -0.75
    if max_log_std is None:
        return

    with torch.no_grad():
        policy.log_std.clamp_(min=-5.0, max=float(max_log_std))
