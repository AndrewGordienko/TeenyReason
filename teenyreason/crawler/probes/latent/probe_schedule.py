"""Probe-budget and online-training schedule helpers."""

from __future__ import annotations

import numpy as np

from .probe_belief import LatentPerformanceMemory


def choose_probe_count(
    z: np.ndarray,
    performance_memory: LatentPerformanceMemory,
    base_probe_episodes: int,
    max_probe_episodes: int,
    novelty_probe_threshold: float,
    low_return_probe_threshold: float,
) -> tuple[int, float, float]:
    """Decide how many probe windows to run before the control episode."""
    novelty = performance_memory.novelty(z)
    expected_return = performance_memory.expected_return(z)
    probe_count = base_probe_episodes
    if novelty >= novelty_probe_threshold:
        probe_count += 1
    if len(performance_memory) >= 16 and expected_return < low_return_probe_threshold:
        probe_count += 1
    return min(max_probe_episodes, probe_count), novelty, expected_return


def choose_policy_epochs(
    base_ppo_epochs: int,
    expected_return: float,
    uncertainty: float,
    exploit_return_threshold: float,
    uncertainty_focus_threshold: float,
) -> int:
    """Give promising low-uncertainty episodes a little more PPO update budget."""
    epochs = base_ppo_epochs
    if expected_return >= exploit_return_threshold:
        epochs += 1
        if uncertainty < uncertainty_focus_threshold:
            epochs += 1
    return min(base_ppo_epochs + 2, epochs)


def adjust_entropy_coef(
    base_entropy_coef: float,
    novelty: float,
    expected_return: float,
    uncertainty: float,
    novelty_probe_threshold: float,
    low_return_probe_threshold: float,
    exploit_return_threshold: float,
    uncertainty_focus_threshold: float,
) -> float:
    """Slightly retune exploration pressure from novelty/return/uncertainty."""
    entropy_coef = base_entropy_coef
    if novelty >= novelty_probe_threshold:
        entropy_coef *= 1.08
    if expected_return < low_return_probe_threshold:
        entropy_coef *= 1.08
    if uncertainty >= uncertainty_focus_threshold:
        entropy_coef *= 1.03
    if expected_return >= exploit_return_threshold and uncertainty < 0.5 * uncertainty_focus_threshold:
        entropy_coef *= 0.75
    return float(np.clip(entropy_coef, 1e-4, 0.05))


def should_promote_episode_to_elite(
    episode_return: float,
    completed_returns,
    best_return_so_far: float,
    min_elite_return: float,
    current_episode: int,
    warmup_episodes: int,
    std_scale: float,
) -> tuple[bool, float]:
    """Choose whether this episode should enter the self-imitation buffer."""
    if current_episode <= warmup_episodes:
        return False, min_elite_return
    recent_returns = completed_returns[-20:]
    recent_avg = float(np.mean(recent_returns)) if recent_returns else 0.0
    recent_std = float(np.std(recent_returns)) if recent_returns else 0.0
    threshold = max(min_elite_return, recent_avg + std_scale * recent_std, 0.9 * best_return_so_far)
    return episode_return >= threshold, threshold
