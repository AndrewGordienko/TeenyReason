"""Failure-frontier mining for generic repair attempts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factors import ControlFactorModel
from ..collection.trajectory import ReplayTrajectory, trajectories_to_batch


@dataclass(frozen=True)
class FailureWindow:
    """A replayable prefix where intervention should happen before failure."""

    trajectory_index: int
    start: int
    end: int
    priority: float
    target_delta: np.ndarray
    reference_return: float
    reference_survival: int


class FailureFrontierMiner:
    """Find high-value repair points from bad or collapsing trajectories."""

    def __init__(self, *, horizon: int, max_windows: int):
        self.horizon = max(1, int(horizon))
        self.max_windows = max(1, int(max_windows))

    def mine(
        self,
        trajectories: list[ReplayTrajectory],
        factor_model: ControlFactorModel,
    ) -> list[FailureWindow]:
        if not trajectories:
            return []
        batch = trajectories_to_batch(trajectories)
        scores = factor_model.transition_scores(batch)
        offset = 0
        windows: list[FailureWindow] = []
        for trajectory_index, trajectory in enumerate(trajectories):
            length = int(trajectory.length)
            local_priority = score_trajectory_frontier(
                trajectory,
                scores["factor_priority"][offset : offset + length],
                horizon=self.horizon,
            )
            for index in top_unique_indices(local_priority, count=min(4, length)):
                start = max(0, int(index) - 1)
                end = min(length, start + self.horizon)
                if end <= start:
                    continue
                bad_delta = np.mean(scores["delta_z"][offset + start : offset + end], axis=0)
                windows.append(
                    FailureWindow(
                        trajectory_index=int(trajectory_index),
                        start=int(start),
                        end=int(end),
                        priority=float(local_priority[index]),
                        target_delta=factor_model.repair_target_delta(bad_delta),
                        reference_return=float(np.sum(trajectory.rewards[start:end])),
                        reference_survival=int(end - start),
                    )
                )
            offset += length
        windows.sort(key=lambda item: item.priority, reverse=True)
        return windows[: self.max_windows]


def score_trajectory_frontier(
    trajectory: ReplayTrajectory,
    factor_priority: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    length = int(trajectory.length)
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    rewards = np.asarray(trajectory.rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(trajectory.dones, dtype=np.float32).reshape(-1)
    rtg = np.asarray(trajectory.returns_to_go, dtype=np.float32).reshape(-1)
    reward_bad = np.maximum(float(np.percentile(rewards, 40.0)) - rewards, 0.0)
    rtg_drop = np.maximum(rtg - np.roll(rtg, -1), 0.0)
    rtg_drop[-1] = max(rtg_drop[-1], 0.0)
    terminal_soon = np.zeros((length,), dtype=np.float32)
    terminal_indices = np.flatnonzero(dones > 0.5)
    for terminal in terminal_indices:
        start = max(0, int(terminal) - int(horizon))
        terminal_soon[start : int(terminal) + 1] = np.linspace(0.25, 1.0, int(terminal) - start + 1)
    priority = (
        1.5 * terminal_soon
        + 0.8 * normalize_nonnegative(reward_bad)
        + 0.6 * normalize_nonnegative(rtg_drop)
        + 0.4 * normalize_nonnegative(factor_priority)
    )
    return priority.astype(np.float32)


def top_unique_indices(values: np.ndarray, *, count: int) -> list[int]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return []
    order = np.argsort(values)[::-1]
    out: list[int] = []
    for index in order:
        idx = int(index)
        if all(abs(idx - old) > 1 for old in out):
            out.append(idx)
        if len(out) >= int(count):
            break
    return out


def normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    high = float(np.percentile(values, 90.0)) if values.size else 0.0
    if high <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / high, 0.0, 2.0).astype(np.float32)


__all__ = ["FailureFrontierMiner", "FailureWindow"]
