"""Internal self-demo and inverse repair priors for generic options."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..collection.trajectory import ReplayTrajectory, trajectories_to_batch
from .factors import ControlFactorModel


@dataclass(frozen=True)
class MotorPriorModel:
    """Maps desired generic state deltas to actions seen in real trajectories."""

    actions: np.ndarray
    delta_z: np.ndarray
    rewards: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    @classmethod
    def fit(
        cls,
        trajectories: list[ReplayTrajectory],
        factor_model: ControlFactorModel,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "MotorPriorModel":
        batch = trajectories_to_batch(trajectories)
        return cls(
            actions=np.asarray(batch.actions, dtype=np.float32),
            delta_z=factor_model.delta_z(batch.observations, batch.next_observations),
            rewards=np.asarray(batch.rewards, dtype=np.float32).reshape(-1),
            action_low=np.asarray(action_low, dtype=np.float32).reshape(-1),
            action_high=np.asarray(action_high, dtype=np.float32).reshape(-1),
        )

    def repair_sequences(
        self,
        target_delta: np.ndarray,
        base: np.ndarray,
        *,
        count: int,
        seed: int,
    ) -> list[np.ndarray]:
        """Return action windows likely to move in the requested latent direction."""
        target = np.asarray(target_delta, dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(target)) < 1e-6 or self.actions.size == 0:
            return []
        target = target / float(np.linalg.norm(target) + 1e-4)
        align = self.delta_z @ target
        reward_bonus = normalize_nonnegative(self.rewards)
        scores = align + 0.15 * reward_bonus
        order = np.argsort(scores)[::-1]
        rng = np.random.default_rng(int(seed) + 710_000)
        duration = int(np.asarray(base).shape[0])
        scale = np.maximum(self.action_high - self.action_low, 1e-6)
        rows: list[np.ndarray] = []
        for index in order[: max(1, int(count))]:
            action = np.asarray(self.actions[int(index)], dtype=np.float32).reshape(1, -1)
            sequence = np.repeat(action, duration, axis=0)
            if len(rows) > 0:
                noise = rng.normal(0.0, 0.10, size=sequence.shape).astype(np.float32)
                sequence = sequence + noise * scale
            rows.append(np.clip(sequence, self.action_low, self.action_high).astype(np.float32))
        return rows


def select_self_demo_trajectories(
    trajectories: list[ReplayTrajectory],
    *,
    count: int,
) -> list[ReplayTrajectory]:
    """Use only real high-return/high-survival rollouts as internal demonstrations."""
    if not trajectories:
        return []
    ranked: list[ReplayTrajectory] = []
    for key in (
        lambda item: item.episode_return,
        lambda item: item.length,
        lambda item: float(np.max(item.returns_to_go)) if item.returns_to_go.size else -float("inf"),
    ):
        for trajectory in sorted(trajectories, key=key, reverse=True)[: max(1, int(count))]:
            if not any(item is trajectory for item in ranked):
                ranked.append(trajectory)
    return ranked[: max(1, int(count))]


def normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    low = float(np.percentile(values, 20.0))
    high = float(np.percentile(values, 90.0))
    if high <= low + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 2.0).astype(np.float32)


__all__ = ["MotorPriorModel", "select_self_demo_trajectories"]
