"""Generic controllable-factor scoring for continuous-control trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...gym_mpc import TransitionBatch, normalize_actions
from ..collection.trajectory import ReplayTrajectory, trajectories_to_batch


@dataclass(frozen=True)
class ControlFactorModel:
    """Ranks state deltas by controllability and reward/survival relevance."""

    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    delta_reward_weights: np.ndarray
    delta_terminal_weights: np.ndarray
    delta_scale: np.ndarray

    @classmethod
    def fit(
        cls,
        trajectories: list[ReplayTrajectory],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> "ControlFactorModel":
        batch = trajectories_to_batch(trajectories)
        obs = np.asarray(batch.observations, dtype=np.float32)
        next_obs = np.asarray(batch.next_observations, dtype=np.float32)
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0) + 1e-4
        delta_z = (next_obs - obs) / obs_std.reshape(1, -1)
        rewards = np.asarray(batch.rewards, dtype=np.float32).reshape(-1)
        terminals = np.asarray(batch.dones, dtype=np.float32).reshape(-1)
        return cls(
            obs_mean=obs_mean.astype(np.float32),
            obs_std=obs_std.astype(np.float32),
            action_low=np.asarray(action_low, dtype=np.float32).reshape(-1),
            action_high=np.asarray(action_high, dtype=np.float32).reshape(-1),
            delta_reward_weights=correlation_weights(delta_z, rewards),
            delta_terminal_weights=correlation_weights(delta_z, terminals),
            delta_scale=np.std(delta_z, axis=0).astype(np.float32) + 1e-4,
        )

    def transition_scores(self, batch: TransitionBatch) -> dict[str, np.ndarray]:
        delta_z = self.delta_z(batch.observations, batch.next_observations)
        actions_z = normalize_actions(batch.actions, self.action_low, self.action_high)
        action_energy = np.mean(np.abs(actions_z), axis=1)
        controllability = np.linalg.norm(delta_z / self.delta_scale.reshape(1, -1), axis=1)
        reward_relevance = np.maximum(delta_z @ self.delta_reward_weights, 0.0)
        terminal_relevance = np.maximum(delta_z @ self.delta_terminal_weights, 0.0)
        priority = (0.45 * controllability + 0.35 * reward_relevance + 0.20 * terminal_relevance)
        priority *= 0.5 + action_energy
        return {
            "delta_z": delta_z.astype(np.float32),
            "controllability_score": controllability.astype(np.float32),
            "reward_relevance": reward_relevance.astype(np.float32),
            "terminal_risk_relevance": terminal_relevance.astype(np.float32),
            "factor_priority": priority.astype(np.float32),
        }

    def delta_z(self, observations: np.ndarray, next_observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        next_obs = np.asarray(next_observations, dtype=np.float32)
        return ((next_obs - obs) / self.obs_std.reshape(1, -1)).astype(np.float32)

    def repair_target_delta(self, bad_delta: np.ndarray) -> np.ndarray:
        """Move against terminal-correlated deltas and with reward-correlated deltas."""
        bad = np.asarray(bad_delta, dtype=np.float32).reshape(-1)
        direction = self.delta_reward_weights - np.maximum(self.delta_terminal_weights, 0.0)
        if float(np.linalg.norm(direction)) < 1e-6:
            direction = -bad
        scale = float(np.linalg.norm(bad) + 1e-4)
        direction = direction / float(np.linalg.norm(direction) + 1e-4)
        return (direction * scale).astype(np.float32)


def correlation_weights(deltas: np.ndarray, values: np.ndarray) -> np.ndarray:
    deltas = np.asarray(deltas, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if deltas.size == 0 or values.size == 0 or float(np.std(values)) < 1e-8:
        return np.zeros((deltas.shape[1],), dtype=np.float32)
    centered = values - float(np.mean(values))
    weights = np.mean((deltas - np.mean(deltas, axis=0)) * centered.reshape(-1, 1), axis=0)
    weights /= float(np.std(values) + 1e-4)
    norm = float(np.linalg.norm(weights) + 1e-4)
    return (weights / norm).astype(np.float32)


__all__ = ["ControlFactorModel"]
