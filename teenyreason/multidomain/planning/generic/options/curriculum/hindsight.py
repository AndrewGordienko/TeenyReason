"""Goal-conditioned hindsight actor from internally collected trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....gym_mpc import normalize_actions
from ...collection.trajectory import ReplayTrajectory
from ..factors import ControlFactorModel
from ..models.actor import denormalize_actions, ridge_solve


@dataclass(frozen=True)
class GoalConditionedHindsightPolicy:
    """Linear inverse skill: current obs plus desired factor delta -> action."""

    weights: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    target_dim: int
    train_rows: int
    train_loss: float
    target_delta_std: float

    @classmethod
    def fit(
        cls,
        trajectories: list[ReplayTrajectory],
        factor_model: ControlFactorModel,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        horizon: int,
        ridge: float = 1e-3,
    ) -> "GoalConditionedHindsightPolicy":
        action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        features, targets, target_std = hindsight_training_rows(
            trajectories,
            factor_model,
            action_low,
            action_high,
            horizon=max(1, int(horizon)),
        )
        if features.size == 0:
            feature_dim = 1 + int(factor_model.obs_mean.size) + int(factor_model.obs_mean.size) + 3
            weights = np.zeros((feature_dim, int(action_low.size)), dtype=np.float32)
            loss = 0.0
            rows = 0
        else:
            weights = ridge_solve(features, targets, ridge=float(ridge)).astype(np.float32)
            pred = np.tanh(features @ weights)
            loss = float(np.mean(np.square(pred - targets)))
            rows = int(features.shape[0])
        return cls(
            weights=weights,
            obs_mean=np.asarray(factor_model.obs_mean, dtype=np.float32),
            obs_std=np.asarray(factor_model.obs_std, dtype=np.float32),
            action_low=action_low,
            action_high=action_high,
            target_dim=int(factor_model.obs_mean.size),
            train_rows=int(rows),
            train_loss=float(loss),
            target_delta_std=float(target_std),
        )

    def act(self, observation: np.ndarray, target_delta: np.ndarray, *, phase: int) -> np.ndarray:
        features = hindsight_features(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            np.asarray(target_delta, dtype=np.float32).reshape(1, -1),
            self.obs_mean,
            self.obs_std,
            np.asarray([phase], dtype=np.float32),
        )
        normalized = np.tanh(features @ self.weights)
        return denormalize_actions(normalized, self.action_low, self.action_high)[0]

    def action_sequence(self, observation: np.ndarray, target_delta: np.ndarray, *, duration: int) -> np.ndarray:
        count = max(1, int(duration))
        return np.stack([self.act(observation, target_delta, phase=idx) for idx in range(count)], axis=0).astype(np.float32)

    def diagnostics(self) -> dict[str, object]:
        return {
            "curriculum_hindsight_rows": int(self.train_rows),
            "curriculum_hindsight_train_loss": float(self.train_loss),
            "curriculum_hindsight_target_delta_std": float(self.target_delta_std),
        }


def hindsight_training_rows(
    trajectories: list[ReplayTrajectory],
    factor_model: ControlFactorModel,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_norms: list[float] = []
    for trajectory in trajectories:
        length = int(trajectory.length)
        for start in range(max(0, length - 1)):
            end = min(length, start + max(1, int(horizon)))
            if end <= start:
                continue
            target = factor_delta_to_future(trajectory, factor_model, start, end)
            if not np.all(np.isfinite(target)):
                continue
            local_obs = trajectory.observations[start:end]
            phases = np.arange(local_obs.shape[0], dtype=np.float32)
            repeated_target = np.repeat(target.reshape(1, -1), local_obs.shape[0], axis=0)
            features.append(hindsight_features(local_obs, repeated_target, factor_model.obs_mean, factor_model.obs_std, phases))
            targets.append(normalize_actions(trajectory.actions[start:end], action_low, action_high))
            target_norms.append(float(np.linalg.norm(target)))
    if not features:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, action_low.size), dtype=np.float32), 0.0
    return (
        np.concatenate(features, axis=0).astype(np.float32),
        np.concatenate(targets, axis=0).astype(np.float32),
        float(np.std(np.asarray(target_norms, dtype=np.float32))) if target_norms else 0.0,
    )


def factor_delta_to_future(
    trajectory: ReplayTrajectory,
    factor_model: ControlFactorModel,
    start: int,
    end: int,
) -> np.ndarray:
    start_obs = np.asarray(trajectory.observations[int(start)], dtype=np.float32).reshape(1, -1)
    future_obs = np.asarray(trajectory.next_observations[int(end) - 1], dtype=np.float32).reshape(1, -1)
    return factor_model.delta_z(start_obs, future_obs)[0]


def hindsight_features(
    observations: np.ndarray,
    target_delta: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    target = np.asarray(target_delta, dtype=np.float32)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if target.shape[0] == 1 and obs.shape[0] > 1:
        target = np.repeat(target, obs.shape[0], axis=0)
    phases = np.asarray(phases, dtype=np.float32).reshape(-1, 1)
    obs_z = np.clip((obs - obs_mean.reshape(1, -1)) / np.maximum(obs_std.reshape(1, -1), 1e-4), -5.0, 5.0)
    target_z = np.clip(target, -5.0, 5.0)
    return np.concatenate(
        [
            np.ones((obs.shape[0], 1), dtype=np.float32),
            obs_z,
            target_z,
            np.sin(phases / 4.0),
            np.cos(phases / 4.0),
            phases / 32.0,
        ],
        axis=1,
    ).astype(np.float32)


__all__ = ["GoalConditionedHindsightPolicy"]
