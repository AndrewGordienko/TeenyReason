"""Closed-loop option actors trained from mined real segments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....gym_mpc import normalize_actions
from ..segments import OptionSegment


@dataclass(frozen=True)
class OptionActor:
    """Small per-option linear feedback policy with phase features."""

    weights: dict[int, np.ndarray]
    option_durations: dict[int, int]
    option_roi: dict[int, float]
    option_scores: dict[int, float]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    @classmethod
    def fit(
        cls,
        segments: list[OptionSegment],
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        ridge: float = 1e-3,
    ) -> "OptionActor":
        if not segments:
            raise ValueError("cannot fit OptionActor without option segments")
        observations = np.concatenate([segment.observations for segment in segments], axis=0)
        obs_mean = np.mean(observations, axis=0).astype(np.float32)
        obs_std = (np.std(observations, axis=0) + 1e-4).astype(np.float32)
        action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        weights: dict[int, np.ndarray] = {}
        durations: dict[int, int] = {}
        roi: dict[int, float] = {}
        scores: dict[int, float] = {}
        for option_id in sorted({int(segment.option_id) for segment in segments}):
            group = [segment for segment in segments if int(segment.option_id) == option_id]
            features, targets = actor_training_rows(group, obs_mean, obs_std, action_low, action_high)
            weights[option_id] = ridge_solve(features, targets, ridge=float(ridge)).astype(np.float32)
            durations[option_id] = int(round(np.mean([segment.duration for segment in group])))
            roi[option_id] = float(np.mean([segment.real_roi for segment in group]))
            scores[option_id] = float(np.mean([segment.score for segment in group]))
        return cls(weights, durations, roi, scores, obs_mean, obs_std, action_low, action_high)

    @property
    def option_ids(self) -> list[int]:
        return sorted(self.weights)

    def act(self, option_id: int, observation: np.ndarray, *, phase: int) -> np.ndarray:
        option = int(option_id)
        if option not in self.weights:
            option = self.best_option_id()
        features = option_features(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            self.obs_mean,
            self.obs_std,
            np.asarray([phase], dtype=np.float32),
        )
        normalized = np.tanh(features @ self.weights[option])
        return denormalize_actions(normalized, self.action_low, self.action_high)[0]

    def rollout_option(self, option_id: int, observation: np.ndarray, *, duration: int | None = None) -> np.ndarray:
        count = max(1, int(self.option_durations.get(int(option_id), 1) if duration is None else duration))
        return np.stack([self.act(option_id, observation, phase=idx) for idx in range(count)], axis=0).astype(np.float32)

    def best_option_id(self) -> int:
        return max(self.option_ids, key=lambda option_id: self.option_scores.get(option_id, 0.0))


def actor_training_rows(
    segments: list[OptionSegment],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for segment in segments:
        phases = np.arange(segment.observations.shape[0], dtype=np.float32)
        features.append(option_features(segment.observations, obs_mean, obs_std, phases))
        actions.append(normalize_actions(segment.actions, action_low, action_high))
    return np.concatenate(features, axis=0).astype(np.float32), np.concatenate(actions, axis=0).astype(np.float32)


def option_features(
    observations: np.ndarray,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    phases = np.asarray(phases, dtype=np.float32).reshape(-1)
    obs_z = np.clip((obs - obs_mean.reshape(1, -1)) / obs_std.reshape(1, -1), -5.0, 5.0)
    phase = phases.reshape(-1, 1)
    return np.concatenate(
        [
            np.ones((obs.shape[0], 1), dtype=np.float32),
            obs_z,
            np.sin(phase / 4.0),
            np.cos(phase / 4.0),
            phase / 32.0,
        ],
        axis=1,
    ).astype(np.float32)


def denormalize_actions(actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    return (low + 0.5 * (np.clip(actions, -1.0, 1.0) + 1.0) * (high - low)).astype(np.float32)


def ridge_solve(features: np.ndarray, targets: np.ndarray, *, ridge: float) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    reg = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    return np.linalg.solve(x.T @ x + reg, x.T @ y).astype(np.float32)


__all__ = ["OptionActor"]
