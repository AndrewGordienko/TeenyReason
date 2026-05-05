"""Small closed-loop repair policies for generic skill practice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import IntrinsicGoal


@dataclass(frozen=True)
class ReflexRepairPolicy:
    """Tiny feedback option used for short frontier repair attempts."""

    weights: np.ndarray
    goal_weights: np.ndarray
    phase_weights: np.ndarray
    bias: np.ndarray
    goal_delta: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    smoothing: float = 0.55

    def action(self, observation: np.ndarray, *, step: int, previous_action: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        obs_z = np.clip((obs - self.obs_mean) / np.maximum(self.obs_std, 1e-4), -5.0, 5.0)
        goal = np.asarray(self.goal_delta, dtype=np.float32).reshape(-1)
        phase = phase_features(step)
        raw_z = obs_z @ self.weights + goal @ self.goal_weights + phase @ self.phase_weights + self.bias
        raw = denormalize(np.tanh(raw_z).reshape(1, -1), self.action_low, self.action_high)[0]
        previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
        blend = float(np.clip(self.smoothing, 0.0, 0.95))
        return np.clip(blend * previous + (1.0 - blend) * raw, self.action_low, self.action_high).astype(np.float32)

    def planned_actions(self, model: Any, observation: np.ndarray, *, duration: int) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        previous = np.clip(np.zeros_like(self.action_low, dtype=np.float32), self.action_low, self.action_high)
        actions: list[np.ndarray] = []
        for step in range(max(1, int(duration))):
            action = self.action(obs[0], step=step, previous_action=previous)
            actions.append(action)
            previous = action
            if model is not None and hasattr(model, "predict_batch"):
                pred = model.predict_batch(obs, action.reshape(1, -1))
                obs = np.asarray(pred["next_observation"], dtype=np.float32).reshape(1, -1)
        return np.asarray(actions, dtype=np.float32)


def feedback_repair_candidates(
    config: Any,
    trajectory,
    window,
    goals: list[IntrinsicGoal],
    factor_model,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    seed: int,
) -> list[tuple[ReflexRepairPolicy, IntrinsicGoal]]:
    if not goals:
        return []
    rng = np.random.default_rng(int(seed) + 719 * int(window.start))
    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    obs_dim = int(factor_model.obs_mean.shape[0])
    action_dim = int(action_low.shape[0])
    count = max(2, min(int(getattr(config, "skill_candidate_count", 16)) // 4, 12))
    base = base_bias(trajectory, window, action_low, action_high)
    rows: list[tuple[ReflexRepairPolicy, IntrinsicGoal]] = []
    for index in range(count):
        goal = goals[index % len(goals)]
        goal_dim = int(np.asarray(goal.target_delta, dtype=np.float32).reshape(-1).shape[0])
        scale = float(getattr(config, "reflex_weight_scale", 0.35))
        weights = rng.normal(0.0, scale, size=(obs_dim, action_dim)).astype(np.float32)
        goal_weights = rng.normal(0.0, scale, size=(goal_dim, action_dim)).astype(np.float32)
        phase_weights = rng.normal(0.0, 0.25 * scale, size=(4, action_dim)).astype(np.float32)
        if index == 0:
            weights *= 0.0
            goal_weights *= 0.0
            phase_weights *= 0.0
        policy = ReflexRepairPolicy(
            weights=weights,
            goal_weights=goal_weights,
            phase_weights=phase_weights,
            bias=base + rng.normal(0.0, 0.15, size=(action_dim,)).astype(np.float32),
            goal_delta=np.asarray(goal.target_delta, dtype=np.float32).reshape(-1),
            obs_mean=np.asarray(factor_model.obs_mean, dtype=np.float32).reshape(-1),
            obs_std=np.asarray(factor_model.obs_std, dtype=np.float32).reshape(-1),
            action_low=action_low,
            action_high=action_high,
            smoothing=float(getattr(config, "reflex_action_smoothing", 0.55)),
        )
        rows.append((policy, goal))
    return rows


def action_sequence(candidate: Any, model: Any, observation: np.ndarray, *, duration: int) -> np.ndarray:
    if hasattr(candidate, "planned_actions"):
        return candidate.planned_actions(model, observation, duration=duration)
    sequence = np.asarray(candidate, dtype=np.float32)
    if sequence.shape[0] >= int(duration):
        return sequence[: max(1, int(duration))]
    pad = np.repeat(sequence[-1:].copy(), max(0, int(duration) - sequence.shape[0]), axis=0)
    return np.concatenate([sequence, pad], axis=0)


def base_bias(trajectory, window, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    actions = trajectory.actions[int(window.start) : max(int(window.start) + 1, int(window.end))]
    if actions.size == 0:
        return np.zeros_like(action_low, dtype=np.float32)
    mean_action = np.mean(np.asarray(actions, dtype=np.float32), axis=0).reshape(1, -1)
    return normalize(mean_action, action_low, action_high)[0]


def normalize(actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    return (2.0 * (np.asarray(actions, dtype=np.float32) - low) / np.maximum(high - low, 1e-6) - 1.0).astype(np.float32)


def denormalize(actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    return (low + 0.5 * (np.clip(actions, -1.0, 1.0) + 1.0) * (high - low)).astype(np.float32)


def phase_features(step: int) -> np.ndarray:
    rows: list[float] = []
    for period in (8.0, 16.0):
        phase = 2.0 * np.pi * float(step) / period
        rows.extend([float(np.sin(phase)), float(np.cos(phase))])
    return np.asarray(rows, dtype=np.float32)


__all__ = ["ReflexRepairPolicy", "action_sequence", "feedback_repair_candidates"]
