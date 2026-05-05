"""Compact probe action vocabularies for Gym action spaces."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def get_action_values(env, action_bins: int, env_name: str | None = None):
    """Return probe action prototypes for the given environment.

    Discrete environments return `None` because their native action indices are
    already the probe vocabulary. Continuous environments return a compact table
    of prototype actions derived from the Box bounds.
    """
    del env_name
    action_space = env.action_space
    if hasattr(action_space, "n"):
        return None
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("Only Box or Discrete action spaces are supported")
    if int(np.prod(action_space.shape)) == 1:
        return scalar_action_values(action_space, action_bins)
    return box_action_values(action_space, action_bins)


def get_action_dim(env, action_bins: int, env_name: str | None = None) -> int:
    """Number of discrete probe actions available in this environment."""
    action_values = get_action_values(env, action_bins, env_name=env_name)
    if action_values is None:
        return int(env.action_space.n)
    return int(len(action_values))


def action_index_to_env_action(action_idx: int, action_values):
    """Convert a probe action index into the real action sent to the env."""
    if action_values is None:
        return int(action_idx)
    return np.asarray(action_values[action_idx], dtype=np.float32).reshape(-1)


def scalar_action_values(action_space, action_bins: int) -> np.ndarray:
    low = float(action_space.low[0])
    high = float(action_space.high[0])
    return np.linspace(low, high, action_bins, dtype=np.float32).reshape(-1, 1)


def box_action_values(action_space: gym.spaces.Box, action_bins: int) -> np.ndarray:
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    center = center_box_action(action_space)
    midpoint = 0.5 * (low + high)
    prototypes: list[np.ndarray] = [center]
    if not np.allclose(midpoint, center, atol=1e-6):
        prototypes.append(midpoint.astype(np.float32))

    for axis in widest_axes(low, high):
        low_action = center.copy()
        high_action = center.copy()
        low_action[axis] = low[axis]
        high_action[axis] = high[axis]
        prototypes.extend((low_action, high_action))
        if len(prototypes) >= int(action_bins):
            break

    if len(prototypes) < int(action_bins):
        prototypes.extend((low.copy(), high.copy()))
    if len(prototypes) < int(action_bins) and low.shape[0] > 1:
        prototypes.extend(alternating_corner_actions(low, high, center))
    return unique_action_rows(prototypes)[: max(1, int(action_bins))]


def center_box_action(action_space: gym.spaces.Box) -> np.ndarray:
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    return np.clip(np.zeros_like(low, dtype=np.float32), low, high)


def widest_axes(low: np.ndarray, high: np.ndarray) -> list[int]:
    return sorted(range(low.shape[0]), key=lambda axis: float(abs(high[axis] - low[axis])), reverse=True)


def alternating_corner_actions(low: np.ndarray, high: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alternating = center.copy()
    mirrored = center.copy()
    for axis in range(low.shape[0]):
        alternating[axis] = high[axis] if axis % 2 == 0 else low[axis]
        mirrored[axis] = low[axis] if axis % 2 == 0 else high[axis]
    return alternating, mirrored


def unique_action_rows(values: list[np.ndarray]) -> np.ndarray:
    deduped: list[np.ndarray] = []
    for value in values:
        row = np.asarray(value, dtype=np.float32).reshape(-1)
        if any(np.allclose(row, existing, atol=1e-6) for existing in deduped):
            continue
        deduped.append(row)
    return np.stack(deduped, axis=0).astype(np.float32)


__all__ = ["action_index_to_env_action", "get_action_dim", "get_action_values"]
