"""Environment helpers used by the rest of the repo."""

import gymnasium as gym
import numpy as np

from .continuous_cartpole import ContinuousCartPoleEnv


CONTINUOUS_CARTPOLE_NAME = "ContinuousCartPole-v0"
CONTINUOUS_LUNAR_LANDER_NAME = "LunarLanderContinuous-v3"
BIPEDAL_WALKER_NAME = "BipedalWalker-v3"

ENV_DISPLAY_NAMES = {
    CONTINUOUS_CARTPOLE_NAME: "Continuous CartPole",
    CONTINUOUS_LUNAR_LANDER_NAME: "Continuous LunarLander",
    BIPEDAL_WALKER_NAME: "Bipedal Walker",
}


def make_env(
    env_name: str,
    max_episode_steps: int = 500,
    render_mode: str | None = None,
):
    """Construct a Gymnasium environment, including the custom CartPole variant."""
    if env_name == CONTINUOUS_CARTPOLE_NAME:
        env = ContinuousCartPoleEnv(render_mode=render_mode)
        return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if render_mode is None:
        return gym.make(env_name)
    return gym.make(env_name, render_mode=render_mode)


def get_env_display_name(env_name: str) -> str:
    """Human-friendly label used by logs and the dashboard."""
    return ENV_DISPLAY_NAMES.get(env_name, env_name)


def _center_box_action(action_space: gym.spaces.Box) -> np.ndarray:
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    return np.clip(np.zeros_like(low, dtype=np.float32), low, high)


def _unique_action_rows(values: list[np.ndarray]) -> np.ndarray:
    deduped: list[np.ndarray] = []
    for value in values:
        row = np.asarray(value, dtype=np.float32).reshape(-1)
        if any(np.allclose(row, existing, atol=1e-6) for existing in deduped):
            continue
        deduped.append(row)
    return np.stack(deduped, axis=0).astype(np.float32)


def _build_scalar_action_values(action_space, action_bins: int) -> np.ndarray:
    """Evenly sample a 1-D continuous action range into a small probe grid."""
    low = float(action_space.low[0])
    high = float(action_space.high[0])
    return np.linspace(low, high, action_bins, dtype=np.float32).reshape(-1, 1)


def _build_box_action_values(action_space: gym.spaces.Box, action_bins: int) -> np.ndarray:
    """Derive a compact action vocabulary directly from Box bounds."""
    flat_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    flat_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    center = _center_box_action(action_space)
    midpoint = 0.5 * (flat_low + flat_high)
    prototypes: list[np.ndarray] = [center]
    if not np.allclose(midpoint, center, atol=1e-6):
        prototypes.append(midpoint.astype(np.float32))

    axis_order = sorted(
        range(flat_low.shape[0]),
        key=lambda axis: float(abs(flat_high[axis] - flat_low[axis])),
        reverse=True,
    )
    for axis in axis_order:
        low_action = center.copy()
        high_action = center.copy()
        low_action[axis] = flat_low[axis]
        high_action[axis] = flat_high[axis]
        prototypes.extend((low_action, high_action))
        if len(prototypes) >= int(action_bins):
            break

    if len(prototypes) < int(action_bins):
        prototypes.extend((flat_low.copy(), flat_high.copy()))
    if len(prototypes) < int(action_bins) and flat_low.shape[0] > 1:
        alternating = center.copy()
        mirrored = center.copy()
        for axis in range(flat_low.shape[0]):
            alternating[axis] = flat_high[axis] if axis % 2 == 0 else flat_low[axis]
            mirrored[axis] = flat_low[axis] if axis % 2 == 0 else flat_high[axis]
        prototypes.extend((alternating, mirrored))

    return _unique_action_rows(prototypes)[: max(1, int(action_bins))]


def get_action_values(env, action_bins: int, env_name: str | None = None):
    """Return the probe action prototypes for the given environment.

    Discrete environments return `None` because their native action indices are
    already the probe vocabulary. Continuous environments instead return a small
    table of prototype actions.
    """
    action_space = env.action_space
    if hasattr(action_space, "n"):
        return None

    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("Only Box or Discrete action spaces are supported")

    action_dim = int(np.prod(action_space.shape))
    if action_dim == 1:
        return _build_scalar_action_values(action_space, action_bins)
    return _build_box_action_values(action_space, action_bins)


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
