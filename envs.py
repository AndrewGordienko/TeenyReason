"""Environment helpers used by the rest of the repo.

Two ideas live here:

- how to build the environments the experiments train on
- how to map a small discrete probe action index into a real environment action

The latent/probe code wants a compact action vocabulary even for continuous
control tasks, so this module defines a small library of representative actions.
"""

import gymnasium as gym
import numpy as np

from continuous_cartpole import ContinuousCartPoleEnv


CONTINUOUS_CARTPOLE_NAME = "ContinuousCartPole-v0"
CONTINUOUS_LUNAR_LANDER_NAME = "LunarLanderContinuous-v3"
BIPEDAL_WALKER_NAME = "BipedalWalker-v3"


def make_env(env_name: str, max_episode_steps: int = 500):
    """Construct a Gymnasium environment, including the custom CartPole variant."""
    if env_name == CONTINUOUS_CARTPOLE_NAME:
        env = ContinuousCartPoleEnv()
        return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return gym.make(env_name)


def _build_scalar_action_values(action_space, action_bins: int) -> np.ndarray:
    """Evenly sample a 1-D continuous action range into a small probe grid."""
    low = float(action_space.low[0])
    high = float(action_space.high[0])
    return np.linspace(low, high, action_bins, dtype=np.float32).reshape(-1, 1)


def _build_lunar_lander_action_values() -> np.ndarray:
    """Hand-picked probe actions for LunarLander's 2-D thruster controls."""
    return np.asarray(
        [
            [-1.0, 0.0],   # idle
            [0.0, 0.0],    # half main engine
            [1.0, 0.0],    # full main engine
            [-1.0, -1.0],  # left side engine
            [-1.0, 1.0],   # right side engine
            [0.0, -1.0],   # hover-left
            [0.0, 1.0],    # hover-right
            [1.0, -1.0],   # full-left
            [1.0, 1.0],    # full-right
        ],
        dtype=np.float32,
    )


def _build_bipedal_walker_action_values() -> np.ndarray:
    """Small library of recognizable gait-like actions for BipedalWalker probes."""
    return np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],      # neutral
            [0.5, -0.5, -0.2, 0.2],    # small left stride
            [-0.2, 0.2, 0.5, -0.5],    # small right stride
            [1.0, -1.0, -0.5, 0.5],    # hard left stride
            [-0.5, 0.5, 1.0, -1.0],    # hard right stride
            [0.0, 1.0, 0.0, 1.0],      # crouch
            [0.0, -1.0, 0.0, -1.0],    # extend
            [1.0, 0.0, 1.0, 0.0],      # hips forward
            [-1.0, 0.0, -1.0, 0.0],    # hips back
        ],
        dtype=np.float32,
    )


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
        # Scalar continuous control can share a simple evenly spaced probe grid.
        return _build_scalar_action_values(action_space, action_bins)

    if env_name == CONTINUOUS_LUNAR_LANDER_NAME:
        # Multi-axis environments use a hand-picked library of probe actions instead.
        return _build_lunar_lander_action_values()

    if env_name == BIPEDAL_WALKER_NAME:
        return _build_bipedal_walker_action_values()

    raise ValueError(
        "Multi-dimensional continuous control requires an environment-specific action prototype set"
    )


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
