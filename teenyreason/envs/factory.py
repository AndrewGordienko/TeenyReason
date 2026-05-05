"""Environment construction helpers."""

from __future__ import annotations

import gymnasium as gym

from .continuous_cartpole import ContinuousCartPoleEnv
from .names import CONTINUOUS_CARTPOLE_NAME


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


__all__ = ["make_env"]
