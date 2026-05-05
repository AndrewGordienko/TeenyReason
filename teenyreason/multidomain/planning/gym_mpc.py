"""Generic continuous-control transition helpers."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from ...envs import make_env


@dataclass(frozen=True)
class TransitionBatch:
    """Flat transition table from generic crawler probes."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    episode_returns: np.ndarray


def collect_probe_transitions(config) -> tuple[TransitionBatch, np.ndarray, np.ndarray]:
    """Collect generic action probes from any Box-observation/Box-action env."""
    env = make_env(config.env_name, max_episode_steps=int(config.probe_steps))
    try:
        assert_box_spaces(env)
        action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        rng = np.random.default_rng(int(config.seed))
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        next_observations: list[np.ndarray] = []
        dones: list[float] = []
        episode_returns: list[float] = []
        for episode in range(max(1, int(config.probe_episodes))):
            obs, _info = env.reset(seed=int(config.seed + episode))
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            episode_return = 0.0
            for step in range(max(1, int(config.probe_steps))):
                action = probe_action(
                    rng,
                    action_low,
                    action_high,
                    episode=episode,
                    step=step,
                )
                next_obs, reward, terminated, truncated, _info = env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                done = bool(terminated or truncated)
                observations.append(obs.copy())
                actions.append(action.copy())
                rewards.append(float(reward))
                next_observations.append(next_obs.copy())
                dones.append(float(done))
                episode_return += float(reward)
                obs = next_obs
                if done:
                    break
            episode_returns.append(float(episode_return))
        return (
            TransitionBatch(
                observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                next_observations=np.asarray(next_observations, dtype=np.float32),
                dones=np.asarray(dones, dtype=np.float32),
                episode_returns=np.asarray(episode_returns, dtype=np.float32),
            ),
            action_low,
            action_high,
        )
    finally:
        env.close()


def assert_box_spaces(env: gym.Env) -> None:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("generic control requires a Box observation space")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError("generic control requires a Box action space")


def probe_action(
    rng: np.random.Generator,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    episode: int,
    step: int,
) -> np.ndarray:
    action_dim = int(action_low.shape[0])
    center = np.clip(np.zeros((action_dim,), dtype=np.float32), action_low, action_high)
    mode = int((episode + step) % (2 * action_dim + 3))
    if mode == 0:
        return center.astype(np.float32)
    axis = (mode - 1) // 2
    if 0 <= axis < action_dim:
        action = center.copy()
        action[axis] = action_low[axis] if mode % 2 == 1 else action_high[axis]
        return action.astype(np.float32)
    return rng.uniform(low=action_low, high=action_high).astype(np.float32)


def normalize_actions(
    actions: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    scale = np.maximum(high - low, 1e-6)
    return (2.0 * (actions - low) / scale - 1.0).astype(np.float32)


__all__ = [
    "TransitionBatch",
    "assert_box_spaces",
    "collect_probe_transitions",
    "normalize_actions",
    "probe_action",
]
