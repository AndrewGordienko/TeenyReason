"""Replayable trajectory data for generic continuous-control collectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .....envs import make_env
from ...gym_mpc import TransitionBatch, assert_box_spaces, probe_action


@dataclass(frozen=True)
class ReplayTrajectory:
    """One replayable episode from a deterministic seed and action trace."""

    seed: int
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    episode_return: float
    returns_to_go: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


def collect_probe_trajectories(
    config,
    *,
    episodes: int | None = None,
) -> tuple[list[ReplayTrajectory], np.ndarray, np.ndarray]:
    """Collect replayable probe trajectories from any Box-observation/Box-action env."""
    episode_count = max(1, int(config.probe_episodes if episodes is None else episodes))
    env = make_env(config.env_name, max_episode_steps=max(1, int(config.probe_steps)))
    try:
        assert_box_spaces(env)
        action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        rng = np.random.default_rng(int(config.seed))
        trajectories = []
        for episode in range(episode_count):
            seed = int(config.seed + episode)
            trajectories.append(collect_probe_trajectory(env, rng, config, action_low, action_high, episode, seed))
        return trajectories, action_low, action_high
    finally:
        env.close()


def collect_probe_trajectory(
    env,
    rng: np.random.Generator,
    config,
    action_low: np.ndarray,
    action_high: np.ndarray,
    episode: int,
    seed: int,
) -> ReplayTrajectory:
    obs, _info = env.reset(seed=int(seed))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    dones: list[float] = []
    for step in range(max(1, int(config.probe_steps))):
        action = probe_action(rng, action_low, action_high, episode=episode, step=step)
        next_obs, reward, terminated, truncated, _info = env.step(action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        done = bool(terminated or truncated)
        observations.append(obs.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        next_observations.append(next_obs.copy())
        dones.append(float(done))
        obs = next_obs
        if done:
            break
    return make_trajectory(
        seed=seed,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
        discount=float(config.discount),
    )


def make_trajectory(
    *,
    seed: int,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    next_observations: list[np.ndarray],
    dones: list[float],
    discount: float,
) -> ReplayTrajectory:
    reward_array = np.asarray(rewards, dtype=np.float32)
    return ReplayTrajectory(
        seed=int(seed),
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=reward_array,
        next_observations=np.asarray(next_observations, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        episode_return=float(np.sum(reward_array)),
        returns_to_go=returns_to_go(reward_array, np.asarray(dones, dtype=np.float32), discount=discount),
    )


def trajectories_to_batch(trajectories: list[ReplayTrajectory]) -> TransitionBatch:
    """Flatten replayable trajectories into the current world-model training table."""
    if not trajectories:
        raise ValueError("cannot build a transition batch from no trajectories")
    return TransitionBatch(
        observations=np.concatenate([item.observations for item in trajectories], axis=0),
        actions=np.concatenate([item.actions for item in trajectories], axis=0),
        rewards=np.concatenate([item.rewards for item in trajectories], axis=0),
        next_observations=np.concatenate([item.next_observations for item in trajectories], axis=0),
        dones=np.concatenate([item.dones for item in trajectories], axis=0),
        episode_returns=np.asarray([item.episode_return for item in trajectories], dtype=np.float32),
    )


def rows_to_trajectory(
    rows: list[dict[str, np.ndarray | float]],
    *,
    seed: int,
    discount: float,
) -> ReplayTrajectory | None:
    if not rows:
        return None
    return make_trajectory(
        seed=seed,
        observations=[np.asarray(row["observation"], dtype=np.float32) for row in rows],
        actions=[np.asarray(row["action"], dtype=np.float32) for row in rows],
        rewards=[float(row["reward"]) for row in rows],
        next_observations=[np.asarray(row["next_observation"], dtype=np.float32) for row in rows],
        dones=[float(row["done"]) for row in rows],
        discount=discount,
    )


def transition_rows(trajectory: ReplayTrajectory, *, end: int | None = None) -> list[dict[str, np.ndarray | float]]:
    limit = trajectory.length if end is None else max(0, min(int(end), trajectory.length))
    return [
        {
            "observation": trajectory.observations[idx].copy(),
            "action": trajectory.actions[idx].copy(),
            "reward": float(trajectory.rewards[idx]),
            "next_observation": trajectory.next_observations[idx].copy(),
            "done": float(trajectory.dones[idx]),
        }
        for idx in range(limit)
    ]


def returns_to_go(rewards: np.ndarray, dones: np.ndarray, *, discount: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(rewards.size - 1, -1, -1):
        if idx < rewards.size - 1 and float(dones[idx]) > 0.5:
            running = 0.0
        running = float(rewards[idx]) + float(discount) * running
        out[idx] = float(running)
    return out


__all__ = [
    "ReplayTrajectory",
    "collect_probe_trajectories",
    "rows_to_trajectory",
    "trajectories_to_batch",
    "transition_rows",
]
