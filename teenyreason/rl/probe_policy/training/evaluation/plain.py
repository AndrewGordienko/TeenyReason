"""Deterministic plain-policy evaluation."""

import numpy as np
import torch

from .....crawler.probes.data import apply_env_params
from .....crawler.probes.latent import select_episode_physics
from .....envs import make_env
from ....core import (
    PlainGaussianActorCritic,
    RunningNormalizer,
    mean_to_continuous_action,
    sanitize_numpy,
)


def evaluate_plain_policy(
    policy: PlainGaussianActorCritic,
    state_normalizer: RunningNormalizer,
    env_name: str,
    action_low: np.ndarray,
    action_high: np.ndarray,
    randomize_physics: bool,
    base_physics,
    eval_episodes: int,
    seed: int,
) -> tuple[list[float], int]:
    """Run short deterministic eval episodes before declaring the baseline solved."""
    env = make_env(env_name)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    total_steps = 0
    device = next(policy.parameters()).device

    for eval_episode in range(eval_episodes):
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        apply_env_params(env, episode_physics)
        raw_state, _info = env.reset(seed=seed + eval_episode)
        raw_state = np.asarray(raw_state, dtype=np.float32)
        done = False
        episode_return = 0.0

        while not done:
            state = sanitize_numpy(state_normalizer.normalize(raw_state))
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, _value = policy(state_t)
            action = mean_to_continuous_action(mean, action_low, action_high)
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            total_steps += 1
            raw_state = np.asarray(next_raw_state, dtype=np.float32)
            episode_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(episode_return)

    env.close()
    return returns, total_steps
