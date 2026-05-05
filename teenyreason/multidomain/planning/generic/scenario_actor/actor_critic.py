"""Small actor-critic core used by the scenario actor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ...gym_mpc import TransitionBatch, normalize_actions
from ..config import AdvancedGymMPCConfig
from ..control.actor import denormalize_actions
from ..model import returns_to_go


class _PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class _CriticNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1))


@dataclass
class ActorCriticAgent:
    actor: _PolicyNet
    actor_target: _PolicyNet
    critic1: _CriticNet
    critic2: _CriticNet
    critic1_target: _CriticNet
    critic2_target: _CriticNet
    action_low: np.ndarray
    action_high: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer

    @classmethod
    def create(
        cls,
        batch: TransitionBatch,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: AdvancedGymMPCConfig,
    ) -> "ActorCriticAgent":
        torch.manual_seed(int(config.seed) + 770_000)
        obs = np.asarray(batch.observations, dtype=np.float32)
        obs_mean = np.mean(obs, axis=0).astype(np.float32)
        obs_std = (np.std(obs, axis=0) + 1e-4).astype(np.float32)
        obs_dim = int(obs.shape[1])
        action_dim = int(np.asarray(action_low).reshape(-1).shape[0])
        hidden = max(8, int(config.hidden_dim))
        actor = _PolicyNet(obs_dim, action_dim, hidden)
        actor_target = _PolicyNet(obs_dim, action_dim, hidden)
        critic1 = _CriticNet(obs_dim, action_dim, hidden)
        critic2 = _CriticNet(obs_dim, action_dim, hidden)
        critic1_target = _CriticNet(obs_dim, action_dim, hidden)
        critic2_target = _CriticNet(obs_dim, action_dim, hidden)
        copy_params(actor_target, actor)
        copy_params(critic1_target, critic1)
        copy_params(critic2_target, critic2)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config.actor_critic_lr))
        critic_optimizer = torch.optim.Adam(
            list(critic1.parameters()) + list(critic2.parameters()),
            lr=float(config.actor_critic_lr),
        )
        return cls(
            actor=actor,
            actor_target=actor_target,
            critic1=critic1,
            critic2=critic2,
            critic1_target=critic1_target,
            critic2_target=critic2_target,
            action_low=np.asarray(action_low, dtype=np.float32).reshape(-1),
            action_high=np.asarray(action_high, dtype=np.float32).reshape(-1),
            obs_mean=obs_mean,
            obs_std=obs_std,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = self.obs_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1))
        with torch.no_grad():
            action_z = self.actor(obs).cpu().numpy()
        return denormalize_actions(action_z, self.action_low, self.action_high)[0]

    def obs_tensor(self, observations: np.ndarray) -> torch.Tensor:
        obs = (np.asarray(observations, dtype=np.float32) - self.obs_mean.reshape(1, -1)) / np.maximum(
            self.obs_std.reshape(1, -1),
            1e-4,
        )
        return torch.as_tensor(obs, dtype=torch.float32)


@dataclass
class ReplayBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    priorities: np.ndarray
    is_imagined: np.ndarray
    imagined_count: int = 0

    @classmethod
    def from_batch(cls, batch: TransitionBatch, config: AdvancedGymMPCConfig) -> "ReplayBuffer":
        rewards = np.asarray(batch.rewards, dtype=np.float32).reshape(-1)
        returns = returns_to_go(rewards, batch.dones, discount=float(config.discount))
        return cls(
            observations=np.asarray(batch.observations, dtype=np.float32),
            actions=np.asarray(batch.actions, dtype=np.float32),
            rewards=rewards,
            next_observations=np.asarray(batch.next_observations, dtype=np.float32),
            dones=np.asarray(batch.dones, dtype=np.float32).reshape(-1),
            priorities=transition_priorities(returns, rewards, batch.dones),
            is_imagined=np.zeros((int(rewards.shape[0]),), dtype=np.bool_),
        )

    def append_batch(self, batch: TransitionBatch, config: AdvancedGymMPCConfig) -> None:
        rewards = np.asarray(batch.rewards, dtype=np.float32).reshape(-1)
        returns = returns_to_go(rewards, batch.dones, discount=float(config.discount))
        self.append(
            batch.observations,
            batch.actions,
            rewards,
            batch.next_observations,
            batch.dones,
            transition_priorities(returns, rewards, batch.dones),
            imagined=False,
        )

    def append(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        priorities: np.ndarray,
        *,
        imagined: bool,
    ) -> None:
        count = int(np.asarray(obs).reshape(-1, self.observations.shape[1]).shape[0])
        if count <= 0:
            return
        self.observations = np.concatenate([self.observations, np.asarray(obs, dtype=np.float32).reshape(count, -1)], axis=0)
        self.actions = np.concatenate([self.actions, np.asarray(actions, dtype=np.float32).reshape(count, -1)], axis=0)
        self.rewards = np.concatenate([self.rewards, np.asarray(rewards, dtype=np.float32).reshape(count)], axis=0)
        self.next_observations = np.concatenate([self.next_observations, np.asarray(next_obs, dtype=np.float32).reshape(count, -1)], axis=0)
        self.dones = np.concatenate([self.dones, np.asarray(dones, dtype=np.float32).reshape(count)], axis=0)
        self.priorities = np.concatenate([self.priorities, np.asarray(priorities, dtype=np.float32).reshape(count)], axis=0)
        self.is_imagined = np.concatenate([self.is_imagined, np.full((count,), bool(imagined), dtype=np.bool_)], axis=0)
        if imagined:
            self.imagined_count += count

    def sample(self, rng: np.random.Generator, batch_size: int, *, real_only: bool = False, imagined_only: bool = False) -> dict[str, np.ndarray]:
        indices = self.source_indices(real_only=real_only, imagined_only=imagined_only)
        weights = np.square(np.maximum(self.priorities[indices], 1e-4))
        probs = weights / float(np.sum(weights))
        idx = rng.choice(indices, size=max(1, int(batch_size)), replace=True, p=probs)
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones": self.dones[idx],
        }

    def source_indices(self, *, real_only: bool = False, imagined_only: bool = False) -> np.ndarray:
        if real_only and imagined_only:
            raise ValueError("real_only and imagined_only cannot both be true")
        if real_only:
            indices = np.flatnonzero(~self.is_imagined)
        elif imagined_only:
            indices = np.flatnonzero(self.is_imagined)
        else:
            indices = np.arange(int(self.observations.shape[0]), dtype=np.int64)
        if indices.size == 0:
            return np.arange(int(self.observations.shape[0]), dtype=np.int64)
        return indices.astype(np.int64)

    @property
    def real_count(self) -> int:
        return int(np.sum(~self.is_imagined))

    @property
    def real_observations(self) -> np.ndarray:
        return self.observations[self.source_indices(real_only=True)]

    @property
    def reward_scale(self) -> float:
        real = self.rewards[self.source_indices(real_only=True)]
        return max(float(np.std(real)), 1.0)


def transition_priorities(returns: np.ndarray, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float32).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    if returns.size == 0:
        return np.ones((1,), dtype=np.float32)
    z = (returns - float(np.median(returns))) / float(np.std(returns) + 1e-4)
    reward_z = np.abs(rewards) / float(np.std(rewards) + 1e-4)
    priority = 1.0 + np.clip(z, 0.0, 4.0) + 0.25 * np.clip(reward_z, 0.0, 4.0) + 0.75 * dones
    return (priority / float(np.mean(priority))).astype(np.float32)


def copy_params(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def soft_update(target: nn.Module, source: nn.Module, *, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - float(tau)).add_(source_param.data, alpha=float(tau))


__all__ = ["ActorCriticAgent", "ReplayBuffer", "copy_params", "soft_update"]
