"""Return-weighted actor prior for generic continuous-control MPC."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ...gym_mpc import TransitionBatch, normalize_actions
from ..model import EnsembleMLPWorldModel, returns_to_go


class _ActorNet(nn.Module):
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


@dataclass
class ActorPolicyModel:
    """Small behavior actor trained from high-return crawler data."""

    net: _ActorNet
    action_low: np.ndarray
    action_high: np.ndarray
    stats: dict[str, float]

    @classmethod
    def fit(
        cls,
        batch: TransitionBatch,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        discount: float,
        hidden_dim: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ) -> tuple["ActorPolicyModel", float, dict[str, float]]:
        torch.manual_seed(int(seed))
        observations = np.asarray(batch.observations, dtype=np.float32)
        actions = np.asarray(batch.actions, dtype=np.float32)
        action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        targets = normalize_actions(actions, action_low, action_high)
        returns = returns_to_go(batch.rewards, batch.dones, discount=float(discount))
        weights = actor_sample_weights(returns, sample_weights=sample_weights)

        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32).reshape(-1, 1)
        dataset = TensorDataset(obs_tensor, target_tensor, weight_tensor)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(int(observations.shape[0]), 1),
            replacement=True,
        )
        loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), sampler=sampler)

        net = _ActorNet(
            obs_dim=int(observations.shape[1]),
            action_dim=int(actions.shape[1]),
            hidden_dim=max(4, int(hidden_dim)),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
        last_loss = train_actor_batches(net, optimizer, loader, epochs=max(1, int(epochs)))

        stats = {
            "actor_policy": 1.0,
            "actor_train_loss": float(last_loss),
            "actor_target_return_mean": float(np.mean(returns)) if returns.size else 0.0,
            "actor_target_return_std": float(np.std(returns)) if returns.size else 0.0,
            "actor_target_return_max": float(np.max(returns)) if returns.size else 0.0,
            "actor_weight_mean": float(np.mean(weights)) if weights.size else 0.0,
            "actor_weight_max": float(np.max(weights)) if weights.size else 0.0,
        }
        model = cls(
            net=net.eval(),
            action_low=action_low,
            action_high=action_high,
            stats=stats,
        )
        return model, float(last_loss), stats

    @classmethod
    def fit_supervised(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ) -> tuple["ActorPolicyModel", float, dict[str, float]]:
        """Train an actor directly from teacher actions."""
        torch.manual_seed(int(seed))
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        targets = normalize_actions(actions, action_low, action_high)
        weights = supervised_weights(sample_weights, count=int(observations.shape[0]))
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32).reshape(-1, 1)
        dataset = TensorDataset(obs_tensor, target_tensor, weight_tensor)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(int(observations.shape[0]), 1),
            replacement=True,
        )
        loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), sampler=sampler)
        net = _ActorNet(
            obs_dim=int(observations.shape[1]),
            action_dim=int(actions.shape[1]),
            hidden_dim=max(4, int(hidden_dim)),
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
        last_loss = train_actor_batches(net, optimizer, loader, epochs=max(1, int(epochs)))
        stats = {
            "actor_policy": 1.0,
            "actor_train_loss": float(last_loss),
            "actor_weight_mean": float(np.mean(weights)) if weights.size else 0.0,
            "actor_weight_max": float(np.max(weights)) if weights.size else 0.0,
        }
        model = cls(net=net.eval(), action_low=action_low, action_high=action_high, stats=stats)
        return model, float(last_loss), stats

    def refine_supervised(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ) -> tuple["ActorPolicyModel", float, dict[str, float]]:
        """Continue actor training from this actor's weights."""
        torch.manual_seed(int(seed))
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        targets = normalize_actions(actions, self.action_low, self.action_high)
        weights = supervised_weights(sample_weights, count=int(observations.shape[0]))
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32).reshape(-1, 1)
        dataset = TensorDataset(obs_tensor, target_tensor, weight_tensor)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=max(int(observations.shape[0]), 1),
            replacement=True,
        )
        loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), sampler=sampler)
        net = copy.deepcopy(self.net).train()
        optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))
        last_loss = train_actor_batches(net, optimizer, loader, epochs=max(1, int(epochs)))
        stats = {
            "actor_policy": 1.0,
            "actor_train_loss": float(last_loss),
            "actor_weight_mean": float(np.mean(weights)) if weights.size else 0.0,
            "actor_weight_max": float(np.max(weights)) if weights.size else 0.0,
            "actor_refined_from_prior": 1.0,
        }
        model = ActorPolicyModel(net=net.eval(), action_low=self.action_low.copy(), action_high=self.action_high.copy(), stats=stats)
        return model, float(last_loss), stats

    def predict(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        with torch.no_grad():
            normalized = self.net(torch.as_tensor(observations, dtype=torch.float32)).cpu().numpy()
        return denormalize_actions(normalized, self.action_low, self.action_high)

    def plan_sequence(
        self,
        model: EnsembleMLPWorldModel,
        observation: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        rows: list[np.ndarray] = []
        for _step in range(max(1, int(horizon))):
            action = self.predict(obs)[0].astype(np.float32)
            rows.append(action)
            pred = model.predict_batch(obs, action.reshape(1, -1))
            obs = np.asarray(pred["next_observation"], dtype=np.float32).reshape(1, -1)
        return np.stack(rows, axis=0).astype(np.float32)


def actor_sample_weights(
    returns: np.ndarray,
    *,
    sample_weights: np.ndarray | None,
) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float32).reshape(-1)
    if returns.size == 0:
        return np.ones((1,), dtype=np.float32)
    centered = returns - float(np.median(returns))
    scale = float(np.std(returns) + 1e-4)
    advantage = np.clip(centered / scale, 0.0, 4.0)
    weights = (1.0 + advantage).astype(np.float32)
    if sample_weights is not None:
        extra = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if extra.shape == weights.shape and float(np.mean(extra)) > 0.0:
            weights = weights * (extra / float(np.mean(extra)))
    weights = np.maximum(weights, 1e-3).astype(np.float32)
    return weights / float(np.mean(weights))


def supervised_weights(sample_weights: np.ndarray | None, *, count: int) -> np.ndarray:
    if sample_weights is None:
        return np.ones((max(1, int(count)),), dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    if int(weights.shape[0]) != int(count):
        raise ValueError("sample_weights length must match teacher rows")
    weights = np.maximum(weights, 1e-3).astype(np.float32)
    return weights / float(np.mean(weights))


def train_actor_batches(
    net: _ActorNet,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    *,
    epochs: int,
) -> float:
    last_loss = 0.0
    for _epoch in range(max(1, int(epochs))):
        epoch_losses: list[float] = []
        for obs_batch, target_batch, weight_batch in loader:
            pred = net(obs_batch)
            loss = torch.mean(weight_batch * torch.square(pred - target_batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
        last_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    return float(last_loss)


def denormalize_actions(
    normalized_actions: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
    action_low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    actions = action_low + 0.5 * (np.clip(normalized_actions, -1.0, 1.0) + 1.0) * (action_high - action_low)
    return np.clip(actions, action_low, action_high).astype(np.float32)


__all__ = ["ActorPolicyModel"]
