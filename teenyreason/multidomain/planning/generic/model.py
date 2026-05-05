"""Ensemble dynamics model for generic continuous-control MPC."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gym_mpc import TransitionBatch, normalize_actions


class _DynamicsNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class EnsembleMLPWorldModel:
    """Bootstrap ensemble predicting delta-observation, reward, and done risk."""

    def __init__(
        self,
        *,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        reward_mean: float,
        reward_std: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        nets: list[_DynamicsNet],
    ):
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(obs_std.astype(np.float32), 1e-4)
        self.reward_mean = float(reward_mean)
        self.reward_std = max(float(reward_std), 1e-4)
        self.action_low = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)
        self.nets = nets
        self.obs_dim = int(self.obs_mean.shape[0])

    @classmethod
    def fit(
        cls,
        batch: TransitionBatch,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        ensemble_size: int,
        hidden_dim: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ) -> tuple["EnsembleMLPWorldModel", float]:
        torch.manual_seed(int(seed))
        obs = np.asarray(batch.observations, dtype=np.float32)
        next_obs = np.asarray(batch.next_observations, dtype=np.float32)
        actions = np.asarray(batch.actions, dtype=np.float32)
        rewards = np.asarray(batch.rewards, dtype=np.float32).reshape(-1, 1)
        dones = np.asarray(batch.dones, dtype=np.float32).reshape(-1, 1)

        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0) + 1e-4
        reward_mean = float(np.mean(rewards))
        reward_std = float(np.std(rewards) + 1e-4)
        obs_z = (obs - obs_mean) / obs_std
        next_z = (next_obs - obs_mean) / obs_std
        action_z = normalize_actions(actions, action_low, action_high)
        x = torch.tensor(np.concatenate([obs_z, action_z], axis=1), dtype=torch.float32)
        delta_y = torch.tensor(next_z - obs_z, dtype=torch.float32)
        reward_y = torch.tensor((rewards - reward_mean) / reward_std, dtype=torch.float32)
        done_y = torch.tensor(dones, dtype=torch.float32)
        weight_y = make_weight_tensor(sample_weights, count=int(x.shape[0]))

        input_dim = int(x.shape[1])
        output_dim = int(delta_y.shape[1] + 2)
        nets: list[_DynamicsNet] = []
        losses: list[float] = []
        for member in range(max(1, int(ensemble_size))):
            net = _DynamicsNet(input_dim, output_dim, max(8, int(hidden_dim)))
            optimizer = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=1e-4)
            generator = torch.Generator().manual_seed(int(seed) * 9_973 + member)
            losses.append(
                train_member(
                    net,
                    optimizer,
                    x,
                    delta_y,
                    reward_y,
                    done_y,
                    weight_y,
                    generator=generator,
                    epochs=max(1, int(epochs)),
                    batch_size=max(1, int(batch_size)),
                )
            )
            net.eval()
            nets.append(net)

        return (
            cls(
                obs_mean=obs_mean,
                obs_std=obs_std,
                reward_mean=reward_mean,
                reward_std=reward_std,
                action_low=np.asarray(action_low, dtype=np.float32),
                action_high=np.asarray(action_high, dtype=np.float32),
                nets=nets,
            ),
            float(np.mean(losses)) if losses else 0.0,
        )

    def predict_batch(self, observations: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(observations, dtype=np.float32).reshape(-1, self.obs_dim)
        actions = np.asarray(actions, dtype=np.float32).reshape(obs.shape[0], -1)
        obs_z = (obs - self.obs_mean) / self.obs_std
        action_z = normalize_actions(actions, self.action_low, self.action_high)
        pred_mean, pred_var = self._predict_np(obs_z, action_z)
        delta = pred_mean[:, : self.obs_dim]
        next_z = obs_z + delta
        reward = pred_mean[:, self.obs_dim] * self.reward_std + self.reward_mean
        done_risk = sigmoid_np(pred_mean[:, self.obs_dim + 1])
        uncertainty = np.mean(pred_var[:, : self.obs_dim], axis=1) + pred_var[:, self.obs_dim]
        return {
            "next_observation": next_z * self.obs_std + self.obs_mean,
            "reward": reward.astype(np.float32),
            "done_risk": done_risk.astype(np.float32),
            "uncertainty": uncertainty.astype(np.float32),
        }

    def score_sequences(
        self,
        observation: np.ndarray,
        action_sequences: np.ndarray,
        *,
        discount: float,
        done_penalty: float,
        uncertainty_penalty: float,
        value_model: "ValueBootstrapModel | None" = None,
        action_value_model: "ActionValueModel | None" = None,
        action_value_weight: float = 0.0,
        manifold_model=None,
        off_manifold_penalty: float = 0.0,
        value_overestimate_penalty: float = 0.0,
        value_confidence: float = 1.0,
        value_calibration: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        components = self.score_sequence_components(
            observation,
            action_sequences,
            discount=discount,
            done_penalty=done_penalty,
            uncertainty_penalty=uncertainty_penalty,
            value_model=value_model,
            action_value_model=action_value_model,
            action_value_weight=action_value_weight,
            manifold_model=manifold_model,
            off_manifold_penalty=off_manifold_penalty,
            value_overestimate_penalty=value_overestimate_penalty,
            value_confidence=value_confidence,
            value_calibration=value_calibration,
        )
        return components["score"].astype(np.float32), components["uncertainty_total"].astype(np.float32)

    def score_sequence_components(
        self,
        observation: np.ndarray,
        action_sequences: np.ndarray,
        *,
        discount: float,
        done_penalty: float,
        uncertainty_penalty: float,
        value_model: "ValueBootstrapModel | None" = None,
        action_value_model: "ActionValueModel | None" = None,
        action_value_weight: float = 0.0,
        manifold_model=None,
        off_manifold_penalty: float = 0.0,
        value_overestimate_penalty: float = 0.0,
        value_confidence: float = 1.0,
        value_calibration: bool = False,
    ) -> dict[str, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        obs_z = np.repeat((obs - self.obs_mean) / self.obs_std, action_sequences.shape[0], axis=0)
        count = int(action_sequences.shape[0])
        reward_total = np.zeros((count,), dtype=np.float32)
        uncertainty_total = np.zeros((count,), dtype=np.float32)
        done_penalty_total = np.zeros((count,), dtype=np.float32)
        uncertainty_penalty_total = np.zeros((count,), dtype=np.float32)
        off_manifold_total = np.zeros((count,), dtype=np.float32)
        raw_action_value_total = np.zeros((count,), dtype=np.float32)
        if action_value_model is not None and float(action_value_weight) != 0.0:
            first_obs = np.repeat(obs, action_sequences.shape[0], axis=0)
            first_actions = action_sequences[:, 0, :]
            raw_action_value = action_value_model.predict(first_obs, first_actions)
            raw_action_value_total += float(action_value_weight) * raw_action_value
        weight = 1.0
        for step in range(action_sequences.shape[1]):
            action_z = normalize_actions(action_sequences[:, step, :], self.action_low, self.action_high)
            pred_mean, pred_var = self._predict_np(obs_z, action_z)
            delta = pred_mean[:, : self.obs_dim]
            reward = pred_mean[:, self.obs_dim] * self.reward_std + self.reward_mean
            done_risk = sigmoid_np(pred_mean[:, self.obs_dim + 1])
            uncertainty = np.mean(pred_var[:, : self.obs_dim], axis=1) + pred_var[:, self.obs_dim]
            uncertainty_total += float(weight) * uncertainty
            reward_total += float(weight) * reward
            done_penalty_total += float(weight) * float(done_penalty) * done_risk
            uncertainty_penalty_total += float(weight) * float(uncertainty_penalty) * uncertainty
            obs_z = obs_z + delta
            if manifold_model is not None and float(off_manifold_penalty) > 0.0:
                predicted_obs = obs_z * self.obs_std + self.obs_mean
                off_manifold_total += float(weight) * float(off_manifold_penalty) * manifold_model.distance(predicted_obs)
            weight *= float(discount)
        raw_value_total = np.zeros((count,), dtype=np.float32)
        if value_model is not None:
            final_obs = obs_z * self.obs_std + self.obs_mean
            raw_value_total += float(weight) * value_model.predict(final_obs)
        confidence = float(np.clip(value_confidence, 0.0, 1.0)) if bool(value_calibration) else 1.0
        value_total = raw_value_total * confidence
        action_value_total = raw_action_value_total * confidence
        overestimate_total = value_overestimate_penalty_total(
            raw_value_total,
            raw_action_value_total,
            value_model,
            action_value_model,
            penalty=float(value_overestimate_penalty),
        )
        score = reward_total + value_total + action_value_total - done_penalty_total - uncertainty_penalty_total - off_manifold_total - overestimate_total
        return {
            "score": score.astype(np.float32),
            "reward_total": reward_total.astype(np.float32),
            "value_total": value_total.astype(np.float32),
            "action_value_total": action_value_total.astype(np.float32),
            "raw_value_total": raw_value_total.astype(np.float32),
            "raw_action_value_total": raw_action_value_total.astype(np.float32),
            "value_confidence": np.full((count,), confidence, dtype=np.float32),
            "done_penalty_total": done_penalty_total.astype(np.float32),
            "uncertainty_penalty_total": uncertainty_penalty_total.astype(np.float32),
            "uncertainty_total": uncertainty_total.astype(np.float32),
            "off_manifold_penalty_total": off_manifold_total.astype(np.float32),
            "value_overestimate_penalty_total": overestimate_total.astype(np.float32),
        }

    def sequence_summary(
        self,
        observation: np.ndarray,
        action_sequence: np.ndarray,
        *,
        discount: float,
        done_penalty: float,
        uncertainty_penalty: float,
        value_model: "ValueBootstrapModel | None" = None,
        action_value_model: "ActionValueModel | None" = None,
        action_value_weight: float = 0.0,
        manifold_model=None,
        off_manifold_penalty: float = 0.0,
        value_overestimate_penalty: float = 0.0,
        value_confidence: float = 1.0,
        value_calibration: bool = False,
    ) -> dict[str, float]:
        sequence = np.asarray(action_sequence, dtype=np.float32).reshape(1, action_sequence.shape[0], -1)
        components = self.score_sequence_components(
            observation,
            sequence,
            discount=discount,
            done_penalty=done_penalty,
            uncertainty_penalty=uncertainty_penalty,
            value_model=value_model,
            action_value_model=action_value_model,
            action_value_weight=action_value_weight,
            manifold_model=manifold_model,
            off_manifold_penalty=off_manifold_penalty,
            value_overestimate_penalty=value_overestimate_penalty,
            value_confidence=value_confidence,
            value_calibration=value_calibration,
        )
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        pred = self.predict_batch(obs, sequence[:, 0, :])
        action_value = (
            action_value_model.predict(obs, sequence[:, 0, :])[0]
            if action_value_model is not None
            else 0.0
        )
        return {
            "predicted_score": float(components["score"][0]),
            "predicted_uncertainty": float(components["uncertainty_total"][0]),
            "predicted_reward_total": float(components["reward_total"][0]),
            "predicted_value_total": float(components["value_total"][0]),
            "predicted_action_value_total": float(components["action_value_total"][0]),
            "predicted_raw_value_total": float(components["raw_value_total"][0]),
            "predicted_raw_action_value_total": float(components["raw_action_value_total"][0]),
            "predicted_value_confidence": float(components["value_confidence"][0]),
            "predicted_done_penalty_total": float(components["done_penalty_total"][0]),
            "predicted_uncertainty_penalty_total": float(components["uncertainty_penalty_total"][0]),
            "predicted_off_manifold_penalty_total": float(components["off_manifold_penalty_total"][0]),
            "predicted_value_overestimate_penalty_total": float(components["value_overestimate_penalty_total"][0]),
            "predicted_done_risk": float(pred["done_risk"][0]),
            "predicted_first_reward": float(pred["reward"][0]),
            "predicted_action_value": float(action_value),
        }

    def normalized_transition_error(
        self,
        predicted_next: np.ndarray,
        actual_next: np.ndarray,
    ) -> float:
        predicted = np.asarray(predicted_next, dtype=np.float32).reshape(-1)
        actual = np.asarray(actual_next, dtype=np.float32).reshape(-1)
        error = (predicted - actual) / self.obs_std
        return float(np.mean(np.square(error)))

    def _predict_np(self, obs_z: np.ndarray, action_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.tensor(np.concatenate([obs_z, action_z], axis=1), dtype=torch.float32)
        with torch.no_grad():
            preds = torch.stack([net(x) for net in self.nets], dim=0)
        pred_np = preds.detach().cpu().numpy().astype(np.float32)
        return np.mean(pred_np, axis=0), np.var(pred_np, axis=0)


class ValueBootstrapModel:
    """Generic value head trained from discounted returns-to-go."""

    def __init__(
        self,
        *,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        target_mean: float,
        target_std: float,
        target_max: float,
        net: _DynamicsNet,
    ):
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(obs_std.astype(np.float32), 1e-4)
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-4)
        self.target_max = float(target_max)
        self.net = net

    @classmethod
    def fit(
        cls,
        batch: TransitionBatch,
        *,
        discount: float,
        hidden_dim: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        sample_weights: np.ndarray | None = None,
    ) -> tuple["ValueBootstrapModel", float, dict[str, float]]:
        torch.manual_seed(int(seed))
        obs = np.asarray(batch.observations, dtype=np.float32)
        returns = returns_to_go(batch.rewards, batch.dones, discount=float(discount)).reshape(-1, 1)
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0) + 1e-4
        target_mean = float(np.mean(returns))
        target_std = float(np.std(returns) + 1e-4)
        x = torch.tensor((obs - obs_mean) / obs_std, dtype=torch.float32)
        y = torch.tensor((returns - target_mean) / target_std, dtype=torch.float32)
        weights = make_weight_tensor(sample_weights, count=int(x.shape[0]))
        net = _DynamicsNet(int(x.shape[1]), 1, max(8, int(hidden_dim)))
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=1e-4)
        generator = torch.Generator().manual_seed(int(seed) + 67_000)
        loss = train_value_member(
            net,
            optimizer,
            x,
            y,
            weights,
            generator=generator,
            epochs=max(1, int(epochs)),
            batch_size=max(1, int(batch_size)),
        )
        net.eval()
        stats = {
            "value_target_mean": target_mean,
            "value_target_std": target_std,
            "value_target_max": float(np.max(returns)) if returns.size else 0.0,
        }
        return cls(obs_mean=obs_mean, obs_std=obs_std, target_mean=target_mean, target_std=target_std, target_max=stats["value_target_max"], net=net), loss, stats

    def predict(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32).reshape(-1, self.obs_mean.shape[0])
        x = torch.tensor((obs - self.obs_mean) / self.obs_std, dtype=torch.float32)
        with torch.no_grad():
            pred = self.net(x).detach().cpu().numpy().reshape(-1)
        return (pred * self.target_std + self.target_mean).astype(np.float32)


class ActionValueModel:
    """Generic Q-style head trained from observed action returns-to-go."""

    def __init__(
        self,
        *,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        target_mean: float,
        target_std: float,
        target_max: float,
        net: _DynamicsNet,
    ):
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.maximum(obs_std.astype(np.float32), 1e-4)
        self.action_low = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-4)
        self.target_max = float(target_max)
        self.net = net

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
    ) -> tuple["ActionValueModel", float, dict[str, float]]:
        torch.manual_seed(int(seed))
        obs = np.asarray(batch.observations, dtype=np.float32)
        actions = np.asarray(batch.actions, dtype=np.float32)
        returns = returns_to_go(batch.rewards, batch.dones, discount=float(discount)).reshape(-1, 1)
        obs_mean = np.mean(obs, axis=0)
        obs_std = np.std(obs, axis=0) + 1e-4
        target_mean = float(np.mean(returns))
        target_std = float(np.std(returns) + 1e-4)
        obs_z = (obs - obs_mean) / obs_std
        action_z = normalize_actions(actions, action_low, action_high)
        x = torch.tensor(np.concatenate([obs_z, action_z], axis=1), dtype=torch.float32)
        y = torch.tensor((returns - target_mean) / target_std, dtype=torch.float32)
        weights = make_weight_tensor(sample_weights, count=int(x.shape[0]))
        net = _DynamicsNet(int(x.shape[1]), 1, max(8, int(hidden_dim)))
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=1e-4)
        generator = torch.Generator().manual_seed(int(seed) + 91_000)
        loss = train_value_member(
            net,
            optimizer,
            x,
            y,
            weights,
            generator=generator,
            epochs=max(1, int(epochs)),
            batch_size=max(1, int(batch_size)),
        )
        net.eval()
        stats = {
            "action_value_target_mean": target_mean,
            "action_value_target_std": target_std,
            "action_value_target_max": float(np.max(returns)) if returns.size else 0.0,
        }
        return (
            cls(
                obs_mean=obs_mean,
                obs_std=obs_std,
                action_low=np.asarray(action_low, dtype=np.float32),
                action_high=np.asarray(action_high, dtype=np.float32),
                target_mean=target_mean,
                target_std=target_std,
                target_max=stats["action_value_target_max"],
                net=net,
            ),
            loss,
            stats,
        )

    def predict(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32).reshape(-1, self.obs_mean.shape[0])
        actions = np.asarray(actions, dtype=np.float32).reshape(obs.shape[0], -1)
        obs_z = (obs - self.obs_mean) / self.obs_std
        action_z = normalize_actions(actions, self.action_low, self.action_high)
        x = torch.tensor(np.concatenate([obs_z, action_z], axis=1), dtype=torch.float32)
        with torch.no_grad():
            pred = self.net(x).detach().cpu().numpy().reshape(-1)
        return (pred * self.target_std + self.target_mean).astype(np.float32)


def value_overestimate_penalty_total(
    value_total: np.ndarray,
    action_value_total: np.ndarray,
    value_model: ValueBootstrapModel | None,
    action_value_model: ActionValueModel | None,
    *,
    penalty: float,
) -> np.ndarray:
    if float(penalty) <= 0.0:
        return np.zeros_like(np.asarray(value_total, dtype=np.float32))
    value = np.asarray(value_total, dtype=np.float32)
    action_value = np.asarray(action_value_total, dtype=np.float32)
    value_cap = float(getattr(value_model, "target_max", np.inf)) if value_model is not None else np.inf
    action_cap = float(getattr(action_value_model, "target_max", np.inf)) if action_value_model is not None else np.inf
    excess = np.zeros_like(value, dtype=np.float32)
    if np.isfinite(value_cap):
        excess += np.maximum(value - value_cap, 0.0)
    if np.isfinite(action_cap):
        excess += np.maximum(action_value - action_cap, 0.0)
    return (float(penalty) * excess).astype(np.float32)


def train_member(
    net: _DynamicsNet,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    delta_y: torch.Tensor,
    reward_y: torch.Tensor,
    done_y: torch.Tensor,
    weight_y: torch.Tensor,
    *,
    generator: torch.Generator,
    epochs: int,
    batch_size: int,
) -> float:
    count = int(x.shape[0])
    last_loss = torch.tensor(0.0)
    for _epoch in range(max(1, int(epochs))):
        indices = torch.randint(0, count, (count,), generator=generator)
        for start in range(0, count, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            pred = net(x[batch_idx])
            delta_pred = pred[:, : delta_y.shape[1]]
            reward_pred = pred[:, delta_y.shape[1] : delta_y.shape[1] + 1]
            done_logit = pred[:, delta_y.shape[1] + 1 :]
            row_weight = weight_y[batch_idx]
            loss = (
                weighted_mse(delta_pred, delta_y[batch_idx], row_weight)
                + 0.35 * weighted_mse(reward_pred, reward_y[batch_idx], row_weight)
                + 0.15
                * weighted_bce_with_logits(done_logit, done_y[batch_idx], row_weight)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            last_loss = loss.detach()
    return float(last_loss.item())


def train_value_member(
    net: _DynamicsNet,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    target_y: torch.Tensor,
    weight_y: torch.Tensor,
    *,
    generator: torch.Generator,
    epochs: int,
    batch_size: int,
) -> float:
    count = int(x.shape[0])
    last_loss = torch.tensor(0.0)
    for _epoch in range(max(1, int(epochs))):
        indices = torch.randint(0, count, (count,), generator=generator)
        for start in range(0, count, int(batch_size)):
            batch_idx = indices[start : start + int(batch_size)]
            pred = net(x[batch_idx])
            loss = weighted_mse(pred, target_y[batch_idx], weight_y[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            last_loss = loss.detach()
    return float(last_loss.item())


def returns_to_go(rewards: np.ndarray, dones: np.ndarray, *, discount: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(rewards.shape[0] - 1, -1, -1):
        if idx < rewards.shape[0] - 1 and dones[idx] > 0.5:
            running = 0.0
        running = float(rewards[idx]) + float(discount) * running
        returns[idx] = running
    return returns


def make_weight_tensor(sample_weights: np.ndarray | None, *, count: int) -> torch.Tensor:
    if sample_weights is None:
        return torch.ones((count, 1), dtype=torch.float32)
    weights = np.asarray(sample_weights, dtype=np.float32).reshape(-1, 1)
    if weights.shape[0] != count:
        raise ValueError("sample_weights length must match transition count")
    weights = weights / max(float(np.mean(weights)), 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


def weighted_mse(prediction: torch.Tensor, target: torch.Tensor, row_weight: torch.Tensor) -> torch.Tensor:
    per_row = torch.mean(torch.square(prediction - target), dim=1, keepdim=True)
    return torch.mean(per_row * row_weight)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    row_weight: torch.Tensor,
) -> torch.Tensor:
    per_row = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return torch.mean(per_row * row_weight)


def sigmoid_np(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


__all__ = ["ActionValueModel", "EnsembleMLPWorldModel", "ValueBootstrapModel", "returns_to_go"]
