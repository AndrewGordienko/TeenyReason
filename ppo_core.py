"""Shared PPO models and optimization utilities.

The training loops in `probe_ppo.py` collect episodes and then call into this
module to:

- normalize observations/rewards
- pack episode data into one training batch
- evaluate squashed Gaussian actions
- run the PPO update itself
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


@dataclass
class EpisodeBatch:
    """Everything PPO needs from one or more collected episodes."""
    states: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    beliefs: np.ndarray | None


def init_linear(layer: nn.Linear, gain: float, bias: float = 0.0) -> nn.Linear:
    """Apply the repo's standard linear-layer initialization."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


class RunningNormalizer:
    """Online mean/variance tracker for observations or rewards."""
    def __init__(self, shape, clip: float = 5.0, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.clip = float(clip)

    def update(self, values):
        values_np = np.asarray(values, dtype=np.float64)
        if values_np.ndim == 1:
            values_np = values_np[None, :]
        if values_np.shape[0] == 0:
            return

        # Update running moments online so rollouts can be normalized incrementally.
        batch_mean = values_np.mean(axis=0)
        batch_var = values_np.var(axis=0)
        batch_count = float(values_np.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (mean_a + mean_b + correction) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count

    def normalize(self, values):
        values_np = np.asarray(values, dtype=np.float32)
        normalized = (values_np - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def scale_only(self, values):
        values_np = np.asarray(values, dtype=np.float32)
        scaled = values_np / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)


class PlainGaussianActorCritic(nn.Module):
    """Standard state-only actor-critic used for the baseline PPO run."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            init_linear(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.actor_mean = init_linear(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.value_head = init_linear(nn.Linear(hidden_dim, 1), gain=1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        mean = self.actor_mean(features)
        value = self.value_head(features).squeeze(-1)
        return mean, value


class ProbeConditionedGaussianActorCritic(nn.Module):
    """Actor-critic that receives both the environment state and probe belief."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        belief_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.actor_net = nn.Sequential(
            init_linear(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.actor_mean = init_linear(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.actor_belief_delta = init_linear(nn.Linear(belief_dim, action_dim), gain=0.01)

        self.value_state_net = nn.Sequential(
            init_linear(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.value_belief_to_gamma = init_linear(
            nn.Linear(belief_dim, hidden_dim), gain=0.5
        )
        self.value_belief_to_beta = init_linear(
            nn.Linear(belief_dim, hidden_dim), gain=0.5
        )
        self.value_head_net = nn.Sequential(
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.value_head = init_linear(nn.Linear(hidden_dim, 1), gain=1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict action mean and value under the current state/belief."""
        actor_features = self.actor_net(state)
        half = belief.shape[-1] // 2
        spread = belief[..., half:]
        uncertainty = spread.mean(dim=-1, keepdim=True)
        trust = torch.clamp(1.0 - 2.0 * uncertainty, 0.0, 1.0)

        # Let the policy stay primarily state-driven and use the probe belief
        # only as a small residual nudge.
        mean = self.actor_mean(actor_features)
        actor_delta = 0.10 * trust * torch.tanh(self.actor_belief_delta(belief))
        mean = mean + actor_delta

        value_features = self.value_state_net(state)
        gamma = trust * torch.tanh(self.value_belief_to_gamma(belief))
        beta = trust * torch.tanh(self.value_belief_to_beta(belief))
        value_features = value_features * (1.0 + 0.75 * gamma) + 0.75 * beta
        value_features = self.value_head_net(value_features)
        value = self.value_head(value_features).squeeze(-1)
        return mean, value


def validate_continuous_env(env):
    """Extract action bounds and ensure the environment is continuous-control."""
    action_space = env.action_space
    if not isinstance(action_space, gym.spaces.Box):
        raise ValueError("PPO path currently supports only Box action spaces")
    action_low = np.asarray(action_space.low, dtype=np.float32)
    action_high = np.asarray(action_space.high, dtype=np.float32)
    return action_low, action_high


def atanh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse tanh for squashed-action log-prob evaluation."""
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def action_scale_bias(
    action_low: np.ndarray,
    action_high: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert env action bounds into affine scale/bias tensors."""
    low = torch.tensor(action_low, dtype=torch.float32, device=device)
    high = torch.tensor(action_high, dtype=torch.float32, device=device)
    scale = 0.5 * (high - low)
    bias = 0.5 * (high + low)
    return scale, bias


def build_tanh_normal(mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
    """Build the unsquashed Gaussian used before tanh action squashing."""
    std = torch.exp(torch.clamp(log_std, -5.0, 1.0))
    return Normal(mean, std)


def sample_continuous_action(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, float]:
    # Sample in unconstrained space, tanh-squash, then rescale into env action bounds.
    dist = build_tanh_normal(mean, log_std)
    raw_action = dist.rsample()
    squashed_action = torch.tanh(raw_action)
    scale, bias = action_scale_bias(action_low, action_high, mean.device)
    action = bias + scale * squashed_action
    log_prob = dist.log_prob(raw_action) - torch.log(scale * (1.0 - squashed_action.pow(2)) + 1e-6)
    return (
        action.squeeze(0).detach().cpu().numpy().astype(np.float32),
        float(log_prob.sum(dim=-1).item()),
    )


def evaluate_continuous_actions(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-prob and entropy for already-sampled bounded actions."""
    dist = build_tanh_normal(mean, log_std)
    scale, bias = action_scale_bias(action_low, action_high, mean.device)
    normalized_action = torch.clamp((actions - bias) / scale, -0.999, 0.999)
    raw_action = atanh(normalized_action)
    log_prob = dist.log_prob(raw_action) - torch.log(scale * (1.0 - normalized_action.pow(2)) + 1e-6)
    return log_prob.sum(dim=-1), dist.entropy().sum(dim=-1)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminals: np.ndarray,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantage estimates and returns for one rollout."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    next_advantage = 0.0
    next_value = bootstrap_value

    # Standard reverse-time GAE pass.
    for idx in range(len(rewards) - 1, -1, -1):
        non_terminal = 1.0 - terminals[idx]
        delta = rewards[idx] + gamma * next_value * non_terminal - values[idx]
        next_advantage = delta + gamma * gae_lambda * non_terminal * next_advantage
        advantages[idx] = next_advantage
        next_value = values[idx]

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def build_episode_batch(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    log_probs: list[float],
    rewards: list[float],
    values: list[float],
    terminals: list[float],
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
    beliefs: list[np.ndarray] | None = None,
) -> EpisodeBatch:
    """Pack one collected episode into the normalized PPO batch structure."""
    rewards_np = np.asarray(rewards, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)
    terminals_np = np.asarray(terminals, dtype=np.float32)
    advantages, returns = compute_gae(
        rewards=rewards_np,
        values=values_np,
        terminals=terminals_np,
        bootstrap_value=bootstrap_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return EpisodeBatch(
        states=np.stack(states, axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.float32),
        old_log_probs=np.asarray(log_probs, dtype=np.float32),
        returns=returns,
        advantages=advantages,
        beliefs=None if beliefs is None else np.stack(beliefs, axis=0).astype(np.float32),
    )


def concat_episode_batches(batches: list[EpisodeBatch]) -> EpisodeBatch:
    """Concatenate several episode batches before an optimizer step."""
    beliefs = None
    if batches[0].beliefs is not None:
        beliefs = np.concatenate([batch.beliefs for batch in batches], axis=0).astype(np.float32)

    return EpisodeBatch(
        states=np.concatenate([batch.states for batch in batches], axis=0).astype(np.float32),
        actions=np.concatenate([batch.actions for batch in batches], axis=0).astype(np.float32),
        old_log_probs=np.concatenate([batch.old_log_probs for batch in batches], axis=0).astype(np.float32),
        returns=np.concatenate([batch.returns for batch in batches], axis=0).astype(np.float32),
        advantages=np.concatenate([batch.advantages for batch in batches], axis=0).astype(np.float32),
        beliefs=beliefs,
    )


def update_ppo_policy(
    model,
    optimizer,
    batch: EpisodeBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
    clip_ratio: float,
    value_loss_weight: float,
    entropy_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    target_kl: float,
    auxiliary_loss_fn=None,
):
    """Run one PPO optimization phase over a collected batch."""
    device = next(model.parameters()).device
    states = torch.tensor(batch.states, dtype=torch.float32, device=device)
    actions = torch.tensor(batch.actions, dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(batch.old_log_probs, dtype=torch.float32, device=device)
    returns = torch.tensor(batch.returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(batch.advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    beliefs = None
    if batch.beliefs is not None:
        beliefs = torch.tensor(batch.beliefs, dtype=torch.float32, device=device)

    total_steps = states.shape[0]
    minibatch_size = min(minibatch_size, total_steps)

    for _ in range(ppo_epochs):
        permutation = torch.randperm(total_steps, device=device)
        stop_early = False

        for start in range(0, total_steps, minibatch_size):
            idx = permutation[start:start + minibatch_size]
            batch_state = states[idx]
            batch_action = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_returns = returns[idx]
            batch_advantages = advantages[idx]

            if beliefs is None:
                mean, value = model(batch_state)
            else:
                mean, value = model(batch_state, beliefs[idx])

            new_log_prob, entropy = evaluate_continuous_actions(
                mean=mean,
                log_std=model.log_std,
                actions=batch_action,
                action_low=action_low,
                action_high=action_high,
            )
            log_ratio = new_log_prob - batch_old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped = ratio * batch_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            # PPO uses the clipped objective to keep each update close to the rollout policy.
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(value, batch_returns)
            loss = policy_loss + value_loss_weight * value_loss - entropy_coef * entropy.mean()

            if auxiliary_loss_fn is not None:
                loss = loss + auxiliary_loss_fn()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            approx_kl = float((batch_old_log_probs - new_log_prob).mean().item())
            if approx_kl > 1.5 * target_kl:
                stop_early = True
                break

        if stop_early:
            break
