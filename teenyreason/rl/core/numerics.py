"""Numerical helpers for PPO action distributions and rollout targets.

This module owns the math that is easy to get subtly wrong: finite-value
sanitization, tanh-squashed Gaussian log-probs, continuous action bounds,
and GAE. Model classes and training loops import these primitives rather
than duplicating ad hoc tensor logic.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def init_linear(layer: nn.Linear, gain: float, bias: float = 0.0) -> nn.Linear:
    """Apply the repo's standard linear-layer initialization."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def sanitize_numpy(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Keep rollout arrays finite before they are converted back into tensors."""
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    ).astype(np.float32)


def sanitize_tensor(values: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Replace non-finite tensor values with a harmless finite fallback."""
    if torch.isfinite(values).all():
        return values
    return torch.nan_to_num(
        values,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )


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
    mean = sanitize_tensor(mean)
    log_std = sanitize_tensor(log_std, fill_value=-1.5)
    std = torch.exp(torch.clamp(log_std, -5.0, 1.0))
    std = sanitize_tensor(std, fill_value=float(np.exp(-1.5)))
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
        sanitize_numpy(action.squeeze(0).detach().cpu().numpy()),
        float(log_prob.sum(dim=-1).item()),
    )


def mean_to_continuous_action(
    mean: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Convert the policy mean into a deterministic bounded environment action."""
    mean = sanitize_tensor(mean)
    squashed_mean = torch.tanh(mean)
    scale, bias = action_scale_bias(action_low, action_high, mean.device)
    action = bias + scale * squashed_mean
    return sanitize_numpy(action.squeeze(0).detach().cpu().numpy())


def evaluate_continuous_actions(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log-prob and entropy for already-sampled bounded actions."""
    scale, bias = action_scale_bias(action_low, action_high, mean.device)
    return evaluate_continuous_actions_with_scale_bias(
        mean=mean,
        log_std=log_std,
        actions=actions,
        scale=scale,
        bias=bias,
    )


def evaluate_continuous_actions_with_scale_bias(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bounded-action log-probs with precomputed action affine terms.

    PPO calls this inside every minibatch. Passing cached `scale`/`bias` avoids
    allocating action-bound tensors repeatedly while keeping the public helper
    above convenient for tests and one-off callers.
    """
    dist = build_tanh_normal(mean, log_std)
    scale = sanitize_tensor(scale)
    bias = sanitize_tensor(bias)
    actions = sanitize_tensor(actions)
    normalized_action = torch.clamp((actions - bias) / scale, -0.999, 0.999)
    raw_action = atanh(normalized_action)
    log_prob = dist.log_prob(raw_action) - torch.log(scale * (1.0 - normalized_action.pow(2)) + 1e-6)
    return sanitize_tensor(log_prob.sum(dim=-1)), sanitize_tensor(dist.entropy().sum(dim=-1))


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
