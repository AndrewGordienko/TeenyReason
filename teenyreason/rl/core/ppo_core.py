"""Shared PPO models and optimization utilities.

The training loops under `teenyreason.rl.probe_policy` collect episodes and
then call into this module to:

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
    old_values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    beliefs: np.ndarray | None
    recurrent_hidden_states: np.ndarray | None = None
    sequence_length: int | None = None


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


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Set one shared learning rate across all optimizer parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


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


class MatchedBeliefActorCritic(nn.Module):
    """Shared actor-critic body used for both baseline and env-expression PPO."""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        belief_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.belief_dim = int(belief_dim)
        self.expression_dim = max(0, self.belief_dim - 2)
        expression_input_dim = max(2, self.expression_dim + 1)

        self.actor_state_backbone = nn.Sequential(
            init_linear(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.actor_base_head = init_linear(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.actor_expression_proj = nn.Sequential(
            init_linear(nn.Linear(expression_input_dim, hidden_dim), gain=0.5),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=0.5),
            nn.Tanh(),
        )
        self.actor_residual_head = init_linear(nn.Linear(hidden_dim * 2, action_dim), gain=0.01)

        self.value_state_backbone = nn.Sequential(
            init_linear(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.value_base_head = init_linear(nn.Linear(hidden_dim, 1), gain=1.0)
        self.value_expression_proj = nn.Sequential(
            init_linear(nn.Linear(expression_input_dim, hidden_dim), gain=0.5),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=0.5),
            nn.Tanh(),
        )
        self.value_residual_head = init_linear(nn.Linear(hidden_dim * 2, 1), gain=1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    def split_expression_input(
        self,
        belief: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split solver input into env-expression vector, confidence, and uncertainty."""
        belief = sanitize_tensor(belief)
        batch_size = belief.shape[0]
        if self.belief_dim <= 0:
            zeros = torch.zeros((batch_size, 0), dtype=belief.dtype, device=belief.device)
            scalar_zeros = torch.zeros((batch_size, 1), dtype=belief.dtype, device=belief.device)
            return zeros, scalar_zeros, scalar_zeros
        if self.belief_dim == 1:
            vector = torch.zeros((batch_size, 0), dtype=belief.dtype, device=belief.device)
            confidence = torch.zeros((batch_size, 1), dtype=belief.dtype, device=belief.device)
            uncertainty = belief[:, :1]
            return vector, confidence, uncertainty
        vector = belief[:, :-2] if self.expression_dim > 0 else torch.zeros(
            (batch_size, 0),
            dtype=belief.dtype,
            device=belief.device,
        )
        confidence = torch.clamp(belief[:, -2:-1], 0.0, 1.0)
        uncertainty = belief[:, -1:]
        return sanitize_tensor(vector), sanitize_tensor(confidence), sanitize_tensor(uncertainty)

    def forward_with_belief(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict action mean and value under the current state/env-expression."""
        state = sanitize_tensor(state)
        belief = sanitize_tensor(belief)
        expression_vector, expression_confidence, expression_uncertainty = self.split_expression_input(belief)
        expression_context_input = (
            torch.cat([expression_vector, expression_uncertainty], dim=-1)
            if expression_vector.shape[-1] > 0
            else expression_uncertainty
        )

        actor_features = sanitize_tensor(self.actor_state_backbone(state))
        actor_base = sanitize_tensor(self.actor_base_head(actor_features))
        actor_expression = sanitize_tensor(self.actor_expression_proj(expression_context_input))
        actor_residual = sanitize_tensor(
            self.actor_residual_head(torch.cat([actor_features, actor_expression], dim=-1))
        )
        mean = sanitize_tensor(actor_base + expression_confidence * actor_residual)

        value_features = sanitize_tensor(self.value_state_backbone(state))
        value_base = sanitize_tensor(self.value_base_head(value_features).squeeze(-1))
        value_expression = sanitize_tensor(self.value_expression_proj(expression_context_input))
        value_residual = sanitize_tensor(
            self.value_residual_head(torch.cat([value_features, value_expression], dim=-1)).squeeze(-1)
        )
        value = sanitize_tensor(value_base + expression_confidence.squeeze(-1) * value_residual)
        return mean, value


class PlainGaussianActorCritic(MatchedBeliefActorCritic):
    """Baseline actor-critic that matches the probe architecture but masks belief."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        belief_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            belief_dim=belief_dim,
            hidden_dim=hidden_dim,
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = sanitize_tensor(state)
        zero_belief = torch.zeros(
            (state.shape[0], self.belief_dim),
            dtype=state.dtype,
            device=state.device,
        )
        return self.forward_with_belief(state, zero_belief)


class ProbeConditionedGaussianActorCritic(MatchedBeliefActorCritic):
    """Actor-critic that receives both the environment state and probe belief."""

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_with_belief(state, belief)


class BeliefNativeActorCritic(nn.Module):
    """Belief-first actor-critic with context modulation and recurrent adaptation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        mechanics_dim: int,
        affordance_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.mechanics_dim = int(mechanics_dim)
        self.affordance_dim = int(affordance_dim)
        self.context_dim = int(self.mechanics_dim + self.affordance_dim + 2)
        self.hidden_dim = int(hidden_dim)
        self.recurrent_dim = int(hidden_dim)

        self.state_encoder = nn.Sequential(
            init_linear(nn.Linear(state_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.context_encoder = nn.Sequential(
            init_linear(nn.Linear(self.context_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.state_bridge = nn.Sequential(
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
        )
        self.context_to_film = init_linear(nn.Linear(hidden_dim, hidden_dim * 2), gain=0.5)
        self.context_gate = nn.Sequential(
            init_linear(nn.Linear(hidden_dim * 2 + 2, hidden_dim), gain=0.5),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), gain=0.5),
        )
        self.context_to_hidden = nn.Sequential(
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=0.5),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, self.recurrent_dim), gain=0.5),
            nn.Tanh(),
        )
        self.recurrent = nn.GRUCell(hidden_dim, self.recurrent_dim)
        trunk_dim = self.recurrent_dim + hidden_dim
        self.actor_head = nn.Sequential(
            init_linear(nn.Linear(trunk_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, action_dim), gain=0.01),
        )
        self.value_head = nn.Sequential(
            init_linear(nn.Linear(trunk_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), gain=1.0),
        )
        self.surprise_head = nn.Sequential(
            init_linear(nn.Linear(trunk_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), gain=0.5),
        )
        self.recovery_head = nn.Sequential(
            init_linear(nn.Linear(trunk_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), gain=0.5),
        )
        self.query_head = nn.Sequential(
            init_linear(nn.Linear(trunk_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), gain=0.5),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    def split_controller_context(
        self,
        belief: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the flat controller context into its structured fields."""
        belief = sanitize_tensor(belief)
        mechanics_end = self.mechanics_dim
        affordance_end = mechanics_end + self.affordance_dim
        mechanics = belief[:, :mechanics_end]
        affordance = belief[:, mechanics_end:affordance_end]
        confidence = torch.clamp(belief[:, affordance_end:affordance_end + 1], 0.0, 1.0)
        uncertainty = belief[:, affordance_end + 1:affordance_end + 2]
        return (
            sanitize_tensor(mechanics),
            sanitize_tensor(affordance),
            sanitize_tensor(confidence),
            sanitize_tensor(uncertainty),
        )

    def encode_context(
        self,
        belief: torch.Tensor,
    ) -> torch.Tensor:
        """Encode mechanics, affordances, and trust scalars into one context feature."""
        mechanics, affordance, confidence, uncertainty = self.split_controller_context(belief)
        flat_context = torch.cat([mechanics, affordance, confidence, uncertainty], dim=-1)
        return sanitize_tensor(self.context_encoder(flat_context))

    def compute_context_gate(
        self,
        state_features: torch.Tensor,
        context_features: torch.Tensor,
        confidence: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """Predict how much the controller should trust the context right now."""
        gate_features = torch.cat(
            [state_features, context_features, confidence, uncertainty],
            dim=-1,
        )
        learned_gate = torch.sigmoid(self.context_gate(gate_features))
        trust_prior = torch.clamp(confidence, 0.0, 1.0) * torch.exp(
            -torch.clamp(uncertainty, min=0.0)
        )
        return sanitize_tensor(torch.clamp(learned_gate * trust_prior, 0.0, 1.0))

    def init_recurrent_state(
        self,
        belief: torch.Tensor,
    ) -> torch.Tensor:
        """Initialize recurrent controller state from the belief context."""
        context_features = self.encode_context(belief)
        return sanitize_tensor(self.context_to_hidden(context_features))

    def refresh_recurrent_state(
        self,
        belief: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        blend: float = 0.35,
    ) -> torch.Tensor:
        """Refresh recurrent state from a new context without fully discarding history."""
        fresh_hidden = self.init_recurrent_state(belief)
        if hidden_state is None:
            return fresh_hidden
        blend = float(np.clip(blend, 0.0, 1.0))
        return sanitize_tensor((1.0 - blend) * sanitize_tensor(hidden_state) + blend * fresh_hidden)

    def forward_with_hidden(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Predict action/value while carrying a recurrent hidden state."""
        state = sanitize_tensor(state)
        belief = sanitize_tensor(belief)
        state_features = sanitize_tensor(self.state_encoder(state))
        context_features = self.encode_context(belief)
        _mechanics, _affordance, confidence, uncertainty = self.split_controller_context(belief)
        film_scale, film_shift = self.context_to_film(context_features).chunk(2, dim=-1)
        modulated_state = sanitize_tensor(
            state_features * (1.0 + 0.5 * torch.tanh(film_scale)) + 0.5 * torch.tanh(film_shift)
        )
        context_gate = self.compute_context_gate(
            state_features,
            context_features,
            confidence,
            uncertainty,
        )
        state_bridge = sanitize_tensor(self.state_bridge(state_features))
        fused_state = sanitize_tensor(
            (1.0 - context_gate) * state_features + context_gate * modulated_state
        )
        fused_context = sanitize_tensor(
            (1.0 - context_gate) * state_bridge + context_gate * context_features
        )
        if hidden_state is None:
            hidden_state = self.init_recurrent_state(belief)
        next_hidden = sanitize_tensor(self.recurrent(fused_state, sanitize_tensor(hidden_state)))
        trunk = sanitize_tensor(torch.cat([next_hidden, fused_context], dim=-1))
        mean = sanitize_tensor(self.actor_head(trunk))
        value = sanitize_tensor(self.value_head(trunk).squeeze(-1))
        aux = {
            "surprise": sanitize_tensor(self.surprise_head(trunk).squeeze(-1)),
            "recovery": sanitize_tensor(self.recovery_head(trunk).squeeze(-1)),
            "query_logit": sanitize_tensor(self.query_head(trunk).squeeze(-1)),
            "context_gate": sanitize_tensor(context_gate.squeeze(-1)),
        }
        return mean, value, next_hidden, aux

    def forward_sequence(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        hidden_state: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Unroll the recurrent controller over one or more fixed-length sequences."""
        state = sanitize_tensor(state)
        belief = sanitize_tensor(belief)
        hidden_state = sanitize_tensor(hidden_state)
        if state.ndim != 3 or belief.ndim != 3:
            raise ValueError("forward_sequence expects [batch, time, dim] inputs")
        batch_size, time_steps, _state_dim = state.shape
        mean_steps = []
        value_steps = []
        surprise_steps = []
        recovery_steps = []
        query_steps = []
        gate_steps = []
        next_hidden = hidden_state
        if mask is None:
            mask = torch.ones(
                (batch_size, time_steps),
                dtype=state.dtype,
                device=state.device,
            )
        for step_idx in range(time_steps):
            step_mean, step_value, proposed_hidden, step_aux = self.forward_with_hidden(
                state[:, step_idx, :],
                belief[:, step_idx, :],
                hidden_state=next_hidden,
            )
            step_mask = mask[:, step_idx : step_idx + 1]
            next_hidden = sanitize_tensor(
                step_mask * proposed_hidden + (1.0 - step_mask) * next_hidden
            )
            mean_steps.append(step_mean)
            value_steps.append(step_value)
            surprise_steps.append(step_aux["surprise"])
            recovery_steps.append(step_aux["recovery"])
            query_steps.append(step_aux["query_logit"])
            gate_steps.append(step_aux["context_gate"])
        return (
            sanitize_tensor(torch.stack(mean_steps, dim=1)),
            sanitize_tensor(torch.stack(value_steps, dim=1)),
            next_hidden,
            {
                "surprise": sanitize_tensor(torch.stack(surprise_steps, dim=1)),
                "recovery": sanitize_tensor(torch.stack(recovery_steps, dim=1)),
                "query_logit": sanitize_tensor(torch.stack(query_steps, dim=1)),
                "context_gate": sanitize_tensor(torch.stack(gate_steps, dim=1)),
            },
        )

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, value, _next_hidden, _aux = self.forward_with_hidden(state, belief, hidden_state=None)
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
    dist = build_tanh_normal(mean, log_std)
    scale, bias = action_scale_bias(action_low, action_high, mean.device)
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
    recurrent_hidden_states: list[np.ndarray] | None = None,
    sequence_length: int | None = None,
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
        old_values=values_np.astype(np.float32),
        returns=returns,
        advantages=advantages,
        beliefs=None if beliefs is None else np.stack(beliefs, axis=0).astype(np.float32),
        recurrent_hidden_states=(
            None
            if recurrent_hidden_states is None
            else np.stack(recurrent_hidden_states, axis=0).astype(np.float32)
        ),
        sequence_length=None if sequence_length is None else int(sequence_length),
    )


def concat_episode_batches(batches: list[EpisodeBatch]) -> EpisodeBatch:
    """Concatenate several episode batches before an optimizer step."""
    beliefs = None
    if batches[0].beliefs is not None:
        beliefs = np.concatenate([batch.beliefs for batch in batches], axis=0).astype(np.float32)
    recurrent_hidden_states = None
    if batches[0].recurrent_hidden_states is not None:
        recurrent_hidden_states = np.concatenate(
            [batch.recurrent_hidden_states for batch in batches],
            axis=0,
        ).astype(np.float32)

    return EpisodeBatch(
        states=np.concatenate([batch.states for batch in batches], axis=0).astype(np.float32),
        actions=np.concatenate([batch.actions for batch in batches], axis=0).astype(np.float32),
        old_log_probs=np.concatenate([batch.old_log_probs for batch in batches], axis=0).astype(np.float32),
        old_values=np.concatenate([batch.old_values for batch in batches], axis=0).astype(np.float32),
        returns=np.concatenate([batch.returns for batch in batches], axis=0).astype(np.float32),
        advantages=np.concatenate([batch.advantages for batch in batches], axis=0).astype(np.float32),
        beliefs=beliefs,
        recurrent_hidden_states=recurrent_hidden_states,
        sequence_length=batches[0].sequence_length,
    )


def _shuffle_context_codes(sequence: torch.Tensor) -> torch.Tensor:
    """Shuffle only the code portion of one controller-context sequence."""
    if sequence.shape[-1] <= 2:
        return sequence
    shuffled = sequence.clone()
    permutation = torch.randperm(sequence.shape[-1] - 2, device=sequence.device)
    shuffled[..., :-2] = shuffled[..., :-2][..., permutation]
    return shuffled


def corrupt_controller_context_sequences(
    beliefs: torch.Tensor,
    *,
    zero_prob: float,
    shuffle_prob: float,
    stale_prob: float,
) -> torch.Tensor:
    """Apply belief-native training corruption so the policy learns a safe fallback."""
    if beliefs.ndim != 3:
        return beliefs
    zero_prob = max(0.0, float(zero_prob))
    shuffle_prob = max(0.0, float(shuffle_prob))
    stale_prob = max(0.0, float(stale_prob))
    if zero_prob <= 0.0 and shuffle_prob <= 0.0 and stale_prob <= 0.0:
        return beliefs
    corrupted = beliefs.clone()
    thresholds = np.cumsum([zero_prob, shuffle_prob, stale_prob]).tolist()
    for seq_idx in range(corrupted.shape[0]):
        sample = float(torch.rand((), device=beliefs.device).item())
        if sample < thresholds[0]:
            corrupted[seq_idx] = 0.0
        elif sample < thresholds[1]:
            corrupted[seq_idx] = _shuffle_context_codes(corrupted[seq_idx])
        elif sample < thresholds[2]:
            corrupted[seq_idx] = corrupted[seq_idx, :1, :].expand_as(corrupted[seq_idx])
    return sanitize_tensor(corrupted)


def prepare_recurrent_minibatch(
    *,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    beliefs: torch.Tensor,
    recurrent_hidden_states: torch.Tensor,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Pack flat rollout tensors into fixed-length padded sequences."""
    total_steps = int(states.shape[0])
    sequence_length = max(1, int(sequence_length))
    num_sequences = int(np.ceil(float(total_steps) / float(sequence_length)))
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    belief_dim = beliefs.shape[-1]
    hidden_dim = recurrent_hidden_states.shape[-1]
    states_seq = torch.zeros(
        (num_sequences, sequence_length, state_dim),
        dtype=states.dtype,
        device=states.device,
    )
    actions_seq = torch.zeros(
        (num_sequences, sequence_length, action_dim),
        dtype=actions.dtype,
        device=actions.device,
    )
    old_log_probs_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=old_log_probs.dtype,
        device=old_log_probs.device,
    )
    old_values_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=old_values.dtype,
        device=old_values.device,
    )
    returns_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=returns.dtype,
        device=returns.device,
    )
    advantages_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=advantages.dtype,
        device=advantages.device,
    )
    beliefs_seq = torch.zeros(
        (num_sequences, sequence_length, belief_dim),
        dtype=beliefs.dtype,
        device=beliefs.device,
    )
    hidden_seq = torch.zeros(
        (num_sequences, hidden_dim),
        dtype=recurrent_hidden_states.dtype,
        device=recurrent_hidden_states.device,
    )
    mask_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=states.dtype,
        device=states.device,
    )
    for seq_idx, start in enumerate(range(0, total_steps, sequence_length)):
        end = min(start + sequence_length, total_steps)
        valid = end - start
        states_seq[seq_idx, :valid] = states[start:end]
        actions_seq[seq_idx, :valid] = actions[start:end]
        old_log_probs_seq[seq_idx, :valid] = old_log_probs[start:end]
        old_values_seq[seq_idx, :valid] = old_values[start:end]
        returns_seq[seq_idx, :valid] = returns[start:end]
        advantages_seq[seq_idx, :valid] = advantages[start:end]
        beliefs_seq[seq_idx, :valid] = beliefs[start:end]
        hidden_seq[seq_idx] = recurrent_hidden_states[start]
        mask_seq[seq_idx, :valid] = 1.0
    return {
        "states": states_seq,
        "actions": actions_seq,
        "old_log_probs": old_log_probs_seq,
        "old_values": old_values_seq,
        "returns": returns_seq,
        "advantages": advantages_seq,
        "beliefs": beliefs_seq,
        "hidden": hidden_seq,
        "mask": mask_seq,
    }


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
    value_clip_ratio: float | None = None,
    auxiliary_loss_fn=None,
    expression_consistency_weight: float = 0.0,
    expression_consistency_threshold: float = 0.35,
    controller_context_zero_prob: float = 0.0,
    controller_context_shuffle_prob: float = 0.0,
    controller_context_stale_prob: float = 0.0,
    controller_sequence_length: int | None = None,
):
    """Run one PPO optimization phase over a collected batch."""
    device = next(model.parameters()).device
    states = torch.tensor(sanitize_numpy(batch.states), dtype=torch.float32, device=device)
    actions = torch.tensor(sanitize_numpy(batch.actions), dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(sanitize_numpy(batch.old_log_probs), dtype=torch.float32, device=device)
    old_values = torch.tensor(sanitize_numpy(batch.old_values), dtype=torch.float32, device=device)
    returns = torch.tensor(sanitize_numpy(batch.returns), dtype=torch.float32, device=device)
    advantages = torch.tensor(sanitize_numpy(batch.advantages), dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    beliefs = None
    if batch.beliefs is not None:
        beliefs = torch.tensor(sanitize_numpy(batch.beliefs), dtype=torch.float32, device=device)
    recurrent_hidden_states = None
    if batch.recurrent_hidden_states is not None:
        recurrent_hidden_states = torch.tensor(
            sanitize_numpy(batch.recurrent_hidden_states),
            dtype=torch.float32,
            device=device,
        )

    if (
        beliefs is not None
        and recurrent_hidden_states is not None
        and isinstance(model, BeliefNativeActorCritic)
    ):
        sequence_batch = prepare_recurrent_minibatch(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            old_values=old_values,
            returns=returns,
            advantages=advantages,
            beliefs=beliefs,
            recurrent_hidden_states=recurrent_hidden_states,
            sequence_length=(
                batch.sequence_length
                if controller_sequence_length is None
                else int(controller_sequence_length)
            )
            or 32,
        )
        total_sequences = int(sequence_batch["states"].shape[0])
        minibatch_size = min(minibatch_size, total_sequences)

        for _ in range(ppo_epochs):
            permutation = torch.randperm(total_sequences, device=device)
            stop_early = False

            for start in range(0, total_sequences, minibatch_size):
                idx = permutation[start:start + minibatch_size]
                batch_states = sequence_batch["states"][idx]
                batch_actions = sequence_batch["actions"][idx]
                batch_old_log_probs = sequence_batch["old_log_probs"][idx]
                batch_old_values = sequence_batch["old_values"][idx]
                batch_returns = sequence_batch["returns"][idx]
                batch_advantages = sequence_batch["advantages"][idx]
                batch_beliefs = sequence_batch["beliefs"][idx]
                batch_hidden = sequence_batch["hidden"][idx]
                batch_mask = sequence_batch["mask"][idx]
                batch_beliefs = corrupt_controller_context_sequences(
                    batch_beliefs,
                    zero_prob=controller_context_zero_prob,
                    shuffle_prob=controller_context_shuffle_prob,
                    stale_prob=controller_context_stale_prob,
                )

                mean, value, _next_hidden, _aux = model.forward_sequence(
                    batch_states,
                    batch_beliefs,
                    batch_hidden,
                    mask=batch_mask,
                )
                flat_mask = batch_mask.reshape(-1) > 0
                flat_mean = mean.reshape(-1, mean.shape[-1])[flat_mask]
                flat_value = value.reshape(-1)[flat_mask]
                flat_actions = batch_actions.reshape(-1, batch_actions.shape[-1])[flat_mask]
                flat_old_log_probs = batch_old_log_probs.reshape(-1)[flat_mask]
                flat_old_values = batch_old_values.reshape(-1)[flat_mask]
                flat_returns = batch_returns.reshape(-1)[flat_mask]
                flat_advantages = batch_advantages.reshape(-1)[flat_mask]

                new_log_prob, entropy = evaluate_continuous_actions(
                    mean=flat_mean,
                    log_std=model.log_std,
                    actions=flat_actions,
                    action_low=action_low,
                    action_high=action_high,
                )
                log_ratio = new_log_prob - flat_old_log_probs
                ratio = torch.exp(log_ratio)
                unclipped = ratio * flat_advantages
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * flat_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                if value_clip_ratio is not None and value_clip_ratio > 0.0:
                    value_delta = torch.clamp(
                        flat_value - flat_old_values,
                        min=-float(value_clip_ratio),
                        max=float(value_clip_ratio),
                    )
                    value_clipped = flat_old_values + value_delta
                    value_loss_unclipped = (flat_value - flat_returns).pow(2)
                    value_loss_clipped = (value_clipped - flat_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = nn.functional.mse_loss(flat_value, flat_returns)
                loss = policy_loss + value_loss_weight * value_loss - entropy_coef * entropy.mean()

                if auxiliary_loss_fn is not None:
                    loss = loss + auxiliary_loss_fn()
                if not torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                approx_kl = float(sanitize_tensor((flat_old_log_probs - new_log_prob).mean()).item())
                if approx_kl > 1.5 * target_kl:
                    stop_early = True
                    break

            if stop_early:
                break
        return

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
            batch_old_values = old_values[idx]
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
            if value_clip_ratio is not None and value_clip_ratio > 0.0:
                value_delta = torch.clamp(
                    value - batch_old_values,
                    min=-float(value_clip_ratio),
                    max=float(value_clip_ratio),
                )
                value_clipped = batch_old_values + value_delta
                value_loss_unclipped = (value - batch_returns).pow(2)
                value_loss_clipped = (value_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = nn.functional.mse_loss(value, batch_returns)
            loss = policy_loss + value_loss_weight * value_loss - entropy_coef * entropy.mean()

            if beliefs is not None and expression_consistency_weight > 0.0 and batch.beliefs.shape[-1] >= 2:
                batch_belief = beliefs[idx]
                expression_confidence = torch.clamp(batch_belief[:, -2], 0.0, 1.0)
                low_confidence_weight = torch.clamp(
                    (float(expression_consistency_threshold) - expression_confidence)
                    / max(float(expression_consistency_threshold), 1e-6),
                    min=0.0,
                )
                if torch.any(low_confidence_weight > 0.0):
                    zero_expression_belief = batch_belief.clone()
                    if zero_expression_belief.shape[-1] > 2:
                        zero_expression_belief[:, :-2] = 0.0
                    zero_expression_belief[:, -2] = 0.0
                    base_mean, base_value = model(batch_state, zero_expression_belief)
                    mean_gap = (mean - base_mean).pow(2).mean(dim=-1)
                    value_gap = (value - base_value).pow(2)
                    consistency_loss = ((mean_gap + value_gap) * low_confidence_weight).mean()
                    loss = loss + float(expression_consistency_weight) * consistency_loss

            if auxiliary_loss_fn is not None:
                loss = loss + auxiliary_loss_fn()

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            approx_kl = float(sanitize_tensor((batch_old_log_probs - new_log_prob).mean()).item())
            if approx_kl > 1.5 * target_kl:
                stop_early = True
                break

        if stop_early:
            break
