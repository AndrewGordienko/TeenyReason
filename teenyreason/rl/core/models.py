"""Actor-critic model definitions shared by the PPO training paths.

`MatchedBeliefActorCritic` keeps the plain and probe-conditioned policies
architecturally comparable. `BeliefNativeActorCritic` is the recurrent
belief-first controller used by the full-system path. Keeping models here,
away from rollout packing and optimizer code, makes architecture bugs much
easier to isolate.
"""

import numpy as np
import torch
import torch.nn as nn

from .numerics import init_linear, sanitize_tensor


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
