"""Cheap belief-conditioned controller with state-first candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..core import (
    action_scale_bias,
    init_linear,
    sanitize_numpy,
    sanitize_tensor,
)
from .simulator_fanout import candidate_score


MIN_CONTROLLER_ACTION_MARGIN = 0.03
FULL_CONTROLLER_ACTION_MARGIN = 0.10


@dataclass(frozen=True)
class AffordanceSelection:
    """One explicit action-selection result for the cheap controller."""

    action: np.ndarray
    actor_action: np.ndarray
    mean: torch.Tensor
    value: torch.Tensor
    next_hidden: torch.Tensor
    trust: float
    controller_used: float
    action_divergence: float
    candidate_actions: np.ndarray
    candidate_returns: np.ndarray
    candidate_risks: np.ndarray
    candidate_recoverability: np.ndarray
    candidate_scores: np.ndarray
    best_idx: int


class BeliefAffordanceController(nn.Module):
    """Cheap recurrent controller with a state student and belief residual."""

    def __init__(
        self,
        *,
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
        self.state_trunk_dim = int(hidden_dim * 2)

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
        self.recurrent = nn.GRUCell(hidden_dim, hidden_dim)
        self.actor_head = nn.Sequential(
            init_linear(nn.Linear(self.state_trunk_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, action_dim), gain=0.01),
        )
        self.value_head = nn.Sequential(
            init_linear(nn.Linear(self.state_trunk_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), gain=1.0),
        )
        self.trust_head = nn.Sequential(
            init_linear(
                nn.Linear(self.state_trunk_dim + hidden_dim + 2, hidden_dim),
                gain=np.sqrt(2.0),
            ),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), gain=0.5),
        )
        self.state_candidate_head = nn.Sequential(
            init_linear(
                nn.Linear(self.state_trunk_dim + action_dim, hidden_dim),
                gain=np.sqrt(2.0),
            ),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 3), gain=0.5),
        )
        self.belief_residual_head = nn.Sequential(
            init_linear(
                nn.Linear(self.state_trunk_dim + hidden_dim + action_dim + 2, hidden_dim),
                gain=np.sqrt(2.0),
            ),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(2.0)),
            nn.Tanh(),
            init_linear(nn.Linear(hidden_dim, 1), gain=0.25),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

    def split_controller_context(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the flat controller input back into its structured fields."""
        context = sanitize_tensor(context)
        mechanics_end = self.mechanics_dim
        affordance_end = mechanics_end + self.affordance_dim
        mechanics = context[:, :mechanics_end]
        affordance = context[:, mechanics_end:affordance_end]
        confidence = torch.clamp(context[:, affordance_end:affordance_end + 1], 0.0, 1.0)
        uncertainty = context[:, affordance_end + 1:affordance_end + 2]
        return (
            sanitize_tensor(mechanics),
            sanitize_tensor(affordance),
            sanitize_tensor(confidence),
            sanitize_tensor(uncertainty),
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return sanitize_tensor(self.context_encoder(sanitize_tensor(context)))

    def init_recurrent_state(self, context: torch.Tensor) -> torch.Tensor:
        batch_size = int(context.shape[0])
        return torch.zeros(
            (batch_size, self.hidden_dim),
            dtype=context.dtype,
            device=context.device,
        )

    def refresh_recurrent_state(
        self,
        context: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        blend: float = 0.35,
    ) -> torch.Tensor:
        fresh_hidden = self.init_recurrent_state(context)
        if hidden_state is None:
            return fresh_hidden
        blend = float(np.clip(blend, 0.0, 1.0))
        return sanitize_tensor((1.0 - blend) * sanitize_tensor(hidden_state) + blend * fresh_hidden)

    def forward_with_hidden(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run one recurrent step with a state-only trunk and belief-side trust."""
        state = sanitize_tensor(state)
        context = sanitize_tensor(context)
        state_features = sanitize_tensor(self.state_encoder(state))
        context_features = self.encode_context(context)
        _mechanics, _affordance, confidence, uncertainty = self.split_controller_context(context)
        if hidden_state is None:
            hidden_state = self.init_recurrent_state(context)
        next_hidden = sanitize_tensor(self.recurrent(state_features, sanitize_tensor(hidden_state)))
        state_trunk = sanitize_tensor(torch.cat([next_hidden, state_features], dim=-1))
        trust_input = torch.cat(
            [state_trunk, context_features, confidence, uncertainty],
            dim=-1,
        )
        trust_prior = torch.clamp(confidence, 0.0, 1.0) * torch.exp(
            -torch.clamp(uncertainty, min=0.0)
        )
        mean = sanitize_tensor(self.actor_head(state_trunk))
        value = sanitize_tensor(self.value_head(state_trunk).squeeze(-1))
        trust = sanitize_tensor(
            torch.clamp(
                torch.sigmoid(self.trust_head(trust_input)).squeeze(-1) * trust_prior.squeeze(-1),
                0.0,
                1.0,
            )
        )
        return mean, value, next_hidden, {
            "trust": trust,
            "state_trunk": state_trunk,
            "context_features": context_features,
            "confidence": confidence.squeeze(-1),
            "uncertainty": uncertainty.squeeze(-1),
        }

    def forward_sequence(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        hidden_state: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Unroll the recurrent controller over one or more fixed-length sequences."""
        if state.ndim != 3 or context.ndim != 3:
            raise ValueError("forward_sequence expects [batch, time, dim] inputs")
        batch_size, time_steps, _ = state.shape
        if mask is None:
            mask = torch.ones((batch_size, time_steps), dtype=state.dtype, device=state.device)
        mean_steps = []
        value_steps = []
        trust_steps = []
        state_trunk_steps = []
        context_feature_steps = []
        confidence_steps = []
        uncertainty_steps = []
        next_hidden = sanitize_tensor(hidden_state)
        for step_idx in range(time_steps):
            step_mean, step_value, proposed_hidden, step_aux = self.forward_with_hidden(
                state[:, step_idx, :],
                context[:, step_idx, :],
                hidden_state=next_hidden,
            )
            step_mask = mask[:, step_idx : step_idx + 1]
            next_hidden = sanitize_tensor(
                step_mask * proposed_hidden + (1.0 - step_mask) * next_hidden
            )
            mean_steps.append(step_mean)
            value_steps.append(step_value)
            trust_steps.append(step_aux["trust"])
            state_trunk_steps.append(step_aux["state_trunk"])
            context_feature_steps.append(step_aux["context_features"])
            confidence_steps.append(step_aux["confidence"])
            uncertainty_steps.append(step_aux["uncertainty"])
        return (
            sanitize_tensor(torch.stack(mean_steps, dim=1)),
            sanitize_tensor(torch.stack(value_steps, dim=1)),
            next_hidden,
            {
                "trust": sanitize_tensor(torch.stack(trust_steps, dim=1)),
                "state_trunk": sanitize_tensor(torch.stack(state_trunk_steps, dim=1)),
                "context_features": sanitize_tensor(torch.stack(context_feature_steps, dim=1)),
                "confidence": sanitize_tensor(torch.stack(confidence_steps, dim=1)),
                "uncertainty": sanitize_tensor(torch.stack(uncertainty_steps, dim=1)),
            },
        )

    def evaluate_candidates(
        self,
        state_trunk: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict state-only return, risk, and recoverability for fixed candidates."""
        if candidate_actions.ndim != 3:
            raise ValueError("candidate actions must have shape [batch, candidates, action_dim]")
        batch_size, num_candidates, action_dim = candidate_actions.shape
        expanded_trunk = sanitize_tensor(state_trunk).unsqueeze(1).expand(-1, num_candidates, -1)
        candidate_input = torch.cat([expanded_trunk, sanitize_tensor(candidate_actions)], dim=-1)
        flat_output = self.state_candidate_head(candidate_input.reshape(-1, candidate_input.shape[-1]))
        output = sanitize_tensor(flat_output.reshape(batch_size, num_candidates, 3))
        predicted_return = sanitize_tensor(output[..., 0])
        predicted_risk = sanitize_tensor(torch.sigmoid(output[..., 1]))
        predicted_recoverability = sanitize_tensor(torch.sigmoid(output[..., 2]))
        return predicted_return, predicted_risk, predicted_recoverability

    def evaluate_candidate_scores(
        self,
        *,
        state_trunk: torch.Tensor,
        context_features: torch.Tensor,
        candidate_actions: torch.Tensor,
        trust: torch.Tensor,
        confidence: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score candidates with a state student plus trust-gated belief residual."""
        predicted_return, predicted_risk, predicted_recoverability = self.evaluate_candidates(
            state_trunk,
            candidate_actions,
        )
        state_scores = sanitize_tensor(
            predicted_return
            + 0.5 * predicted_recoverability
            - 2.0 * predicted_risk
        )
        batch_size, num_candidates, _action_dim = candidate_actions.shape
        expanded_state_trunk = sanitize_tensor(state_trunk).unsqueeze(1).expand(-1, num_candidates, -1)
        expanded_context = sanitize_tensor(context_features).unsqueeze(1).expand(-1, num_candidates, -1)
        if confidence is None:
            confidence = torch.ones((batch_size,), dtype=state_trunk.dtype, device=state_trunk.device)
        if uncertainty is None:
            uncertainty = torch.zeros((batch_size,), dtype=state_trunk.dtype, device=state_trunk.device)
        trust_input = torch.stack(
            [
                sanitize_tensor(confidence).reshape(-1),
                sanitize_tensor(uncertainty).reshape(-1),
            ],
            dim=-1,
        )
        expanded_trust_input = trust_input.unsqueeze(1).expand(-1, num_candidates, -1)
        residual_input = torch.cat(
            [
                expanded_state_trunk,
                expanded_context,
                sanitize_tensor(candidate_actions),
                expanded_trust_input,
            ],
            dim=-1,
        )
        flat_residual = self.belief_residual_head(
            residual_input.reshape(-1, residual_input.shape[-1])
        )
        belief_residual = sanitize_tensor(flat_residual.reshape(batch_size, num_candidates))
        trust = sanitize_tensor(trust).reshape(-1, 1)
        final_scores = sanitize_tensor(state_scores + trust * belief_residual)
        return {
            "return": predicted_return,
            "risk": predicted_risk,
            "recoverability": predicted_recoverability,
            "state_scores": state_scores,
            "belief_residual": belief_residual,
            "final_scores": final_scores,
        }


def generate_candidate_actions(
    *,
    mean_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Generate the fixed five-action proposal set around the actor mean."""
    mean_action = sanitize_numpy(np.asarray(mean_action, dtype=np.float32).reshape(1, -1))
    action_low = sanitize_numpy(np.asarray(action_low, dtype=np.float32).reshape(1, -1))
    action_high = sanitize_numpy(np.asarray(action_high, dtype=np.float32).reshape(1, -1))
    action_span = action_high - action_low
    offsets = np.asarray([0.0, -0.25, 0.25, -0.50, 0.50], dtype=np.float32).reshape(-1, 1)
    candidates = mean_action + offsets * action_span
    return sanitize_numpy(np.clip(candidates, action_low, action_high))


def choose_affordance_action(
    *,
    controller: BeliefAffordanceController,
    state_t: torch.Tensor,
    context_t: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
    hidden_state: torch.Tensor | None,
    force_state_only: bool = False,
) -> AffordanceSelection:
    """Select one cheap candidate-scored action using trust-gated control."""
    with torch.no_grad():
        mean, value, next_hidden, aux = controller.forward_with_hidden(
            state_t,
            context_t,
            hidden_state=hidden_state,
        )
        actor_action = np.asarray(
            mean_to_action(mean, action_low=action_low, action_high=action_high),
            dtype=np.float32,
        ).reshape(-1)
        candidate_actions = generate_candidate_actions(
            mean_action=actor_action,
            action_low=action_low,
            action_high=action_high,
        )
        candidate_t = torch.tensor(
            candidate_actions[None, :, :],
            dtype=torch.float32,
            device=state_t.device,
        )
        if hasattr(controller, "evaluate_candidate_scores") and "state_trunk" in aux:
            score_outputs = controller.evaluate_candidate_scores(
                state_trunk=aux["state_trunk"],
                context_features=aux["context_features"],
                candidate_actions=candidate_t,
                trust=aux["trust"],
                confidence=aux.get("confidence"),
                uncertainty=aux.get("uncertainty"),
            )
            predicted_return = score_outputs["return"]
            predicted_risk = score_outputs["risk"]
            predicted_recoverability = score_outputs["recoverability"]
            state_scores = sanitize_numpy(score_outputs["state_scores"].squeeze(0).cpu().numpy())
            scores = sanitize_numpy(score_outputs["final_scores"].squeeze(0).cpu().numpy())
        else:
            predicted_return, predicted_risk, predicted_recoverability = controller.evaluate_candidates(
                aux["trunk"],
                candidate_t,
            )
            state_scores = candidate_score(
                predicted_return.squeeze(0).cpu().numpy(),
                predicted_risk.squeeze(0).cpu().numpy(),
                predicted_recoverability.squeeze(0).cpu().numpy(),
            )
            scores = state_scores.copy()
        best_idx = int(np.argmax(scores))
        best_candidate = candidate_actions[best_idx].reshape(-1)
        trust = float(aux["trust"].squeeze(0).item())
        actor_score = float(scores[0]) if scores.size else 0.0
        best_score = float(scores[best_idx]) if scores.size else actor_score
        best_margin = float(best_score - actor_score)
        if force_state_only:
            state_best_idx = int(np.argmax(state_scores))
            chosen_action = candidate_actions[state_best_idx].reshape(-1).copy()
            controller_used = 0.0
            best_idx = state_best_idx
            scores = state_scores.copy()
        elif trust < 0.15 or best_idx == 0 or best_margin < MIN_CONTROLLER_ACTION_MARGIN:
            chosen_action = actor_action.copy()
            controller_used = 0.0
        elif trust < 0.35 or best_margin < FULL_CONTROLLER_ACTION_MARGIN:
            chosen_action = sanitize_numpy(0.5 * actor_action + 0.5 * best_candidate).reshape(-1)
            controller_used = 0.5
        else:
            chosen_action = best_candidate.copy()
            controller_used = 1.0
    return AffordanceSelection(
        action=sanitize_numpy(chosen_action),
        actor_action=sanitize_numpy(actor_action),
        mean=mean.detach(),
        value=value.detach(),
        next_hidden=next_hidden.detach(),
        trust=trust,
        controller_used=controller_used,
        action_divergence=float(np.mean(np.abs(chosen_action - actor_action))),
        candidate_actions=sanitize_numpy(candidate_actions),
        candidate_returns=sanitize_numpy(predicted_return.squeeze(0).cpu().numpy()),
        candidate_risks=sanitize_numpy(predicted_risk.squeeze(0).cpu().numpy()),
        candidate_recoverability=sanitize_numpy(predicted_recoverability.squeeze(0).cpu().numpy()),
        candidate_scores=sanitize_numpy(scores),
        best_idx=best_idx,
    )


def mean_to_action(
    mean: torch.Tensor,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Map the controller mean into one bounded environment action."""
    mean = sanitize_tensor(mean)
    squashed_mean = torch.tanh(mean)
    scale, bias = action_scale_bias(action_low, action_high, mean.device)
    action = bias + scale * squashed_mean
    return sanitize_numpy(action.squeeze(0).detach().cpu().numpy())
