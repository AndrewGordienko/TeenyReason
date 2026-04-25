"""Small rollout-supervision and surprise helpers for belief-planner training."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..core.ppo_core import sanitize_numpy, sanitize_tensor


def rollout_supervision_and_alignment(
    *,
    model,
    initial_state: torch.Tensor,
    action_sequences: torch.Tensor,
    learned_context: torch.Tensor,
    oracle_context: torch.Tensor,
    target_next_states: torch.Tensor,
    target_rewards: torch.Tensor,
    target_terms: torch.Tensor,
    target_recoverability: torch.Tensor,
    gamma: float,
    reward_weight: float,
    recoverability_weight: float,
    termination_weight: float,
    disagreement_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supervise short rollouts and align learned-vs-oracle control semantics."""
    if action_sequences.ndim != 3 or action_sequences.shape[1] <= 0:
        zero = sanitize_tensor(initial_state.sum() * 0.0)
        return zero, zero

    learned_state = sanitize_tensor(initial_state)
    oracle_state = sanitize_tensor(initial_state)
    rollout_loss = sanitize_tensor(initial_state.sum() * 0.0)
    learned_score = torch.zeros(
        (initial_state.shape[0],),
        dtype=torch.float32,
        device=initial_state.device,
    )
    oracle_score = torch.zeros_like(learned_score)

    horizon = int(action_sequences.shape[1])
    for step_idx in range(horizon):
        action_t = sanitize_tensor(action_sequences[:, step_idx, :])
        learned_pred = model.predict_summary(learned_state, action_t, learned_context)
        oracle_pred = model.predict_summary(oracle_state, action_t, oracle_context)
        target_next_t = sanitize_tensor(target_next_states[:, step_idx, :])
        target_reward_t = sanitize_tensor(target_rewards[:, step_idx])
        target_term_t = sanitize_tensor(target_terms[:, step_idx])
        target_recover_t = sanitize_tensor(target_recoverability[:, step_idx])
        discount = float(gamma**step_idx)

        learned_next = sanitize_tensor(learned_state + learned_pred["next_state_delta"])
        oracle_next = sanitize_tensor(oracle_state + oracle_pred["next_state_delta"])
        rollout_loss = sanitize_tensor(
            rollout_loss
            + discount
            * (
                F.mse_loss(learned_next, target_next_t)
                + F.mse_loss(oracle_next, target_next_t)
                + 0.50 * F.mse_loss(learned_pred["reward"], target_reward_t)
                + 0.50 * F.mse_loss(oracle_pred["reward"], target_reward_t)
                + 0.25 * F.binary_cross_entropy_with_logits(learned_pred["term_logit"], target_term_t)
                + 0.25 * F.binary_cross_entropy_with_logits(oracle_pred["term_logit"], target_term_t)
                + 0.25 * F.mse_loss(learned_pred["recoverability"], target_recover_t)
                + 0.25 * F.mse_loss(oracle_pred["recoverability"], target_recover_t)
            )
        )
        learned_score = sanitize_tensor(
            learned_score
            + discount
            * (
                reward_weight * learned_pred["reward"]
                + recoverability_weight * learned_pred["recoverability"]
                - termination_weight * torch.sigmoid(learned_pred["term_logit"])
                - disagreement_weight * learned_pred["disagreement"]
            )
        )
        oracle_score = sanitize_tensor(
            oracle_score
            + discount
            * (
                reward_weight * oracle_pred["reward"]
                + recoverability_weight * oracle_pred["recoverability"]
                - termination_weight * torch.sigmoid(oracle_pred["term_logit"])
                - disagreement_weight * oracle_pred["disagreement"]
            )
        )
        learned_state = learned_next
        oracle_state = oracle_next

    rollout_loss = sanitize_tensor(rollout_loss / float(max(horizon, 1)))
    score_alignment_loss = F.mse_loss(learned_score, oracle_score.detach())
    return rollout_loss, sanitize_tensor(score_alignment_loss)


def planner_prediction_surprise(
    *,
    normalized_state: np.ndarray,
    prediction: dict[str, torch.Tensor],
    normalized_next_state: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> float:
    """Score how implausible one observed step was under the planner model."""
    predicted_delta = sanitize_numpy(
        prediction["next_state_delta"].detach().cpu().numpy().reshape(-1)
    )
    predicted_reward = float(prediction["reward"].detach().cpu().numpy().reshape(-1)[0])
    predicted_term = float(
        torch.sigmoid(prediction["term_logit"]).detach().cpu().numpy().reshape(-1)[0]
    )
    disagreement = float(prediction["disagreement"].detach().cpu().numpy().reshape(-1)[0])
    predicted_next = sanitize_numpy(normalized_state.reshape(-1) + predicted_delta)
    target_next = sanitize_numpy(np.asarray(normalized_next_state, dtype=np.float32).reshape(-1))
    terminal_target = float(bool(terminated or truncated))
    state_error = float(np.mean(np.abs(predicted_next - target_next)))
    reward_error = abs(predicted_reward - float(reward))
    term_error = abs(predicted_term - terminal_target)
    return float(state_error + 0.50 * reward_error + 0.50 * term_error + 0.25 * disagreement)


def surprise_z_score(
    surprise_value: float,
    history: Sequence[float],
    *,
    min_history: int = 8,
) -> float:
    """Return a simple z-score for surprise spikes once enough history exists."""
    if len(history) < max(1, int(min_history)):
        return 0.0
    history_arr = sanitize_numpy(np.asarray(list(history), dtype=np.float32))
    history_std = float(np.std(history_arr))
    if not np.isfinite(history_std) or history_std <= 1e-6:
        return 0.0
    return float((float(surprise_value) - float(np.mean(history_arr))) / history_std)


__all__ = [
    "planner_prediction_surprise",
    "rollout_supervision_and_alignment",
    "surprise_z_score",
]
