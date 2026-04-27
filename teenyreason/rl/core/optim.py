"""PPO optimizer step implementation.

The training loops collect data; this module owns the minibatch update
mechanics. It handles the two PPO surfaces in the repo: ordinary flat
rollouts and belief-native recurrent controller rollouts.
"""

import numpy as np
import torch
import torch.nn as nn

from .batches import corrupt_controller_context_sequences, prepare_recurrent_minibatch
from .models import BeliefNativeActorCritic
from .numerics import (
    action_scale_bias,
    evaluate_continuous_actions_with_scale_bias,
    sanitize_numpy,
    sanitize_tensor,
)
from .types import EpisodeBatch


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    """Set one shared learning rate across all optimizer parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


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
    action_scale, action_bias = action_scale_bias(action_low, action_high, device)
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

                new_log_prob, entropy = evaluate_continuous_actions_with_scale_bias(
                    mean=flat_mean,
                    log_std=model.log_std,
                    actions=flat_actions,
                    scale=action_scale,
                    bias=action_bias,
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

            new_log_prob, entropy = evaluate_continuous_actions_with_scale_bias(
                mean=mean,
                log_std=model.log_std,
                actions=batch_action,
                scale=action_scale,
                bias=action_bias,
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
