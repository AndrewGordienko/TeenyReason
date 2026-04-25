"""Window-level optimization pass for belief-model training."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from .belief_common import sanitize_tensor
from .belief_losses import (
    gaussian_moment_regularizer,
    info_nce_loss,
    pairwise_env_geometry_loss,
    supervised_same_env_contrastive_loss,
)
from .belief_training_common import modules_are_finite, sanitize_modules_


def run_window_training_epoch(
    *,
    encoder,
    predictor,
    latent_transition_model,
    affordance_predictor,
    decision_predictor,
    return_predictor,
    risk_predictor,
    contrastive_query,
    contrastive_key,
    env_projector,
    mode_adversary,
    optimizer: torch.optim.Optimizer,
    train_modules: list[nn.Module],
    window_states: torch.Tensor,
    window_actions: torch.Tensor,
    window_rewards: torch.Tensor,
    prefix_states: torch.Tensor,
    prefix_actions: torch.Tensor,
    prefix_rewards: torch.Tensor,
    env_instance_id: torch.Tensor,
    probe_mode_idx: torch.Tensor,
    current_state: torch.Tensor,
    current_action: torch.Tensor,
    target_delta: torch.Tensor,
    target_env_params: torch.Tensor,
    target_affordances: torch.Tensor,
    target_decision: torch.Tensor,
    target_return: torch.Tensor,
    target_risk: torch.Tensor,
    target_future_summary: torch.Tensor,
    batch_size: int,
    max_grad_norm: float,
    affordance_loss_weight: float,
    decision_loss_weight: float,
    return_loss_weight: float,
    risk_loss_weight: float,
    kl_loss_weight: float,
    contrastive_loss_weight: float,
    env_consistency_loss_weight: float,
    env_geometry_loss_weight: float,
    mode_adversary_loss_weight: float,
    latent_rollout_loss_weight: float,
    latent_gaussian_loss_weight: float,
) -> dict[str, float]:
    """Train the per-window latent heads for one epoch."""
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    num_windows = int(window_states.shape[0])
    device = window_states.device
    permutation = torch.randperm(num_windows, device=device)

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_env_consistency_loss = 0.0
    total_env_geometry_loss = 0.0
    total_mode_adversary_loss = 0.0
    total_latent_rollout_loss = 0.0
    total_latent_gaussian_loss = 0.0
    skipped_window_batches = 0

    for start in range(0, num_windows, batch_size):
        idx = permutation[start:start + batch_size]
        batch_states = window_states[idx]
        batch_actions = window_actions[idx]
        batch_rewards = window_rewards[idx]
        batch_prefix_states = prefix_states[idx]
        batch_prefix_actions = prefix_actions[idx]
        batch_prefix_rewards = prefix_rewards[idx]
        batch_env_instance_id = env_instance_id[idx]
        batch_probe_mode_idx = probe_mode_idx[idx]
        batch_current_state = current_state[idx]
        batch_current_action = current_action[idx]
        batch_target_delta = target_delta[idx]
        batch_target_affordances = target_affordances[idx]
        batch_target_decision = target_decision[idx]
        batch_target_return = target_return[idx]
        batch_target_risk = target_risk[idx]
        batch_target_future_summary = target_future_summary[idx]

        mean, logvar = encoder.encode_posterior(batch_states, batch_actions, rewards=batch_rewards)
        mean = sanitize_tensor(mean)
        logvar = sanitize_tensor(logvar)
        step_mean, step_logvar = encoder.encode_step_posteriors(batch_states, batch_actions, rewards=batch_rewards)
        step_mean = sanitize_tensor(step_mean)
        step_logvar = sanitize_tensor(step_logvar)
        prefix_mean, _prefix_logvar = encoder.encode_posterior(
            batch_prefix_states,
            batch_prefix_actions,
            rewards=batch_prefix_rewards,
        )
        prefix_mean = sanitize_tensor(prefix_mean)
        z = sanitize_tensor(encoder.sample_latent(mean, logvar))
        delta_preds = sanitize_tensor(predictor.predict_all(batch_current_state, batch_current_action, z))
        affordance_pred = sanitize_tensor(affordance_predictor(z))
        decision_pred = sanitize_tensor(decision_predictor(z))
        return_pred = sanitize_tensor(return_predictor(z))
        risk_pred = sanitize_tensor(risk_predictor(z))

        delta_loss = torch.stack(
            [mse_loss(delta_preds[member_idx], batch_target_delta) for member_idx in range(predictor.ensemble_size)],
            dim=0,
        ).mean()
        affordance_loss = mse_loss(affordance_pred, batch_target_affordances)
        decision_loss = mse_loss(decision_pred, batch_target_decision)
        return_loss = mse_loss(return_pred, batch_target_return)
        risk_loss = bce_loss(risk_pred, batch_target_risk)
        kl_loss = 0.5 * torch.mean(torch.exp(logvar) + mean.pow(2) - 1.0 - logvar)
        contrastive_loss = info_nce_loss(
            contrastive_query(prefix_mean),
            contrastive_key(batch_target_future_summary),
        )
        env_consistency_loss = supervised_same_env_contrastive_loss(
            embeddings=env_projector(mean),
            env_instance_id=batch_env_instance_id,
            probe_mode_idx=batch_probe_mode_idx,
        )
        env_geometry_loss = pairwise_env_geometry_loss(
            latent_mean=mean,
            normalized_env_params=target_env_params[idx],
        )
        mode_adversary_loss = ce_loss(mode_adversary(mean, reverse_scale=1.0), batch_probe_mode_idx)
        latent_gaussian_loss = gaussian_moment_regularizer(
            torch.cat([mean, prefix_mean], dim=0)
        )
        latent_rollout_loss = mean.sum() * 0.0
        if step_mean.shape[1] > 1:
            rollout_input_mean = step_mean[:, :-1, :].reshape(-1, step_mean.shape[-1])
            rollout_state = batch_states[:, 1:-1, :].reshape(-1, batch_states.shape[-1])
            rollout_action = batch_actions[:, 1:].reshape(-1)
            rollout_reward = batch_rewards[:, 1:].reshape(-1)
            target_rollout_mean = step_mean[:, 1:, :].reshape(-1, step_mean.shape[-1])
            target_rollout_logvar = step_logvar[:, 1:, :].reshape(-1, step_logvar.shape[-1])
            pred_rollout_mean, pred_rollout_logvar = latent_transition_model(
                latent_mean=rollout_input_mean,
                state=rollout_state,
                action=rollout_action,
                reward=rollout_reward,
            )
            latent_rollout_loss = (
                mse_loss(pred_rollout_mean, target_rollout_mean)
                + 0.5 * mse_loss(pred_rollout_logvar, target_rollout_logvar)
            )

        loss = sanitize_tensor(
            delta_loss
            + affordance_loss_weight * affordance_loss
            + decision_loss_weight * decision_loss
            + return_loss_weight * return_loss
            + risk_loss_weight * risk_loss
            + kl_loss_weight * kl_loss
            + contrastive_loss_weight * contrastive_loss
            + env_consistency_loss_weight * env_consistency_loss
            + env_geometry_loss_weight * env_geometry_loss
            + mode_adversary_loss_weight * mode_adversary_loss
            + latent_rollout_loss_weight * latent_rollout_loss
            + latent_gaussian_loss_weight * latent_gaussian_loss
        )
        if not torch.isfinite(loss):
            skipped_window_batches += 1
            optimizer.zero_grad(set_to_none=True)
            sanitize_modules_(train_modules)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(
            [param for module in train_modules for param in module.parameters()],
            max_grad_norm,
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            skipped_window_batches += 1
            optimizer.zero_grad(set_to_none=True)
            sanitize_modules_(train_modules)
            continue
        optimizer.step()
        if not modules_are_finite(train_modules):
            skipped_window_batches += 1
            sanitize_modules_(train_modules)
            optimizer.zero_grad(set_to_none=True)
            continue

        total_loss += float(loss.item()) * len(idx)
        total_contrastive_loss += float(contrastive_loss.item()) * len(idx)
        total_env_consistency_loss += float(env_consistency_loss.item()) * len(idx)
        total_env_geometry_loss += float(env_geometry_loss.item()) * len(idx)
        total_mode_adversary_loss += float(mode_adversary_loss.item()) * len(idx)
        total_latent_rollout_loss += float(latent_rollout_loss.item()) * len(idx)
        total_latent_gaussian_loss += float(latent_gaussian_loss.item()) * len(idx)

    return {
        "loss": total_loss / max(num_windows, 1),
        "contrastive_loss": total_contrastive_loss / max(num_windows, 1),
        "env_consistency_loss": total_env_consistency_loss / max(num_windows, 1),
        "env_geometry_loss": total_env_geometry_loss / max(num_windows, 1),
        "mode_adversary_loss": total_mode_adversary_loss / max(num_windows, 1),
        "latent_rollout_loss": total_latent_rollout_loss / max(num_windows, 1),
        "latent_gaussian_loss": total_latent_gaussian_loss / max(num_windows, 1),
        "skipped_window_batches": float(skipped_window_batches),
    }
