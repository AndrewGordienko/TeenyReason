"""Online belief updates and window encoding helpers."""

from __future__ import annotations

import numpy as np
import torch

from ...models.env_belief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ...representation import DeltaPredictorEnsemble, WorldEncoder
from .probe_belief import (
    aggregate_env_belief,
    build_belief_vector,
    sanitize_array,
    sanitize_belief_vector,
    update_belief_with_posterior,
)


def init_recurrent_belief_hidden(encoder: WorldEncoder, device: torch.device) -> torch.Tensor:
    """Create an empty recurrent belief state used for online context updates."""
    return encoder.init_hidden(batch_size=1, device=device)


def clone_recurrent_belief_hidden(belief_hidden: torch.Tensor | None) -> torch.Tensor | None:
    """Clone a recurrent hidden state so probe planning can branch safely."""
    if belief_hidden is None:
        return None
    return belief_hidden.detach().clone()


def update_recurrent_belief_from_transition(
    encoder: WorldEncoder,
    device: torch.device,
    belief_hidden: torch.Tensor,
    prev_state: np.ndarray,
    action_idx: int,
    reward: float,
    next_state: np.ndarray,
    prior_belief: np.ndarray | None = None,
    alpha: float = 0.5,
) -> tuple[np.ndarray, torch.Tensor, tuple[np.ndarray, np.ndarray]]:
    """Update the recurrent belief with one observed transition."""
    encoder.eval()
    with torch.no_grad():
        prev_state_t = torch.tensor(prev_state[None, :], dtype=torch.float32, device=device)
        next_state_t = torch.tensor(next_state[None, :], dtype=torch.float32, device=device)
        action_t = torch.tensor([action_idx], dtype=torch.long, device=device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
        belief_hidden, mean_t, logvar_t = encoder.update_belief(
            prev_state=prev_state_t,
            next_state=next_state_t,
            action=action_t,
            reward=reward_t,
            hidden=belief_hidden,
        )

    mean = sanitize_array(mean_t.squeeze(0).cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    logvar = sanitize_array(logvar_t.squeeze(0).cpu().numpy().astype(np.float32), nan=0.0, posinf=2.0, neginf=-5.0)
    if prior_belief is None:
        belief = build_belief_vector([(mean, logvar)])
    else:
        belief = update_belief_with_posterior(prior_belief, mean, logvar, alpha)
    return sanitize_belief_vector(belief), belief_hidden, (mean, logvar)


def update_recurrent_belief_from_window(
    encoder: WorldEncoder,
    device: torch.device,
    belief_hidden: torch.Tensor,
    window_states: np.ndarray,
    window_actions: np.ndarray,
    window_rewards: np.ndarray | None,
    prior_belief: np.ndarray | None = None,
    alpha: float = 0.5,
) -> tuple[np.ndarray, torch.Tensor, tuple[np.ndarray, np.ndarray]]:
    """Roll the recurrent belief through an entire probe window."""
    belief = prior_belief
    posterior = None
    rewards = None if window_rewards is None else np.asarray(window_rewards, dtype=np.float32)

    for step_idx, action_idx in enumerate(np.asarray(window_actions, dtype=np.int64)):
        reward = 0.0 if rewards is None else float(rewards[step_idx])
        belief, belief_hidden, posterior = update_recurrent_belief_from_transition(
            encoder=encoder,
            device=device,
            belief_hidden=belief_hidden,
            prev_state=np.asarray(window_states[step_idx], dtype=np.float32),
            action_idx=int(action_idx),
            reward=reward,
            next_state=np.asarray(window_states[step_idx + 1], dtype=np.float32),
            prior_belief=belief,
            alpha=alpha,
        )

    if posterior is None:
        raise ValueError("Cannot update recurrent belief from an empty window")
    return belief, belief_hidden, posterior


def encode_window(
    encoder: WorldEncoder,
    device: torch.device,
    window_states: np.ndarray,
    window_actions: np.ndarray,
    window_rewards: np.ndarray | None = None,
) -> np.ndarray:
    """Run one probe window through the frozen encoder and return its latent."""
    encoder.eval()
    with torch.no_grad():
        states = torch.tensor(window_states[None, ...], dtype=torch.float32, device=device)
        actions = torch.tensor(window_actions[None, ...], dtype=torch.long, device=device)
        rewards = None if window_rewards is None else torch.tensor(window_rewards[None, ...], dtype=torch.float32, device=device)
        z = encoder(states, actions, rewards=rewards).squeeze(0).cpu().numpy().astype(np.float32)
    return z


def encode_window_posterior(
    encoder: WorldEncoder,
    device: torch.device,
    window_states: np.ndarray,
    window_actions: np.ndarray,
    window_rewards: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the posterior mean/log-variance for one probe trajectory."""
    encoder.eval()
    with torch.no_grad():
        states = torch.tensor(window_states[None, ...], dtype=torch.float32, device=device)
        actions = torch.tensor(window_actions[None, ...], dtype=torch.long, device=device)
        rewards = None if window_rewards is None else torch.tensor(window_rewards[None, ...], dtype=torch.float32, device=device)
        mean, logvar = encoder.encode_posterior(states, actions, rewards=rewards)
    return mean.squeeze(0).cpu().numpy().astype(np.float32), logvar.squeeze(0).cpu().numpy().astype(np.float32)


def maybe_update_online_belief(
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    device: torch.device,
    belief_hidden: torch.Tensor,
    belief_posteriors: list[tuple[np.ndarray, np.ndarray]],
    prev_state: np.ndarray,
    action_idx: int,
    reward: float,
    next_state: np.ndarray,
    belief: np.ndarray,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    episode_step: int,
) -> tuple[np.ndarray, torch.Tensor, list[tuple[np.ndarray, np.ndarray]]]:
    """Refresh the belief online from the recent real trajectory when scheduled."""
    _, next_hidden, posterior = update_recurrent_belief_from_transition(
        encoder=encoder,
        device=device,
        belief_hidden=belief_hidden,
        prev_state=prev_state,
        action_idx=action_idx,
        reward=reward,
        next_state=next_state,
        prior_belief=None,
        alpha=1.0,
    )
    if episode_step % online_z_update_freq != 0:
        return belief, next_hidden, belief_posteriors

    updated_posteriors = list(belief_posteriors)
    updated_posteriors.append((posterior[0], posterior[1]))
    if len(updated_posteriors) > 6:
        updated_posteriors = updated_posteriors[-6:]

    aggregated_belief, _payload = aggregate_env_belief(
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        device=device,
        posterior_views=updated_posteriors,
    )
    blended_belief = ((1.0 - online_z_update_alpha) * np.asarray(belief, dtype=np.float32) + online_z_update_alpha * aggregated_belief).astype(np.float32)
    return sanitize_belief_vector(blended_belief), next_hidden, updated_posteriors


def select_episode_physics(rng: np.random.Generator, randomize_physics: bool, base_physics):
    """Choose either a randomized physics instance or the shared base one."""
    from ..data.probe_env import sample_env_params

    if randomize_physics:
        return sample_env_params(rng, base_physics)
    return base_physics
