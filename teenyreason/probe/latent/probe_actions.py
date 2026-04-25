"""Active probe collection and action scoring helpers."""

from __future__ import annotations

import numpy as np
import torch

from ...envs import action_index_to_env_action
from ...representation import DeltaPredictorEnsemble, WorldEncoder
from ..explorer import build_probe_planner
from .probe_belief import belief_mean_z, belief_posterior_std
from ..data.probe_env import apply_env_params
from .probe_online import clone_recurrent_belief_hidden, init_recurrent_belief_hidden, update_recurrent_belief_from_transition
from ..data.probe_policy import ProbePolicy


MAX_SECOND_STEP_LOOKAHEAD_ACTIONS = 32


def collect_probe_window(
    env,
    probe_policy: ProbePolicy,
    rng: np.random.Generator,
    window_size: int,
    episode_physics,
    action_values: np.ndarray,
    probe_mode: str,
    max_probe_retries: int = 12,
):
    """Collect one full probe window, retrying if the environment ends too early."""
    for _ in range(max_probe_retries):
        apply_env_params(env, episode_physics)
        state, _info = env.reset()
        states = [np.asarray(state, dtype=np.float32)]
        actions = []
        done = False

        for step_idx in range(window_size):
            action_idx = probe_policy.sample_action(probe_mode, step_idx, rng)
            env_action = action_index_to_env_action(action_idx, action_values)
            next_state, _reward, terminated, truncated, _info = env.step(env_action)
            states.append(np.asarray(next_state, dtype=np.float32))
            actions.append(int(action_idx))
            done = bool(terminated or truncated)
            if done:
                break

        if done:
            continue
        if len(actions) == window_size and len(states) == window_size + 1:
            return np.stack(states, axis=0).astype(np.float32), np.asarray(actions, dtype=np.int64), False

    return None, None, True


def score_probe_actions(
    encoder: WorldEncoder,
    predictor: DeltaPredictorEnsemble | None,
    device: torch.device,
    current_state: np.ndarray,
    belief: np.ndarray | None,
    belief_hidden: torch.Tensor | None,
    action_vocab_size: int,
    recent_actions: list[int],
    action_prior_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Score candidate probe actions by disagreement, novelty, and posterior shrinkage."""
    base_scores = np.zeros(action_vocab_size, dtype=np.float32)
    if predictor is not None and belief is not None and belief_hidden is not None:
        repeated_states = np.repeat(np.asarray(current_state, dtype=np.float32)[None, :], action_vocab_size, axis=0)
        repeated_latent = np.repeat(belief_mean_z(belief)[None, :], action_vocab_size, axis=0)
        state_t = torch.tensor(repeated_states, dtype=torch.float32, device=device)
        action_t = torch.arange(action_vocab_size, dtype=torch.long, device=device)
        latent_t = torch.tensor(repeated_latent, dtype=torch.float32, device=device)
        with torch.no_grad():
            delta_preds = predictor.predict_all(state_t, action_t, latent_t)
            predicted_delta = delta_preds.mean(dim=0)
            predicted_next_state = state_t + predicted_delta
            repeated_hidden = belief_hidden.repeat(1, action_vocab_size, 1)
            zero_reward = torch.zeros(action_vocab_size, dtype=torch.float32, device=device)
            next_hidden, next_mean, next_logvar = encoder.update_belief(
                prev_state=state_t,
                next_state=predicted_next_state,
                action=action_t,
                reward=zero_reward,
                hidden=repeated_hidden,
            )
        disagreement = delta_preds.std(dim=0).mean(dim=1).cpu().numpy().astype(np.float32)
        transition_energy = predicted_delta.abs().mean(dim=1).cpu().numpy().astype(np.float32)
        current_std = float(np.mean(belief_posterior_std(belief)))
        predicted_std = torch.exp(0.5 * next_logvar).mean(dim=1).cpu().numpy().astype(np.float32)
        posterior_shrinkage = np.clip(current_std - predicted_std, 0.0, None)
        latent_shift = torch.norm(next_mean - latent_t, dim=1).cpu().numpy().astype(np.float32)
        base_scores = disagreement + 0.15 * transition_energy + 0.35 * posterior_shrinkage + 0.10 * latent_shift
        if action_vocab_size <= MAX_SECOND_STEP_LOOKAHEAD_ACTIONS:
            base_scores += 0.20 * _score_second_probe_step(
                encoder=encoder,
                predictor=predictor,
                device=device,
                predicted_next_state=predicted_next_state,
                predicted_next_hidden=next_hidden,
                predicted_next_mean=next_mean,
                predicted_next_logvar=next_logvar,
                action_vocab_size=action_vocab_size,
            )

    counts = np.bincount(recent_actions, minlength=action_vocab_size).astype(np.float32)
    novelty_bonus = 1.0 / (1.0 + counts)
    total_scores = base_scores + 0.10 * novelty_bonus
    if action_prior_scores is not None:
        prior_scores = np.asarray(action_prior_scores, dtype=np.float32).reshape(-1)
        if prior_scores.shape[0] != action_vocab_size:
            raise ValueError("Action prior scores must match the action vocabulary size")
        total_scores += 0.45 * (prior_scores - float(np.mean(prior_scores)))
    return np.nan_to_num(total_scores.astype(np.float32), nan=-1.0, posinf=5.0, neginf=-5.0)


def safe_choice_weights(scores: np.ndarray, temperature: float, fallback_scores: np.ndarray | None = None) -> np.ndarray:
    """Convert scores into a stable sampling distribution with sane fallbacks."""
    safe_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if safe_scores.size == 0:
        raise ValueError("Cannot sample from an empty action-score vector")
    safe_scores = np.nan_to_num(safe_scores, nan=-1.0, posinf=5.0, neginf=-5.0)
    if fallback_scores is not None:
        fallback_scores = np.asarray(fallback_scores, dtype=np.float32).reshape(-1)
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=1.0, neginf=-1.0)

    logits = safe_scores - float(np.max(safe_scores))
    weights = np.exp(logits / max(float(temperature), 1e-3)).astype(np.float32)
    weight_sum = float(np.sum(weights))
    if np.isfinite(weight_sum) and weight_sum > 1e-6:
        return weights / weight_sum

    if fallback_scores is not None and fallback_scores.shape == safe_scores.shape:
        fallback_logits = fallback_scores - float(np.max(fallback_scores))
        fallback_weights = np.exp(fallback_logits / 0.35).astype(np.float32)
        fallback_sum = float(np.sum(fallback_weights))
        if np.isfinite(fallback_sum) and fallback_sum > 1e-6:
            return fallback_weights / fallback_sum

    uniform = np.ones_like(safe_scores, dtype=np.float32)
    return uniform / float(max(uniform.shape[0], 1))


def _score_second_probe_step(
    encoder: WorldEncoder,
    predictor: DeltaPredictorEnsemble,
    device: torch.device,
    predicted_next_state: torch.Tensor,
    predicted_next_hidden: torch.Tensor,
    predicted_next_mean: torch.Tensor,
    predicted_next_logvar: torch.Tensor,
    action_vocab_size: int,
) -> np.ndarray:
    """Estimate how much extra identification value remains after one more probe step."""
    future_scores = np.zeros(action_vocab_size, dtype=np.float32)
    candidate_actions = torch.arange(action_vocab_size, dtype=torch.long, device=device)
    zero_reward = torch.zeros(action_vocab_size, dtype=torch.float32, device=device)

    with torch.no_grad():
        for first_action_idx in range(action_vocab_size):
            state_t = predicted_next_state[first_action_idx].unsqueeze(0).repeat(action_vocab_size, 1)
            latent_t = predicted_next_mean[first_action_idx].unsqueeze(0).repeat(action_vocab_size, 1)
            hidden_t = predicted_next_hidden[:, first_action_idx:first_action_idx + 1, :].repeat(1, action_vocab_size, 1)
            current_std = float(torch.exp(0.5 * predicted_next_logvar[first_action_idx]).mean().item())
            delta_preds = predictor.predict_all(state_t, candidate_actions, latent_t)
            predicted_delta = delta_preds.mean(dim=0)
            predicted_next_state_2 = state_t + predicted_delta
            _hidden_2, next_mean_2, next_logvar_2 = encoder.update_belief(
                prev_state=state_t,
                next_state=predicted_next_state_2,
                action=candidate_actions,
                reward=zero_reward,
                hidden=hidden_t,
            )
            disagreement = delta_preds.std(dim=0).mean(dim=1).detach().cpu().numpy().astype(np.float32)
            posterior_std = torch.exp(0.5 * next_logvar_2).mean(dim=1).detach().cpu().numpy().astype(np.float32)
            posterior_shrinkage = np.clip(current_std - posterior_std, 0.0, None)
            latent_shift = torch.norm(next_mean_2 - latent_t, dim=1).detach().cpu().numpy().astype(np.float32)
            future_scores[first_action_idx] = float(np.max(disagreement + 0.35 * posterior_shrinkage + 0.10 * latent_shift))
    return future_scores


def choose_active_probe_action(
    encoder: WorldEncoder,
    predictor: DeltaPredictorEnsemble | None,
    device: torch.device,
    rng: np.random.Generator,
    current_state: np.ndarray,
    belief: np.ndarray | None,
    belief_hidden: torch.Tensor | None,
    action_vocab_size: int,
    recent_actions: list[int],
    action_prior_scores: np.ndarray | None = None,
) -> int:
    """Choose an exploratory action using disagreement and a little randomness."""
    if belief is None or len(recent_actions) < 2 or predictor is None or belief_hidden is None:
        if action_prior_scores is None:
            return int(rng.integers(0, action_vocab_size))
        prior = np.asarray(action_prior_scores, dtype=np.float32).reshape(-1)
        weights = safe_choice_weights(prior, temperature=0.35)
        return int(rng.choice(np.arange(action_vocab_size), p=weights))

    scores = score_probe_actions(
        encoder=encoder,
        predictor=predictor,
        device=device,
        current_state=current_state,
        belief=belief,
        belief_hidden=belief_hidden,
        action_vocab_size=action_vocab_size,
        recent_actions=recent_actions,
        action_prior_scores=action_prior_scores,
    )
    weights = safe_choice_weights(scores, temperature=0.35, fallback_scores=action_prior_scores)
    return int(rng.choice(np.arange(action_vocab_size), p=weights))


def _pad_partial_probe_window(
    *,
    states: list[np.ndarray],
    actions: list[int],
    rewards: list[float],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn one partial real rollout into a fixed-size fallback probe window."""
    state_list = [np.asarray(state, dtype=np.float32).copy() for state in states]
    action_list = [int(action_idx) for action_idx in actions]
    reward_list = [float(reward) for reward in rewards]
    if len(action_list) <= 0 or len(state_list) != len(action_list) + 1:
        raise ValueError("Partial probe window must contain at least one transition")

    pad_action = int(action_list[-1])
    pad_state = state_list[-1].copy()
    while len(action_list) < int(window_size):
        action_list.append(pad_action)
        reward_list.append(0.0)
        state_list.append(pad_state.copy())
    return (
        np.stack(state_list, axis=0).astype(np.float32),
        np.asarray(action_list, dtype=np.int64),
        np.asarray(reward_list, dtype=np.float32),
    )


def collect_adaptive_probe_window(
    env,
    encoder: WorldEncoder,
    predictor: DeltaPredictorEnsemble | None,
    device: torch.device,
    rng: np.random.Generator,
    window_size: int,
    episode_physics,
    action_values: np.ndarray,
    env_name: str | None = None,
    prior_belief: np.ndarray | None = None,
    prior_hidden: torch.Tensor | None = None,
    planner=None,
    max_probe_retries: int = 12,
    trace_writer=None,
    trace_context: dict | None = None,
):
    """Collect one probe window using disagreement-driven active actions."""
    action_vocab_size = int(env.action_space.n) if action_values is None else int(len(action_values))
    steps_used = 0
    best_partial_states: list[np.ndarray] | None = None
    best_partial_actions: list[int] | None = None
    best_partial_rewards: list[float] | None = None
    best_partial_len = 0
    min_salvage_steps = max(2, int(window_size) // 2)
    local_planner = planner
    if local_planner is None:
        local_planner = build_probe_planner(
            action_space=env.action_space,
            action_values=action_values,
            rng=rng,
        )

    for _ in range(max_probe_retries):
        apply_env_params(env, episode_physics)
        state, _info = env.reset()
        states = [np.asarray(state, dtype=np.float32)]
        actions: list[int] = []
        rewards: list[float] = []
        current_belief = None if prior_belief is None else np.asarray(prior_belief, dtype=np.float32).copy()
        current_hidden = clone_recurrent_belief_hidden(prior_hidden)
        if current_hidden is None:
            current_hidden = init_recurrent_belief_hidden(encoder=encoder, device=device)
        if local_planner is not None:
            local_planner.begin_rollout()
        done = False

        for step_idx in range(window_size):
            del step_idx
            action_prior_scores = None if local_planner is None else local_planner.action_prior_scores(states[-1], actions)
            action_idx = choose_active_probe_action(
                encoder=encoder,
                predictor=predictor,
                device=device,
                rng=rng,
                current_state=states[-1],
                belief=current_belief,
                belief_hidden=current_hidden,
                action_vocab_size=action_vocab_size,
                recent_actions=actions,
                action_prior_scores=action_prior_scores,
            )
            env_action = action_index_to_env_action(action_idx, action_values)
            next_state, reward, terminated, truncated, _info = env.step(env_action)
            steps_used += 1
            states.append(np.asarray(next_state, dtype=np.float32))
            actions.append(int(action_idx))
            rewards.append(float(reward))
            if trace_writer is not None:
                trace_writer.record_probe_collection_step(
                    phase="online_active_probe",
                    state=np.asarray(states[-2], dtype=np.float32),
                    action_value=float(np.asarray(env_action, dtype=np.float32).reshape(-1)[0]),
                    action_index=int(action_idx),
                    probe_mode=(
                        str(local_planner.current_goal)
                        if local_planner is not None and local_planner.current_goal is not None
                        else "active_probe"
                    ),
                    env_params=None,
                    env_instance_id=int((trace_context or {}).get("env_instance_id", -1)),
                    episode_id=int((trace_context or {}).get("episode_id", -1)),
                    step_idx=int((trace_context or {}).get("step_offset", 0)) + len(actions) - 1,
                    reward=float(reward),
                )
            if local_planner is not None:
                local_planner.observe_transition(
                    prev_state=np.asarray(states[-2], dtype=np.float32),
                    action_idx=int(action_idx),
                    next_state=np.asarray(states[-1], dtype=np.float32),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            done = bool(terminated or truncated)
            if done:
                break

            current_belief, current_hidden, _posterior = update_recurrent_belief_from_transition(
                encoder=encoder,
                device=device,
                belief_hidden=current_hidden,
                prev_state=np.asarray(states[-2], dtype=np.float32),
                action_idx=int(action_idx),
                reward=float(reward),
                next_state=np.asarray(states[-1], dtype=np.float32),
                prior_belief=current_belief,
                alpha=0.5,
            )

        if done:
            if len(actions) > best_partial_len:
                best_partial_states = [np.asarray(state, dtype=np.float32).copy() for state in states]
                best_partial_actions = [int(action_idx) for action_idx in actions]
                best_partial_rewards = [float(reward) for reward in rewards]
                best_partial_len = len(actions)
            continue
        if len(actions) == window_size and len(states) == window_size + 1:
            return (
                np.stack(states, axis=0).astype(np.float32),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                False,
                steps_used,
            )

    if (
        best_partial_states is not None
        and best_partial_actions is not None
        and best_partial_rewards is not None
        and best_partial_len >= min_salvage_steps
    ):
        padded_states, padded_actions, padded_rewards = _pad_partial_probe_window(
            states=best_partial_states,
            actions=best_partial_actions,
            rewards=best_partial_rewards,
            window_size=window_size,
        )
        return padded_states, padded_actions, padded_rewards, False, steps_used

    return None, None, None, True, steps_used


def nearest_probe_action_idx(action: np.ndarray, action_values: np.ndarray) -> int:
    """Map a continuous policy action back to the nearest discrete probe action."""
    action_np = np.asarray(action, dtype=np.float32).reshape(1, -1)
    prototypes = np.asarray(action_values, dtype=np.float32)
    if prototypes.ndim == 1:
        prototypes = prototypes.reshape(-1, 1)
    distances = np.linalg.norm(prototypes - action_np, axis=1)
    return int(np.argmin(distances))
