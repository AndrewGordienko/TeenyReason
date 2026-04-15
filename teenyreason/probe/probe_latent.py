"""Latent belief helpers.

The encoder produces a raw latent vector `z` from one probe window. This module
defines what the control policy actually uses on top of that:

- latent normalization and similarity
- memory of which latent regions led to good returns
- aggregation of several probe latents into one belief vector
- optional online belief updates during the control episode
"""

import random
from collections import deque

import numpy as np
import torch

from ..envs import action_index_to_env_action
from .probe_data import ProbePolicy, apply_env_params, sample_env_params
from .cartpole_scientist import build_probe_planner
from ..models.env_belief import (
    EnvBeliefAggregator,
    EnvParamPredictorEnsemble,
    aggregate_env_posteriors,
)
from ..representation import DeltaPredictorEnsemble, WorldEncoder


def sanitize_array(
    values: np.ndarray,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> np.ndarray:
    """Replace non-finite values in a NumPy array and keep float32 semantics."""
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    ).astype(np.float32)


def sanitize_belief_vector(belief: np.ndarray) -> np.ndarray:
    """Keep belief vectors finite and keep the uncertainty half non-negative."""
    belief_np = sanitize_array(belief, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    if belief_np.size == 0:
        return belief_np
    half = belief_np.shape[0] // 2
    if half <= 0:
        return belief_np
    mean = sanitize_array(belief_np[:half], nan=0.0, posinf=0.0, neginf=0.0)
    spread = sanitize_array(belief_np[half:], nan=0.0, posinf=1.0, neginf=0.0)
    spread = np.clip(spread, 1e-4, 5.0).astype(np.float32)
    return np.concatenate([mean, spread], axis=0).astype(np.float32)


def normalize_latent(latent: np.ndarray) -> np.ndarray:
    """Normalize a latent to unit length unless it is effectively zero."""
    latent_np = sanitize_array(latent, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(latent_np))
    if not np.isfinite(norm) or norm <= 1e-6:
        return np.zeros_like(latent_np, dtype=np.float32)
    # Most downstream comparisons treat latent direction as the meaningful signal.
    return (latent_np / norm).astype(np.float32)


class LatentPerformanceMemory:
    """Nearest-neighbor memory from latent regions to achieved episode return."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, z: np.ndarray, episode_return: float):
        self.buffer.append((normalize_latent(z), float(episode_return)))

    def __len__(self):
        return len(self.buffer)

    def _similarities(self, z: np.ndarray):
        """Return cosine-like similarities between `z` and stored latent codes."""
        if not self.buffer:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

        normalized_z = normalize_latent(z)
        memory_z = np.stack([item[0] for item in self.buffer], axis=0)
        returns = np.asarray([item[1] for item in self.buffer], dtype=np.float32)
        similarities = memory_z @ normalized_z
        return similarities.astype(np.float32), returns

    def novelty(self, z: np.ndarray) -> float:
        """High when the latent is far from everything seen before."""
        similarities, _returns = self._similarities(z)
        if similarities.size == 0:
            return 1.0
        return float(1.0 - np.max(similarities))

    def expected_return(self, z: np.ndarray, top_k: int = 8) -> float:
        """Estimate likely return by averaging nearby latent memories."""
        similarities, returns = self._similarities(z)
        if similarities.size == 0:
            return 0.0

        top_k = min(top_k, similarities.size)
        top_idx = np.argsort(similarities)[-top_k:]
        top_sims = similarities[top_idx]
        top_returns = returns[top_idx]
        weights = np.clip((top_sims + 1.0) * 0.5, 0.05, None)
        return float(np.sum(weights * top_returns) / np.sum(weights))


class EliteTrajectoryBuffer:
    """Replay buffer for unusually strong trajectories used by self-imitation."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push_episode(
        self,
        states: np.ndarray,
        beliefs: np.ndarray,
        actions: np.ndarray,
        returns_to_go: np.ndarray,
        episode_weight: float,
    ):
        for idx in range(len(states)):
            self.buffer.append(
                (
                    np.asarray(states[idx], dtype=np.float32),
                    np.asarray(beliefs[idx], dtype=np.float32),
                    np.asarray(actions[idx], dtype=np.float32),
                    float(returns_to_go[idx]),
                    float(episode_weight),
                )
            )

    def sample(self, batch_size: int):
        weights = [item[-1] for item in self.buffer]
        batch = random.choices(list(self.buffer), weights=weights, k=batch_size)
        states, beliefs, actions, returns_to_go, sample_weight = zip(*batch)
        return (
            np.stack(states, axis=0),
            np.stack(beliefs, axis=0),
            np.stack(actions, axis=0),
            np.asarray(returns_to_go, dtype=np.float32),
            np.asarray(sample_weight, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def build_belief_vector(latents: list[np.ndarray]) -> np.ndarray:
    """Aggregate several probe latents into one policy-conditioning vector."""
    if not latents:
        raise ValueError("Cannot build a belief vector from an empty list")

    first_item = latents[0]
    if isinstance(first_item, tuple):
        means = np.stack([np.asarray(item[0], dtype=np.float32) for item in latents], axis=0)
        logvars = np.stack([np.asarray(item[1], dtype=np.float32) for item in latents], axis=0)
        precisions = np.exp(-logvars)
        combined_var = 1.0 / np.clip(np.sum(precisions, axis=0), 1e-6, None)
        combined_mean = combined_var * np.sum(means * precisions, axis=0)
        combined_std = np.sqrt(np.clip(combined_var, 1e-6, None))
        return np.concatenate([combined_mean, combined_std], axis=0).astype(np.float32)

    # Legacy path: if callers still pass raw latents, treat them like a point estimate.
    stacked = np.stack([normalize_latent(latent) for latent in latents], axis=0).astype(np.float32)
    mean_z = normalize_latent(np.mean(stacked, axis=0))
    spread_z = np.clip(np.std(stacked, axis=0), 0.0, 1.0)
    return np.concatenate([mean_z, spread_z], axis=0).astype(np.float32)


def aggregate_env_belief(
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    device: torch.device,
    posterior_views: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Aggregate several window posteriors into one env-level belief vector."""
    if not posterior_views:
        raise ValueError("Cannot aggregate an env belief without any window posteriors")

    window_means = np.stack([np.asarray(item[0], dtype=np.float32) for item in posterior_views], axis=0)
    window_logvars = np.stack([np.asarray(item[1], dtype=np.float32) for item in posterior_views], axis=0)
    payload = aggregate_env_posteriors(
        aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        device=device,
        window_means=window_means,
        window_logvars=window_logvars,
    )
    payload["belief"] = sanitize_belief_vector(payload["belief"])
    payload["env_mean"] = sanitize_array(payload["env_mean"])
    if "env_mean_raw" in payload:
        payload["env_mean_raw"] = sanitize_array(payload["env_mean_raw"])
    payload["env_logvar"] = sanitize_array(payload["env_logvar"])
    payload["view_spread"] = sanitize_array(payload["view_spread"], nan=0.0, posinf=1.0, neginf=0.0)
    payload["env_param_mean"] = sanitize_array(payload["env_param_mean"])
    payload["env_param_std"] = sanitize_array(payload["env_param_std"], nan=0.0, posinf=1.0, neginf=0.0)
    return payload["belief"], payload


def belief_mean_z(belief: np.ndarray) -> np.ndarray:
    """Extract the mean-latent half of a belief vector."""
    belief_np = sanitize_belief_vector(belief)
    half = belief_np.shape[0] // 2
    return np.asarray(belief_np[:half], dtype=np.float32)


def belief_uncertainty(belief: np.ndarray) -> float:
    """Collapse the spread half of the belief into one scalar uncertainty."""
    belief_np = sanitize_belief_vector(belief)
    half = belief_np.shape[0] // 2
    spread = np.asarray(belief_np[half:], dtype=np.float32)
    return float(np.mean(spread))


def belief_posterior_std(belief: np.ndarray) -> np.ndarray:
    """Extract the posterior-std portion of a posterior-style belief vector."""
    belief_np = sanitize_belief_vector(belief)
    half = belief_np.shape[0] // 2
    return np.asarray(belief_np[half:], dtype=np.float32)


def belief_epistemic_std(belief: np.ndarray) -> np.ndarray:
    """Extract the epistemic-disagreement portion of a posterior-style belief."""
    return np.zeros_like(belief_mean_z(belief))


def update_belief_with_latent(
    belief: np.ndarray,
    new_latent: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend a new latent observation into an existing belief."""
    mean_z = normalize_latent(belief_mean_z(belief))
    spread_z = np.asarray(belief[len(mean_z):], dtype=np.float32)
    normalized_new_latent = normalize_latent(new_latent)
    # Spread tracks how much new evidence disagrees with the current belief mean.
    deviation = np.abs(normalized_new_latent - mean_z)
    updated_mean = normalize_latent((1.0 - alpha) * mean_z + alpha * normalized_new_latent)
    updated_spread = np.clip((1.0 - alpha) * spread_z + alpha * deviation, 0.0, 1.0)
    return np.concatenate([updated_mean, updated_spread], axis=0).astype(np.float32)


def update_belief_with_posterior(
    belief: np.ndarray,
    new_mean: np.ndarray,
    new_logvar: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend a new Gaussian posterior into the existing belief."""
    old_mean = belief_mean_z(belief)
    old_std = np.clip(belief_posterior_std(belief), 1e-3, None)
    new_mean = np.asarray(new_mean, dtype=np.float32)
    new_var = np.exp(np.asarray(new_logvar, dtype=np.float32))
    old_var = np.square(old_std)
    blended_precision = (1.0 - alpha) / old_var + alpha / np.clip(new_var, 1e-6, None)
    updated_var = 1.0 / np.clip(blended_precision, 1e-6, None)
    updated_mean = updated_var * (
        (1.0 - alpha) * old_mean / old_var + alpha * new_mean / np.clip(new_var, 1e-6, None)
    )
    updated_std = np.sqrt(np.clip(updated_var, 1e-6, None))
    return np.concatenate([updated_mean, updated_std], axis=0).astype(np.float32)


def init_recurrent_belief_hidden(
    encoder: WorldEncoder,
    device: torch.device,
) -> torch.Tensor:
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

    mean = mean_t.squeeze(0).cpu().numpy().astype(np.float32)
    logvar = logvar_t.squeeze(0).cpu().numpy().astype(np.float32)
    mean = sanitize_array(mean, nan=0.0, posinf=0.0, neginf=0.0)
    logvar = sanitize_array(logvar, nan=0.0, posinf=2.0, neginf=-5.0)
    if prior_belief is None:
        belief = build_belief_vector([(mean, logvar)])
    else:
        belief = update_belief_with_posterior(
            belief=prior_belief,
            new_mean=mean,
            new_logvar=logvar,
            alpha=alpha,
        )
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
    rewards = None
    if window_rewards is not None:
        rewards = np.asarray(window_rewards, dtype=np.float32)

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


def select_episode_physics(
    rng: np.random.Generator,
    randomize_physics: bool,
    base_physics,
):
    """Choose either a randomized physics instance or the shared base one."""
    if randomize_physics:
        return sample_env_params(rng, base_physics)
    return base_physics


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
        rewards = None
        if window_rewards is not None:
            rewards = torch.tensor(window_rewards[None, ...], dtype=torch.float32, device=device)
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
        rewards = None
        if window_rewards is not None:
            rewards = torch.tensor(window_rewards[None, ...], dtype=torch.float32, device=device)
        mean, logvar = encoder.encode_posterior(states, actions, rewards=rewards)
    return (
        mean.squeeze(0).cpu().numpy().astype(np.float32),
        logvar.squeeze(0).cpu().numpy().astype(np.float32),
    )


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
    # Retry until we get a full clean probe window rather than an early-terminated one.
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
            return (
                np.stack(states, axis=0).astype(np.float32),
                np.asarray(actions, dtype=np.int64),
                False,
            )

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
        base_scores = (
            disagreement
            + 0.15 * transition_energy
            + 0.35 * posterior_shrinkage
            + 0.10 * latent_shift
        )
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
    return np.nan_to_num(
        total_scores.astype(np.float32),
        nan=-1.0,
        posinf=5.0,
        neginf=-5.0,
    )


def safe_choice_weights(
    scores: np.ndarray,
    temperature: float,
    fallback_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Convert scores into a stable sampling distribution with sane fallbacks."""
    safe_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    safe_scores = np.nan_to_num(safe_scores, nan=-1.0, posinf=5.0, neginf=-5.0)

    if fallback_scores is not None:
        fallback_scores = np.asarray(fallback_scores, dtype=np.float32).reshape(-1)
        fallback_scores = np.nan_to_num(fallback_scores, nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        fallback_scores = None

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
            hidden_t = predicted_next_hidden[:, first_action_idx:first_action_idx + 1, :].repeat(
                1,
                action_vocab_size,
                1,
            )
            current_std = float(
                torch.exp(0.5 * predicted_next_logvar[first_action_idx]).mean().item()
            )
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
            disagreement = (
                delta_preds.std(dim=0).mean(dim=1).detach().cpu().numpy().astype(np.float32)
            )
            posterior_std = (
                torch.exp(0.5 * next_logvar_2).mean(dim=1).detach().cpu().numpy().astype(np.float32)
            )
            posterior_shrinkage = np.clip(current_std - posterior_std, 0.0, None)
            latent_shift = (
                torch.norm(next_mean_2 - latent_t, dim=1).detach().cpu().numpy().astype(np.float32)
            )
            future_scores[first_action_idx] = float(
                np.max(disagreement + 0.35 * posterior_shrinkage + 0.10 * latent_shift)
            )

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
    temperature = 0.35
    weights = safe_choice_weights(
        scores,
        temperature=temperature,
        fallback_scores=action_prior_scores,
    )
    return int(rng.choice(np.arange(action_vocab_size), p=weights))


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
):
    """Collect one probe window using disagreement-driven active actions."""
    action_vocab_size = int(len(action_values))
    steps_used = 0
    local_planner = planner
    if local_planner is None and env_name is not None:
        local_planner = build_probe_planner(env_name=env_name, action_values=action_values, rng=rng)

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
            action_prior_scores = None
            if local_planner is not None:
                action_prior_scores = local_planner.action_prior_scores(states[-1], actions)
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
            continue

        if len(actions) == window_size and len(states) == window_size + 1:
            return (
                np.stack(states, axis=0).astype(np.float32),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                False,
                steps_used,
            )

    return None, None, None, True, steps_used


def nearest_probe_action_idx(action: np.ndarray, action_values: np.ndarray) -> int:
    """Map a continuous policy action back to the nearest discrete probe action."""
    action_np = np.asarray(action, dtype=np.float32).reshape(1, -1)
    prototypes = np.asarray(action_values, dtype=np.float32)
    if prototypes.ndim == 1:
        prototypes = prototypes.reshape(-1, 1)
    distances = np.linalg.norm(prototypes - action_np, axis=1)
    return int(np.argmin(distances))


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
    # Always keep the recurrent hidden belief state current, but only refresh the
    # exposed policy-conditioning vector on the configured schedule.
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
    blended_belief = (
        (1.0 - online_z_update_alpha) * np.asarray(belief, dtype=np.float32)
        + online_z_update_alpha * aggregated_belief
    ).astype(np.float32)
    return sanitize_belief_vector(blended_belief), next_hidden, updated_posteriors


def choose_probe_count(
    z: np.ndarray,
    performance_memory: LatentPerformanceMemory,
    base_probe_episodes: int,
    max_probe_episodes: int,
    novelty_probe_threshold: float,
    low_return_probe_threshold: float,
) -> tuple[int, float, float]:
    """Decide how many probe windows to run before the control episode."""
    novelty = performance_memory.novelty(z)
    expected_return = performance_memory.expected_return(z)
    probe_count = base_probe_episodes

    if novelty >= novelty_probe_threshold:
        probe_count += 1

    # Avoid turning every early episode into a heavy probe routine before the
    # memory has enough examples to make the return estimate meaningful.
    if len(performance_memory) >= 16 and expected_return < low_return_probe_threshold:
        probe_count += 1
    return min(max_probe_episodes, probe_count), novelty, expected_return


def choose_policy_epochs(
    base_ppo_epochs: int,
    expected_return: float,
    uncertainty: float,
    exploit_return_threshold: float,
    uncertainty_focus_threshold: float,
) -> int:
    """Give promising low-uncertainty episodes a little more PPO update budget."""
    epochs = base_ppo_epochs
    if expected_return >= exploit_return_threshold:
        epochs += 1
        if uncertainty < uncertainty_focus_threshold:
            epochs += 1
    return min(base_ppo_epochs + 2, epochs)


def adjust_entropy_coef(
    base_entropy_coef: float,
    novelty: float,
    expected_return: float,
    uncertainty: float,
    novelty_probe_threshold: float,
    low_return_probe_threshold: float,
    exploit_return_threshold: float,
    uncertainty_focus_threshold: float,
) -> float:
    """Slightly retune exploration pressure from novelty/return/uncertainty."""
    entropy_coef = base_entropy_coef
    if novelty >= novelty_probe_threshold:
        entropy_coef *= 1.08
    if expected_return < low_return_probe_threshold:
        entropy_coef *= 1.08
    if uncertainty >= uncertainty_focus_threshold:
        entropy_coef *= 1.03
    if expected_return >= exploit_return_threshold and uncertainty < 0.5 * uncertainty_focus_threshold:
        entropy_coef *= 0.75
    return float(np.clip(entropy_coef, 1e-4, 0.05))


def should_promote_episode_to_elite(
    episode_return: float,
    completed_returns,
    best_return_so_far: float,
    min_elite_return: float,
    current_episode: int,
    warmup_episodes: int,
    std_scale: float,
) -> tuple[bool, float]:
    """Choose whether this episode should enter the self-imitation buffer."""
    if current_episode <= warmup_episodes:
        return False, min_elite_return

    recent_returns = completed_returns[-20:]
    recent_avg = float(np.mean(recent_returns)) if recent_returns else 0.0
    recent_std = float(np.std(recent_returns)) if recent_returns else 0.0
    threshold = max(
        min_elite_return,
        recent_avg + std_scale * recent_std,
        0.9 * best_return_so_far,
    )
    return episode_return >= threshold, threshold
