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
from ..representation import DeltaPredictorEnsemble, WorldEncoder


def normalize_latent(latent: np.ndarray) -> np.ndarray:
    """Normalize a latent to unit length unless it is effectively zero."""
    latent_np = np.asarray(latent, dtype=np.float32)
    norm = float(np.linalg.norm(latent_np))
    if norm <= 1e-6:
        return latent_np
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
        epistemic = np.std(means, axis=0)
        return np.concatenate([combined_mean, combined_std, epistemic], axis=0).astype(np.float32)

    # Legacy path: if callers still pass raw latents, treat them like a point estimate.
    stacked = np.stack([normalize_latent(latent) for latent in latents], axis=0).astype(np.float32)
    mean_z = normalize_latent(np.mean(stacked, axis=0))
    spread_z = np.clip(np.std(stacked, axis=0), 0.0, 1.0)
    return np.concatenate([mean_z, spread_z], axis=0).astype(np.float32)


def belief_mean_z(belief: np.ndarray) -> np.ndarray:
    """Extract the mean-latent half of a belief vector."""
    if belief.shape[0] % 3 == 0:
        third = belief.shape[0] // 3
        return np.asarray(belief[:third], dtype=np.float32)
    half = belief.shape[0] // 2
    return np.asarray(belief[:half], dtype=np.float32)


def belief_uncertainty(belief: np.ndarray) -> float:
    """Collapse the spread half of the belief into one scalar uncertainty."""
    if belief.shape[0] % 3 == 0:
        third = belief.shape[0] // 3
        posterior_std = np.asarray(belief[third:2 * third], dtype=np.float32)
        epistemic = np.asarray(belief[2 * third:], dtype=np.float32)
        return float(np.mean(posterior_std) + np.mean(epistemic))
    half = belief.shape[0] // 2
    spread = np.asarray(belief[half:], dtype=np.float32)
    return float(np.mean(spread))


def belief_posterior_std(belief: np.ndarray) -> np.ndarray:
    """Extract the posterior-std portion of a posterior-style belief vector."""
    if belief.shape[0] % 3 != 0:
        half = belief.shape[0] // 2
        return np.asarray(belief[half:], dtype=np.float32)
    third = belief.shape[0] // 3
    return np.asarray(belief[third:2 * third], dtype=np.float32)


def belief_epistemic_std(belief: np.ndarray) -> np.ndarray:
    """Extract the epistemic-disagreement portion of a posterior-style belief."""
    if belief.shape[0] % 3 != 0:
        return np.zeros_like(belief_mean_z(belief))
    third = belief.shape[0] // 3
    return np.asarray(belief[2 * third:], dtype=np.float32)


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
    if belief.shape[0] % 3 != 0:
        return update_belief_with_latent(belief, new_mean, alpha)

    old_mean = belief_mean_z(belief)
    old_std = np.clip(belief_posterior_std(belief), 1e-3, None)
    old_epistemic = belief_epistemic_std(belief)
    new_mean = np.asarray(new_mean, dtype=np.float32)
    new_var = np.exp(np.asarray(new_logvar, dtype=np.float32))
    old_var = np.square(old_std)
    blended_precision = (1.0 - alpha) / old_var + alpha / np.clip(new_var, 1e-6, None)
    updated_var = 1.0 / np.clip(blended_precision, 1e-6, None)
    updated_mean = updated_var * (
        (1.0 - alpha) * old_mean / old_var + alpha * new_mean / np.clip(new_var, 1e-6, None)
    )
    updated_std = np.sqrt(np.clip(updated_var, 1e-6, None))
    updated_epistemic = np.clip(
        (1.0 - alpha) * old_epistemic + alpha * np.abs(new_mean - updated_mean),
        0.0,
        3.0,
    )
    return np.concatenate([updated_mean, updated_std, updated_epistemic], axis=0).astype(np.float32)


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
    predictor: DeltaPredictorEnsemble | None,
    device: torch.device,
    current_state: np.ndarray,
    belief: np.ndarray | None,
    action_vocab_size: int,
    recent_actions: list[int],
) -> np.ndarray:
    """Score candidate probe actions by model disagreement plus action diversity."""
    base_scores = np.zeros(action_vocab_size, dtype=np.float32)
    if predictor is not None and belief is not None:
        repeated_states = np.repeat(np.asarray(current_state, dtype=np.float32)[None, :], action_vocab_size, axis=0)
        repeated_latent = np.repeat(belief_mean_z(belief)[None, :], action_vocab_size, axis=0)
        state_t = torch.tensor(repeated_states, dtype=torch.float32, device=device)
        action_t = torch.arange(action_vocab_size, dtype=torch.long, device=device)
        latent_t = torch.tensor(repeated_latent, dtype=torch.float32, device=device)
        with torch.no_grad():
            delta_preds = predictor.predict_all(state_t, action_t, latent_t)
        disagreement = delta_preds.std(dim=0).mean(dim=1).cpu().numpy().astype(np.float32)
        transition_energy = delta_preds.mean(dim=0).abs().mean(dim=1).cpu().numpy().astype(np.float32)
        base_scores = disagreement + 0.15 * transition_energy

    counts = np.bincount(recent_actions, minlength=action_vocab_size).astype(np.float32)
    novelty_bonus = 1.0 / (1.0 + counts)
    return base_scores + 0.10 * novelty_bonus


def choose_active_probe_action(
    predictor: DeltaPredictorEnsemble | None,
    device: torch.device,
    rng: np.random.Generator,
    current_state: np.ndarray,
    belief: np.ndarray | None,
    action_vocab_size: int,
    recent_actions: list[int],
) -> int:
    """Choose an exploratory action using disagreement and a little randomness."""
    if belief is None or len(recent_actions) < 2 or predictor is None:
        return int(rng.integers(0, action_vocab_size))

    scores = score_probe_actions(
        predictor=predictor,
        device=device,
        current_state=current_state,
        belief=belief,
        action_vocab_size=action_vocab_size,
        recent_actions=recent_actions,
    )
    temperature = 0.35
    logits = scores - np.max(scores)
    weights = np.exp(logits / max(temperature, 1e-3))
    weights = weights / np.clip(np.sum(weights), 1e-6, None)
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
    prior_belief: np.ndarray | None = None,
    max_probe_retries: int = 12,
):
    """Collect one probe window using disagreement-driven active actions."""
    action_vocab_size = int(len(action_values))
    steps_used = 0
    for _ in range(max_probe_retries):
        apply_env_params(env, episode_physics)
        state, _info = env.reset()
        states = [np.asarray(state, dtype=np.float32)]
        actions: list[int] = []
        rewards: list[float] = []
        current_belief = None if prior_belief is None else np.asarray(prior_belief, dtype=np.float32)
        done = False

        for step_idx in range(window_size):
            del step_idx
            action_idx = choose_active_probe_action(
                predictor=predictor,
                device=device,
                rng=rng,
                current_state=states[-1],
                belief=current_belief,
                action_vocab_size=action_vocab_size,
                recent_actions=actions,
            )
            env_action = action_index_to_env_action(action_idx, action_values)
            next_state, reward, terminated, truncated, _info = env.step(env_action)
            steps_used += 1
            states.append(np.asarray(next_state, dtype=np.float32))
            actions.append(int(action_idx))
            rewards.append(float(reward))
            done = bool(terminated or truncated)
            if done:
                break

            if len(actions) >= 2:
                posterior = encode_window_posterior(
                    encoder=encoder,
                    device=device,
                    window_states=np.stack(states, axis=0).astype(np.float32),
                    window_actions=np.asarray(actions, dtype=np.int64),
                    window_rewards=np.asarray(rewards, dtype=np.float32),
                )
                if prior_belief is None:
                    current_belief = build_belief_vector([posterior])
                else:
                    current_belief = update_belief_with_posterior(
                        belief=current_belief,
                        new_mean=posterior[0],
                        new_logvar=posterior[1],
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
    device: torch.device,
    recent_states: deque,
    recent_action_idx: deque,
    recent_rewards: deque | None,
    belief: np.ndarray,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    episode_step: int,
) -> np.ndarray:
    """Refresh the belief online from the recent real trajectory when scheduled."""
    if len(recent_action_idx) < recent_action_idx.maxlen:
        return belief
    if episode_step % online_z_update_freq != 0:
        return belief

    # During control, refresh the latent from the recent real trajectory.
    reward_array = None
    if recent_rewards is not None and len(recent_rewards) == recent_action_idx.maxlen:
        reward_array = np.asarray(recent_rewards, dtype=np.float32)

    updated_mean, updated_logvar = encode_window_posterior(
        encoder=encoder,
        device=device,
        window_states=np.stack(recent_states, axis=0),
        window_actions=np.asarray(recent_action_idx, dtype=np.int64),
        window_rewards=reward_array,
    )
    return update_belief_with_posterior(
        belief=belief,
        new_mean=updated_mean,
        new_logvar=updated_logvar,
        alpha=online_z_update_alpha,
    )


def choose_probe_count(
    z: np.ndarray,
    performance_memory: LatentPerformanceMemory,
    base_probe_episodes: int,
    max_probe_episodes: int,
    novelty_probe_threshold: float,
    low_return_probe_threshold: float,
) -> tuple[int, float, float]:
    """Decide how many scripted probes to run before the control episode."""
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
