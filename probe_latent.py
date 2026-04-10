import random
from collections import deque

import numpy as np
import torch

from envs import action_index_to_env_action
from probe_data import ProbePolicy, apply_env_params, sample_env_params
from world_model import WorldEncoder


def normalize_latent(latent: np.ndarray) -> np.ndarray:
    latent_np = np.asarray(latent, dtype=np.float32)
    norm = float(np.linalg.norm(latent_np))
    if norm <= 1e-6:
        return latent_np
    return (latent_np / norm).astype(np.float32)


class LatentPerformanceMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, z: np.ndarray, episode_return: float):
        self.buffer.append((normalize_latent(z), float(episode_return)))

    def __len__(self):
        return len(self.buffer)

    def _similarities(self, z: np.ndarray):
        if not self.buffer:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

        normalized_z = normalize_latent(z)
        memory_z = np.stack([item[0] for item in self.buffer], axis=0)
        returns = np.asarray([item[1] for item in self.buffer], dtype=np.float32)
        similarities = memory_z @ normalized_z
        return similarities.astype(np.float32), returns

    def novelty(self, z: np.ndarray) -> float:
        similarities, _returns = self._similarities(z)
        if similarities.size == 0:
            return 1.0
        return float(1.0 - np.max(similarities))

    def expected_return(self, z: np.ndarray, top_k: int = 8) -> float:
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
    stacked = np.stack([normalize_latent(latent) for latent in latents], axis=0).astype(np.float32)
    mean_z = normalize_latent(np.mean(stacked, axis=0))
    spread_z = np.clip(np.std(stacked, axis=0), 0.0, 1.0)
    return np.concatenate([mean_z, spread_z], axis=0).astype(np.float32)


def belief_mean_z(belief: np.ndarray) -> np.ndarray:
    half = belief.shape[0] // 2
    return np.asarray(belief[:half], dtype=np.float32)


def belief_uncertainty(belief: np.ndarray) -> float:
    half = belief.shape[0] // 2
    spread = np.asarray(belief[half:], dtype=np.float32)
    return float(np.mean(spread))


def update_belief_with_latent(
    belief: np.ndarray,
    new_latent: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mean_z = normalize_latent(belief_mean_z(belief))
    spread_z = np.asarray(belief[len(mean_z):], dtype=np.float32)
    normalized_new_latent = normalize_latent(new_latent)
    deviation = np.abs(normalized_new_latent - mean_z)
    updated_mean = normalize_latent((1.0 - alpha) * mean_z + alpha * normalized_new_latent)
    updated_spread = np.clip((1.0 - alpha) * spread_z + alpha * deviation, 0.0, 1.0)
    return np.concatenate([updated_mean, updated_spread], axis=0).astype(np.float32)


def select_episode_physics(
    rng: np.random.Generator,
    randomize_physics: bool,
    base_physics,
):
    if randomize_physics:
        return sample_env_params(rng, base_physics)
    return base_physics


def encode_window(
    encoder: WorldEncoder,
    device: torch.device,
    window_states: np.ndarray,
    window_actions: np.ndarray,
) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        states = torch.tensor(window_states[None, ...], dtype=torch.float32, device=device)
        actions = torch.tensor(window_actions[None, ...], dtype=torch.long, device=device)
        z = encoder(states, actions).squeeze(0).cpu().numpy().astype(np.float32)
    return z


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


def nearest_probe_action_idx(action: np.ndarray, action_values: np.ndarray) -> int:
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
    belief: np.ndarray,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    episode_step: int,
) -> np.ndarray:
    if len(recent_action_idx) < recent_action_idx.maxlen:
        return belief
    if episode_step % online_z_update_freq != 0:
        return belief

    updated_latent = encode_window(
        encoder=encoder,
        device=device,
        window_states=np.stack(recent_states, axis=0),
        window_actions=np.asarray(recent_action_idx, dtype=np.int64),
    )
    return update_belief_with_latent(
        belief=belief,
        new_latent=updated_latent,
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
