"""Belief-vector helpers and replay buffers for probing."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch

from ....models.envbelief import EnvBeliefAggregator, EnvParamPredictorEnsemble, aggregate_env_posteriors


def sanitize_array(
    values: np.ndarray,
    *,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> np.ndarray:
    """Replace non-finite values in a NumPy array and keep float32 semantics."""
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=nan, posinf=posinf, neginf=neginf).astype(np.float32)


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
    """Replay buffer for unusually strong trajectories used by self-imitation."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push_episode(self, states: np.ndarray, beliefs: np.ndarray, actions: np.ndarray, returns_to_go: np.ndarray, episode_weight: float):
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

    stacked = np.stack([normalize_latent(latent) for latent in latents], axis=0).astype(np.float32)
    mean_z = normalize_latent(np.mean(stacked, axis=0))
    spread_z = np.clip(np.std(stacked, axis=0), 0.0, 1.0)
    return np.concatenate([mean_z, spread_z], axis=0).astype(np.float32)


def probe_group_ids_from_families(
    probe_families: Sequence[str | None],
    family_names: Sequence[str] | None = None,
) -> np.ndarray | None:
    """Map probe-family names onto the ids used by the trained family heads."""
    if not probe_families:
        return None

    if family_names is None:
        family_to_id: dict[str, int] = {}
        ids: list[int] = []
        for family in probe_families:
            name = "" if family is None else str(family)
            if not name:
                ids.append(-1)
                continue
            if name not in family_to_id:
                family_to_id[name] = len(family_to_id)
            ids.append(family_to_id[name])
        return np.asarray(ids, dtype=np.int64)

    family_to_id = {str(name): idx for idx, name in enumerate(family_names)}
    return np.asarray(
        [
            int(family_to_id.get("" if family is None else str(family), -1))
            for family in probe_families
        ],
        dtype=np.int64,
    )


def aggregate_env_belief(
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    device: torch.device,
    posterior_views: list[tuple[np.ndarray, np.ndarray]],
    probe_group_ids: np.ndarray | None = None,
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
        probe_group_ids=probe_group_ids,
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


def update_belief_with_latent(belief: np.ndarray, new_latent: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a new latent observation into an existing belief."""
    mean_z = normalize_latent(belief_mean_z(belief))
    spread_z = np.asarray(belief[len(mean_z):], dtype=np.float32)
    normalized_new_latent = normalize_latent(new_latent)
    deviation = np.abs(normalized_new_latent - mean_z)
    updated_mean = normalize_latent((1.0 - alpha) * mean_z + alpha * normalized_new_latent)
    updated_spread = np.clip((1.0 - alpha) * spread_z + alpha * deviation, 0.0, 1.0)
    return np.concatenate([updated_mean, updated_spread], axis=0).astype(np.float32)


def update_belief_with_posterior(belief: np.ndarray, new_mean: np.ndarray, new_logvar: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a new Gaussian posterior into the existing belief."""
    old_mean = belief_mean_z(belief)
    old_std = np.clip(belief_posterior_std(belief), 1e-3, None)
    new_mean = np.asarray(new_mean, dtype=np.float32)
    new_var = np.exp(np.asarray(new_logvar, dtype=np.float32))
    old_var = np.square(old_std)
    blended_precision = (1.0 - alpha) / old_var + alpha / np.clip(new_var, 1e-6, None)
    updated_var = 1.0 / np.clip(blended_precision, 1e-6, None)
    updated_mean = updated_var * ((1.0 - alpha) * old_mean / old_var + alpha * new_mean / np.clip(new_var, 1e-6, None))
    updated_std = np.sqrt(np.clip(updated_var, 1e-6, None))
    return np.concatenate([updated_mean, updated_std], axis=0).astype(np.float32)
