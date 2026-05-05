"""Model-truth diagnostics for generic world-model control."""

from __future__ import annotations

import numpy as np

from ..gym_mpc import TransitionBatch
from .model import EnsembleMLPWorldModel


def split_transition_batch(
    batch: TransitionBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[TransitionBatch, TransitionBatch | None]:
    count = int(batch.observations.shape[0])
    validation_count = int(round(count * float(validation_fraction)))
    if count < 6 or validation_count < 2 or validation_count >= count:
        return batch, None
    rng = np.random.default_rng(int(seed) + 41_000)
    order = rng.permutation(count)
    validation_idx = np.sort(order[:validation_count])
    train_idx = np.sort(order[validation_count:])
    return take_batch(batch, train_idx), take_batch(batch, validation_idx)


def model_validation_diagnostics(
    model: EnsembleMLPWorldModel,
    batch: TransitionBatch | None,
    *,
    rollout_horizons: tuple[int, ...],
) -> dict[str, object]:
    if batch is None or int(batch.observations.shape[0]) == 0:
        return empty_diagnostics()

    pred = model.predict_batch(batch.observations, batch.actions)
    errors = np.square((pred["next_observation"] - batch.next_observations) / model.obs_std)
    per_row_mse = np.mean(errors, axis=1)
    reward_error = np.abs(pred["reward"] - batch.rewards)
    done_pred = (pred["done_risk"] >= 0.5).astype(np.float32)
    done_accuracy = np.mean(done_pred == batch.dones.astype(np.float32))
    diagnostics: dict[str, object] = {
        "validation_samples": int(batch.observations.shape[0]),
        "one_step_obs_mse": float(np.mean(per_row_mse)),
        "one_step_reward_mae": float(np.mean(reward_error)),
        "done_accuracy": float(done_accuracy),
        "uncertainty_error_corr": safe_corr(pred["uncertainty"], per_row_mse),
    }
    for horizon in rollout_horizons:
        value = rollout_mse(model, batch, horizon=max(1, int(horizon)))
        diagnostics[f"rollout_mse_h{int(horizon)}"] = value
    return diagnostics


def rollout_mse(
    model: EnsembleMLPWorldModel,
    batch: TransitionBatch,
    *,
    horizon: int,
) -> float | None:
    count = int(batch.observations.shape[0])
    if horizon <= 0 or count < horizon:
        return None
    rows: list[float] = []
    for start in range(0, count - horizon + 1):
        if horizon > 1 and np.any(batch.dones[start : start + horizon - 1] > 0.5):
            continue
        obs = batch.observations[start].reshape(1, -1)
        for offset in range(horizon):
            action = batch.actions[start + offset].reshape(1, -1)
            obs = model.predict_batch(obs, action)["next_observation"]
        target = batch.next_observations[start + horizon - 1]
        error = np.square((obs.reshape(-1) - target.reshape(-1)) / model.obs_std)
        rows.append(float(np.mean(error)))
    if not rows:
        return None
    return float(np.mean(rows))


def take_batch(batch: TransitionBatch, indices: np.ndarray) -> TransitionBatch:
    indices = np.asarray(indices, dtype=np.int64)
    return TransitionBatch(
        observations=batch.observations[indices],
        actions=batch.actions[indices],
        rewards=batch.rewards[indices],
        next_observations=batch.next_observations[indices],
        dones=batch.dones[indices],
        episode_returns=batch.episode_returns,
    )


def merge_transition_batches(base: TransitionBatch, extra: TransitionBatch | None) -> TransitionBatch:
    if extra is None or int(extra.observations.shape[0]) == 0:
        return base
    return TransitionBatch(
        observations=np.concatenate([base.observations, extra.observations], axis=0),
        actions=np.concatenate([base.actions, extra.actions], axis=0),
        rewards=np.concatenate([base.rewards, extra.rewards], axis=0),
        next_observations=np.concatenate([base.next_observations, extra.next_observations], axis=0),
        dones=np.concatenate([base.dones, extra.dones], axis=0),
        episode_returns=np.concatenate([base.episode_returns, extra.episode_returns], axis=0),
    )


def batch_from_rows(rows: list[dict[str, np.ndarray | float]]) -> TransitionBatch | None:
    if not rows:
        return None
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float32)
    return TransitionBatch(
        observations=np.asarray([row["observation"] for row in rows], dtype=np.float32),
        actions=np.asarray([row["action"] for row in rows], dtype=np.float32),
        rewards=rewards,
        next_observations=np.asarray([row["next_observation"] for row in rows], dtype=np.float32),
        dones=np.asarray([float(row["done"]) for row in rows], dtype=np.float32),
        episode_returns=np.asarray([float(np.sum(rewards))], dtype=np.float32),
    )


def empty_diagnostics() -> dict[str, object]:
    return {
        "validation_samples": 0,
        "one_step_obs_mse": None,
        "one_step_reward_mae": None,
        "done_accuracy": None,
        "uncertainty_error_corr": None,
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size < 2 or right.size < 2:
        return None
    if float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
        return None
    return float(np.corrcoef(left, right)[0, 1])


__all__ = [
    "batch_from_rows",
    "merge_transition_batches",
    "model_validation_diagnostics",
    "split_transition_batch",
]
