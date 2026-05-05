"""Generic transition event scoring for continuous-control world models."""

from __future__ import annotations

import numpy as np

from ..gym_mpc import TransitionBatch, normalize_actions


def transition_event_weights(
    batch: TransitionBatch,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    strength: float,
    terminal_weight: float,
    quantile: float,
    floor: float,
    action_saturation_floor: float,
) -> tuple[np.ndarray, dict[str, object]]:
    rewards = np.asarray(batch.rewards, dtype=np.float32).reshape(-1)
    observations = np.asarray(batch.observations, dtype=np.float32)
    next_observations = np.asarray(batch.next_observations, dtype=np.float32)
    actions = np.asarray(batch.actions, dtype=np.float32)
    dones = np.asarray(batch.dones, dtype=np.float32).reshape(-1)

    obs_std = np.std(observations, axis=0) + 1e-4
    reward_center = float(np.median(rewards)) if rewards.size else 0.0
    reward_mag = scaled_event(np.abs(rewards - reward_center), quantile=quantile)
    reward_delta = scaled_event(abs_reward_delta(rewards), quantile=quantile)
    state_delta = scaled_event(
        np.linalg.norm((next_observations - observations) / obs_std, axis=1),
        quantile=quantile,
    )
    action_z = normalize_actions(actions, action_low, action_high)
    saturation = np.max(np.abs(action_z), axis=1) if action_z.size else np.zeros_like(rewards)
    action_event = threshold_event(saturation, floor=action_saturation_floor)

    raw_score = np.maximum.reduce([reward_mag, reward_delta, state_delta, action_event, dones])
    score = threshold_event(raw_score, floor=floor)
    weights = 1.0 + float(strength) * score + float(terminal_weight) * dones
    weights = np.asarray(np.clip(weights, 1.0, 1.0 + float(strength) + float(terminal_weight)), dtype=np.float32)
    stats = {
        "event_weight_mean": float(np.mean(weights)) if weights.size else 1.0,
        "event_weight_max": float(np.max(weights)) if weights.size else 1.0,
        "event_fraction": float(np.mean(score > 0.0)) if score.size else 0.0,
        "terminal_event_fraction": float(np.mean(dones > 0.5)) if dones.size else 0.0,
        "reward_event_fraction": float(np.mean(threshold_event(np.maximum(reward_mag, reward_delta), floor=floor) > 0.0))
        if rewards.size
        else 0.0,
        "state_event_fraction": float(np.mean(threshold_event(state_delta, floor=floor) > 0.0))
        if state_delta.size
        else 0.0,
        "action_event_fraction": float(np.mean(action_event > 0.0)) if action_event.size else 0.0,
        "event_floor": float(floor),
        "action_saturation_mean": float(np.mean(saturation)) if saturation.size else 0.0,
    }
    return weights, stats


def scaled_event(values: np.ndarray, *, quantile: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    scale = float(np.percentile(values, float(np.clip(quantile, 0.5, 0.99)) * 100.0))
    if scale <= 1e-8:
        scale = float(np.max(values))
    if scale <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, 0.0, 1.0).astype(np.float32)


def abs_reward_delta(rewards: np.ndarray) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if rewards.size == 0:
        return rewards
    previous = np.concatenate([rewards[:1], rewards[:-1]], axis=0)
    return np.abs(rewards - previous).astype(np.float32)


def threshold_event(values: np.ndarray, *, floor: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    floor = float(np.clip(floor, 0.0, 0.99))
    if values.size == 0:
        return values
    if floor <= 0.0:
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    return np.clip((values - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0).astype(np.float32)


__all__ = ["transition_event_weights"]
