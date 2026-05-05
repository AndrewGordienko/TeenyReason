"""Model fitting helpers for the active scenario-control path."""

from __future__ import annotations

import numpy as np

from ..gym_mpc import TransitionBatch
from .config import AdvancedGymMPCConfig
from .events import transition_event_weights
from .model import ActionValueModel, EnsembleMLPWorldModel, ValueBootstrapModel
from .planner import CEMPlanner


def fit_model(
    config: AdvancedGymMPCConfig,
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    seed_offset: int,
    epochs: int | None = None,
) -> tuple[EnsembleMLPWorldModel, float]:
    sample_weights = model_sample_weights(config, batch, action_low, action_high)
    return EnsembleMLPWorldModel.fit(
        batch,
        action_low=action_low,
        action_high=action_high,
        ensemble_size=int(config.ensemble_size),
        hidden_dim=int(config.hidden_dim),
        epochs=int(config.epochs if epochs is None else epochs),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        seed=int(config.seed + seed_offset),
        sample_weights=sample_weights,
    )


def fit_value_model(
    config: AdvancedGymMPCConfig,
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    seed_offset: int,
    epochs: int | None = None,
) -> tuple[ValueBootstrapModel | None, float, dict[str, float]]:
    if not bool(config.value_bootstrap):
        return None, 0.0, {"value_bootstrap": 0.0}
    sample_weights = model_sample_weights(config, batch, action_low, action_high)
    value_model, value_loss, stats = ValueBootstrapModel.fit(
        batch,
        discount=float(config.discount),
        hidden_dim=int(config.hidden_dim),
        epochs=int(config.epochs if epochs is None else epochs),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        seed=int(config.seed + seed_offset + 43_000),
        sample_weights=sample_weights,
    )
    stats["value_bootstrap"] = 1.0
    return value_model, float(value_loss), stats


def fit_action_value_model(
    config: AdvancedGymMPCConfig,
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    seed_offset: int,
    epochs: int | None = None,
) -> tuple[ActionValueModel | None, float, dict[str, float]]:
    if not bool(config.action_value_bootstrap):
        return None, 0.0, {"action_value_bootstrap": 0.0}
    sample_weights = model_sample_weights(config, batch, action_low, action_high)
    action_value_model, action_value_loss, stats = ActionValueModel.fit(
        batch,
        action_low=action_low,
        action_high=action_high,
        discount=float(config.discount),
        hidden_dim=int(config.hidden_dim),
        epochs=int(config.epochs if epochs is None else epochs),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        seed=int(config.seed + seed_offset + 47_000),
        sample_weights=sample_weights,
    )
    stats["action_value_bootstrap"] = 1.0
    return action_value_model, float(action_value_loss), stats


def make_planner(
    config: AdvancedGymMPCConfig,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> CEMPlanner:
    return CEMPlanner(
        action_low=action_low,
        action_high=action_high,
        horizon=int(config.horizon),
        candidate_count=int(config.candidate_count),
        iterations=int(config.cem_iterations),
        elite_fraction=float(config.elite_fraction),
        noise_floor=float(config.action_noise_floor),
        temporal_chunk_size=int(config.temporal_chunk_size),
        temporal_chunk_candidates=int(config.temporal_chunk_candidates),
        temporal_smoothness_penalty=float(config.temporal_smoothness_penalty),
    )


def model_sample_weights(
    config: AdvancedGymMPCConfig,
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray | None:
    if not bool(config.event_weighting):
        return None
    weights, _stats = transition_event_weights(
        batch,
        action_low=action_low,
        action_high=action_high,
        strength=float(config.event_weight_strength),
        terminal_weight=float(config.event_terminal_weight),
        quantile=float(config.event_quantile),
        floor=float(config.event_floor),
        action_saturation_floor=float(config.event_action_saturation_floor),
    )
    return weights


__all__ = [
    "fit_action_value_model",
    "fit_model",
    "fit_value_model",
    "make_planner",
]
