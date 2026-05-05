"""Continuous weighting for remembered and imagined scenarios."""

from __future__ import annotations

import numpy as np

from .schema import ScenarioVariant, ScenarioWeights, ScenarioWindow


def score_variant_weights(
    window: ScenarioWindow,
    *,
    predicted_lift: float,
    uncertainty: float,
    done_risk: float,
    variant_surprise: float,
    advantage_temperature: float,
    uncertainty_scale: float,
    surprise_scale: float,
) -> ScenarioWeights:
    familiarity = float(np.clip(window.familiarity, 0.05, 1.0))
    plausibility = float(np.exp(-float(uncertainty_scale) * max(0.0, float(uncertainty)) - 2.0 * max(0.0, float(done_risk))))
    usefulness = float(np.exp(np.clip(float(predicted_lift) / max(1e-6, float(advantage_temperature)), -4.0, 4.0)))
    inverse_surprise = float(np.exp(-float(surprise_scale) * max(0.0, float(variant_surprise))))
    return ScenarioWeights(
        familiarity=familiarity,
        plausibility=max(1e-3, plausibility),
        usefulness=max(1e-3, usefulness),
        inverse_surprise=max(1e-3, inverse_surprise),
    )


def normalize_variant_weights(variants: list[ScenarioVariant]) -> np.ndarray:
    if not variants:
        return np.zeros((0,), dtype=np.float32)
    weights = np.asarray([variant.weights.combined for variant in variants], dtype=np.float32)
    weights = np.maximum(weights, 1e-4)
    return (weights / float(np.mean(weights))).astype(np.float32)


__all__ = ["normalize_variant_weights", "score_variant_weights"]
