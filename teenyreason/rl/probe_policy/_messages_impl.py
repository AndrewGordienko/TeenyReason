"""Solver-facing env-expression helpers.

These helpers keep the controller side intentionally simple:

- one compact environment expression
- one scalar confidence
- one scalar uncertainty

The decomposition-heavy analysis lives elsewhere. By the time PPO sees the
expression, it should feel like "here is my current take on the world, plus
how much I trust it" rather than a bundle of specialist diagnostics.
"""

from __future__ import annotations

import numpy as np

from ...crawler.types import EnvExpression
from ..core.ppo_core import sanitize_numpy


DEFAULT_EXPRESSION_UNCERTAINTY_THRESHOLD = 0.20
DEFAULT_READY_CONFIDENCE_FLOOR = 0.32
DEFAULT_READY_SUPPORT_DIVERSITY_FLOOR = 0.70
DEFAULT_READY_POSTERIOR_ENTROPY_CEILING = 3.80
DEFAULT_READY_FULL_FUTURE_ERROR_GOOD = 0.60
DEFAULT_READY_FULL_FUTURE_ERROR_BAD = 1.20
DEFAULT_READY_HELDOUT_FUTURE_ERROR_GOOD = 0.52
DEFAULT_READY_HELDOUT_FUTURE_ERROR_BAD = 1.00
DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_GOOD = 0.50
DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_BAD = 0.95
DEFAULT_READY_POSTERIOR_ENTROPY_GOOD = 3.80
DEFAULT_READY_POSTERIOR_ENTROPY_BAD = 5.00
DEFAULT_READY_GEOMETRY_SCALE_FLOOR = 0.65
DEFAULT_READINESS_SCORE_FLOOR = 0.55
DEFAULT_CONTROLLER_TRUST_FLOOR = 0.30
DEFAULT_NOT_READY_TRUST_CLAMP = 0.35
DEFAULT_NOT_READY_TRUST_MULTIPLIER = 0.55
DEFAULT_LATE_TRAINING_KEEP_SCALE_FLOOR = 0.50
DEFAULT_SHADOW_READINESS_SCORE_FLOOR = 0.48
DEFAULT_SHADOW_SCORE_FLOOR = 0.58
DEFAULT_SHADOW_FUTURE_PROBE_QUALITY_FLOOR = 0.35
DEFAULT_SHADOW_SUBSET_STABILITY_FLOOR = 0.45
DEFAULT_SHADOW_LEAVEOUT_STABILITY_FLOOR = 0.45
DEFAULT_SHADOW_CONFIDENCE_FLOOR = 0.30
DEFAULT_SHADOW_SCALE_CAP = 0.20
DEFAULT_FAIR_POLICY_FUTURE_QUALITY_FLOOR = 0.50
DEFAULT_FAIR_POLICY_SUBSET_STABILITY_FLOOR = 0.45
DEFAULT_FAIR_POLICY_CONFIDENCE_FLOOR = 0.35
DEFAULT_FAIR_POLICY_ONLINE_OFFLINE_GAP_CEILING = 0.05
DEFAULT_FAIR_POLICY_STRONG_READINESS_FLOOR = 0.62
DEFAULT_FAIR_POLICY_STRONG_SUBSET_STABILITY_FLOOR = 0.60
DEFAULT_FAIR_POLICY_SCALE_CAP = 0.35
DEFAULT_DIAGNOSTIC_SUPPORT_COUNT = 1
DEFAULT_READY_SUPPORT_COUNT = 2
DEFAULT_DIAGNOSTIC_CONFIDENCE_FLOOR = 0.25
DEFAULT_DIAGNOSTIC_FUTURE_QUALITY_FLOOR = 0.35
DEFAULT_DIAGNOSTIC_SUBSET_STABILITY_FLOOR = 0.45
DEFAULT_DIAGNOSTIC_SCALE_FLOOR = 0.12
DEFAULT_DIAGNOSTIC_SCALE_CAP = 0.20
DEFAULT_FORCED_EVAL_EXPRESSION_SCALE = 0.15


def solver_expression_input_from_env_expression(
    env_expression: EnvExpression,
    *,
    confidence_scale: float | None = None,
) -> np.ndarray:
    """Build the canonical solver-side env-expression input.

    The controller sees the compact expression vector plus two explicit scalar
    slots: confidence and uncertainty.
    """
    vector = sanitize_numpy(np.asarray(env_expression.vector, dtype=np.float32).reshape(-1))
    confidence = (
        float(env_expression.confidence)
        if confidence_scale is None
        else float(confidence_scale)
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))
    uncertainty = float(env_expression.uncertainty_scalar)
    return sanitize_numpy(
        np.concatenate(
            [
                vector,
                np.asarray([confidence, uncertainty], dtype=np.float32),
            ],
            axis=0,
        )
    )


def solver_belief_input_from_message(
    belief_message: np.ndarray,
    uncertainty_scalar: float,
    message_scale: float = 1.0,
) -> np.ndarray:
    """Compatibility wrapper around the env-expression controller contract."""
    env_expression = EnvExpression(
        vector=sanitize_numpy(np.asarray(belief_message, dtype=np.float32).reshape(-1)),
        confidence=float(np.clip(message_scale, 0.0, 1.0)),
        ready=False,
        uncertainty_scalar=float(uncertainty_scalar),
        compressed=False,
        metadata={},
    )
    return solver_expression_input_from_env_expression(env_expression)


def apply_solver_expression_keep_scale(
    solver_expression: np.ndarray,
    keep_scale: float,
) -> np.ndarray:
    """Scale the expression content while preserving the uncertainty slot."""
    expression = sanitize_numpy(np.asarray(solver_expression, dtype=np.float32).reshape(-1))
    keep_scale = float(np.clip(keep_scale, 0.0, 1.0))
    if expression.size == 0:
        return expression
    if expression.size == 1:
        return sanitize_numpy(expression * keep_scale)
    scaled = expression.copy()
    if expression.size > 2:
        scaled[:-2] = sanitize_numpy(scaled[:-2] * keep_scale)
    scaled[-2] = float(scaled[-2] * keep_scale)
    return sanitize_numpy(scaled)


def apply_solver_message_keep_scale(
    solver_belief: np.ndarray,
    keep_scale: float,
) -> np.ndarray:
    """Compatibility alias for the older message-oriented name."""
    return apply_solver_expression_keep_scale(solver_belief, keep_scale)


def compute_solver_geometry_scale(
    *,
    gap_ratio: float | None = None,
    split_retrieval_margin_deficit: float | None = None,
    split_latent_disagreement: float | None = None,
) -> float:
    """Downweight beliefs whose local geometry says "same world" is still unstable."""
    geometry_penalty = 0.0
    if gap_ratio is not None:
        gap_budget = 0.18
        geometry_penalty += max(0.0, float(gap_ratio) - gap_budget) / gap_budget
    if split_retrieval_margin_deficit is not None:
        deficit_budget = 0.06
        geometry_penalty += (
            max(0.0, float(split_retrieval_margin_deficit) - deficit_budget)
            / deficit_budget
        )
    if split_latent_disagreement is not None:
        disagreement_budget = 0.015
        geometry_penalty += (
            max(0.0, float(split_latent_disagreement) - disagreement_budget)
            / disagreement_budget
        )
    geometry_scale = 1.0 / (1.0 + geometry_penalty)
    return float(np.clip(geometry_scale, 0.02, 1.0))


def _normalize_ascending(
    value: float | None,
    *,
    good: float,
    bad: float,
) -> float:
    """Map smaller-is-better values into a stable 0-1 quality score."""
    if value is None:
        return 1.0
    if bad <= good:
        return 1.0 if float(value) <= good else 0.0
    return float(np.clip((bad - float(value)) / (bad - good), 0.0, 1.0))


def _normalize_descending(
    value: float | None,
    *,
    bad: float,
    good: float,
) -> float:
    """Map larger-is-better values into a stable 0-1 quality score."""
    if value is None:
        return 1.0
    if good <= bad:
        return 1.0 if float(value) >= good else 0.0
    return float(np.clip((float(value) - bad) / (good - bad), 0.0, 1.0))


def compute_env_expression_readiness_components(
    *,
    future_probe_error: float | None = None,
    heldout_family_future_error: float | None = None,
    support_size_matched_future_error: float | None = None,
    support_count: int | None = None,
    support_diversity_ratio: float | None = None,
    posterior_entropy: float | None = None,
    online_subset_stability: float | None = None,
    online_geometry_complete: bool | None = None,
    gap_ratio: float | None = None,
    split_retrieval_margin_deficit: float | None = None,
    split_latent_disagreement: float | None = None,
    leaveout_shift: float | None = None,
    leaveout_param_std_mean: float | None = None,
) -> dict[str, float]:
    """Compute the four child-like readiness axes used for probe handoff."""
    full_future_quality = _normalize_ascending(
        future_probe_error,
        good=DEFAULT_READY_FULL_FUTURE_ERROR_GOOD,
        bad=DEFAULT_READY_FULL_FUTURE_ERROR_BAD,
    )
    heldout_future_quality = _normalize_ascending(
        heldout_family_future_error if heldout_family_future_error is not None else future_probe_error,
        good=DEFAULT_READY_HELDOUT_FUTURE_ERROR_GOOD,
        bad=DEFAULT_READY_HELDOUT_FUTURE_ERROR_BAD,
    )
    support_size_matched_future_quality = _normalize_ascending(
        support_size_matched_future_error
        if support_size_matched_future_error is not None
        else (
            heldout_family_future_error
            if heldout_family_future_error is not None
            else future_probe_error
        ),
        good=DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_GOOD,
        bad=DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_BAD,
    )
    entropy_quality = _normalize_ascending(
        posterior_entropy,
        good=DEFAULT_READY_POSTERIOR_ENTROPY_GOOD,
        bad=max(
            DEFAULT_READY_POSTERIOR_ENTROPY_BAD,
            DEFAULT_READY_POSTERIOR_ENTROPY_CEILING,
        ),
    )
    future_probe_quality = float(
        np.clip(
            (
                0.65 * support_size_matched_future_quality
                + 0.20 * heldout_future_quality
                + 0.15 * full_future_quality
            )
            * (0.75 + 0.25 * entropy_quality),
            0.0,
            1.0,
        )
    )
    offline_subset_stability = compute_solver_geometry_scale(
        gap_ratio=gap_ratio,
        split_retrieval_margin_deficit=split_retrieval_margin_deficit,
        split_latent_disagreement=split_latent_disagreement,
    )
    subset_stability = float(offline_subset_stability)
    if online_subset_stability is not None and bool(online_geometry_complete):
        subset_stability = float(np.clip(float(online_subset_stability), 0.0, 1.0))
    leaveout_penalty = 0.0
    if leaveout_shift is not None:
        leaveout_penalty += max(0.0, float(leaveout_shift) - 0.05) / 0.10
    if leaveout_param_std_mean is not None:
        leaveout_penalty += max(0.0, float(leaveout_param_std_mean) - 0.03) / 0.10
    leaveout_stability = float(np.clip(1.0 / (1.0 + leaveout_penalty), 0.02, 1.0))
    if support_count is not None and int(support_count) < DEFAULT_READY_SUPPORT_COUNT:
        leaveout_stability = max(
            leaveout_stability,
            DEFAULT_READINESS_SCORE_FLOOR,
        )
    support_diversity = _normalize_descending(
        support_diversity_ratio,
        bad=0.55,
        good=max(DEFAULT_READY_SUPPORT_DIVERSITY_FLOOR, 0.90),
    )
    return {
        "future_probe_quality": future_probe_quality,
        "subset_stability": subset_stability,
        "leaveout_stability": leaveout_stability,
        "support_diversity": support_diversity,
    }


def env_expression_readiness_reason(components: dict[str, float]) -> str:
    """Return the weakest readiness axis for debugging and dashboard display."""
    if not components:
        return "unknown"
    weakest_name, weakest_value = min(
        components.items(),
        key=lambda item: (float(item[1]), item[0]),
    )
    if float(weakest_value) >= DEFAULT_READINESS_SCORE_FLOOR:
        return "ok"
    return weakest_name


def compute_message_mode(
    *,
    support_count: int | None,
    confidence: float,
    readiness_components: dict[str, float],
    readiness_score: float,
) -> tuple[str, str]:
    """Collapse the many readiness details into off/diag/on plus one blocker."""
    support_count = 0 if support_count is None else int(support_count)
    future_quality = float(readiness_components.get("future_probe_quality", 0.0))
    subset_stability = float(readiness_components.get("subset_stability", 0.0))
    readiness_reason = env_expression_readiness_reason(readiness_components)
    if support_count < DEFAULT_DIAGNOSTIC_SUPPORT_COUNT:
        return "off", "support_count"
    if future_quality < DEFAULT_DIAGNOSTIC_FUTURE_QUALITY_FLOOR:
        return "diag", "future_probe_quality"
    if subset_stability < DEFAULT_DIAGNOSTIC_SUBSET_STABILITY_FLOOR:
        return "off", "subset_stability"
    if support_count < DEFAULT_READY_SUPPORT_COUNT:
        return "diag", "support_count"
    if float(confidence) < DEFAULT_READY_CONFIDENCE_FLOOR:
        return "diag", "confidence"
    if float(readiness_score) < DEFAULT_READINESS_SCORE_FLOOR:
        return "diag", readiness_reason
    return "on", "enabled"


def compute_env_expression_readiness_score(
    *,
    future_probe_error: float | None = None,
    heldout_family_future_error: float | None = None,
    support_size_matched_future_error: float | None = None,
    support_count: int | None = None,
    support_diversity_ratio: float | None = None,
    posterior_entropy: float | None = None,
    online_subset_stability: float | None = None,
    online_geometry_complete: bool | None = None,
    gap_ratio: float | None = None,
    split_retrieval_margin_deficit: float | None = None,
    split_latent_disagreement: float | None = None,
    leaveout_shift: float | None = None,
    leaveout_param_std_mean: float | None = None,
) -> float:
    """Readiness is gated by the weakest of the key few-shot belief axes."""
    components = compute_env_expression_readiness_components(
        future_probe_error=future_probe_error,
        heldout_family_future_error=heldout_family_future_error,
        support_size_matched_future_error=support_size_matched_future_error,
        support_count=support_count,
        support_diversity_ratio=support_diversity_ratio,
        posterior_entropy=posterior_entropy,
        online_subset_stability=online_subset_stability,
        online_geometry_complete=online_geometry_complete,
        gap_ratio=gap_ratio,
        split_retrieval_margin_deficit=split_retrieval_margin_deficit,
        split_latent_disagreement=split_latent_disagreement,
        leaveout_shift=leaveout_shift,
        leaveout_param_std_mean=leaveout_param_std_mean,
    )
    return float(min(components.values(), default=0.0))


def compute_env_expression_utility_forecast(
    *,
    readiness_components: dict[str, float] | None = None,
) -> float:
    """Estimate whether the env expression is likely to help control locally."""
    readiness_components = readiness_components or {}
    if not readiness_components:
        return 0.0
    future_quality = float(readiness_components.get("future_probe_quality", 0.0))
    subset_stability = float(readiness_components.get("subset_stability", 0.0))
    leaveout_stability = float(readiness_components.get("leaveout_stability", 0.0))
    support_diversity = float(readiness_components.get("support_diversity", 0.0))
    utility_forecast = (
        0.40 * future_quality
        + 0.25 * subset_stability
        + 0.20 * leaveout_stability
        + 0.15 * support_diversity
    )
    return float(np.clip(utility_forecast, 0.0, 1.0))


def compute_shadow_expression_score(
    *,
    readiness_components: dict[str, float] | None = None,
    env_expression: EnvExpression | None = None,
) -> float:
    """Score near-ready env expressions for the shadow diagnostic arm."""
    if readiness_components is None:
        metadata = {} if env_expression is None else dict(env_expression.metadata or {})
        readiness_components = {
            "future_probe_quality": float(metadata.get("future_probe_quality", 0.0)),
            "subset_stability": float(metadata.get("subset_stability", 0.0)),
            "leaveout_stability": float(metadata.get("leaveout_stability", 0.0)),
            "support_diversity": float(metadata.get("support_diversity", 0.0)),
        }
    shadow_score = (
        0.45 * float(readiness_components.get("future_probe_quality", 0.0))
        + 0.30 * float(readiness_components.get("subset_stability", 0.0))
        + 0.15 * float(readiness_components.get("leaveout_stability", 0.0))
        + 0.10 * float(readiness_components.get("support_diversity", 0.0))
    )
    return float(np.clip(shadow_score, 0.0, 1.0))


def shadow_env_expression_diagnostics(
    *,
    env_expression: EnvExpression,
    readiness_score_floor: float = DEFAULT_SHADOW_READINESS_SCORE_FLOOR,
    shadow_score_floor: float = DEFAULT_SHADOW_SCORE_FLOOR,
    future_probe_quality_floor: float = DEFAULT_SHADOW_FUTURE_PROBE_QUALITY_FLOOR,
    subset_stability_floor: float = DEFAULT_SHADOW_SUBSET_STABILITY_FLOOR,
    leaveout_stability_floor: float = DEFAULT_SHADOW_LEAVEOUT_STABILITY_FLOOR,
    confidence_floor: float = DEFAULT_SHADOW_CONFIDENCE_FLOOR,
) -> dict[str, float | bool | str]:
    """Return one explicit shadow-gate decision plus the main blocker."""
    metadata = dict(env_expression.metadata or {})
    message_mode = str(
        metadata.get("message_mode", "on" if bool(env_expression.ready) else "off")
    )
    shadow_score = compute_shadow_expression_score(env_expression=env_expression)
    readiness_score = float(metadata.get("readiness_score", 0.0))
    future_probe_quality = float(metadata.get("future_probe_quality", 0.0))
    subset_stability = float(metadata.get("subset_stability", 0.0))
    leaveout_stability = float(metadata.get("leaveout_stability", 0.0))
    confidence = float(env_expression.confidence)
    readiness_deficit = max(0.0, float(readiness_score_floor) - readiness_score) / max(
        float(readiness_score_floor),
        1e-6,
    )
    score_deficit = max(0.0, float(shadow_score_floor) - shadow_score) / max(
        float(shadow_score_floor),
        1e-6,
    )
    future_quality_deficit = max(
        0.0,
        float(future_probe_quality_floor) - future_probe_quality,
    ) / max(float(future_probe_quality_floor), 1e-6)
    subset_deficit = max(0.0, float(subset_stability_floor) - subset_stability) / max(
        float(subset_stability_floor),
        1e-6,
    )
    leaveout_deficit = max(
        0.0,
        float(leaveout_stability_floor) - leaveout_stability,
    ) / max(float(leaveout_stability_floor), 1e-6)
    confidence_deficit = max(0.0, float(confidence_floor) - confidence) / max(
        float(confidence_floor),
        1e-6,
    )
    enabled = (
        message_mode in {"diag", "on"}
        and
        readiness_deficit <= 0.0
        and score_deficit <= 0.0
        and future_quality_deficit <= 0.0
        and subset_deficit <= 0.0
        and leaveout_deficit <= 0.0
        and confidence_deficit <= 0.0
    )
    blocker = "enabled"
    if not enabled:
        blocker = max(
            (
                ("message_mode", 0.0 if message_mode in {"diag", "on"} else 1.0),
                ("readiness_score", readiness_deficit),
                ("shadow_score", score_deficit),
                ("future_probe_quality", future_quality_deficit),
                ("subset_stability", subset_deficit),
                ("leaveout_stability", leaveout_deficit),
                ("confidence", confidence_deficit),
            ),
            key=lambda item: (float(item[1]), item[0]),
        )[0]
    scale_cap = 0.0
    if enabled:
        scale_cap = min(
            DEFAULT_SHADOW_SCALE_CAP,
            0.45 * confidence,
            0.40 * shadow_score,
        )
    return {
        "enabled": bool(enabled),
        "blocker": str(blocker),
        "message_mode": str(message_mode),
        "readiness_score": float(readiness_score),
        "shadow_score": float(shadow_score),
        "future_probe_quality": float(future_probe_quality),
        "subset_stability": float(subset_stability),
        "leaveout_stability": float(leaveout_stability),
        "confidence": float(confidence),
        "scale_cap": float(np.clip(scale_cap, 0.0, DEFAULT_SHADOW_SCALE_CAP)),
    }


def shadow_env_expression_enabled(
    *,
    env_expression: EnvExpression,
    readiness_score_floor: float = DEFAULT_SHADOW_READINESS_SCORE_FLOOR,
    shadow_score_floor: float = DEFAULT_SHADOW_SCORE_FLOOR,
    future_probe_quality_floor: float = DEFAULT_SHADOW_FUTURE_PROBE_QUALITY_FLOOR,
    subset_stability_floor: float = DEFAULT_SHADOW_SUBSET_STABILITY_FLOOR,
    leaveout_stability_floor: float = DEFAULT_SHADOW_LEAVEOUT_STABILITY_FLOOR,
    confidence_floor: float = DEFAULT_SHADOW_CONFIDENCE_FLOOR,
) -> bool:
    """Return whether the diagnostic shadow arm may expose the env expression."""
    diagnostics = shadow_env_expression_diagnostics(
        env_expression=env_expression,
        readiness_score_floor=readiness_score_floor,
        shadow_score_floor=shadow_score_floor,
        future_probe_quality_floor=future_probe_quality_floor,
        subset_stability_floor=subset_stability_floor,
        leaveout_stability_floor=leaveout_stability_floor,
        confidence_floor=confidence_floor,
    )
    return bool(diagnostics["enabled"])


def compute_env_expression_confidence(
    *,
    uncertainty_scalar: float,
    uncertainty_probe_threshold: float = DEFAULT_EXPRESSION_UNCERTAINTY_THRESHOLD,
    future_probe_error: float | None = None,
    heldout_family_future_error: float | None = None,
    support_size_matched_future_error: float | None = None,
    support_diversity_ratio: float | None = None,
    support_count: int | None = None,
    posterior_entropy: float | None = None,
    online_subset_stability: float | None = None,
    online_geometry_complete: bool | None = None,
    gap_ratio: float | None = None,
    split_retrieval_margin_deficit: float | None = None,
    split_latent_disagreement: float | None = None,
    leaveout_shift: float | None = None,
    leaveout_param_std_mean: float | None = None,
    readiness_score: float | None = None,
    utility_forecast: float | None = None,
) -> float:
    """Estimate how much the controller should trust one env expression."""
    threshold = max(float(uncertainty_probe_threshold), 1e-6)
    uncertainty_excess = max(0.0, float(uncertainty_scalar) - threshold)
    uncertainty_scale = 1.0 / (1.0 + uncertainty_excess / threshold)
    readiness_components = compute_env_expression_readiness_components(
        future_probe_error=future_probe_error,
        heldout_family_future_error=heldout_family_future_error,
        support_size_matched_future_error=support_size_matched_future_error,
        support_count=support_count,
        support_diversity_ratio=support_diversity_ratio,
        posterior_entropy=posterior_entropy,
        online_subset_stability=online_subset_stability,
        online_geometry_complete=online_geometry_complete,
        gap_ratio=gap_ratio,
        split_retrieval_margin_deficit=split_retrieval_margin_deficit,
        split_latent_disagreement=split_latent_disagreement,
        leaveout_shift=leaveout_shift,
        leaveout_param_std_mean=leaveout_param_std_mean,
    )
    if readiness_score is None:
        readiness_score = float(min(readiness_components.values(), default=0.0))
    if utility_forecast is None:
        utility_forecast = compute_env_expression_utility_forecast(
            readiness_components=readiness_components,
        )
    confidence = float(np.clip(uncertainty_scale, 0.05, 1.0))
    confidence *= 0.35 + 0.35 * float(readiness_score) + 0.30 * float(utility_forecast)
    confidence *= 0.45 + 0.55 * float(readiness_components.get("support_diversity", 1.0))
    if support_count is not None:
        support_scale = float(np.clip(0.40 + 0.18 * float(support_count), 0.40, 1.0))
        confidence *= support_scale
    return float(np.clip(confidence, 0.05, 1.0))


def env_expression_is_ready(
    *,
    confidence: float,
    uncertainty_scalar: float,
    future_probe_error: float | None = None,
    heldout_family_future_error: float | None = None,
    support_size_matched_future_error: float | None = None,
    support_count: int | None = None,
    support_diversity_ratio: float | None = None,
    posterior_entropy: float | None = None,
    online_subset_stability: float | None = None,
    online_geometry_complete: bool | None = None,
    gap_ratio: float | None = None,
    split_retrieval_margin_deficit: float | None = None,
    split_latent_disagreement: float | None = None,
    leaveout_shift: float | None = None,
    leaveout_param_std_mean: float | None = None,
    readiness_score: float | None = None,
    confidence_floor: float = DEFAULT_READY_CONFIDENCE_FLOOR,
    uncertainty_probe_threshold: float = DEFAULT_EXPRESSION_UNCERTAINTY_THRESHOLD,
) -> bool:
    """Return whether the current env expression looks strong enough to hand off."""
    if readiness_score is None:
        readiness_score = compute_env_expression_readiness_score(
            future_probe_error=future_probe_error,
            heldout_family_future_error=heldout_family_future_error,
            support_size_matched_future_error=support_size_matched_future_error,
            support_count=support_count,
            support_diversity_ratio=support_diversity_ratio,
            posterior_entropy=posterior_entropy,
            online_subset_stability=online_subset_stability,
            online_geometry_complete=online_geometry_complete,
            gap_ratio=gap_ratio,
            split_retrieval_margin_deficit=split_retrieval_margin_deficit,
            split_latent_disagreement=split_latent_disagreement,
            leaveout_shift=leaveout_shift,
            leaveout_param_std_mean=leaveout_param_std_mean,
        )
    enough_support = support_count is None or int(support_count) >= DEFAULT_READY_SUPPORT_COUNT
    return bool(
        enough_support
        and float(confidence) >= float(confidence_floor)
        and float(readiness_score) >= DEFAULT_READINESS_SCORE_FLOOR
    )


def build_env_expression(
    *,
    predictive_belief,
    metric_belief,
    uncertainty,
    raw_expression: np.ndarray,
    bits_per_dim: int = 0,
    use_residual_sketch: bool = False,
    uncertainty_probe_threshold: float = DEFAULT_EXPRESSION_UNCERTAINTY_THRESHOLD,
    quantize_vector_fn=None,
) -> EnvExpression:
    """Convert crawler-side diagnostics into one controller-facing env expression."""
    if quantize_vector_fn is None:
        compressed_vector = sanitize_numpy(np.asarray(raw_expression, dtype=np.float32).reshape(-1))
        residual = None
    else:
        compressed_vector, residual = quantize_vector_fn(
            raw_expression,
            bits_per_dim=bits_per_dim,
            use_residual_sketch=use_residual_sketch,
        )
    vector = sanitize_numpy(np.asarray(compressed_vector, dtype=np.float32).reshape(-1))
    if residual is not None:
        vector = sanitize_numpy(vector + sanitize_numpy(np.asarray(residual, dtype=np.float32).reshape(-1)))
    posterior_entropy = float(
        predictive_belief.metadata.get("mechanics_posterior_entropy", 0.0)
    )
    full_future_prediction_error = float(
        predictive_belief.metadata.get(
            "full_future_prediction_error",
            predictive_belief.future_probe_error,
        )
    )
    heldout_family_future_error = float(
        predictive_belief.metadata.get(
            "heldout_family_future_error",
            predictive_belief.future_probe_error,
        )
    )
    support_size_matched_future_error = float(
        predictive_belief.metadata.get(
            "support_size_matched_future_error",
            heldout_family_future_error,
        )
    )
    online_subset_stability = float(
        predictive_belief.metadata.get("online_subset_stability", 0.0)
    )
    online_geometry_complete = bool(
        predictive_belief.metadata.get("online_geometry_complete", False)
    )
    online_split_latent_disagreement = float(
        predictive_belief.metadata.get("online_split_latent_disagreement", 0.0)
    )
    online_split_retrieval_margin_deficit = float(
        predictive_belief.metadata.get("online_split_retrieval_margin_deficit", 0.0)
    )
    online_leaveout_shift = float(
        predictive_belief.metadata.get("online_leaveout_shift", 0.0)
    )
    online_observed_family_count = int(
        predictive_belief.metadata.get("online_observed_family_count", 0)
    )
    online_offline_gap = float(
        predictive_belief.metadata.get("online_offline_gap", 0.0)
    )
    fair_handoff_probe_families = tuple(
        str(family)
        for family in predictive_belief.metadata.get("fair_handoff_probe_families", ())
        if str(family)
    )
    teacher_action_agreement = float(
        predictive_belief.metadata.get("teacher_action_agreement", 0.0)
    )
    split_retrieval_margin_deficit = float(
        metric_belief.metadata.get("split_retrieval_margin_deficit", 0.0)
    )
    split_latent_disagreement = float(
        predictive_belief.metadata.get("split_latent_disagreement", 0.0)
    )
    leaveout_shift = float(
        predictive_belief.metadata.get("leaveout_shift", 0.0)
    )
    leaveout_param_std_mean = float(
        predictive_belief.metadata.get("leaveout_param_std_mean", 0.0)
    )
    geometry_scale = compute_solver_geometry_scale(
        gap_ratio=float(metric_belief.gap_ratio),
        split_retrieval_margin_deficit=split_retrieval_margin_deficit,
        split_latent_disagreement=split_latent_disagreement,
    )
    readiness_components = compute_env_expression_readiness_components(
        future_probe_error=float(predictive_belief.future_probe_error),
        heldout_family_future_error=heldout_family_future_error,
        support_size_matched_future_error=support_size_matched_future_error,
        support_count=int(predictive_belief.support_count),
        support_diversity_ratio=float(predictive_belief.support_diversity_ratio),
        posterior_entropy=posterior_entropy,
        online_subset_stability=online_subset_stability,
        online_geometry_complete=online_geometry_complete,
        gap_ratio=float(metric_belief.gap_ratio),
        split_retrieval_margin_deficit=split_retrieval_margin_deficit,
        split_latent_disagreement=split_latent_disagreement,
        leaveout_shift=leaveout_shift,
        leaveout_param_std_mean=leaveout_param_std_mean,
    )
    readiness_score = float(min(readiness_components.values(), default=0.0))
    readiness_reason = env_expression_readiness_reason(readiness_components)
    utility_forecast = compute_env_expression_utility_forecast(
        readiness_components=readiness_components,
    )
    confidence = compute_env_expression_confidence(
        uncertainty_scalar=float(uncertainty.scalar),
        uncertainty_probe_threshold=uncertainty_probe_threshold,
        future_probe_error=float(predictive_belief.future_probe_error),
        heldout_family_future_error=heldout_family_future_error,
        support_size_matched_future_error=support_size_matched_future_error,
        support_diversity_ratio=float(predictive_belief.support_diversity_ratio),
        support_count=int(predictive_belief.support_count),
        posterior_entropy=posterior_entropy,
        online_subset_stability=online_subset_stability,
        online_geometry_complete=online_geometry_complete,
        gap_ratio=float(metric_belief.gap_ratio),
        split_retrieval_margin_deficit=split_retrieval_margin_deficit,
        split_latent_disagreement=split_latent_disagreement,
        leaveout_shift=leaveout_shift,
        leaveout_param_std_mean=leaveout_param_std_mean,
        readiness_score=readiness_score,
        utility_forecast=utility_forecast,
    )
    message_mode, message_blocker = compute_message_mode(
        support_count=int(predictive_belief.support_count),
        confidence=confidence,
        readiness_components=readiness_components,
        readiness_score=readiness_score,
    )
    ready = env_expression_is_ready(
        confidence=confidence,
        uncertainty_scalar=float(uncertainty.scalar),
        future_probe_error=float(predictive_belief.future_probe_error),
        heldout_family_future_error=heldout_family_future_error,
        support_size_matched_future_error=support_size_matched_future_error,
        support_count=int(predictive_belief.support_count),
        support_diversity_ratio=float(predictive_belief.support_diversity_ratio),
        posterior_entropy=posterior_entropy,
        online_subset_stability=online_subset_stability,
        online_geometry_complete=online_geometry_complete,
        gap_ratio=float(metric_belief.gap_ratio),
        split_retrieval_margin_deficit=split_retrieval_margin_deficit,
        split_latent_disagreement=split_latent_disagreement,
        leaveout_shift=leaveout_shift,
        leaveout_param_std_mean=leaveout_param_std_mean,
        readiness_score=readiness_score,
        uncertainty_probe_threshold=uncertainty_probe_threshold,
    )
    fair_policy_diagnostics = fair_env_expression_diagnostics(
        env_expression=EnvExpression(
            vector=vector,
            confidence=confidence,
            ready=bool(message_mode == "on" and ready),
            uncertainty_scalar=float(uncertainty.scalar),
            compressed=bool(bits_per_dim > 0),
            metadata={
                "future_probe_quality": float(readiness_components["future_probe_quality"]),
                "subset_stability": float(readiness_components["subset_stability"]),
                "online_subset_stability": float(online_subset_stability),
                "online_geometry_complete": bool(online_geometry_complete),
                "readiness_score": float(readiness_score),
                "online_offline_gap": float(online_offline_gap),
                "message_mode": str(message_mode),
                "message_blocker": str(message_blocker),
            },
        )
    )
    ready = bool(message_mode == "on" and ready)
    fair_policy_enabled = bool(ready and fair_policy_diagnostics["enabled"])
    return EnvExpression(
        vector=vector,
        confidence=confidence,
        ready=ready,
        uncertainty_scalar=float(uncertainty.scalar),
        compressed=bool(bits_per_dim > 0),
        metadata={
            "bits_per_dim": int(bits_per_dim),
            "residual_vector": None if residual is None else sanitize_numpy(np.asarray(residual, dtype=np.float32).reshape(-1)),
            "raw_expression_norm": float(np.linalg.norm(raw_expression)),
            "compressed_expression_norm": float(np.linalg.norm(vector)),
            "future_probe_error": float(predictive_belief.future_probe_error),
            "full_future_prediction_error": full_future_prediction_error,
            "heldout_family_future_error": heldout_family_future_error,
            "support_size_matched_future_error": support_size_matched_future_error,
            "support_size_matched_future_quality": _normalize_ascending(
                support_size_matched_future_error,
                good=DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_GOOD,
                bad=DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_BAD,
            ),
            "support_count": int(predictive_belief.support_count),
            "support_diversity_ratio": float(predictive_belief.support_diversity_ratio),
            "posterior_entropy": posterior_entropy,
            "predictive_reuse_error": float(predictive_belief.future_probe_error),
            "online_subset_stability": float(online_subset_stability),
            "online_geometry_complete": bool(online_geometry_complete),
            "online_split_latent_disagreement": float(online_split_latent_disagreement),
            "online_split_retrieval_margin_deficit": float(online_split_retrieval_margin_deficit),
            "online_leaveout_shift": float(online_leaveout_shift),
            "online_observed_family_count": int(online_observed_family_count),
            "online_offline_gap": online_offline_gap,
            "fair_handoff_probe_families": fair_handoff_probe_families,
            "teacher_action_agreement": teacher_action_agreement,
            "gap_ratio": float(metric_belief.gap_ratio),
            "split_retrieval_margin_deficit": split_retrieval_margin_deficit,
            "split_latent_disagreement": split_latent_disagreement,
            "future_probe_quality": float(readiness_components["future_probe_quality"]),
            "subset_stability": float(readiness_components["subset_stability"]),
            "leaveout_stability": float(readiness_components["leaveout_stability"]),
            "support_diversity": float(readiness_components["support_diversity"]),
            "leaveout_shift": leaveout_shift,
            "leaveout_param_std_mean": leaveout_param_std_mean,
            "readiness_score": readiness_score,
            "readiness_reason": readiness_reason,
            "utility_forecast": utility_forecast,
            "message_mode": str(message_mode),
            "message_blocker": str(message_blocker),
            "geometry_scale": geometry_scale,
            "fair_policy_enabled": bool(fair_policy_enabled),
            "fair_stop_ready": bool(fair_policy_enabled),
            "fair_stop_blocker": (
                str(fair_policy_diagnostics["blocker"])
                if not fair_policy_enabled
                else "enabled"
            ),
        },
    )


def solver_expression_reliability_kwargs(step_result) -> dict[str, float | int]:
    """Extract controller-trust diagnostics from one crawler step result."""
    env_expression = step_result.env_expression
    return {
        "expression_confidence": float(env_expression.confidence),
        "future_probe_error": float(
            env_expression.metadata.get(
                "future_probe_error",
                step_result.predictive_belief.future_probe_error,
            )
        ),
        "heldout_family_future_error": float(
            env_expression.metadata.get(
                "heldout_family_future_error",
                step_result.predictive_belief.metadata.get(
                    "heldout_family_future_error",
                    step_result.predictive_belief.future_probe_error,
                ),
            )
        ),
        "support_size_matched_future_quality": float(
            env_expression.metadata.get("support_size_matched_future_quality", 0.0)
        ),
        "support_diversity_ratio": float(
            env_expression.metadata.get(
                "support_diversity_ratio",
                step_result.predictive_belief.support_diversity_ratio,
            )
        ),
        "support_count": int(
            env_expression.metadata.get(
                "support_count",
                step_result.predictive_belief.support_count,
            )
        ),
        "posterior_entropy": float(
            env_expression.metadata.get(
                "posterior_entropy",
                step_result.predictive_belief.metadata.get(
                    "mechanics_posterior_entropy",
                    0.0,
                ),
            )
        ),
        "gap_ratio": float(
            env_expression.metadata.get("gap_ratio", step_result.metric_belief.gap_ratio)
        ),
        "split_retrieval_margin_deficit": float(
            env_expression.metadata.get(
                "split_retrieval_margin_deficit",
                step_result.metric_belief.metadata.get(
                    "split_retrieval_margin_deficit",
                    0.0,
                ),
            )
        ),
        "split_latent_disagreement": float(
            env_expression.metadata.get(
                "split_latent_disagreement",
                step_result.predictive_belief.metadata.get(
                    "split_latent_disagreement",
                    0.0,
                ),
            )
        ),
        "readiness_score": float(env_expression.metadata.get("readiness_score", 0.0)),
        "utility_forecast": float(env_expression.metadata.get("utility_forecast", 0.0)),
        "subset_stability": float(env_expression.metadata.get("subset_stability", 0.0)),
        "leaveout_stability": float(env_expression.metadata.get("leaveout_stability", 0.0)),
        "leaveout_shift": float(
            env_expression.metadata.get(
                "leaveout_shift",
                step_result.predictive_belief.metadata.get("leaveout_shift", 0.0),
            )
        ),
        "leaveout_param_std_mean": float(
            env_expression.metadata.get(
                "leaveout_param_std_mean",
                step_result.predictive_belief.metadata.get("leaveout_param_std_mean", 0.0),
            )
        ),
        "predictive_reuse_error": float(
            env_expression.metadata.get(
                "predictive_reuse_error",
                step_result.predictive_belief.future_probe_error,
            )
        ),
        "online_offline_gap": float(
            env_expression.metadata.get("online_offline_gap", 0.0)
        ),
        "online_subset_stability": float(
            env_expression.metadata.get("online_subset_stability", 0.0)
        ),
        "online_geometry_complete": bool(
            env_expression.metadata.get("online_geometry_complete", False)
        ),
        "message_mode": str(env_expression.metadata.get("message_mode", "off")),
        "message_blocker": str(env_expression.metadata.get("message_blocker", "unknown")),
        "fair_policy_enabled": bool(
            env_expression.metadata.get("fair_policy_enabled", False)
        ),
        "geometry_scale": float(
            env_expression.metadata.get(
                "geometry_scale",
                compute_solver_geometry_scale(
                    gap_ratio=step_result.metric_belief.gap_ratio,
                    split_retrieval_margin_deficit=step_result.metric_belief.metadata.get(
                        "split_retrieval_margin_deficit",
                        0.0,
                    ),
                    split_latent_disagreement=step_result.predictive_belief.metadata.get(
                        "split_latent_disagreement",
                        0.0,
                    ),
                ),
            )
        ),
    }


def solver_message_reliability_kwargs(step_result) -> dict[str, float | int]:
    """Compatibility alias for the older message-oriented name."""
    return solver_expression_reliability_kwargs(step_result)


def solver_message_warmup_episodes(total_episodes: int) -> int:
    """Return how long to ramp controller trust during PPO training."""
    total_episodes = max(int(total_episodes), 1)
    return max(8, min(24, total_episodes // 12))


def fair_env_expression_diagnostics(
    *,
    env_expression: EnvExpression,
    readiness_score_floor: float = DEFAULT_READINESS_SCORE_FLOOR,
    future_probe_quality_floor: float = DEFAULT_FAIR_POLICY_FUTURE_QUALITY_FLOOR,
    subset_stability_floor: float = DEFAULT_FAIR_POLICY_SUBSET_STABILITY_FLOOR,
    confidence_floor: float = DEFAULT_FAIR_POLICY_CONFIDENCE_FLOOR,
    online_offline_gap_ceiling: float = DEFAULT_FAIR_POLICY_ONLINE_OFFLINE_GAP_CEILING,
) -> dict[str, float | bool | str]:
    """Return the stricter fair-policy gate for stop/mute decisions."""
    metadata = dict(env_expression.metadata or {})
    message_mode = str(
        metadata.get("message_mode", "on" if bool(env_expression.ready) else "off")
    )
    raw_ready = bool(message_mode == "on")
    readiness_score = float(metadata.get("readiness_score", 0.0))
    future_probe_quality = float(metadata.get("future_probe_quality", 0.0))
    online_subset_stability = float(
        metadata.get(
            "online_subset_stability",
            metadata.get("subset_stability", 0.0),
        )
    )
    confidence = float(env_expression.confidence)
    online_offline_gap = float(metadata.get("online_offline_gap", 0.0))
    online_geometry_complete = bool(metadata.get("online_geometry_complete", False))
    readiness_deficit = max(0.0, float(readiness_score_floor) - readiness_score) / max(
        float(readiness_score_floor),
        1e-6,
    )
    future_quality_deficit = max(
        0.0,
        float(future_probe_quality_floor) - future_probe_quality,
    ) / max(float(future_probe_quality_floor), 1e-6)
    subset_deficit = max(
        0.0,
        float(subset_stability_floor) - online_subset_stability,
    ) / max(float(subset_stability_floor), 1e-6)
    confidence_deficit = max(0.0, float(confidence_floor) - confidence) / max(
        float(confidence_floor),
        1e-6,
    )
    gap_deficit = max(0.0, online_offline_gap - float(online_offline_gap_ceiling)) / max(
        float(online_offline_gap_ceiling),
        1e-6,
    )
    geometry_incomplete_deficit = 0.0 if online_geometry_complete else 1.0
    raw_ready_deficit = 0.0 if raw_ready else 1.0
    enabled = (
        raw_ready_deficit <= 0.0
        and
        readiness_deficit <= 0.0
        and future_quality_deficit <= 0.0
        and subset_deficit <= 0.0
        and confidence_deficit <= 0.0
        and gap_deficit <= 0.0
        and geometry_incomplete_deficit <= 0.0
    )
    blocker = "enabled"
    if not enabled:
        blocker = max(
            (
                ("message_mode", 0.0 if message_mode == "on" else 1.0),
                ("env_expression_ready", raw_ready_deficit),
                ("readiness_score", readiness_deficit),
                ("future_probe_quality", future_quality_deficit),
                ("online_subset_stability", subset_deficit),
                ("confidence", confidence_deficit),
                ("online_offline_gap", gap_deficit),
                ("online_geometry_complete", geometry_incomplete_deficit),
            ),
            key=lambda item: (float(item[1]), item[0]),
        )[0]
    return {
        "enabled": bool(enabled),
        "blocker": str(blocker),
        "message_mode": str(message_mode),
        "env_expression_ready": bool(raw_ready),
        "readiness_score": float(readiness_score),
        "future_probe_quality": float(future_probe_quality),
        "online_subset_stability": float(online_subset_stability),
        "confidence": float(confidence),
        "online_offline_gap": float(online_offline_gap),
        "online_geometry_complete": bool(online_geometry_complete),
    }


def fair_env_expression_enabled(
    *,
    env_expression: EnvExpression,
    controller_trust_floor: float = DEFAULT_CONTROLLER_TRUST_FLOOR,
) -> bool:
    """Return whether fair-mode control should be allowed to use this env expression."""
    metadata = dict(env_expression.metadata or {})
    message_mode = str(
        metadata.get("message_mode", "on" if bool(env_expression.ready) else "off")
    )
    if "fair_policy_enabled" in metadata:
        return (
            message_mode == "on"
            and bool(metadata.get("fair_policy_enabled", False))
            and float(env_expression.confidence) >= float(controller_trust_floor)
        )
    diagnostics = fair_env_expression_diagnostics(env_expression=env_expression)
    return message_mode == "on" and bool(diagnostics["enabled"]) and float(env_expression.confidence) >= float(
        controller_trust_floor
    )


def compute_solver_expression_scale(
    *,
    expression_confidence: float,
    current_episode: int,
    total_episodes: int,
    disable_env_expression: bool = False,
    expression_ready: bool | None = None,
    geometry_scale: float | None = None,
    readiness_score: float | None = None,
    utility_forecast: float | None = None,
    fair_policy_enabled: bool | None = None,
    online_subset_stability: float | None = None,
    ) -> float:
    """Warm the controller into trusting the env expression early enough to matter."""
    if disable_env_expression:
        return 0.0
    warmup_episodes = solver_message_warmup_episodes(total_episodes)
    progress = min(1.0, float(max(current_episode, 1)) / float(max(warmup_episodes, 1)))
    schedule_scale = 0.30 + 0.70 * progress
    trust_scale = float(np.clip(float(expression_confidence), 0.0, 1.0))
    if geometry_scale is not None:
        trust_scale *= float(np.clip(float(geometry_scale), 0.0, 1.0))
    if expression_ready is False:
        readiness_scale = float(np.clip(0.0 if readiness_score is None else readiness_score, 0.0, 1.0))
        utility_scale = float(np.clip(0.0 if utility_forecast is None else utility_forecast, 0.0, 1.0))
        conservative_cap = min(
            DEFAULT_NOT_READY_TRUST_CLAMP,
            0.15 + 0.35 * max(readiness_scale, utility_scale),
        )
        trust_scale = min(trust_scale, conservative_cap)
        trust_scale *= DEFAULT_NOT_READY_TRUST_MULTIPLIER + 0.15 * utility_scale
    scale = float(np.clip(trust_scale * schedule_scale, 0.0, 1.0))
    if bool(fair_policy_enabled):
        marginal_readiness = float(
            0.0 if readiness_score is None else readiness_score
        ) < DEFAULT_FAIR_POLICY_STRONG_READINESS_FLOOR
        marginal_subset = float(
            0.0 if online_subset_stability is None else online_subset_stability
        ) < DEFAULT_FAIR_POLICY_STRONG_SUBSET_STABILITY_FLOOR
        if marginal_readiness or marginal_subset:
            scale = min(scale, DEFAULT_FAIR_POLICY_SCALE_CAP)
    return float(np.clip(scale, 0.0, 1.0))


def compute_solver_message_scale(
    *,
    uncertainty_scalar: float,
    uncertainty_probe_threshold: float,
    current_episode: int,
    total_episodes: int,
    future_probe_error: float | None = None,
    support_diversity_ratio: float | None = None,
    support_count: int | None = None,
    posterior_entropy: float | None = None,
    gap_ratio: float | None = None,
    split_retrieval_margin_deficit: float | None = None,
    split_latent_disagreement: float | None = None,
    leaveout_shift: float | None = None,
    leaveout_param_std_mean: float | None = None,
    expression_confidence: float | None = None,
    expression_ready: bool | None = None,
    readiness_score: float | None = None,
    utility_forecast: float | None = None,
    fair_policy_enabled: bool | None = None,
    online_subset_stability: float | None = None,
    disable_belief_message: bool = False,
) -> float:
    """Compatibility wrapper for older tests and call sites."""
    if expression_confidence is None:
        expression_confidence = compute_env_expression_confidence(
            uncertainty_scalar=uncertainty_scalar,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            future_probe_error=future_probe_error,
            support_diversity_ratio=support_diversity_ratio,
            support_count=support_count,
            posterior_entropy=posterior_entropy,
            gap_ratio=gap_ratio,
            split_retrieval_margin_deficit=split_retrieval_margin_deficit,
            split_latent_disagreement=split_latent_disagreement,
            leaveout_shift=leaveout_shift,
            leaveout_param_std_mean=leaveout_param_std_mean,
            readiness_score=readiness_score,
            utility_forecast=utility_forecast,
        )
    return compute_solver_expression_scale(
        expression_confidence=float(expression_confidence),
        current_episode=current_episode,
        total_episodes=total_episodes,
        expression_ready=expression_ready,
        readiness_score=readiness_score,
        utility_forecast=utility_forecast,
        fair_policy_enabled=fair_policy_enabled,
        online_subset_stability=online_subset_stability,
        geometry_scale=compute_solver_geometry_scale(
            gap_ratio=gap_ratio,
            split_retrieval_margin_deficit=split_retrieval_margin_deficit,
            split_latent_disagreement=split_latent_disagreement,
        ),
        disable_env_expression=disable_belief_message,
    )


def compute_strict_fair_diagnostic_scale(
    *,
    env_expression: EnvExpression,
    base_scale: float,
) -> float:
    """Soften strict-fair DIAG handoffs without forcing one fixed floor."""
    metadata = dict(env_expression.metadata or {})
    future_quality = float(np.clip(metadata.get("future_probe_quality", 0.0), 0.0, 1.0))
    readiness_score = float(np.clip(metadata.get("readiness_score", 0.0), 0.0, 1.0))
    utility_forecast = float(np.clip(metadata.get("utility_forecast", 0.0), 0.0, 1.0))
    subset_stability = float(
        np.clip(
            metadata.get(
                "online_subset_stability",
                metadata.get("subset_stability", 0.0),
            ),
            0.0,
            1.0,
        )
    )
    confidence = float(np.clip(env_expression.confidence, 0.0, 1.0))
    online_offline_gap = max(0.0, float(metadata.get("online_offline_gap", 0.0)))
    geometry_complete = bool(metadata.get("online_geometry_complete", False))

    gap_overage = max(
        0.0,
        online_offline_gap - float(DEFAULT_FAIR_POLICY_ONLINE_OFFLINE_GAP_CEILING),
    )
    gap_scale = 1.0 - min(
        1.0,
        gap_overage / max(float(DEFAULT_FAIR_POLICY_ONLINE_OFFLINE_GAP_CEILING), 1e-6),
    )
    diagnostic_strength = (
        0.30 * future_quality
        + 0.25 * readiness_score
        + 0.20 * utility_forecast
        + 0.15 * subset_stability
        + 0.10 * confidence
    )
    if not geometry_complete:
        diagnostic_strength *= 0.75
    diagnostic_strength *= max(0.25, gap_scale)
    diagnostic_strength = float(np.clip(diagnostic_strength, 0.0, 1.0))

    diag_floor = float(
        np.interp(
            diagnostic_strength,
            [0.0, 1.0],
            [0.04, DEFAULT_DIAGNOSTIC_SCALE_FLOOR],
        )
    )
    diag_cap = float(
        np.interp(
            diagnostic_strength,
            [0.0, 1.0],
            [0.10, DEFAULT_DIAGNOSTIC_SCALE_CAP],
        )
    )
    return float(np.clip(max(float(base_scale), diag_floor), 0.0, diag_cap))


def compute_solver_training_dropout_prob(
    *,
    current_episode: int,
    total_episodes: int,
    message_scale: float,
    base_dropout_prob: float,
) -> float:
    """Return the episode-level chance of muting the env expression during training."""
    if base_dropout_prob <= 0.0:
        return 0.0
    warmup_episodes = solver_message_warmup_episodes(total_episodes)
    progress = min(1.0, float(max(current_episode, 1)) / float(max(warmup_episodes, 1)))
    early_training_boost = 0.10 * (1.0 - progress)
    weak_expression_boost = 0.30 * max(0.0, 0.50 - float(message_scale))
    return float(
        np.clip(float(base_dropout_prob) + early_training_boost + weak_expression_boost, 0.0, 0.60)
    )


def sample_solver_training_message_keep_scale(
    *,
    rng: np.random.Generator,
    current_episode: int,
    total_episodes: int,
    message_scale: float,
    base_dropout_prob: float,
) -> float:
    """Sample one per-episode keep scale for the env expression during training."""
    dropout_prob = compute_solver_training_dropout_prob(
        current_episode=current_episode,
        total_episodes=total_episodes,
        message_scale=message_scale,
        base_dropout_prob=base_dropout_prob,
    )
    if dropout_prob <= 0.0:
        return 1.0
    binary_dropout_cutoff = max(1, int(total_episodes) // 4)
    if int(current_episode) <= binary_dropout_cutoff:
        return 0.0 if float(rng.random()) < dropout_prob else 1.0
    jitter_floor = 1.0 - 0.5 * float(np.clip(dropout_prob / 0.60, 0.0, 1.0))
    jitter_floor = max(DEFAULT_LATE_TRAINING_KEEP_SCALE_FLOOR, jitter_floor)
    return float(rng.uniform(jitter_floor, 1.0))


def build_solver_episode_expression(
    *,
    env_expression: EnvExpression,
    current_episode: int,
    total_episodes: int,
    disable_env_expression: bool = False,
    strict_fair_mode: bool = False,
    shadow_expression_mode: bool = False,
    controller_trust_floor: float = DEFAULT_CONTROLLER_TRUST_FLOOR,
    force_message_mode: str | None = None,
    forced_expression_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Build one solver-facing env-expression vector for an episode."""
    if strict_fair_mode and shadow_expression_mode:
        raise ValueError("Strict fair mode and shadow expression mode are mutually exclusive.")
    if force_message_mode not in {None, "off", "diag", "on"}:
        raise ValueError(f"Unsupported force_message_mode: {force_message_mode}")
    message_mode = str(
        (env_expression.metadata or {}).get(
            "message_mode",
            "on" if bool(env_expression.ready) else "off",
        )
    )
    effective_message_mode = (
        message_mode if force_message_mode is None else str(force_message_mode)
    )
    fair_expression_allowed = fair_env_expression_enabled(
        env_expression=env_expression,
        controller_trust_floor=controller_trust_floor,
    )
    shadow_diagnostics = shadow_env_expression_diagnostics(
        env_expression=env_expression,
    ) if shadow_expression_mode else None
    shadow_expression_allowed = (
        True
        if shadow_diagnostics is None
        else bool(shadow_diagnostics["enabled"])
    )
    if (
        (force_message_mode == "off")
        or (
            force_message_mode is None
            and disable_env_expression
        )
        or (strict_fair_mode and effective_message_mode == "off")
        or (shadow_expression_mode and not shadow_expression_allowed)
    ):
        muted_expression = solver_expression_input_from_env_expression(
            env_expression,
            confidence_scale=0.0,
        )
        muted_expression = apply_solver_expression_keep_scale(
            muted_expression,
            keep_scale=0.0,
        )
        return muted_expression, 0.0
    expression_scale = compute_solver_expression_scale(
        expression_confidence=float(env_expression.confidence),
        current_episode=current_episode,
        total_episodes=total_episodes,
        expression_ready=bool(env_expression.ready),
        geometry_scale=float(env_expression.metadata.get("geometry_scale", 1.0)),
        readiness_score=float(env_expression.metadata.get("readiness_score", 0.0)),
        utility_forecast=float(env_expression.metadata.get("utility_forecast", 0.0)),
        fair_policy_enabled=bool(env_expression.metadata.get("fair_policy_enabled", False)),
        online_subset_stability=float(
            env_expression.metadata.get("online_subset_stability", 0.0)
        ),
        disable_env_expression=disable_env_expression,
    )
    if shadow_diagnostics is not None:
        expression_scale = min(
            float(expression_scale),
            float(shadow_diagnostics["scale_cap"]),
        )
    if forced_expression_scale is not None:
        expression_scale = float(np.clip(float(forced_expression_scale), 0.0, 1.0))
    elif strict_fair_mode and effective_message_mode == "diag":
        expression_scale = compute_strict_fair_diagnostic_scale(
            env_expression=env_expression,
            base_scale=float(expression_scale),
        )
    solver_expression = solver_expression_input_from_env_expression(
        env_expression,
        confidence_scale=expression_scale,
    )
    return solver_expression, expression_scale


def build_solver_episode_belief(
    *,
    belief_message: np.ndarray,
    uncertainty_scalar: float,
    uncertainty_probe_threshold: float,
    current_episode: int,
    total_episodes: int,
    future_probe_error: float | None = None,
    support_diversity_ratio: float | None = None,
    support_count: int | None = None,
    posterior_entropy: float | None = None,
    gap_ratio: float | None = None,
    split_retrieval_margin_deficit: float | None = None,
    split_latent_disagreement: float | None = None,
    disable_belief_message: bool = False,
) -> tuple[np.ndarray, float]:
    """Compatibility wrapper around the env-expression controller contract."""
    confidence = compute_env_expression_confidence(
        uncertainty_scalar=uncertainty_scalar,
        uncertainty_probe_threshold=uncertainty_probe_threshold,
        future_probe_error=future_probe_error,
        support_diversity_ratio=support_diversity_ratio,
        support_count=support_count,
        posterior_entropy=posterior_entropy,
        gap_ratio=gap_ratio,
        split_retrieval_margin_deficit=split_retrieval_margin_deficit,
        split_latent_disagreement=split_latent_disagreement,
    )
    env_expression = EnvExpression(
        vector=sanitize_numpy(np.asarray(belief_message, dtype=np.float32).reshape(-1)),
        confidence=confidence,
        ready=env_expression_is_ready(
            confidence=confidence,
            uncertainty_scalar=uncertainty_scalar,
            future_probe_error=future_probe_error,
            support_count=support_count,
            support_diversity_ratio=support_diversity_ratio,
            posterior_entropy=posterior_entropy,
            gap_ratio=gap_ratio,
            split_retrieval_margin_deficit=split_retrieval_margin_deficit,
            split_latent_disagreement=split_latent_disagreement,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
        ),
        uncertainty_scalar=float(uncertainty_scalar),
        compressed=False,
        metadata={
            "future_probe_error": float(future_probe_error or 0.0),
            "support_count": int(support_count or 0),
            "support_diversity_ratio": float(support_diversity_ratio or 0.0),
            "posterior_entropy": float(posterior_entropy or 0.0),
            "gap_ratio": float(gap_ratio or 0.0),
            "split_retrieval_margin_deficit": float(split_retrieval_margin_deficit or 0.0),
            "split_latent_disagreement": float(split_latent_disagreement or 0.0),
            "geometry_scale": compute_solver_geometry_scale(
                gap_ratio=gap_ratio,
                split_retrieval_margin_deficit=split_retrieval_margin_deficit,
                split_latent_disagreement=split_latent_disagreement,
            ),
        },
    )
    return build_solver_episode_expression(
        env_expression=env_expression,
        current_episode=current_episode,
        total_episodes=total_episodes,
        disable_env_expression=disable_belief_message,
    )
