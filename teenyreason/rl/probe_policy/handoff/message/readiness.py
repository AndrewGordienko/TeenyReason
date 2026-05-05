"""Readiness, confidence, and diagnostic gates for env-expression handoff."""

import numpy as np

from .....crawler.types import EnvExpression
from .constants import (
    DEFAULT_CONTROLLER_TRUST_FLOOR,
    DEFAULT_DIAGNOSTIC_CONFIDENCE_FLOOR,
    DEFAULT_DIAGNOSTIC_FUTURE_QUALITY_FLOOR,
    DEFAULT_DIAGNOSTIC_SCALE_CAP,
    DEFAULT_DIAGNOSTIC_SCALE_FLOOR,
    DEFAULT_DIAGNOSTIC_SUBSET_STABILITY_FLOOR,
    DEFAULT_DIAGNOSTIC_SUBSET_STABILITY_HARD_FLOOR,
    DEFAULT_DIAGNOSTIC_SUPPORT_COUNT,
    DEFAULT_EXPRESSION_UNCERTAINTY_THRESHOLD,
    DEFAULT_FAIR_POLICY_CONFIDENCE_FLOOR,
    DEFAULT_FAIR_POLICY_FUTURE_QUALITY_FLOOR,
    DEFAULT_FAIR_POLICY_ONLINE_OFFLINE_GAP_CEILING,
    DEFAULT_FAIR_POLICY_SCALE_CAP,
    DEFAULT_FAIR_POLICY_STRONG_READINESS_FLOOR,
    DEFAULT_FAIR_POLICY_STRONG_SUBSET_STABILITY_FLOOR,
    DEFAULT_LATE_TRAINING_KEEP_SCALE_FLOOR,
    DEFAULT_NOT_READY_TRUST_CLAMP,
    DEFAULT_NOT_READY_TRUST_MULTIPLIER,
    DEFAULT_READINESS_SCORE_FLOOR,
    DEFAULT_READY_CONFIDENCE_FLOOR,
    DEFAULT_READY_FULL_FUTURE_ERROR_BAD,
    DEFAULT_READY_FULL_FUTURE_ERROR_GOOD,
    DEFAULT_READY_HELDOUT_FUTURE_ERROR_BAD,
    DEFAULT_READY_HELDOUT_FUTURE_ERROR_GOOD,
    DEFAULT_READY_POSTERIOR_ENTROPY_BAD,
    DEFAULT_READY_POSTERIOR_ENTROPY_CEILING,
    DEFAULT_READY_POSTERIOR_ENTROPY_GOOD,
    DEFAULT_READY_SUPPORT_COUNT,
    DEFAULT_READY_SUPPORT_DIVERSITY_FLOOR,
    DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_BAD,
    DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_GOOD,
    DEFAULT_SHADOW_CONFIDENCE_FLOOR,
    DEFAULT_SHADOW_FUTURE_PROBE_QUALITY_FLOOR,
    DEFAULT_SHADOW_LEAVEOUT_STABILITY_FLOOR,
    DEFAULT_SHADOW_READINESS_SCORE_FLOOR,
    DEFAULT_SHADOW_SCALE_CAP,
    DEFAULT_SHADOW_SCORE_FLOOR,
    DEFAULT_SHADOW_SUBSET_STABILITY_FLOOR,
)


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
    online_leaveout_stability: float | None = None,
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
    if online_leaveout_stability is not None and bool(online_geometry_complete):
        leaveout_stability = float(np.clip(float(online_leaveout_stability), 0.02, 1.0))
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
        if subset_stability >= DEFAULT_DIAGNOSTIC_SUBSET_STABILITY_HARD_FLOOR:
            return "diag", "subset_stability"
        return "off", "subset_stability"
    if support_count < DEFAULT_READY_SUPPORT_COUNT:
        if (
            support_count >= DEFAULT_DIAGNOSTIC_SUPPORT_COUNT
            and float(confidence) >= DEFAULT_FAIR_POLICY_CONFIDENCE_FLOOR
            and float(readiness_score) >= DEFAULT_FAIR_POLICY_STRONG_READINESS_FLOOR
            and future_quality >= DEFAULT_FAIR_POLICY_FUTURE_QUALITY_FLOOR
            and subset_stability >= DEFAULT_FAIR_POLICY_STRONG_SUBSET_STABILITY_FLOOR
        ):
            return "on", "enabled"
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
    online_leaveout_stability: float | None = None,
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
        online_leaveout_stability=online_leaveout_stability,
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
    belief_source = str(metadata.get("belief_source", "learned"))
    subset_blocker = (
        "particle_subset_stability"
        if belief_source == "sysid"
        else "learned_latent_geometry"
    )
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
    online_leaveout_stability: float | None = None,
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
        online_leaveout_stability=online_leaveout_stability,
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
    online_leaveout_stability: float | None = None,
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
            online_leaveout_stability=online_leaveout_stability,
            online_geometry_complete=online_geometry_complete,
            gap_ratio=gap_ratio,
            split_retrieval_margin_deficit=split_retrieval_margin_deficit,
            split_latent_disagreement=split_latent_disagreement,
            leaveout_shift=leaveout_shift,
            leaveout_param_std_mean=leaveout_param_std_mean,
        )
    enough_support = support_count is None or int(support_count) >= DEFAULT_READY_SUPPORT_COUNT
    if not enough_support and support_count is not None:
        enough_support = (
            int(support_count) >= DEFAULT_DIAGNOSTIC_SUPPORT_COUNT
            and float(readiness_score) >= DEFAULT_FAIR_POLICY_STRONG_READINESS_FLOOR
        )
    return bool(
        enough_support
        and float(confidence) >= float(confidence_floor)
        and float(readiness_score) >= DEFAULT_READINESS_SCORE_FLOOR
    )
