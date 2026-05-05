"""Runtime scaling, dropout, and episode message construction."""

import numpy as np

from .....crawler.types import EnvExpression
from ....core import sanitize_numpy
from .constants import (
    DEFAULT_CONTROLLER_TRUST_FLOOR,
    DEFAULT_DIAGNOSTIC_SCALE_CAP,
    DEFAULT_DIAGNOSTIC_SCALE_FLOOR,
    DEFAULT_EXPRESSION_UNCERTAINTY_THRESHOLD,
    DEFAULT_FAIR_POLICY_CONFIDENCE_FLOOR,
    DEFAULT_FAIR_POLICY_FUTURE_QUALITY_FLOOR,
    DEFAULT_FAIR_POLICY_LEAVEOUT_STABILITY_FLOOR,
    DEFAULT_FAIR_POLICY_ONLINE_OFFLINE_GAP_CEILING,
    DEFAULT_FAIR_POLICY_SCALE_CAP,
    DEFAULT_FAIR_POLICY_STRONG_READINESS_FLOOR,
    DEFAULT_FAIR_POLICY_STRONG_SUBSET_STABILITY_FLOOR,
    DEFAULT_FORCED_EVAL_EXPRESSION_SCALE,
    DEFAULT_LATE_TRAINING_KEEP_SCALE_FLOOR,
    DEFAULT_NOT_READY_TRUST_CLAMP,
    DEFAULT_NOT_READY_TRUST_MULTIPLIER,
)
from .input import apply_solver_expression_keep_scale, solver_expression_input_from_env_expression
from .readiness import (
    compute_env_expression_confidence,
    compute_solver_geometry_scale,
    env_expression_is_ready,
    shadow_env_expression_diagnostics,
)


def solver_message_warmup_episodes(total_episodes: int) -> int:
    """Return how long to ramp controller trust during PPO training."""
    total_episodes = max(int(total_episodes), 1)
    return max(8, min(24, total_episodes // 12))


def fair_env_expression_diagnostics(
    *,
    env_expression: EnvExpression,
    readiness_score_floor: float = DEFAULT_FAIR_POLICY_STRONG_READINESS_FLOOR,
    future_probe_quality_floor: float = DEFAULT_FAIR_POLICY_FUTURE_QUALITY_FLOOR,
    subset_stability_floor: float = DEFAULT_FAIR_POLICY_STRONG_SUBSET_STABILITY_FLOOR,
    leaveout_stability_floor: float = DEFAULT_FAIR_POLICY_LEAVEOUT_STABILITY_FLOOR,
    confidence_floor: float = DEFAULT_FAIR_POLICY_CONFIDENCE_FLOOR,
    online_offline_gap_ceiling: float = DEFAULT_FAIR_POLICY_ONLINE_OFFLINE_GAP_CEILING,
) -> dict[str, float | bool | str]:
    """Return the stricter fair-policy gate for stop/mute decisions."""
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
    raw_ready = bool(message_mode == "on")
    readiness_score = float(metadata.get("readiness_score", 0.0))
    future_probe_quality = float(metadata.get("future_probe_quality", 0.0))
    online_subset_stability = float(
        metadata.get(
            "online_subset_stability",
            metadata.get("subset_stability", 0.0),
        )
    )
    leaveout_stability = float(
        metadata.get("leaveout_stability", readiness_score)
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
    leaveout_deficit = max(
        0.0,
        float(leaveout_stability_floor) - leaveout_stability,
    ) / max(float(leaveout_stability_floor), 1e-6)
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
        and leaveout_deficit <= 0.0
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
                (subset_blocker, subset_deficit),
                ("leaveout_stability", leaveout_deficit),
                ("confidence", confidence_deficit),
                ("online_offline_gap", gap_deficit),
                (
                    "particle_subset_stability"
                    if belief_source == "sysid"
                    else "online_geometry_complete",
                    geometry_incomplete_deficit,
                ),
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
        "subset_blocker_name": subset_blocker,
        "leaveout_stability": float(leaveout_stability),
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
        or (
            strict_fair_mode
            and effective_message_mode == "off"
            and forced_expression_scale is None
        )
        or (
            strict_fair_mode
            and effective_message_mode == "on"
            and not fair_expression_allowed
            and forced_expression_scale is None
        )
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
