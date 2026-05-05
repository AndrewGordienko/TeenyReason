"""Build controller-facing environment expressions from crawler beliefs."""

import numpy as np

from .....crawler.types import EnvExpression
from ....core import sanitize_numpy
from .constants import (
    DEFAULT_EXPRESSION_UNCERTAINTY_THRESHOLD,
    DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_BAD,
    DEFAULT_READY_SUPPORT_MATCHED_FUTURE_ERROR_GOOD,
)
from .readiness import (
    _normalize_ascending,
    compute_env_expression_confidence,
    compute_env_expression_readiness_components,
    compute_env_expression_utility_forecast,
    compute_message_mode,
    compute_solver_geometry_scale,
    env_expression_is_ready,
    env_expression_readiness_reason,
)
from .runtime import fair_env_expression_diagnostics


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
    belief_source = str(
        predictive_belief.metadata.get(
            "belief_source",
            "sysid"
            if str(predictive_belief.metadata.get("belief_mode", "")) == "particle_sysid"
            else "learned",
        )
    )
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
    online_leaveout_stability = predictive_belief.metadata.get(
        "online_leaveout_stability",
        None,
    )
    if online_leaveout_stability is not None:
        online_leaveout_stability = float(online_leaveout_stability)
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
    if belief_source == "sysid":
        particle_subset_stability = float(
            predictive_belief.metadata.get(
                "particle_subset_stability",
                predictive_belief.metadata.get("online_subset_stability", 0.0),
            )
        )
        geometry_scale = max(
            float(geometry_scale),
            float(np.clip(particle_subset_stability, 0.0, 1.0)),
        )
    readiness_components = compute_env_expression_readiness_components(
        future_probe_error=float(predictive_belief.future_probe_error),
        heldout_family_future_error=heldout_family_future_error,
        support_size_matched_future_error=support_size_matched_future_error,
        support_count=int(predictive_belief.support_count),
        support_diversity_ratio=float(predictive_belief.support_diversity_ratio),
        posterior_entropy=posterior_entropy,
        online_subset_stability=online_subset_stability,
        online_leaveout_stability=online_leaveout_stability,
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
        online_leaveout_stability=online_leaveout_stability,
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
    if message_blocker == "subset_stability":
        message_blocker = (
            "particle_subset_stability"
            if belief_source == "sysid"
            else "learned_latent_geometry"
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
        online_leaveout_stability=online_leaveout_stability,
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
                "belief_source": belief_source,
                "future_probe_quality": float(readiness_components["future_probe_quality"]),
                "subset_stability": float(readiness_components["subset_stability"]),
                "leaveout_stability": float(readiness_components["leaveout_stability"]),
                "online_subset_stability": float(online_subset_stability),
                "online_leaveout_stability": float(
                    readiness_components["leaveout_stability"]
                    if online_leaveout_stability is None
                    else online_leaveout_stability
                ),
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
            "belief_source": belief_source,
            "solver_message_source": str(
                predictive_belief.metadata.get("solver_message_source", belief_source)
            ),
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
            "online_leaveout_stability": float(
                readiness_components["leaveout_stability"]
                if online_leaveout_stability is None
                else online_leaveout_stability
            ),
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


def solver_expression_reliability_kwargs(step_result) -> dict[str, float | int | str]:
    """Extract controller-trust diagnostics from one crawler step result."""
    env_expression = step_result.env_expression
    return {
        "belief_source": str(env_expression.metadata.get("belief_source", "learned")),
        "solver_message_source": str(
            env_expression.metadata.get(
                "solver_message_source",
                env_expression.metadata.get("belief_source", "learned"),
            )
        ),
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


def solver_message_reliability_kwargs(step_result) -> dict[str, float | int | str]:
    """Compatibility alias for the older message-oriented name."""
    return solver_expression_reliability_kwargs(step_result)
