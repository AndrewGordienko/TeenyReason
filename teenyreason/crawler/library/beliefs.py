"""Belief and solver-message builders for crawler bundles."""

from __future__ import annotations

import numpy as np
import torch

from ...rl.probe_policy.handoff.message import build_env_expression as build_controller_env_expression
from ..types import (
    BeliefMessage,
    ControllerBeliefContext,
    EnvExpression,
    LegacyCrawlerStepResult,
    MetricBelief,
    PredictiveBelief,
    UncertaintyEstimate,
)
from .helpers import belief_source_from_mode, quantize_vector, sanitize_array


class BeliefBuilderMixin:
    """Build canonical crawler beliefs and solver-facing messages."""

    def build_predictive_belief(self, payload: dict[str, np.ndarray]) -> PredictiveBelief:
        """Convert one raw aggregation payload into the canonical predictive belief."""
        leaveout_param_std = sanitize_array(
            payload.get("leaveout_param_std", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        fair_handoff_probe_families = tuple(
            str(family)
            for family in np.asarray(
                payload.get("fair_handoff_probe_families", np.asarray([], dtype="U")),
                dtype="U",
            ).tolist()
            if str(family)
        )
        return PredictiveBelief(
            mean_raw=sanitize_array(payload["env_mean_raw"]),
            mean_unit=sanitize_array(payload["env_mean"]),
            logvar=sanitize_array(payload["env_logvar"]),
            view_spread=sanitize_array(payload["view_spread"]),
            env_param_mean=sanitize_array(payload["env_param_mean"]),
            env_param_std=sanitize_array(payload["env_param_std"]),
            future_probe_error=float(np.mean(payload.get("future_probe_error", np.asarray([0.0], dtype=np.float32)))),
            support_count=int(np.asarray(payload.get("support_count", np.asarray([0], dtype=np.int32))).reshape(-1)[0]),
            support_diversity_ratio=float(
                np.asarray(payload.get("support_group_ratio", np.asarray([1.0], dtype=np.float32))).reshape(-1)[0]
            ),
            metadata={
                "factor_mean": sanitize_array(
                    payload.get("factor_mean", np.zeros((0,), dtype=np.float32))
                ),
                "factor_std": sanitize_array(
                    payload.get("factor_std", np.zeros((0,), dtype=np.float32))
                ),
                "mechanics_posterior_mean": sanitize_array(
                    payload.get("mechanics_posterior_mean", np.zeros((0,), dtype=np.float32))
                ),
                "mechanics_posterior_std": sanitize_array(
                    payload.get("mechanics_posterior_std", np.zeros((0,), dtype=np.float32))
                ),
                "mechanics_posterior_logvar": sanitize_array(
                    payload.get("mechanics_posterior_logvar", np.zeros((0,), dtype=np.float32))
                ),
                "mechanics_posterior_entropy": float(
                    np.asarray(payload.get("mechanics_posterior_entropy", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "subset_shift": float(np.asarray(payload.get("subset_shift", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]),
                "split_latent_disagreement": float(
                    np.asarray(payload.get("split_latent_disagreement", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "split_param_disagreement": float(
                    np.asarray(payload.get("split_param_disagreement", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "leaveout_param_std_mean": float(np.mean(leaveout_param_std)) if leaveout_param_std.size else 0.0,
                "leaveout_shift": float(
                    np.asarray(payload.get("leaveout_shift", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "full_future_prediction_error": float(
                    np.asarray(
                        payload.get("full_future_prediction_error", np.asarray([0.0], dtype=np.float32))
                    ).reshape(-1)[0]
                ),
                "observed_family_future_error": float(
                    np.asarray(
                        payload.get("observed_family_future_error", np.asarray([0.0], dtype=np.float32))
                    ).reshape(-1)[0]
                ),
                "heldout_family_future_error": float(
                    np.asarray(
                        payload.get("heldout_family_future_error", np.asarray([0.0], dtype=np.float32))
                    ).reshape(-1)[0]
                ),
                "support_size_matched_future_error": float(
                    np.asarray(
                        payload.get("support_size_matched_future_error", np.asarray([0.0], dtype=np.float32))
                    ).reshape(-1)[0]
                ),
                "online_subset_stability": float(
                    np.asarray(
                        payload.get("online_subset_stability", np.asarray([0.0], dtype=np.float32))
                    ).reshape(-1)[0]
                ),
                "online_geometry_complete": bool(
                    np.asarray(
                        payload.get("online_geometry_complete", np.asarray([0.0], dtype=np.float32))
                    ).reshape(-1)[0]
                    > 0.5
                ),
                "online_split_latent_disagreement": float(
                    np.asarray(
                        payload.get(
                            "online_split_latent_disagreement",
                            np.asarray([0.0], dtype=np.float32),
                        )
                    ).reshape(-1)[0]
                ),
                "online_split_retrieval_margin_deficit": float(
                    np.asarray(
                        payload.get(
                            "online_split_retrieval_margin_deficit",
                            np.asarray([0.0], dtype=np.float32),
                        )
                    ).reshape(-1)[0]
                ),
                "online_leaveout_shift": float(
                    np.asarray(
                        payload.get("online_leaveout_shift", np.asarray([0.0], dtype=np.float32))
                    ).reshape(-1)[0]
                ),
                "online_observed_family_count": int(
                    np.asarray(
                        payload.get("online_observed_family_count", np.asarray([0], dtype=np.int32))
                    ).reshape(-1)[0]
                ),
                "online_offline_gap": float(
                    np.asarray(payload.get("online_offline_gap", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "fair_handoff_probe_families": fair_handoff_probe_families,
                "teacher_action_agreement": float(
                    np.asarray(
                        payload.get("teacher_action_agreement", np.asarray([0.0], dtype=np.float32))
                    ).reshape(-1)[0]
                ),
                "belief_mode": str(
                    np.asarray(payload.get("belief_mode", np.asarray([self.belief_mode], dtype="U")), dtype="U").reshape(-1)[0]
                ),
                "belief_source": str(
                    np.asarray(
                        payload.get(
                            "belief_source",
                            np.asarray(
                                [belief_source_from_mode(self.belief_mode)],
                                dtype="U",
                            ),
                        ),
                        dtype="U",
                    ).reshape(-1)[0]
                ),
                "solver_message_source": str(
                    np.asarray(
                        payload.get(
                            "solver_message_source",
                            payload.get(
                                "belief_source",
                                np.asarray(
                                    [belief_source_from_mode(self.belief_mode)],
                                    dtype="U",
                                ),
                            ),
                        ),
                        dtype="U",
                    ).reshape(-1)[0]
                ),
                "sysid_belief_available": bool(
                    np.asarray(payload.get("sysid_belief_available", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0] > 0.5
                ),
                "learned_belief_available": bool(
                    np.asarray(payload.get("learned_belief_available", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0] > 0.5
                ),
                "particle_param_mean": sanitize_array(
                    payload.get("particle_param_mean", np.zeros((0,), dtype=np.float32))
                ),
                "particle_param_std": sanitize_array(
                    payload.get("particle_param_std", np.zeros((0,), dtype=np.float32))
                ),
                "particle_param_mean_norm": sanitize_array(
                    payload.get("particle_param_mean_norm", np.zeros((0,), dtype=np.float32))
                ),
                "particle_param_std_norm": sanitize_array(
                    payload.get("particle_param_std_norm", np.zeros((0,), dtype=np.float32))
                ),
                "particle_weights": sanitize_array(
                    payload.get("particle_weights", np.zeros((0,), dtype=np.float32))
                ),
                "particle_particles_norm": sanitize_array(
                    payload.get("particle_particles_norm", np.zeros((0, 0), dtype=np.float32))
                ),
                "particle_particles_raw": sanitize_array(
                    payload.get("particle_particles_raw", np.zeros((0, 0), dtype=np.float32))
                ),
                "particle_entropy": float(
                    np.asarray(payload.get("particle_entropy", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "particle_entropy_norm": float(
                    np.asarray(payload.get("particle_entropy_norm", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "particle_ess_ratio": float(
                    np.asarray(payload.get("particle_ess_ratio", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "particle_top_weight": float(
                    np.asarray(payload.get("particle_top_weight", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "particle_leaveout_shift": float(
                    np.asarray(payload.get("particle_leaveout_shift", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "particle_subset_stability": float(
                    np.asarray(payload.get("particle_subset_stability", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "sysid_validation_top1": float(
                    np.asarray(payload.get("sysid_validation_top1", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "sysid_validation_margin": float(
                    np.asarray(payload.get("sysid_validation_margin", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "sysid_validation_nll": float(
                    np.asarray(payload.get("sysid_validation_nll", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
                "sysid_trusted": bool(
                    np.asarray(payload.get("sysid_trusted", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0] > 0.5
                ),
            },
        )

    def build_metric_belief(self, payload: dict[str, np.ndarray]) -> MetricBelief:
        """Convert one raw aggregation payload into the auxiliary metric belief."""
        metric_mean = sanitize_array(payload.get("env_metric_mean", payload["env_mean_raw"]))
        metric_mean_unit = sanitize_array(payload.get("env_metric_mean_unit", payload.get("env_mean", metric_mean)))
        split_a = sanitize_array(payload.get("env_metric_split_mean_a", metric_mean))
        split_b = sanitize_array(payload.get("env_metric_split_mean_b", metric_mean))
        nearest_between = float(
            np.asarray(payload.get("nearest_between_distance", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
        )
        gap_ratio = float(
            np.asarray(payload.get("gap_ratio", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
        )
        return MetricBelief(
            mean_raw=metric_mean,
            mean_unit=metric_mean_unit,
            split_mean_a=split_a,
            split_mean_b=split_b,
            nearest_between_distance=nearest_between,
            gap_ratio=gap_ratio,
            metadata={
                "split_retrieval_margin_deficit": float(
                    np.asarray(payload.get("split_retrieval_margin_deficit", np.asarray([0.0], dtype=np.float32))).reshape(-1)[0]
                ),
            },
        )

    def build_uncertainty_estimate(self, payload: dict[str, np.ndarray]) -> UncertaintyEstimate:
        """Convert one raw aggregation payload into the operational uncertainty object."""
        belief = sanitize_array(payload["belief"]).reshape(-1)
        half = belief.shape[0] // 2
        uncertainty_vector = belief[half:] if half > 0 else np.zeros((0,), dtype=np.float32)
        uncertainty_scalar = float(np.mean(uncertainty_vector)) if uncertainty_vector.size else 0.0
        feature_names = tuple(
            str(name)
            for name in np.asarray(
                payload.get("uncertainty_feature_names", np.asarray([], dtype="U")),
                dtype="U",
            ).tolist()
        )
        feature_weights = sanitize_array(
            payload.get("uncertainty_feature_weights", np.zeros((len(feature_names),), dtype=np.float32))
        )
        return UncertaintyEstimate(
            vector=uncertainty_vector.astype(np.float32),
            scalar=uncertainty_scalar,
            feature_names=feature_names,
            feature_weights=feature_weights,
            metadata={
                "belief_dim": int(belief.shape[0]),
            },
        )

    def build_env_expression(
        self,
        predictive_belief: PredictiveBelief,
        metric_belief: MetricBelief,
        uncertainty: UncertaintyEstimate,
        *,
        bits_per_dim: int = 0,
        use_residual_sketch: bool = False,
    ) -> EnvExpression:
        """Project the predictive belief into one controller-facing env expression."""
        predictive_mean = sanitize_array(predictive_belief.mean_raw).reshape(1, -1)
        uncertainty_scalar = np.asarray([[float(uncertainty.scalar)]], dtype=np.float32)
        particle_mode = str(predictive_belief.metadata.get("belief_mode", "")) == "particle_sysid"
        with torch.no_grad():
            if self.belief_message_projector is not None and not particle_mode:
                message_t = self.belief_message_projector(
                    torch.tensor(predictive_mean, dtype=torch.float32, device=self.device),
                    torch.tensor(uncertainty_scalar, dtype=torch.float32, device=self.device),
                )
                message = sanitize_array(message_t.squeeze(0).cpu().numpy())
            else:
                message = sanitize_array(predictive_belief.mean_raw)
        return build_controller_env_expression(
            predictive_belief=predictive_belief,
            metric_belief=metric_belief,
            uncertainty=uncertainty,
            raw_expression=message,
            bits_per_dim=bits_per_dim,
            use_residual_sketch=use_residual_sketch,
            quantize_vector_fn=quantize_vector,
        )

    def build_belief_message(
        self,
        predictive_belief: PredictiveBelief,
        uncertainty: UncertaintyEstimate,
        *,
        bits_per_dim: int = 0,
        use_residual_sketch: bool = False,
        metric_belief: MetricBelief | None = None,
    ) -> BeliefMessage:
        """Compatibility alias for the older belief-message name."""
        if metric_belief is None:
            metric_belief = MetricBelief(
                mean_raw=sanitize_array(predictive_belief.mean_raw),
                mean_unit=sanitize_array(predictive_belief.mean_unit),
                split_mean_a=sanitize_array(predictive_belief.mean_unit),
                split_mean_b=sanitize_array(predictive_belief.mean_unit),
                nearest_between_distance=0.0,
                gap_ratio=0.0,
                metadata={},
            )
        return self.build_env_expression(
            predictive_belief,
            metric_belief,
            uncertainty,
            bits_per_dim=bits_per_dim,
            use_residual_sketch=use_residual_sketch,
        )

    def build_controller_context(
        self,
        predictive_belief: PredictiveBelief,
        metric_belief: MetricBelief,
        uncertainty: UncertaintyEstimate,
        *,
        env_expression: EnvExpression | None = None,
    ) -> ControllerBeliefContext:
        """Build the richer controller-facing belief context for the full-system path."""
        predictive_mean = sanitize_array(predictive_belief.mean_raw).reshape(1, -1)
        uncertainty_scalar = np.asarray([[float(uncertainty.scalar)]], dtype=np.float32)
        particle_mode = str(predictive_belief.metadata.get("belief_mode", "")) == "particle_sysid"
        if self.controller_context_projector is not None and not particle_mode:
            with torch.no_grad():
                mechanics_t, affordance_t = self.controller_context_projector(
                    torch.tensor(predictive_mean, dtype=torch.float32, device=self.device),
                    torch.tensor(uncertainty_scalar, dtype=torch.float32, device=self.device),
                )
            mechanics_code = sanitize_array(mechanics_t.squeeze(0).cpu().numpy())
            affordance_code = sanitize_array(affordance_t.squeeze(0).cpu().numpy())
        else:
            mechanics_code = sanitize_array(predictive_belief.mean_raw)
            affordance_code = sanitize_array(predictive_belief.mean_raw)
        if env_expression is None:
            env_expression = self.build_env_expression(
                predictive_belief,
                metric_belief,
                uncertainty,
                bits_per_dim=0,
                use_residual_sketch=False,
            )
        control_confidence = self.estimate_controller_trust(
            predictive_belief,
            metric_belief,
            uncertainty,
        )
        return ControllerBeliefContext(
            mechanics_code=mechanics_code,
            affordance_code=affordance_code,
            confidence=float(control_confidence),
            uncertainty_scalar=float(uncertainty.scalar),
            metadata={
                "source_kind": "particle_sysid" if particle_mode else "learned",
                "belief_source": str(
                    predictive_belief.metadata.get(
                        "belief_source",
                        "sysid" if particle_mode else "learned",
                    )
                ),
                "solver_message_source": str(
                    predictive_belief.metadata.get(
                        "solver_message_source",
                        predictive_belief.metadata.get(
                            "belief_source",
                            "sysid" if particle_mode else "learned",
                        ),
                    )
                ),
                "env_expression_ready": bool(env_expression.ready),
                "env_expression_confidence": float(env_expression.confidence),
                "readiness_score": float(env_expression.metadata.get("readiness_score", 0.0)),
                "utility_forecast": float(env_expression.metadata.get("utility_forecast", 0.0)),
                "support_count": int(predictive_belief.support_count),
                "support_diversity_ratio": float(predictive_belief.support_diversity_ratio),
                "future_probe_error": float(predictive_belief.future_probe_error),
                "particle_entropy": float(predictive_belief.metadata.get("particle_entropy", 0.0)),
                "particle_ess_ratio": float(predictive_belief.metadata.get("particle_ess_ratio", 0.0)),
                "gap_ratio": float(metric_belief.gap_ratio),
                "nearest_between_distance": float(metric_belief.nearest_between_distance),
                "control_confidence": float(control_confidence),
            },
        )

    def build_oracle_controller_context(
        self,
        env_params: np.ndarray | Sequence[float],
    ) -> ControllerBeliefContext:
        """Build controller context directly from true hidden env parameters."""
        normalized_env_params = self.normalize_env_params(env_params).reshape(1, -1)
        if self.oracle_context_projector is not None:
            with torch.no_grad():
                mechanics_t, affordance_t = self.oracle_context_projector(
                    torch.tensor(
                        normalized_env_params,
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
            mechanics_code = sanitize_array(mechanics_t.squeeze(0).cpu().numpy())
            affordance_code = sanitize_array(affordance_t.squeeze(0).cpu().numpy())
        else:
            mechanics_code = sanitize_array(normalized_env_params.reshape(-1))
            affordance_code = sanitize_array(normalized_env_params.reshape(-1))
        return ControllerBeliefContext(
            mechanics_code=mechanics_code,
            affordance_code=affordance_code,
            confidence=1.0,
            uncertainty_scalar=0.0,
            metadata={
                "source_kind": "oracle",
                "belief_source": "oracle",
                "solver_message_source": "oracle",
                "normalized_env_params": sanitize_array(normalized_env_params.reshape(-1)),
            },
        )

    def build_step_result(
        self,
        *,
        payload: dict[str, np.ndarray],
        expected_family_gain: dict[str, dict[str, float]],
        realized_family_gain: dict[str, float],
        stop_reason: str | None,
        bits_per_dim: int = 0,
        use_residual_sketch: bool = False,
    ) -> LegacyCrawlerStepResult:
        """Convert one aggregation payload plus crawler bookkeeping into a step object."""
        predictive_belief = self.build_predictive_belief(payload)
        metric_belief = self.build_metric_belief(payload)
        uncertainty = self.build_uncertainty_estimate(payload)
        env_expression = self.build_env_expression(
            predictive_belief,
            metric_belief,
            uncertainty,
            bits_per_dim=bits_per_dim,
            use_residual_sketch=use_residual_sketch,
        )
        controller_context = self.build_controller_context(
            predictive_belief,
            metric_belief,
            uncertainty,
            env_expression=env_expression,
        )
        return LegacyCrawlerStepResult(
            predictive_belief=predictive_belief,
            metric_belief=metric_belief,
            uncertainty=uncertainty,
            env_expression=env_expression,
            controller_context=controller_context,
            expected_family_gain=expected_family_gain,
            realized_family_gain=realized_family_gain,
            stop_reason=stop_reason,
        )
