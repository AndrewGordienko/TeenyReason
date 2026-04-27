"""Library-facing crawler bundle and reusable env-expression helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ..models.belief_world_model import (
    BeliefMessageProjector,
    ContrastiveProjector,
    ControllerBeliefContextProjector,
    ControllerTrustPredictor,
    FamilyConditionedOutcomePredictor,
    FamilyConditionedValuePredictor,
    OracleControllerContextProjector,
    OutcomePredictor,
)
from ..models.env_belief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ..models.sysid import (
    ProbeLikelihoodModel,
    SysIdFeatureStats,
    build_latin_hypercube_particles,
    particle_payload_from_windows,
    train_probe_likelihood_model,
)
from ..probe.probe_latent import aggregate_env_belief, encode_window_posterior
from ..representation import DeltaPredictorEnsemble, WorldEncoder, train_encoder_predictor
from ..representation.metrics import deterministic_rotation
from ..rl.probe_policy.messages import build_env_expression as build_controller_env_expression
from .types import (
    BeliefMessage,
    ControllerBeliefContext,
    EvidenceBatch,
    EvidenceWindow,
    EnvExpression,
    LegacyCrawlerStepResult,
    MetricBelief,
    PredictiveBelief,
    UncertaintyEstimate,
)


def sanitize_array(values: np.ndarray | Sequence[float], fill_value: float = 0.0) -> np.ndarray:
    """Convert arrays into finite float32 vectors."""
    return np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    ).astype(np.float32)


def estimate_probe_family_cost(family: str) -> float:
    """Assign one simple relative interaction cost to each probe family."""
    family_name = str(family)
    if "passive" in family_name:
        return 0.70
    if "chirp" in family_name:
        return 0.90
    if "impulse" in family_name:
        return 1.00
    if "boundary" in family_name:
        return 1.15
    if "brake" in family_name:
        return 1.10
    return 1.00


def cartpole_family_factor_prior(
    family: str,
    factor_count: int,
) -> np.ndarray:
    """Map probe families onto a small set of CartPole scientist factors."""
    base = {
        "passive_decay": np.asarray([1.00, 0.10, 0.85, 0.25], dtype=np.float32),
        "impulse_left": np.asarray([0.35, 1.00, 0.80, 0.20], dtype=np.float32),
        "impulse_right": np.asarray([0.35, 1.00, 0.80, 0.20], dtype=np.float32),
        "chirp": np.asarray([0.45, 0.90, 0.70, 0.35], dtype=np.float32),
        "boundary_push": np.asarray([0.25, 0.50, 0.60, 1.00], dtype=np.float32),
        "cart_brake": np.asarray([0.25, 0.75, 1.00, 0.80], dtype=np.float32),
    }.get(str(family), np.ones((4,), dtype=np.float32))
    if factor_count <= base.shape[0]:
        return base[:factor_count].astype(np.float32)
    repeats = int(np.ceil(float(factor_count) / float(base.shape[0])))
    return np.tile(base, repeats)[:factor_count].astype(np.float32)


def quantize_vector(
    values: np.ndarray,
    *,
    bits_per_dim: int,
    use_residual_sketch: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Quantize one message vector with optional residual sketch."""
    vector = sanitize_array(values).reshape(-1)
    if bits_per_dim <= 0 or vector.size == 0:
        return vector, None

    rotation = deterministic_rotation(vector.size, seed=bits_per_dim)
    rotated = rotation @ vector
    levels = max(2, 2 ** int(bits_per_dim))
    scale = float(np.max(np.abs(rotated)))
    if not np.isfinite(scale) or scale <= 1e-6:
        return vector, None

    normalized = np.clip(rotated / scale, -1.0, 1.0)
    grid = np.linspace(-1.0, 1.0, num=levels, dtype=np.float32)
    quantized_idx = np.abs(normalized[:, None] - grid[None, :]).argmin(axis=1)
    quantized_rotated = grid[quantized_idx] * scale
    coarse = rotation.T @ quantized_rotated
    coarse = sanitize_array(coarse)
    if not use_residual_sketch:
        return coarse, None

    residual = sanitize_array(vector - coarse)
    residual_scale = float(np.max(np.abs(residual)))
    if not np.isfinite(residual_scale) or residual_scale <= 1e-6:
        return coarse, np.zeros_like(vector, dtype=np.float32)
    residual_bits = max(2, bits_per_dim // 2)
    residual_levels = max(2, 2 ** residual_bits)
    residual_grid = np.linspace(-1.0, 1.0, num=residual_levels, dtype=np.float32)
    residual_norm = np.clip(residual / residual_scale, -1.0, 1.0)
    residual_idx = np.abs(residual_norm[:, None] - residual_grid[None, :]).argmin(axis=1)
    residual_quantized = residual_grid[residual_idx] * residual_scale
    return coarse, sanitize_array(residual_quantized)


def mean_pairwise_distance(values: np.ndarray) -> float:
    """Return the mean off-diagonal distance across a small set of vectors."""
    rows = sanitize_array(values)
    if rows.ndim != 2 or rows.shape[0] < 2:
        return 0.0
    deltas = rows[:, None, :] - rows[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    mask = ~np.eye(rows.shape[0], dtype=bool)
    if not np.any(mask):
        return 0.0
    return float(np.mean(distances[mask]))


def build_evidence_batch(
    *,
    windows: dict[str, np.ndarray],
    env_name: str,
    window_size: int,
    action_vocab_size: int,
) -> EvidenceBatch:
    """Convert stored NumPy windows into library-facing evidence objects."""
    evidence_windows: list[EvidenceWindow] = []
    for idx in range(int(windows["states"].shape[0])):
        evidence_windows.append(
            EvidenceWindow(
                states=sanitize_array(windows["states"][idx]),
                actions=np.asarray(windows["actions"][idx], dtype=np.int64),
                rewards=sanitize_array(windows["rewards"][idx]),
                terminated=bool(windows["terminated"][idx]),
                truncated=bool(windows["truncated"][idx]),
                probe_family=str(np.asarray(windows["probe_mode"][idx]).item()),
                env_instance_id=int(windows["env_instance_id"][idx]),
            )
        )
    return EvidenceBatch(
        windows=tuple(evidence_windows),
        env_name=env_name,
        window_size=int(window_size),
        action_vocab_size=int(action_vocab_size),
    )


@dataclass
class CrawlerModelBundle:
    """All crawler-side models needed to build env beliefs from probe evidence."""

    encoder: WorldEncoder
    predictor: DeltaPredictorEnsemble
    belief_aggregator: EnvBeliefAggregator
    env_param_predictor: EnvParamPredictorEnsemble
    env_future_predictor: nn.Module | None
    env_family_future_predictor: nn.Module | None
    family_value_predictor: nn.Module | None
    env_metric_projector: nn.Module | None
    belief_message_projector: nn.Module | None
    controller_context_projector: nn.Module | None
    device: torch.device
    z_dim: int
    window_size: int
    action_vocab_size: int
    belief_message_dim: int
    controller_context_dim: int
    family_names: tuple[str, ...] = ()
    oracle_context_projector: nn.Module | None = None
    controller_trust_predictor: nn.Module | None = None
    env_param_normalizer_mean: np.ndarray | None = None
    env_param_normalizer_std: np.ndarray | None = None
    belief_mode: str = "latent_pool"
    sysid_model: ProbeLikelihoodModel | None = None
    sysid_stats: SysIdFeatureStats | None = None
    sysid_particles_raw: np.ndarray | None = None
    sysid_trusted: bool = False
    sysid_validation_metrics: dict[str, float] | None = None
    sysid_likelihood_scale: float = 0.35

    @property
    def env_expression_dim(self) -> int:
        """Canonical controller-facing expression width."""
        return int(self.belief_message_dim)

    @property
    def env_expression_projector(self):
        """Canonical controller-facing projector alias."""
        return self.belief_message_projector

    @property
    def full_system_controller_dim(self) -> int:
        """Canonical flat controller-context width for the belief-native controller."""
        return int(self.controller_context_dim)

    def normalize_env_params(
        self,
        env_params: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Normalize raw environment parameters into the oracle-projector space."""
        raw = sanitize_array(env_params).reshape(-1)
        if self.env_param_normalizer_mean is None or self.env_param_normalizer_std is None:
            return raw
        mean = sanitize_array(self.env_param_normalizer_mean).reshape(-1)
        std = sanitize_array(self.env_param_normalizer_std).reshape(-1)
        if mean.shape[0] != raw.shape[0]:
            shared_dim = min(int(mean.shape[0]), int(raw.shape[0]))
            mean = mean[:shared_dim]
            std = std[:shared_dim]
            raw = raw[:shared_dim]
        std = np.where(np.abs(std) < 1e-6, 1.0, std).astype(np.float32)
        return sanitize_array((raw - mean) / std)

    def estimate_controller_trust(
        self,
        predictive_belief: PredictiveBelief,
        metric_belief: MetricBelief,
        uncertainty: UncertaintyEstimate,
    ) -> float:
        """Estimate full-system control trust separately from fair readiness."""
        predictive_mean = sanitize_array(predictive_belief.mean_raw).reshape(1, -1)
        uncertainty_scalar = np.asarray([[float(uncertainty.scalar)]], dtype=np.float32)
        if self.controller_trust_predictor is not None:
            with torch.no_grad():
                trust_t = self.controller_trust_predictor(
                    torch.tensor(predictive_mean, dtype=torch.float32, device=self.device),
                    torch.tensor(uncertainty_scalar, dtype=torch.float32, device=self.device),
                )
            return float(np.clip(trust_t.squeeze(0).item(), 0.0, 1.0))

        support_factor = min(float(predictive_belief.support_count) / 4.0, 1.0)
        future_quality = np.exp(-max(0.0, float(predictive_belief.future_probe_error)))
        geometry_penalty = min(max(float(metric_belief.gap_ratio), 0.0), 4.0) / 4.0
        uncertainty_penalty = min(max(float(uncertainty.scalar), 0.0), 2.0) / 2.0
        trust = (
            0.20
            + 0.35 * future_quality
            + 0.20 * float(predictive_belief.support_diversity_ratio)
            + 0.15 * support_factor
            - 0.15 * geometry_penalty
            - 0.15 * uncertainty_penalty
        )
        return float(np.clip(trust, 0.0, 1.0))

    def encode_probe_window(
        self,
        window_states,
        window_actions,
        window_rewards=None,
    ):
        """Return the posterior mean/logvar for one probe window."""
        return encode_window_posterior(
            encoder=self.encoder,
            device=self.device,
            window_states=window_states,
            window_actions=window_actions,
            window_rewards=window_rewards,
        )

    def build_env_belief(
        self,
        posterior_views,
        *,
        probe_group_ids: np.ndarray | None = None,
        bits_per_dim: int = 0,
        use_residual_sketch: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Aggregate probe-window posteriors into one env-level belief vector."""
        belief, payload = aggregate_env_belief(
            belief_aggregator=self.belief_aggregator,
            env_param_predictor=self.env_param_predictor,
            device=self.device,
            posterior_views=posterior_views,
            probe_group_ids=probe_group_ids,
        )
        step_result = self.build_step_result(
            payload=payload,
            expected_family_gain={},
            realized_family_gain={},
            stop_reason=None,
            bits_per_dim=bits_per_dim,
            use_residual_sketch=use_residual_sketch,
        )
        payload["env_expression"] = step_result.env_expression.vector.astype(np.float32)
        payload["env_expression_confidence"] = np.asarray(
            [step_result.env_expression.confidence],
            dtype=np.float32,
        )
        payload["env_expression_uncertainty"] = np.asarray(
            [step_result.env_expression.uncertainty_scalar],
            dtype=np.float32,
        )
        payload["belief_message"] = step_result.env_expression.vector.astype(np.float32)
        payload["belief_message_confidence"] = np.asarray(
            [step_result.env_expression.confidence],
            dtype=np.float32,
        )
        payload["belief_message_uncertainty"] = np.asarray(
            [step_result.env_expression.uncertainty_scalar],
            dtype=np.float32,
        )
        if step_result.env_expression.residual_vector is not None:
            payload["env_expression_residual"] = step_result.env_expression.residual_vector.astype(np.float32)
            payload["belief_message_residual"] = step_result.env_expression.residual_vector.astype(np.float32)
        return belief, payload

    def build_particle_env_belief(
        self,
        probe_window_records,
        *,
        bits_per_dim: int = 0,
        use_residual_sketch: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Build an env belief by likelihood-scoring candidate CartPole mechanics."""
        if self.sysid_model is None or self.sysid_stats is None or self.sysid_particles_raw is None:
            raise ValueError("Particle system-ID belief was requested without a trained sysid bundle")
        belief, payload = particle_payload_from_windows(
            records=tuple(probe_window_records),
            model=self.sysid_model,
            stats=self.sysid_stats,
            particles_raw=self.sysid_particles_raw,
            z_dim=self.z_dim,
            trusted=bool(self.sysid_trusted),
            validation_metrics=self.sysid_validation_metrics or {},
            likelihood_scale=float(self.sysid_likelihood_scale),
        )
        step_result = self.build_step_result(
            payload=payload,
            expected_family_gain={},
            realized_family_gain={},
            stop_reason=None,
            bits_per_dim=bits_per_dim,
            use_residual_sketch=use_residual_sketch,
        )
        payload["env_expression"] = step_result.env_expression.vector.astype(np.float32)
        payload["env_expression_confidence"] = np.asarray(
            [step_result.env_expression.confidence],
            dtype=np.float32,
        )
        payload["env_expression_uncertainty"] = np.asarray(
            [step_result.env_expression.uncertainty_scalar],
            dtype=np.float32,
        )
        payload["belief_message"] = step_result.env_expression.vector.astype(np.float32)
        payload["belief_message_confidence"] = payload["env_expression_confidence"]
        payload["belief_message_uncertainty"] = payload["env_expression_uncertainty"]
        return belief, payload

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
                "normalized_env_params": sanitize_array(normalized_env_params.reshape(-1)),
            },
        )

    def build_mechanics_hypothesis_latents(
        self,
        predictive_belief: PredictiveBelief,
    ) -> np.ndarray:
        """Build a small set of explicit competing world hypotheses in latent space."""
        posterior_mean = sanitize_array(
            predictive_belief.metadata.get("mechanics_posterior_mean", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        posterior_std = sanitize_array(
            predictive_belief.metadata.get("mechanics_posterior_std", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        if posterior_mean.size == 0 or posterior_std.size == 0:
            return sanitize_array(predictive_belief.mean_raw).reshape(1, -1)

        candidate_means = [posterior_mean]
        top_factor_count = min(2, posterior_std.shape[0])
        ranked_factors = np.argsort(-posterior_std)[:top_factor_count]
        for factor_idx in ranked_factors.tolist():
            offset = np.zeros_like(posterior_mean)
            offset[factor_idx] = posterior_std[factor_idx]
            candidate_means.append(posterior_mean + offset)
            candidate_means.append(posterior_mean - offset)

        candidate_mean_t = torch.tensor(
            np.stack(candidate_means, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        candidate_std_t = torch.tensor(
            np.repeat(posterior_std[None, :], candidate_mean_t.shape[0], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            latent_t = self.belief_aggregator.mechanics_updater.posterior_to_latent(
                torch.cat([candidate_mean_t, candidate_std_t], dim=-1)
            )
        return sanitize_array(latent_t.cpu().numpy())

    def family_hypothesis_separation(
        self,
        predictive_belief: PredictiveBelief,
    ) -> dict[str, float]:
        """Estimate which probe families best separate the remaining world hypotheses."""
        if self.env_family_future_predictor is None or not self.family_names:
            return {family: 0.0 for family in self.family_names}

        hypothesis_latents = self.build_mechanics_hypothesis_latents(predictive_belief)
        if hypothesis_latents.shape[0] < 2:
            return {family: 0.0 for family in self.family_names}

        repeated_latents = np.repeat(
            hypothesis_latents[None, :, :],
            repeats=len(self.family_names),
            axis=0,
        )
        family_ids = np.repeat(
            np.arange(len(self.family_names), dtype=np.int64)[:, None],
            repeats=hypothesis_latents.shape[0],
            axis=1,
        )
        with torch.no_grad():
            future_t = self.env_family_future_predictor(
                torch.tensor(repeated_latents.reshape(-1, repeated_latents.shape[-1]), dtype=torch.float32, device=self.device),
                torch.tensor(family_ids.reshape(-1), dtype=torch.long, device=self.device),
            )
        future = sanitize_array(
            future_t.cpu().numpy().reshape(len(self.family_names), hypothesis_latents.shape[0], -1)
        )
        return {
            family: mean_pairwise_distance(future[idx])
            for idx, family in enumerate(self.family_names)
        }

    def score_particle_probe_families(
        self,
        predictive_belief: PredictiveBelief,
        *,
        family_counts: dict[str, int] | None = None,
        global_family_counts: dict[str, int] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Score families by how strongly they separate particle hypotheses."""
        del global_family_counts
        if self.sysid_model is None or self.sysid_stats is None or not self.family_names:
            return {}
        family_counts = family_counts or {}
        particles = sanitize_array(
            predictive_belief.metadata.get("particle_particles_norm", np.zeros((0, 0), dtype=np.float32))
        )
        weights = sanitize_array(
            predictive_belief.metadata.get("particle_weights", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        if particles.ndim != 2 or particles.shape[0] == 0 or weights.shape[0] != particles.shape[0]:
            return {}
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-6:
            weights = np.full((particles.shape[0],), 1.0 / float(particles.shape[0]), dtype=np.float32)
        else:
            weights = (weights / weight_sum).astype(np.float32)

        query_means = sanitize_array(self.sysid_stats.family_query_mean_norm)
        if query_means.ndim != 2 or query_means.shape[0] < len(self.family_names):
            return {}

        self.sysid_model.eval()
        particle_t = torch.tensor(particles, dtype=torch.float32, device=self.device)
        weight_t = torch.tensor(weights, dtype=torch.float32, device=self.device).reshape(-1, 1)
        gains: dict[str, dict[str, float]] = {}
        for family_idx, family in enumerate(self.family_names):
            family_query = query_means[family_idx]
            query_t = torch.tensor(
                np.repeat(family_query[None, :], particles.shape[0], axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            family_t = torch.full((particles.shape[0],), int(family_idx), dtype=torch.long, device=self.device)
            with torch.no_grad():
                mean_t, logvar_t = self.sysid_model.predict(particle_t, query_t, family_t)
                center_t = torch.sum(weight_t * mean_t, dim=0, keepdim=True)
                between_t = torch.sum(weight_t * torch.square(mean_t - center_t), dim=0).mean()
                noise_t = torch.sum(weight_t * torch.exp(logvar_t), dim=0).mean()
            between = float(max(0.0, between_t.item()))
            noise = float(max(1e-4, noise_t.item()))
            info = between / (noise + 1e-4)
            entropy_reduction = float(np.log1p(max(0.0, info)))
            hypothesis_separation = float(np.sqrt(max(0.0, between)))
            cost = estimate_probe_family_cost(family)
            coverage_bonus = 0.08 if int(family_counts.get(family, 0)) <= 0 else 0.0
            score = entropy_reduction + 0.35 * hypothesis_separation
            selection_score = score + coverage_bonus - 0.35 * cost
            value_per_step = selection_score / max(cost, 1e-6)
            gains[family] = {
                "predicted_mechanics_reduction": entropy_reduction,
                "raw_predicted_future_error_reduction": entropy_reduction,
                "predicted_future_error_reduction": entropy_reduction,
                "future_gain_for_choice": entropy_reduction,
                "predicted_split_reduction": hypothesis_separation,
                "predicted_entropy_reduction": entropy_reduction,
                "predicted_hypothesis_separation": hypothesis_separation,
                "diversity_bonus": 0.0,
                "coverage_bonus": coverage_bonus,
                "quota_bonus": 0.0,
                "repeat_penalty": 0.0,
                "global_repeat_penalty": 0.0,
                "realized_gain_calibration": 1.0,
                "realized_gain_bonus": 0.0,
                "raw_future_error_estimate": float(predictive_belief.future_probe_error),
                "future_error_estimate": float(predictive_belief.future_probe_error),
                "signature_norm": hypothesis_separation,
                "estimated_probe_cost": cost,
                "predicted_marginal_value": selection_score,
                "value_per_probe_step": value_per_step,
                "score": score,
                "selection_score": selection_score,
            }
        return gains

    def score_probe_families(
        self,
        predictive_belief: PredictiveBelief,
        uncertainty: UncertaintyEstimate,
        *,
        family_counts: dict[str, int] | None = None,
        global_family_counts: dict[str, int] | None = None,
        family_error_history: dict[str, float] | None = None,
        family_realized_gain_history: dict[str, float] | None = None,
        use_learned_family_value: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Score probe families by expected belief improvement rather than novelty alone."""
        del uncertainty
        if str(self.belief_mode) == "particle_sysid" and str(
            predictive_belief.metadata.get("belief_mode", "")
        ) == "particle_sysid":
            particle_scores = self.score_particle_probe_families(
                predictive_belief,
                family_counts=family_counts,
                global_family_counts=global_family_counts,
            )
            if particle_scores:
                return particle_scores

        family_counts = family_counts or {}
        global_family_counts = global_family_counts or {}
        family_error_history = family_error_history or {}
        family_realized_gain_history = family_realized_gain_history or {}
        if not self.family_names:
            return {}
        min_family_count = min((int(family_counts.get(family, 0)) for family in self.family_names), default=0)
        min_global_family_count = min((int(global_family_counts.get(family, 0)) for family in self.family_names), default=0)
        total_global_family_count = sum(max(0, int(global_family_counts.get(family, 0))) for family in self.family_names)
        target_probe_families = {"boundary_push", "cart_brake"}
        target_quota_floor = max(1, total_global_family_count // max(3 * max(len(self.family_names), 1), 1))

        mechanics_posterior_std = sanitize_array(
            predictive_belief.metadata.get("mechanics_posterior_std", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        mechanics_posterior_logvar = sanitize_array(
            predictive_belief.metadata.get("mechanics_posterior_logvar", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        mechanics_uncertainty = float(
            np.mean(mechanics_posterior_std)
        ) if mechanics_posterior_std.size else float(np.mean(np.abs(predictive_belief.env_param_std)))
        mechanics_entropy = float(predictive_belief.metadata.get("mechanics_posterior_entropy", 0.0))
        split_uncertainty = float(predictive_belief.metadata.get("split_latent_disagreement", 0.0))
        factor_uncertainty = sanitize_array(
            predictive_belief.metadata.get("factor_std", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        predictive_mean = sanitize_array(predictive_belief.mean_raw).reshape(1, -1)
        future_signature_norm = {}
        if self.env_family_future_predictor is not None:
            with torch.no_grad():
                family_idx_t = torch.arange(len(self.family_names), dtype=torch.long, device=self.device)
                repeated_belief_t = torch.tensor(
                    np.repeat(predictive_mean, len(self.family_names), axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                family_future_t = self.env_family_future_predictor(repeated_belief_t, family_idx_t)
            family_future = sanitize_array(family_future_t.cpu().numpy())
            family_center = family_future.mean(axis=0, keepdims=True)
            family_signature_norm = {
                family: float(np.linalg.norm(family_future[idx] - family_center[0]))
                for idx, family in enumerate(self.family_names)
            }
        else:
            family_signature_norm = {family: 1.0 for family in self.family_names}

        family_hypothesis_separation = self.family_hypothesis_separation(predictive_belief)

        entropy_reduction_by_family = {family: 0.0 for family in self.family_names}
        if mechanics_posterior_logvar.size > 0:
            with torch.no_grad():
                entropy_drop_t = self.belief_aggregator.mechanics_updater.expected_family_information_gain(
                    torch.tensor(
                        mechanics_posterior_logvar[None, :],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
            entropy_drop = sanitize_array(entropy_drop_t.squeeze(0).cpu().numpy())
            for family_idx, family in enumerate(self.family_names[: entropy_drop.shape[0]]):
                entropy_reduction_by_family[family] = float(entropy_drop[family_idx])

        family_value_estimates = {
            family: {
                "mechanics": mechanics_uncertainty,
                "future": predictive_belief.future_probe_error,
                "belief_shift": split_uncertainty,
            }
            for family in self.family_names
        }
        if use_learned_family_value and self.family_value_predictor is not None:
            family_value_context = sanitize_array(
                np.concatenate(
                    [
                        predictive_belief.mean_raw.reshape(-1),
                        np.asarray(
                            [
                                float(np.mean(np.abs(predictive_belief.env_param_std))),
                                float(predictive_belief.future_probe_error),
                                float(predictive_belief.metadata.get("mechanics_posterior_entropy", 0.0)),
                                float(predictive_belief.support_diversity_ratio),
                            ],
                            dtype=np.float32,
                        ),
                    ],
                    axis=0,
                )
            )
            repeated_context = np.repeat(family_value_context[None, :], len(self.family_names), axis=0)
            with torch.no_grad():
                family_value_t = self.family_value_predictor(
                    torch.tensor(repeated_context, dtype=torch.float32, device=self.device),
                    torch.arange(len(self.family_names), dtype=torch.long, device=self.device),
                )
            family_value = sanitize_array(family_value_t.cpu().numpy())
            for family_idx, family in enumerate(self.family_names[: family_value.shape[0]]):
                family_value_estimates[family] = {
                    "mechanics": float(family_value[family_idx, 0]),
                    "future": float(family_value[family_idx, 1]),
                    "belief_shift": float(family_value[family_idx, 2]),
                }

        gains: dict[str, dict[str, float]] = {}
        for family in self.family_names:
            family_count = int(family_counts.get(family, 0))
            global_family_count = int(global_family_counts.get(family, 0))
            novelty_bonus = 1.0 / (1.0 + float(family_count))
            predictive_signature = float(family_signature_norm.get(family, 0.0))
            family_value = family_value_estimates.get(family, {})
            family_factor_gain = float(family_value.get("mechanics", mechanics_uncertainty))
            raw_future_error = float(
                family_error_history.get(
                    family,
                    family_value.get("future", predictive_belief.future_probe_error),
                )
            )
            future_error = raw_future_error
            is_unseen_active_family = family != "passive_decay" and family_count <= 0
            if is_unseen_active_family:
                future_error = max(
                    raw_future_error,
                    0.35 * float(predictive_belief.future_probe_error),
                )
            estimated_probe_cost = estimate_probe_family_cost(family)
            hypothesis_separation = float(family_hypothesis_separation.get(family, 0.0))
            realized_gain = float(family_realized_gain_history.get(family, 0.0))
            predicted_mechanics_reduction = family_factor_gain * (0.55 + 0.30 * predictive_signature)
            raw_predicted_future_error_reduction = raw_future_error * (
                0.35 + 0.25 * predictive_signature
            )
            predicted_future_error_reduction = future_error * (
                0.35 + 0.25 * predictive_signature
            )
            future_gain_for_choice = max(
                float(raw_predicted_future_error_reduction),
                0.30 * float(future_error),
            )
            predicted_split_reduction = float(family_value.get("belief_shift", split_uncertainty)) * (
                0.35 + 0.35 * hypothesis_separation
            )
            predicted_entropy_reduction = float(
                entropy_reduction_by_family.get(family, mechanics_entropy * (0.15 + 0.45 * family_factor_gain))
            )
            raw_total_gain = (
                0.22 * predicted_mechanics_reduction
                + 0.18 * predicted_future_error_reduction
                + 0.18 * predicted_split_reduction
                + 0.25 * predicted_entropy_reduction
                + 0.15 * hypothesis_separation
                + 0.02 * novelty_bonus
            )
            if family in family_realized_gain_history:
                realized_gain_calibration = float(
                    np.clip(
                        0.60 + realized_gain / max(raw_total_gain, 0.10),
                        0.55,
                        1.45,
                    )
                )
            else:
                realized_gain_calibration = 1.0
            coverage_bonus = 0.08 * max(0, (min_family_count + 1) - family_count)
            repeat_penalty = 0.06 * max(0, family_count - min_family_count)
            global_repeat_penalty = 0.03 * max(0, global_family_count - min_global_family_count)
            quota_bonus = 0.0
            if family in target_probe_families and global_family_count < target_quota_floor:
                quota_bonus = 0.14 + 0.03 * float(target_quota_floor - global_family_count)
            realized_gain_bonus = 0.18 * realized_gain
            total_gain = raw_total_gain * realized_gain_calibration + realized_gain_bonus
            selection_score = total_gain + coverage_bonus + quota_bonus - repeat_penalty - global_repeat_penalty
            cost_adjusted_gain = selection_score - 0.35 * estimated_probe_cost
            value_per_probe_step = cost_adjusted_gain / max(estimated_probe_cost, 1e-6)
            gains[family] = {
                "predicted_mechanics_reduction": float(predicted_mechanics_reduction),
                "raw_predicted_future_error_reduction": float(raw_predicted_future_error_reduction),
                "predicted_future_error_reduction": float(predicted_future_error_reduction),
                "future_gain_for_choice": float(future_gain_for_choice),
                "predicted_split_reduction": float(predicted_split_reduction),
                "predicted_entropy_reduction": float(predicted_entropy_reduction),
                "predicted_hypothesis_separation": float(hypothesis_separation),
                "diversity_bonus": float(novelty_bonus),
                "coverage_bonus": float(coverage_bonus),
                "quota_bonus": float(quota_bonus),
                "repeat_penalty": float(repeat_penalty),
                "global_repeat_penalty": float(global_repeat_penalty),
                "realized_gain_calibration": float(realized_gain_calibration),
                "realized_gain_bonus": float(realized_gain_bonus),
                "raw_future_error_estimate": float(raw_future_error),
                "future_error_estimate": float(future_error),
                "signature_norm": float(predictive_signature),
                "estimated_probe_cost": float(estimated_probe_cost),
                "predicted_marginal_value": float(cost_adjusted_gain),
                "value_per_probe_step": float(value_per_probe_step),
                "score": float(total_gain),
                "selection_score": float(selection_score),
            }
        return gains

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


def train_crawler_library(
    *,
    windows,
    z_dim: int,
    window_size: int,
    action_vocab_size: int,
    belief_mode: str = "latent_pool",
    sysid_epochs: int = 0,
    sysid_batch_size: int = 256,
    sysid_lr: float = 3e-4,
    sysid_negative_count: int = 15,
    sysid_particle_count: int = 128,
    sysid_likelihood_scale: float = 0.35,
    progress_callback=None,
    **train_kwargs,
) -> CrawlerModelBundle:
    """Train the crawler-side representation stack and return it as one bundle."""
    (
        encoder,
        predictor,
        belief_aggregator,
        env_param_predictor,
        env_future_predictor,
        env_family_future_predictor,
        family_value_predictor,
        env_metric_projector,
        belief_message_projector,
        controller_context_projector,
        oracle_context_projector,
        controller_trust_predictor,
        env_param_normalizer,
        device,
    ) = train_encoder_predictor(
        windows=windows,
        z_dim=z_dim,
        action_vocab_size=action_vocab_size,
        progress_callback=progress_callback,
        **train_kwargs,
    )
    family_names = tuple(sorted({str(mode) for mode in np.asarray(windows["probe_mode"], dtype="U").tolist()}))
    sysid_model = None
    sysid_stats = None
    sysid_particles_raw = None
    sysid_trusted = False
    sysid_validation_metrics: dict[str, float] | None = None
    if str(belief_mode) == "particle_sysid" and int(sysid_epochs) > 0:
        sysid_result = train_probe_likelihood_model(
            windows=windows,
            action_vocab_size=action_vocab_size,
            epochs=int(sysid_epochs),
            batch_size=int(sysid_batch_size),
            lr=float(sysid_lr),
            negative_count=int(sysid_negative_count),
            hidden_dim=128,
            seed=0,
        )
        sysid_model = sysid_result.model
        sysid_stats = sysid_result.stats
        sysid_particles_raw = build_latin_hypercube_particles(
            sysid_stats,
            count=int(sysid_particle_count),
            seed=173,
        )
        sysid_trusted = bool(sysid_result.trusted)
        sysid_validation_metrics = dict(sysid_result.metrics)
        print(
            "sysid validation | "
            f"trusted={sysid_trusted} | "
            f"top1={sysid_validation_metrics.get('validation_top1', 0.0):.3f} | "
            f"margin={sysid_validation_metrics.get('validation_margin', 0.0):.3f} | "
            f"nll={sysid_validation_metrics.get('validation_nll', 0.0):.3f}"
        )

    belief_message_dim = (
        int(belief_message_projector.output_dim)
        if belief_message_projector is not None and hasattr(belief_message_projector, "output_dim")
        else int(z_dim)
    )
    controller_context_dim = int(2 * z_dim + 2)
    return CrawlerModelBundle(
        encoder=encoder,
        predictor=predictor,
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        env_future_predictor=env_future_predictor,
        env_family_future_predictor=env_family_future_predictor,
        family_value_predictor=family_value_predictor,
        env_metric_projector=env_metric_projector,
        belief_message_projector=belief_message_projector,
        controller_context_projector=controller_context_projector,
        oracle_context_projector=oracle_context_projector,
        controller_trust_predictor=controller_trust_predictor,
        device=device,
        z_dim=z_dim,
        window_size=window_size,
        action_vocab_size=action_vocab_size,
        belief_message_dim=belief_message_dim,
        controller_context_dim=controller_context_dim,
        family_names=family_names,
        env_param_normalizer_mean=sanitize_array(env_param_normalizer["mean"]),
        env_param_normalizer_std=sanitize_array(env_param_normalizer["std"]),
        belief_mode=str(belief_mode),
        sysid_model=sysid_model,
        sysid_stats=sysid_stats,
        sysid_particles_raw=None if sysid_particles_raw is None else sanitize_array(sysid_particles_raw),
        sysid_trusted=sysid_trusted,
        sysid_validation_metrics=sysid_validation_metrics,
        sysid_likelihood_scale=float(sysid_likelihood_scale),
    )


def load_crawler_bundle_from_checkpoint(
    *,
    checkpoint: dict,
    state_dim: int,
    action_vocab_size: int,
    device: torch.device,
) -> CrawlerModelBundle:
    """Rebuild a saved crawler bundle from one probe-policy checkpoint."""
    window_size = int(checkpoint["window_size"])
    z_dim = int(checkpoint["z_dim"])
    encoder = WorldEncoder(
        state_dim=state_dim,
        window_size=window_size,
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    predictor_ensemble_size = int(checkpoint.get("predictor_ensemble_size", 0))
    predictor = DeltaPredictorEnsemble(
        ensemble_size=predictor_ensemble_size,
        state_dim=state_dim,
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    predictor.eval()

    belief_aggregator = EnvBeliefAggregator(
        window_z_dim=z_dim,
        param_dim=int(checkpoint.get("env_param_dim", 5)),
        num_families=max(1, int(checkpoint.get("env_family_count", 8))),
    ).to(device)
    belief_aggregator.load_state_dict(checkpoint["belief_aggregator_state_dict"], strict=False)
    belief_aggregator.eval()

    env_param_predictor = EnvParamPredictorEnsemble(
        ensemble_size=int(checkpoint["env_param_predictor_ensemble_size"]),
        input_dim=z_dim,
        output_dim=int(checkpoint["env_param_dim"]),
    ).to(device)
    env_param_predictor.load_state_dict(checkpoint["env_param_predictor_state_dict"])
    env_param_predictor.eval()

    env_future_predictor = None
    if checkpoint.get("env_future_predictor_state_dict") is not None:
        env_future_predictor = OutcomePredictor(
            input_dim=z_dim,
            output_dim=int(checkpoint["env_future_summary_dim"]),
        ).to(device)
        env_future_predictor.load_state_dict(checkpoint["env_future_predictor_state_dict"])
        env_future_predictor.eval()

    env_family_future_predictor = None
    if checkpoint.get("env_family_future_predictor_state_dict") is not None:
        env_family_future_predictor = FamilyConditionedOutcomePredictor(
            input_dim=z_dim,
            num_families=int(checkpoint["env_family_count"]),
            output_dim=int(checkpoint["env_future_summary_dim"]),
        ).to(device)
        env_family_future_predictor.load_state_dict(checkpoint["env_family_future_predictor_state_dict"])
        env_family_future_predictor.eval()

    env_metric_projector = None
    if checkpoint.get("env_metric_projector_state_dict") is not None:
        env_metric_projector = ContrastiveProjector(
            input_dim=z_dim,
            output_dim=int(checkpoint.get("env_metric_dim", z_dim)),
        ).to(device)
        env_metric_projector.load_state_dict(checkpoint["env_metric_projector_state_dict"])
        env_metric_projector.eval()

    family_value_predictor = None
    if checkpoint.get("family_value_predictor_state_dict") is not None:
        family_value_predictor = FamilyConditionedValuePredictor(
            input_dim=int(checkpoint.get("family_value_input_dim", z_dim + 4)),
            num_families=max(1, int(checkpoint.get("env_family_count", 8))),
            output_dim=int(checkpoint.get("family_value_output_dim", 3)),
        ).to(device)
        family_value_predictor.load_state_dict(checkpoint["family_value_predictor_state_dict"])
        family_value_predictor.eval()

    belief_message_projector = None
    if checkpoint.get("belief_message_projector_state_dict") is not None:
        belief_message_projector = BeliefMessageProjector(
            input_dim=z_dim,
            output_dim=int(checkpoint.get("belief_message_dim", z_dim)),
        ).to(device)
        belief_message_projector.load_state_dict(checkpoint["belief_message_projector_state_dict"])
        belief_message_projector.eval()

    controller_context_projector = None
    if checkpoint.get("controller_context_projector_state_dict") is not None:
        controller_context_projector = ControllerBeliefContextProjector(
            input_dim=z_dim,
            mechanics_dim=int(checkpoint.get("controller_mechanics_dim", z_dim)),
            affordance_dim=int(checkpoint.get("controller_affordance_dim", z_dim)),
        ).to(device)
        controller_context_projector.load_state_dict(
            checkpoint["controller_context_projector_state_dict"]
        )
        controller_context_projector.eval()

    oracle_context_projector = None
    if checkpoint.get("oracle_context_projector_state_dict") is not None:
        oracle_context_projector = OracleControllerContextProjector(
            input_dim=int(checkpoint.get("env_param_dim", z_dim)),
            mechanics_dim=int(checkpoint.get("controller_mechanics_dim", z_dim)),
            affordance_dim=int(checkpoint.get("controller_affordance_dim", z_dim)),
        ).to(device)
        oracle_context_projector.load_state_dict(
            checkpoint["oracle_context_projector_state_dict"]
        )
        oracle_context_projector.eval()

    controller_trust_predictor = None
    if checkpoint.get("controller_trust_predictor_state_dict") is not None:
        controller_trust_predictor = ControllerTrustPredictor(
            input_dim=z_dim,
        ).to(device)
        controller_trust_predictor.load_state_dict(
            checkpoint["controller_trust_predictor_state_dict"]
        )
        controller_trust_predictor.eval()

    family_names = tuple(
        str(item)
        for item in np.asarray(
            checkpoint.get("probe_family_names", np.asarray([], dtype="U")),
            dtype="U",
        ).tolist()
    )
    belief_mode = str(checkpoint.get("belief_mode", "latent_pool"))
    sysid_stats = None
    sysid_model = None
    sysid_particles_raw = None
    sysid_feature_stats = checkpoint.get("sysid_feature_stats")
    if sysid_feature_stats is not None:
        sysid_stats = SysIdFeatureStats.from_dict(sysid_feature_stats)
    sysid_state_dict = checkpoint.get("sysid_model_state_dict")
    if sysid_stats is not None and sysid_state_dict is not None:
        sysid_model = ProbeLikelihoodModel(
            param_dim=int(sysid_stats.env_param_mean.shape[0]),
            query_dim=int(sysid_stats.query_mean.shape[0]),
            outcome_dim=int(sysid_stats.outcome_mean.shape[0]),
            num_families=len(sysid_stats.family_names),
            hidden_dim=128,
        ).to(device)
        sysid_model.load_state_dict(sysid_state_dict)
        sysid_model.eval()
    if checkpoint.get("sysid_particles_raw") is not None:
        sysid_particles_raw = sanitize_array(checkpoint.get("sysid_particles_raw"))
    sysid_validation_metrics = checkpoint.get("sysid_validation_metrics") or {}
    return CrawlerModelBundle(
        encoder=encoder,
        predictor=predictor,
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        env_future_predictor=env_future_predictor,
        env_family_future_predictor=env_family_future_predictor,
        family_value_predictor=family_value_predictor,
        env_metric_projector=env_metric_projector,
        belief_message_projector=belief_message_projector,
        controller_context_projector=controller_context_projector,
        oracle_context_projector=oracle_context_projector,
        controller_trust_predictor=controller_trust_predictor,
        device=device,
        z_dim=z_dim,
        window_size=window_size,
        action_vocab_size=action_vocab_size,
        belief_message_dim=int(checkpoint.get("belief_message_dim", z_dim)),
        controller_context_dim=int(checkpoint.get("controller_context_dim", 2 * z_dim + 2)),
        family_names=family_names,
        env_param_normalizer_mean=sanitize_array(
            checkpoint.get(
                "env_param_normalizer_mean",
                np.zeros((int(checkpoint.get("env_param_dim", z_dim)),), dtype=np.float32),
            )
        ),
        env_param_normalizer_std=sanitize_array(
            checkpoint.get(
                "env_param_normalizer_std",
                np.ones((int(checkpoint.get("env_param_dim", z_dim)),), dtype=np.float32),
            )
        ),
        belief_mode=belief_mode,
        sysid_model=sysid_model,
        sysid_stats=sysid_stats,
        sysid_particles_raw=sysid_particles_raw,
        sysid_trusted=bool(checkpoint.get("sysid_trusted", False)),
        sysid_validation_metrics={
            str(key): float(value)
            for key, value in dict(sysid_validation_metrics).items()
        },
        sysid_likelihood_scale=float(checkpoint.get("sysid_likelihood_scale", 0.35)),
    )
