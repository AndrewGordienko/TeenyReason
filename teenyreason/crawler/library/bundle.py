"""Crawler model bundle for the RL-facing crawler library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from ...models.envbelief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ...models.sysid import ProbeLikelihoodModel, SysIdFeatureStats, particle_payload_from_windows
from ...cognition.representation import DeltaPredictorEnsemble, WorldEncoder
from ..probes.latent import aggregate_env_belief, encode_window_posterior
from ..types import (
    MetricBelief,
    PredictiveBelief,
    UncertaintyEstimate,
)
from .beliefs import BeliefBuilderMixin
from .helpers import sanitize_array
from .scoring import ProbeScoringMixin


@dataclass
class CrawlerModelBundle(BeliefBuilderMixin, ProbeScoringMixin):
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
        payload["belief_source"] = np.asarray(["learned"], dtype="U")
        payload["solver_message_source"] = np.asarray(["learned"], dtype="U")
        payload["sysid_belief_available"] = np.asarray([0.0], dtype=np.float32)
        payload["learned_belief_available"] = np.asarray([1.0], dtype=np.float32)
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
        payload["belief_source"] = np.asarray(["sysid"], dtype="U")
        payload["solver_message_source"] = np.asarray(["sysid"], dtype="U")
        payload["sysid_belief_available"] = np.asarray([1.0], dtype=np.float32)
        payload["learned_belief_available"] = np.asarray([0.0], dtype=np.float32)
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
