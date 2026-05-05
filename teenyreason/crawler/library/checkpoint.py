"""Checkpoint loading for the RL-facing crawler library."""

from __future__ import annotations

import numpy as np
import torch

from ...models.belief import (
    BeliefMessageProjector,
    ContrastiveProjector,
    ControllerBeliefContextProjector,
    ControllerTrustPredictor,
    FamilyConditionedOutcomePredictor,
    FamilyConditionedValuePredictor,
    OracleControllerContextProjector,
    OutcomePredictor,
)
from ...models.envbelief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ...models.sysid import ProbeLikelihoodModel, SysIdFeatureStats
from ...cognition.representation import DeltaPredictorEnsemble, WorldEncoder
from .bundle import CrawlerModelBundle
from .helpers import sanitize_array

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
