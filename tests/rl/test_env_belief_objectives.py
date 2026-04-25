import unittest

import torch

from teenyreason.models.belief.belief_training_env_config import (
    EnvBeliefPhaseConfig,
    build_env_belief_phase_config,
    build_primary_env_loss_terms,
)


def _build_unit_primary_loss_terms(phase: EnvBeliefPhaseConfig) -> dict[str, torch.Tensor]:
    one = torch.tensor(1.0, dtype=torch.float32)
    return build_primary_env_loss_terms(
        phase=phase,
        physics_loss_weight=1.0,
        env_consistency_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        env_retrieval_loss_weight=1.0,
        env_param_loss=one,
        mechanics_posterior_loss=one,
        env_split_loss=one,
        env_future_loss=one,
        env_family_future_loss=one,
        env_leaveout_future_loss=one,
        env_leaveout_consistency_loss=one,
        env_split_contrastive_loss=one,
        env_retrieval_loss=one,
        env_retrieval_margin_loss=one,
        env_gap_ratio_loss=one,
        env_unit_gap_ratio_loss=one,
        env_unit_retrieval_margin_loss=one,
        env_metric_geometry_loss=one,
        env_mode_adversary_loss=one,
        uncertainty_calibration_loss=one,
        env_expression_loss=one,
        controller_mechanics_loss=one,
        controller_affordance_loss=one,
        controller_successor_loss=one,
        oracle_mechanics_loss=one,
        oracle_affordance_loss=one,
        controller_oracle_distill_loss=one,
        controller_trust_loss=one,
        env_gaussian_loss=one,
    )


class EnvBeliefObjectiveTests(unittest.TestCase):
    def test_primary_env_loss_terms_include_leaveout_consistency(self):
        phase = build_env_belief_phase_config(epoch_index=0, total_epochs=9)
        loss_terms = _build_unit_primary_loss_terms(phase)

        self.assertIn("env_leaveout_consistency", loss_terms)
        self.assertAlmostEqual(
            float(loss_terms["env_leaveout_consistency"].item()),
            float(phase.predictive_scale * 0.20),
        )

    def test_leakage_control_is_active_before_metric_phase(self):
        clean_phase = build_env_belief_phase_config(
            epoch_index=0,
            total_epochs=9,
            probe_leakage=0.02,
        )
        leaky_phase = build_env_belief_phase_config(
            epoch_index=0,
            total_epochs=9,
            probe_leakage=0.30,
        )

        clean_terms = _build_unit_primary_loss_terms(clean_phase)
        leaky_terms = _build_unit_primary_loss_terms(leaky_phase)

        self.assertEqual(clean_phase.metric_scale, 0.0)
        self.assertGreater(float(clean_terms["env_leakage_control"].item()), 0.0)
        self.assertGreater(
            float(leaky_terms["env_leakage_control"].item()),
            float(clean_terms["env_leakage_control"].item()),
        )
