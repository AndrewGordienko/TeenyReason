import unittest

import torch

from teenyreason.models.belief.objectives.losses import env_param_anchor_loss
from teenyreason.models.belief.training.env_config import (
    EnvBeliefPhaseConfig,
    build_env_belief_phase_config,
    build_primary_env_loss_terms,
    cap_primary_env_loss_terms,
)


def _build_unit_primary_loss_terms(
    phase: EnvBeliefPhaseConfig,
    *,
    representation_repair_mode: bool = False,
) -> dict[str, torch.Tensor]:
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
        raw_env_retrieval_loss=one,
        env_retrieval_margin_loss=one,
        raw_env_retrieval_margin_loss=one,
        env_gap_ratio_loss=one,
        raw_env_gap_ratio_loss=one,
        env_unit_gap_ratio_loss=one,
        env_unit_retrieval_margin_loss=one,
        env_metric_geometry_loss=one,
        env_spread_loss=one,
        raw_env_spread_loss=one,
        env_uniformity_loss=one,
        raw_env_uniformity_loss=one,
        env_vicreg_loss=one,
        raw_env_vicreg_loss=one,
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
        representation_repair_mode=representation_repair_mode,
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

    def test_split_retrieval_gate_starts_metric_pressure_early(self):
        clean_phase = build_env_belief_phase_config(
            epoch_index=0,
            total_epochs=9,
            split_retrieval_top1=0.40,
        )
        weak_phase = build_env_belief_phase_config(
            epoch_index=0,
            total_epochs=9,
            split_retrieval_top1=0.02,
        )

        clean_terms = _build_unit_primary_loss_terms(clean_phase)
        weak_terms = _build_unit_primary_loss_terms(weak_phase)

        self.assertEqual(clean_phase.metric_scale, 0.0)
        self.assertGreater(weak_phase.metric_scale, 0.0)
        self.assertEqual(weak_phase.metric_gate_reason, "split_retrieval")
        self.assertGreater(
            float(weak_terms["raw_env_retrieval"].item()),
            float(clean_terms["raw_env_retrieval"].item()),
        )
        self.assertGreater(float(weak_terms["env_spread"].item()), 0.0)

    def test_metric_phase_activates_anti_collapse_terms(self):
        phase = build_env_belief_phase_config(epoch_index=5, total_epochs=9, split_retrieval_top1=0.0)
        loss_terms = _build_unit_primary_loss_terms(phase)

        for name in (
            "env_spread",
            "raw_env_spread",
            "env_uniformity",
            "env_vicreg",
            "raw_env_vicreg",
            "raw_env_retrieval_margin",
        ):
            self.assertIn(name, loss_terms)
            self.assertGreater(float(loss_terms[name].item()), 0.0)

    def test_repair_mode_focuses_representation_terms(self):
        phase = build_env_belief_phase_config(
            epoch_index=5,
            total_epochs=9,
            split_retrieval_top1=0.0,
            probe_leakage=0.02,
        )
        loss_terms = _build_unit_primary_loss_terms(
            phase,
            representation_repair_mode=True,
        )

        for name in (
            "env_retrieval",
            "raw_env_retrieval",
            "env_retrieval_margin",
            "env_gap_ratio",
            "env_spread",
            "env_vicreg",
        ):
            self.assertGreater(float(loss_terms[name].item()), 0.0)

        for name in (
            "env_expression",
            "controller_mechanics",
            "controller_affordance",
            "controller_successor",
            "oracle_mechanics",
            "oracle_affordance",
            "controller_oracle_distill",
            "controller_trust",
            "env_gaussian",
            "env_leakage_control",
        ):
            self.assertEqual(float(loss_terms[name].item()), 0.0)

    def test_repair_cap_exempts_retrieval_pressure(self):
        loss_terms = {
            "env_retrieval": torch.tensor(100.0),
            "env_spread": torch.tensor(80.0),
            "physics": torch.tensor(100.0),
            "env_future": torch.tensor(1.0),
        }
        capped = cap_primary_env_loss_terms(
            loss_terms,
            max_term_fraction=0.35,
            exempt_names={"env_retrieval", "env_spread"},
        )

        self.assertEqual(float(capped["env_retrieval"].item()), 100.0)
        self.assertEqual(float(capped["env_spread"].item()), 80.0)
        self.assertLess(float(capped["physics"].item()), 100.0)

    def test_env_param_anchor_loss_pushes_collapsed_embeddings(self):
        embeddings = torch.zeros((5, 8), dtype=torch.float32, requires_grad=True)
        env_params = torch.tensor(
            [
                [-1.0, 0.0],
                [-0.5, 0.5],
                [0.0, -0.5],
                [0.5, 1.0],
                [1.0, -1.0],
            ],
            dtype=torch.float32,
        )

        loss = env_param_anchor_loss(embeddings, env_params)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.item()), 0.0)
        self.assertGreater(float(embeddings.grad.abs().sum().item()), 0.0)

    def test_env_param_anchor_loss_penalizes_constant_offset_cloud(self):
        env_params = torch.tensor(
            [
                [-1.0, 0.0],
                [-0.5, 0.5],
                [0.0, -0.5],
                [0.5, 1.0],
                [1.0, -1.0],
            ],
            dtype=torch.float32,
        )
        collapsed = torch.full((5, 8), 0.25, dtype=torch.float32, requires_grad=True)
        varied = torch.cat([env_params, env_params, env_params, env_params], dim=1).detach()

        collapsed_loss = env_param_anchor_loss(collapsed, env_params)
        varied_loss = env_param_anchor_loss(varied, env_params)
        collapsed_loss.backward()

        self.assertGreater(float(collapsed_loss.item()), float(varied_loss.item()))
        self.assertGreater(float(collapsed.grad.abs().sum().item()), 0.0)
