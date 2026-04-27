import unittest
from types import SimpleNamespace

import numpy as np
import torch

from teenyreason.app.benchmark import benchmark_profile_flags, classify_probe_run
from teenyreason.crawler import MetricBelief, PredictiveBelief, UncertaintyEstimate
from teenyreason.crawler.library import CrawlerModelBundle
from teenyreason.models.belief.belief_training_env_config import (
    build_env_belief_phase_config,
)
from teenyreason.models.envbelief.env_belief_subsets import (
    build_cross_family_subset_masks,
    build_env_subset_masks,
    build_split_source_mask,
    build_support_budget_mask,
)
from teenyreason.rl.probe_policy.budget import (
    choose_fair_probe_family,
    choose_next_probe_family,
    choose_quota_probe_family,
    choose_seed_probe_family,
    desired_family_coverage_budget,
    minimum_family_coverage_ratio,
    selectable_unseen_active_probe_families,
    should_continue_probing_adaptive,
    should_require_seed_probe_family,
    should_stop_probing_fair,
)
from teenyreason.rl.probe_policy.handoff_diagnostics import (
    compute_online_geometry_diagnostics,
)
from teenyreason.rl.probe_policy.messages import solver_belief_input_from_message
from teenyreason.rl.probe_policy.reporting import (
    average_family_scalar_counter,
    average_family_score_counter,
    default_family_metric_counter,
    default_family_score_counter,
    update_family_scalar_counter,
    update_family_score_counter,
)
from teenyreason.probe.probe_latent import probe_group_ids_from_families


class ProbeLogicTests(unittest.TestCase):
    class _StubCrawlerBundle:
        def __init__(self, family_names):
            self.family_names = tuple(family_names)

    def test_solver_belief_input_appends_uncertainty_scalar(self):
        message = np.asarray([0.5, -0.25, 1.0], dtype=np.float32)
        belief_input = solver_belief_input_from_message(message, 0.75)
        np.testing.assert_allclose(
            belief_input,
            np.asarray([0.5, -0.25, 1.0, 1.0, 0.75], dtype=np.float32),
        )

    def test_probe_group_ids_use_trained_family_order(self):
        ids = probe_group_ids_from_families(
            ["chirp", "passive_decay", "unknown_family", None],
            family_names=("passive_decay", "chirp"),
        )

        np.testing.assert_array_equal(
            ids,
            np.asarray([1, 0, -1, -1], dtype=np.int64),
        )

    def test_probe_run_is_not_latent_win_when_fair_mode_never_enabled_expression(self):
        classification = classify_probe_run(
            baseline_episode=120,
            baseline_steps=9000,
            probe_episode=80,
            probe_steps=11000,
            probe_no_expression_episode=140,
            probe_no_expression_steps=16000,
            probe_fair_ready_handoff_fraction=0.0,
            probe_fair_expression_enabled_fraction=0.0,
        )

        self.assertEqual(classification, "protocol_win")

    def test_probe_run_requires_noexpr_arm_before_latent_win_label(self):
        classification = classify_probe_run(
            baseline_episode=120,
            baseline_steps=9000,
            probe_episode=80,
            probe_steps=11000,
            probe_no_expression_episode=None,
            probe_no_expression_steps=None,
            probe_env_expression_delta=12.0,
            probe_fair_ready_handoff_fraction=0.4,
            probe_fair_expression_enabled_fraction=0.2,
        )

        self.assertEqual(classification, "protocol_win")

    def test_fast_profile_runs_probe_no_expression_training(self):
        flags = benchmark_profile_flags("fast")

        self.assertTrue(flags["run_probe_no_expression_training"])

    def test_family_scorer_floors_unseen_active_future_estimate(self):
        bundle = CrawlerModelBundle(
            encoder=None,
            predictor=None,
            belief_aggregator=None,
            env_param_predictor=None,
            env_future_predictor=None,
            env_family_future_predictor=None,
            family_value_predictor=None,
            env_metric_projector=None,
            belief_message_projector=None,
            controller_context_projector=None,
            device=torch.device("cpu"),
            z_dim=4,
            window_size=4,
            action_vocab_size=3,
            belief_message_dim=4,
            controller_context_dim=10,
            family_names=("passive_decay", "chirp"),
        )
        predictive_belief = SimpleNamespace(
            mean_raw=np.zeros((4,), dtype=np.float32),
            env_param_std=np.ones((2,), dtype=np.float32) * 0.2,
            future_probe_error=1.0,
            support_diversity_ratio=0.9,
            metadata={
                "mechanics_posterior_std": np.ones((2,), dtype=np.float32) * 0.2,
                "mechanics_posterior_logvar": np.asarray([], dtype=np.float32),
                "mechanics_posterior_entropy": 0.6,
                "split_latent_disagreement": 0.1,
                "factor_std": np.asarray([], dtype=np.float32),
            },
        )
        uncertainty = SimpleNamespace(scalar=0.2)

        gains = bundle.score_probe_families(
            predictive_belief,
            uncertainty,
            family_counts={"passive_decay": 1, "chirp": 0},
            global_family_counts={"passive_decay": 3, "chirp": 0},
            family_error_history={"chirp": 0.0},
            use_learned_family_value=False,
        )

        self.assertAlmostEqual(gains["chirp"]["raw_future_error_estimate"], 0.0, places=6)
        self.assertAlmostEqual(gains["chirp"]["future_error_estimate"], 0.35, places=6)
        self.assertAlmostEqual(
            gains["chirp"]["raw_predicted_future_error_reduction"],
            0.0,
            places=6,
        )
        self.assertGreater(
            gains["chirp"]["predicted_future_error_reduction"],
            0.0,
        )
        self.assertAlmostEqual(
            gains["chirp"]["future_gain_for_choice"],
            0.30 * gains["chirp"]["future_error_estimate"],
            places=6,
        )

    def test_oracle_controller_context_is_deterministic_and_labeled(self):
        bundle = CrawlerModelBundle(
            encoder=None,
            predictor=None,
            belief_aggregator=None,
            env_param_predictor=None,
            env_future_predictor=None,
            env_family_future_predictor=None,
            family_value_predictor=None,
            env_metric_projector=None,
            belief_message_projector=None,
            controller_context_projector=None,
            oracle_context_projector=None,
            controller_trust_predictor=None,
            device=torch.device("cpu"),
            z_dim=4,
            window_size=4,
            action_vocab_size=3,
            belief_message_dim=4,
            controller_context_dim=10,
            family_names=("passive_decay", "chirp"),
            env_param_normalizer_mean=np.asarray([1.0, 2.0], dtype=np.float32),
            env_param_normalizer_std=np.asarray([2.0, 4.0], dtype=np.float32),
        )

        context_a = bundle.build_oracle_controller_context(np.asarray([3.0, 6.0], dtype=np.float32))
        context_b = bundle.build_oracle_controller_context(np.asarray([3.0, 6.0], dtype=np.float32))

        np.testing.assert_allclose(context_a.vector, context_b.vector)
        self.assertEqual(context_a.metadata["source_kind"], "oracle")
        self.assertAlmostEqual(context_a.confidence, 1.0)
        self.assertAlmostEqual(context_a.uncertainty_scalar, 0.0)

    def test_learned_controller_context_confidence_is_not_env_expression_confidence(self):
        bundle = CrawlerModelBundle(
            encoder=None,
            predictor=None,
            belief_aggregator=None,
            env_param_predictor=None,
            env_future_predictor=None,
            env_family_future_predictor=None,
            family_value_predictor=None,
            env_metric_projector=None,
            belief_message_projector=None,
            controller_context_projector=None,
            oracle_context_projector=None,
            controller_trust_predictor=None,
            device=torch.device("cpu"),
            z_dim=4,
            window_size=4,
            action_vocab_size=3,
            belief_message_dim=4,
            controller_context_dim=10,
            family_names=("passive_decay", "chirp"),
        )
        predictive_belief = PredictiveBelief(
            mean_raw=np.asarray([0.2, -0.1, 0.4, 0.3], dtype=np.float32),
            mean_unit=np.asarray([0.2, -0.1, 0.4, 0.3], dtype=np.float32),
            logvar=np.zeros((4,), dtype=np.float32),
            view_spread=np.zeros((4,), dtype=np.float32),
            env_param_mean=np.zeros((2,), dtype=np.float32),
            env_param_std=np.ones((2,), dtype=np.float32) * 0.2,
            future_probe_error=0.4,
            support_count=3,
            support_diversity_ratio=0.85,
            metadata={},
        )
        metric_belief = MetricBelief(
            mean_raw=np.asarray([0.2, -0.1, 0.4, 0.3], dtype=np.float32),
            mean_unit=np.asarray([0.2, -0.1, 0.4, 0.3], dtype=np.float32),
            split_mean_a=np.asarray([0.2, -0.1, 0.4, 0.3], dtype=np.float32),
            split_mean_b=np.asarray([0.2, -0.1, 0.4, 0.3], dtype=np.float32),
            nearest_between_distance=0.4,
            gap_ratio=0.3,
            metadata={},
        )
        uncertainty = UncertaintyEstimate(
            vector=np.asarray([0.2], dtype=np.float32),
            scalar=0.2,
            feature_names=("uncertainty",),
            feature_weights=np.asarray([1.0], dtype=np.float32),
            metadata={},
        )

        context = bundle.build_controller_context(
            predictive_belief,
            metric_belief,
            uncertainty,
            env_expression=SimpleNamespace(confidence=0.91, ready=False, metadata={}),
        )

        self.assertEqual(context.metadata["source_kind"], "learned")
        self.assertNotAlmostEqual(context.confidence, 0.91, places=4)
        self.assertAlmostEqual(context.metadata["env_expression_confidence"], 0.91, places=6)

    def test_balanced_split_builder_duplicates_family_coverage_across_halves(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.float32)
        group_ids = torch.tensor([[0, 0, 1, 1, 2, 3]], dtype=torch.int64)

        mask_a, mask_b = build_env_subset_masks(
            mask,
            group_ids=group_ids,
            generator=torch.Generator().manual_seed(0),
        )

        idx_a = set(torch.nonzero(mask_a[0] > 0, as_tuple=False).squeeze(-1).tolist())
        idx_b = set(torch.nonzero(mask_b[0] > 0, as_tuple=False).squeeze(-1).tolist())
        groups_a = set(group_ids[0, sorted(idx_a)].tolist())
        groups_b = set(group_ids[0, sorted(idx_b)].tolist())

        self.assertTrue(idx_a.isdisjoint(idx_b))
        self.assertLessEqual(abs(len(idx_a) - len(idx_b)), 1)
        self.assertTrue({0, 1}.issubset(groups_a))
        self.assertTrue({0, 1}.issubset(groups_b))

    def test_cross_family_split_builder_keeps_probe_families_disjoint(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.float32)
        group_ids = torch.tensor([[0, 0, 1, 1, 2, 3]], dtype=torch.int64)

        mask_a, mask_b = build_cross_family_subset_masks(
            mask,
            group_ids=group_ids,
            generator=torch.Generator().manual_seed(0),
        )

        idx_a = torch.nonzero(mask_a[0] > 0, as_tuple=False).squeeze(-1)
        idx_b = torch.nonzero(mask_b[0] > 0, as_tuple=False).squeeze(-1)
        groups_a = set(group_ids[0, idx_a].tolist())
        groups_b = set(group_ids[0, idx_b].tolist())

        self.assertTrue(set(idx_a.tolist()).isdisjoint(set(idx_b.tolist())))
        self.assertTrue(groups_a.isdisjoint(groups_b))
        self.assertGreater(len(groups_a), 0)
        self.assertGreater(len(groups_b), 0)

    def test_balanced_split_builder_preserves_single_view_fallback(self):
        mask = torch.tensor([[1]], dtype=torch.float32)
        group_ids = torch.tensor([[7]], dtype=torch.int64)

        mask_a, mask_b = build_env_subset_masks(
            mask,
            group_ids=group_ids,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertEqual(float(mask_a[0, 0]), 1.0)
        self.assertEqual(float(mask_b[0, 0]), 1.0)

    def test_support_budget_mask_spends_canonical_budget_once(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.float32)
        group_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int64)

        support_mask = build_support_budget_mask(
            mask,
            support_size=2,
            subset_count=3,
            group_ids=group_ids,
            generator=torch.Generator().manual_seed(0),
        )

        chosen = torch.nonzero(support_mask[0] > 0, as_tuple=False).squeeze(-1)
        chosen_groups = set(group_ids[0, chosen].tolist())

        self.assertEqual(int(chosen.numel()), 2)
        self.assertEqual(chosen.tolist(), [0, 1])
        self.assertEqual(len(chosen_groups), 2)

    def test_support_budget_mask_prefers_early_unique_probe_families(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.float32)
        group_ids = torch.tensor([[2, 2, 5, 3, 4, 5]], dtype=torch.int64)

        support_mask = build_support_budget_mask(
            mask,
            support_size=3,
            subset_count=1,
            group_ids=group_ids,
            generator=torch.Generator().manual_seed(99),
        )

        chosen = torch.nonzero(support_mask[0] > 0, as_tuple=False).squeeze(-1)

        self.assertEqual(chosen.tolist(), [0, 2, 3])

    def test_split_source_mask_uses_heldout_views_when_available(self):
        mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.float32)
        support_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.float32)

        split_source = build_split_source_mask(mask, support_mask)

        self.assertEqual(split_source[0].tolist(), [1, 1, 1, 1, 1, 1])
        self.assertEqual(split_source[1].tolist(), [1, 1, 1, 1, 0, 0])

    def test_fair_probe_stops_early_when_expression_is_ready(self):
        should_stop, reason = should_stop_probing_fair(
            probe_count=1,
            min_seed_support=1,
            max_probe_episodes=5,
            uncertainty_scalar=0.08,
            uncertainty_probe_threshold=0.10,
            posterior_entropy=0.30,
            best_expected_gain=0.20,
            best_entropy_reduction=0.10,
            best_hypothesis_separation=0.10,
            best_value_per_probe_step=0.18,
            future_probe_error=0.10,
            support_diversity_ratio=0.75,
            has_selectable_family=True,
            expression_ready=True,
        )
        self.assertTrue(should_stop)
        self.assertEqual(reason, "expression_ready")

    def test_live_geometry_diagnostics_use_balanced_splits_and_leaveout_views(self):
        posterior_views = [
            (
                np.asarray([0.10, 0.020], dtype=np.float32),
                np.zeros((2,), dtype=np.float32),
            ),
            (
                np.asarray([0.12, 0.020], dtype=np.float32),
                np.zeros((2,), dtype=np.float32),
            ),
            (
                np.asarray([0.11, 0.021], dtype=np.float32),
                np.zeros((2,), dtype=np.float32),
            ),
            (
                np.asarray([0.13, 0.019], dtype=np.float32),
                np.zeros((2,), dtype=np.float32),
            ),
        ]
        diagnostics = compute_online_geometry_diagnostics(
            posterior_views=posterior_views,
            probe_families=np.asarray(
                ["chirp", "chirp", "cart_brake", "cart_brake"],
                dtype="U",
            ),
            aggregate_latent_fn=lambda views, _group_ids: np.mean(
                np.stack([item[0] for item in views], axis=0),
                axis=0,
                keepdims=True,
            ).astype(np.float32),
        )

        self.assertTrue(diagnostics["online_geometry_complete"])
        self.assertEqual(diagnostics["online_observed_family_count"], 2)
        self.assertGreater(diagnostics["online_split_latent_disagreement"], 0.0)
        self.assertGreater(diagnostics["online_leaveout_shift"], 0.0)
        self.assertGreater(diagnostics["online_subset_stability"], 0.0)
        self.assertLessEqual(diagnostics["online_subset_stability"], 1.0)

    def test_live_geometry_diagnostics_fallback_is_conservative_when_support_is_too_small(self):
        diagnostics = compute_online_geometry_diagnostics(
            posterior_views=[
                (
                    np.asarray([0.10, 0.020], dtype=np.float32),
                    np.zeros((2,), dtype=np.float32),
                )
            ],
            probe_families=np.asarray(["chirp"], dtype="U"),
            aggregate_latent_fn=lambda views, _group_ids: np.mean(
                np.stack([item[0] for item in views], axis=0),
                axis=0,
                keepdims=True,
            ).astype(np.float32),
        )

        self.assertFalse(diagnostics["online_geometry_complete"])
        self.assertEqual(diagnostics["online_observed_family_count"], 1)
        self.assertEqual(diagnostics["online_subset_stability"], 0.0)

    def test_fair_probe_first_family_comes_from_active_shortlist(self):
        chosen = choose_fair_probe_family(
            family_names=("passive_decay", "chirp", "impulse_left"),
            expected_family_gain={},
            family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 0},
            probe_count=0,
        )

        self.assertEqual(chosen, "chirp")

    def test_fair_probe_first_family_avoids_boundary_push_collapse_when_specialist_is_close(self):
        chosen = choose_fair_probe_family(
            family_names=("boundary_push", "chirp", "cart_brake"),
            expected_family_gain={
                "boundary_push": {
                    "predicted_marginal_value": 0.60,
                    "value_per_probe_step": 0.60,
                    "selection_score": 0.60,
                    "score": 0.60,
                    "predicted_entropy_reduction": 0.20,
                    "predicted_hypothesis_separation": 0.05,
                    "future_gain_for_choice": 0.70,
                    "estimated_probe_cost": 1.00,
                },
                "chirp": {
                    "predicted_marginal_value": 0.58,
                    "value_per_probe_step": 0.58,
                    "selection_score": 0.58,
                    "score": 0.58,
                    "predicted_entropy_reduction": 0.40,
                    "predicted_hypothesis_separation": 0.20,
                    "future_gain_for_choice": 0.55,
                    "estimated_probe_cost": 0.90,
                },
                "cart_brake": {
                    "predicted_marginal_value": 0.56,
                    "value_per_probe_step": 0.56,
                    "selection_score": 0.56,
                    "score": 0.56,
                    "predicted_entropy_reduction": 0.28,
                    "predicted_hypothesis_separation": 0.18,
                    "future_gain_for_choice": 0.50,
                    "estimated_probe_cost": 1.05,
                },
            },
            family_counts={"boundary_push": 0, "chirp": 0, "cart_brake": 0},
            probe_count=0,
        )

        self.assertEqual(chosen, "chirp")

    def test_fair_probe_second_family_prioritizes_future_gain_from_a_different_family(self):
        chosen = choose_fair_probe_family(
            family_names=("passive_decay", "chirp", "impulse_left", "boundary_push"),
            expected_family_gain={
                "chirp": {
                    "predicted_entropy_reduction": 0.30,
                    "predicted_future_error_reduction": 0.35,
                    "future_gain_for_choice": 0.35,
                    "predicted_hypothesis_separation": 0.10,
                    "estimated_probe_cost": 0.90,
                },
                "impulse_left": {
                    "predicted_entropy_reduction": 0.30,
                    "predicted_future_error_reduction": 0.25,
                    "future_gain_for_choice": 0.25,
                    "predicted_hypothesis_separation": 0.80,
                    "estimated_probe_cost": 0.70,
                },
                "boundary_push": {
                    "predicted_entropy_reduction": 0.20,
                    "predicted_future_error_reduction": 0.90,
                    "future_gain_for_choice": 0.90,
                    "predicted_hypothesis_separation": 0.90,
                    "estimated_probe_cost": 0.60,
                },
            },
            family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 0, "boundary_push": 1},
            probe_count=1,
        )

        self.assertEqual(chosen, "chirp")

    def test_fair_second_probe_consumes_particle_expected_gain_rows(self):
        chosen = choose_fair_probe_family(
            family_names=("passive_decay", "weak_new", "strong_new", "seen_family"),
            expected_family_gain={
                "weak_new": {
                    "predicted_mechanics_reduction": 0.16,
                    "predicted_entropy_reduction": 0.18,
                    "predicted_hypothesis_separation": 0.07,
                    "future_gain_for_choice": 0.18,
                    "predicted_marginal_value": 0.20,
                    "value_per_probe_step": 0.20,
                    "score": 0.20,
                    "selection_score": 0.20,
                    "estimated_probe_cost": 1.00,
                },
                "strong_new": {
                    "predicted_mechanics_reduction": 0.44,
                    "predicted_entropy_reduction": 0.46,
                    "predicted_hypothesis_separation": 0.21,
                    "future_gain_for_choice": 0.46,
                    "predicted_marginal_value": 0.48,
                    "value_per_probe_step": 0.48,
                    "score": 0.48,
                    "selection_score": 0.48,
                    "estimated_probe_cost": 1.00,
                },
            },
            family_counts={
                "passive_decay": 0,
                "weak_new": 0,
                "strong_new": 0,
                "seen_family": 1,
            },
            probe_count=1,
            global_family_counts={"weak_new": 0, "strong_new": 8},
        )

        self.assertEqual(chosen, "strong_new")

    def test_fair_probe_second_family_avoids_globally_overused_weak_family(self):
        chosen = choose_fair_probe_family(
            family_names=("center", "neg_1", "pos_1"),
            expected_family_gain={
                "center": {
                    "predicted_entropy_reduction": 0.32,
                    "predicted_future_error_reduction": 0.34,
                    "future_gain_for_choice": 0.34,
                    "predicted_hypothesis_separation": 0.05,
                    "estimated_probe_cost": 1.00,
                    "selection_score": 0.08,
                },
                "neg_1": {
                    "predicted_entropy_reduction": 0.30,
                    "predicted_future_error_reduction": 0.33,
                    "future_gain_for_choice": 0.33,
                    "predicted_hypothesis_separation": 0.05,
                    "estimated_probe_cost": 1.00,
                    "selection_score": 0.09,
                },
                "pos_1": {
                    "predicted_entropy_reduction": 0.28,
                    "predicted_future_error_reduction": 0.22,
                    "future_gain_for_choice": 0.22,
                    "predicted_hypothesis_separation": 0.04,
                    "estimated_probe_cost": 1.00,
                    "selection_score": 0.06,
                },
            },
            family_counts={"center": 0, "neg_1": 0, "pos_1": 1},
            probe_count=1,
            global_family_counts={"center": 40, "neg_1": 4, "pos_1": 4},
            family_realized_gain_history={"center": 0.0, "neg_1": 0.08, "pos_1": 0.06},
            recent_families=("center",),
        )

        self.assertEqual(chosen, "neg_1")

    def test_fair_probe_first_family_prefers_underused_scalar_bucket_when_values_are_close(self):
        chosen = choose_fair_probe_family(
            family_names=("neg_1", "neg_2", "pos_1"),
            expected_family_gain={
                "neg_1": {
                    "predicted_marginal_value": 0.42,
                    "value_per_probe_step": 0.42,
                    "selection_score": 0.42,
                    "score": 0.42,
                    "predicted_entropy_reduction": 0.28,
                    "predicted_hypothesis_separation": 0.10,
                    "future_gain_for_choice": 0.42,
                    "estimated_probe_cost": 1.0,
                },
                "neg_2": {
                    "predicted_marginal_value": 0.40,
                    "value_per_probe_step": 0.40,
                    "selection_score": 0.40,
                    "score": 0.40,
                    "predicted_entropy_reduction": 0.27,
                    "predicted_hypothesis_separation": 0.10,
                    "future_gain_for_choice": 0.40,
                    "estimated_probe_cost": 1.0,
                },
                "pos_1": {
                    "predicted_marginal_value": 0.39,
                    "value_per_probe_step": 0.39,
                    "selection_score": 0.39,
                    "score": 0.39,
                    "predicted_entropy_reduction": 0.27,
                    "predicted_hypothesis_separation": 0.10,
                    "future_gain_for_choice": 0.39,
                    "estimated_probe_cost": 1.0,
                },
            },
            family_counts={"neg_1": 0, "neg_2": 0, "pos_1": 0},
            probe_count=0,
            global_family_counts={"neg_1": 20, "neg_2": 18, "pos_1": 1},
        )

        self.assertEqual(chosen, "pos_1")

    def test_fair_probe_first_family_skips_center_when_scalar_directional_probes_exist(self):
        chosen = choose_fair_probe_family(
            family_names=("center", "neg_1", "pos_1"),
            expected_family_gain={
                "center": {
                    "predicted_marginal_value": 1.00,
                    "value_per_probe_step": 1.00,
                    "selection_score": 1.00,
                    "score": 1.00,
                    "predicted_entropy_reduction": 0.80,
                    "predicted_hypothesis_separation": 0.30,
                    "future_gain_for_choice": 1.00,
                    "estimated_probe_cost": 1.0,
                },
                "neg_1": {
                    "predicted_marginal_value": 0.30,
                    "value_per_probe_step": 0.30,
                    "selection_score": 0.30,
                    "score": 0.30,
                    "predicted_entropy_reduction": 0.30,
                    "predicted_hypothesis_separation": 0.10,
                    "future_gain_for_choice": 0.30,
                    "estimated_probe_cost": 1.0,
                },
                "pos_1": {
                    "predicted_marginal_value": 0.25,
                    "value_per_probe_step": 0.25,
                    "selection_score": 0.25,
                    "score": 0.25,
                    "predicted_entropy_reduction": 0.25,
                    "predicted_hypothesis_separation": 0.08,
                    "future_gain_for_choice": 0.25,
                    "estimated_probe_cost": 1.0,
                },
            },
            family_counts={"center": 0, "neg_1": 0, "pos_1": 0},
            probe_count=0,
        )

        self.assertEqual(chosen, "neg_1")

    def test_fair_probe_second_family_prefers_opposite_scalar_bucket_after_positive_probe(self):
        chosen = choose_fair_probe_family(
            family_names=("center", "neg_1", "pos_2"),
            expected_family_gain={
                "center": {
                    "predicted_entropy_reduction": 0.35,
                    "predicted_future_error_reduction": 0.36,
                    "future_gain_for_choice": 0.36,
                    "predicted_split_reduction": 0.20,
                    "predicted_mechanics_reduction": 0.18,
                    "predicted_hypothesis_separation": 0.06,
                    "estimated_probe_cost": 1.0,
                    "selection_score": 0.08,
                },
                "neg_1": {
                    "predicted_entropy_reduction": 0.33,
                    "predicted_future_error_reduction": 0.33,
                    "future_gain_for_choice": 0.33,
                    "predicted_split_reduction": 0.24,
                    "predicted_mechanics_reduction": 0.22,
                    "predicted_hypothesis_separation": 0.08,
                    "estimated_probe_cost": 1.0,
                    "selection_score": 0.08,
                },
                "pos_2": {
                    "predicted_entropy_reduction": 0.30,
                    "predicted_future_error_reduction": 0.30,
                    "future_gain_for_choice": 0.30,
                    "predicted_split_reduction": 0.18,
                    "predicted_mechanics_reduction": 0.16,
                    "predicted_hypothesis_separation": 0.05,
                    "estimated_probe_cost": 1.0,
                    "selection_score": 0.07,
                },
            },
            family_counts={"center": 0, "neg_1": 0, "pos_2": 1},
            probe_count=1,
            global_family_counts={"center": 3, "neg_1": 3, "pos_2": 12},
            recent_families=("pos_2",),
        )

        self.assertEqual(chosen, "neg_1")

    def test_fair_probe_second_family_uses_directional_before_center(self):
        chosen = choose_fair_probe_family(
            family_names=("center", "neg_1", "pos_1"),
            expected_family_gain={
                "center": {
                    "predicted_entropy_reduction": 0.90,
                    "predicted_future_error_reduction": 0.90,
                    "future_gain_for_choice": 0.90,
                    "predicted_split_reduction": 0.40,
                    "predicted_mechanics_reduction": 0.40,
                    "predicted_hypothesis_separation": 0.20,
                    "estimated_probe_cost": 1.0,
                    "selection_score": 0.90,
                },
                "neg_1": {
                    "predicted_entropy_reduction": 0.30,
                    "predicted_future_error_reduction": 0.30,
                    "future_gain_for_choice": 0.30,
                    "predicted_split_reduction": 0.20,
                    "predicted_mechanics_reduction": 0.20,
                    "predicted_hypothesis_separation": 0.10,
                    "estimated_probe_cost": 1.0,
                    "selection_score": 0.30,
                },
            },
            family_counts={"center": 0, "neg_1": 0, "pos_1": 1},
            probe_count=1,
        )

        self.assertEqual(chosen, "neg_1")

    def test_phase_config_holds_controller_when_split_retrieval_is_still_flat(self):
        phase = build_env_belief_phase_config(
            epoch_index=20,
            total_epochs=30,
            probe_leakage=0.02,
            split_retrieval_top1=0.0,
        )

        self.assertEqual(phase.name, "phase_c_controller")
        self.assertGreaterEqual(phase.metric_scale, 0.45)
        self.assertLessEqual(phase.controller_scale, 0.35)
        self.assertEqual(phase.metric_gate_reason, "split_retrieval")

    def test_phase_config_caps_expression_and_controller_when_probe_leakage_is_high(self):
        phase = build_env_belief_phase_config(
            epoch_index=29,
            total_epochs=30,
            probe_leakage=0.55,
            split_retrieval_top1=0.22,
        )

        self.assertEqual(phase.name, "phase_c_controller")
        self.assertTrue(phase.metric_gate_active)
        self.assertEqual(phase.metric_gate_reason, "probe_leakage")
        self.assertLessEqual(phase.metric_scale, 0.30)
        self.assertLessEqual(phase.env_expression_scale, 0.70)
        self.assertLessEqual(phase.controller_scale, 0.30)

    def test_fair_probe_continues_after_first_probe_when_expression_is_not_ready(self):
        should_stop, reason = should_stop_probing_fair(
            probe_count=1,
            min_seed_support=1,
            max_probe_episodes=5,
            uncertainty_scalar=0.18,
            uncertainty_probe_threshold=0.10,
            posterior_entropy=0.80,
            best_expected_gain=0.25,
            best_entropy_reduction=0.18,
            best_hypothesis_separation=0.22,
            best_value_per_probe_step=0.25,
            future_probe_error=0.30,
            support_diversity_ratio=0.75,
            has_selectable_family=True,
            expression_ready=False,
        )
        self.assertFalse(should_stop)
        self.assertIsNone(reason)

    def test_fair_probe_does_not_stop_on_raw_ready_without_strict_fair_gate(self):
        should_stop, reason = should_stop_probing_fair(
            probe_count=1,
            min_seed_support=1,
            max_probe_episodes=5,
            uncertainty_scalar=0.18,
            uncertainty_probe_threshold=0.10,
            posterior_entropy=0.80,
            best_expected_gain=0.25,
            best_entropy_reduction=0.18,
            best_hypothesis_separation=0.22,
            best_value_per_probe_step=0.25,
            future_probe_error=0.30,
            support_diversity_ratio=0.75,
            has_selectable_family=True,
            fair_stop_ready=False,
            expression_ready=True,
        )

        self.assertFalse(should_stop)
        self.assertIsNone(reason)

    def test_fair_probe_hands_off_after_second_probe_even_when_not_ready(self):
        should_stop, reason = should_stop_probing_fair(
            probe_count=2,
            min_seed_support=1,
            max_probe_episodes=5,
            uncertainty_scalar=0.22,
            uncertainty_probe_threshold=0.10,
            posterior_entropy=0.85,
            best_expected_gain=0.43,
            best_entropy_reduction=0.52,
            best_hypothesis_separation=0.02,
            best_value_per_probe_step=0.26,
            best_marginal_value=0.18,
            best_selection_score=0.43,
            best_realized_gain=0.10,
            future_probe_error=0.68,
            support_diversity_ratio=0.80,
            family_coverage_ratio=1.0,
            min_family_coverage_ratio=minimum_family_coverage_ratio(
                family_coverage_budget=2,
                min_seed_support=1,
            ),
            has_selectable_family=True,
            expression_ready=False,
        )
        self.assertTrue(should_stop)
        self.assertEqual(reason, "fair_two_probe_handoff")

    def test_fair_probe_hands_off_when_no_second_family_remains(self):
        should_stop, reason = should_stop_probing_fair(
            probe_count=1,
            min_seed_support=1,
            max_probe_episodes=5,
            uncertainty_scalar=0.22,
            uncertainty_probe_threshold=0.10,
            posterior_entropy=0.70,
            best_expected_gain=0.20,
            best_entropy_reduction=0.10,
            best_hypothesis_separation=0.12,
            best_value_per_probe_step=0.18,
            best_marginal_value=0.12,
            best_selection_score=0.14,
            best_realized_gain=0.05,
            future_probe_error=0.25,
            support_diversity_ratio=0.55,
            family_coverage_ratio=0.50,
            min_family_coverage_ratio=0.75,
            has_selectable_family=False,
            expression_ready=False,
        )
        self.assertTrue(should_stop)
        self.assertEqual(reason, "fair_two_probe_handoff")

    def test_seed_probe_family_prefers_unseen_active_family_before_passive(self):
        chosen = choose_seed_probe_family(
            ("passive_decay", "chirp", "impulse_left"),
            family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 0},
            expected_family_gain={
                "passive_decay": {"predicted_marginal_value": 0.20},
                "chirp": {"predicted_marginal_value": 0.10},
                "impulse_left": {"predicted_marginal_value": 0.05},
            },
            global_family_counts={"passive_decay": 10, "chirp": 1, "impulse_left": 1},
        )
        self.assertEqual(chosen, "chirp")

    def test_seed_probe_family_does_not_force_negative_active_family(self):
        chosen = choose_seed_probe_family(
            ("passive_decay", "chirp", "impulse_left"),
            family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 0},
            expected_family_gain={
                "passive_decay": {"predicted_marginal_value": 0.12, "value_per_probe_step": 0.16},
                "chirp": {"predicted_marginal_value": -0.04, "value_per_probe_step": -0.05},
                "impulse_left": {"predicted_marginal_value": -0.02, "value_per_probe_step": -0.01},
            },
            global_family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 0},
        )
        self.assertEqual(chosen, "passive_decay")

    def test_seed_probe_family_does_not_force_near_zero_active_family(self):
        chosen = choose_seed_probe_family(
            ("passive_decay", "chirp", "impulse_left"),
            family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 0},
            expected_family_gain={
                "passive_decay": {"predicted_marginal_value": 0.12, "value_per_probe_step": 0.16},
                "chirp": {"predicted_marginal_value": 0.01, "value_per_probe_step": 0.015},
                "impulse_left": {"predicted_marginal_value": 0.02, "value_per_probe_step": 0.02},
            },
            global_family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 0},
        )
        self.assertEqual(chosen, "passive_decay")

    def test_selectable_unseen_active_probe_families_filters_dead_active_families(self):
        selectable = selectable_unseen_active_probe_families(
            ("passive_decay", "chirp", "impulse_left", "boundary_push"),
            expected_family_gain={
                "chirp": {"predicted_marginal_value": -0.03, "value_per_probe_step": -0.04},
                "impulse_left": {"predicted_marginal_value": 0.06, "value_per_probe_step": 0.08},
                "boundary_push": {"predicted_marginal_value": 0.07, "value_per_probe_step": 0.09},
            },
            family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 1, "boundary_push": 0},
        )
        self.assertEqual(selectable, ["boundary_push"])

    def test_choose_next_probe_family_does_not_let_quota_force_third_probe(self):
        crawler_bundle = self._StubCrawlerBundle(
            ("passive_decay", "chirp", "boundary_push", "cart_brake")
        )
        chosen = choose_next_probe_family(
            crawler_bundle=crawler_bundle,
            expected_family_gain={
                "passive_decay": {"predicted_marginal_value": 0.18, "value_per_probe_step": 0.26},
                "chirp": {"predicted_marginal_value": 0.09, "value_per_probe_step": 0.09},
                "boundary_push": {"predicted_marginal_value": 0.05, "value_per_probe_step": 0.05},
                "cart_brake": {"predicted_marginal_value": 0.04, "value_per_probe_step": 0.04},
            },
            family_counts={"passive_decay": 0, "chirp": 0, "boundary_push": 0, "cart_brake": 0},
            global_family_counts={"passive_decay": 0, "chirp": 5, "boundary_push": 0, "cart_brake": 0},
            require_seed_family=False,
            allow_quota_family=False,
        )
        self.assertEqual(chosen, "passive_decay")

    def test_should_require_seed_probe_family_turns_off_when_no_good_active_family_remains(self):
        require_seed = should_require_seed_probe_family(
            probe_count=1,
            family_coverage_budget=3,
            family_names=("passive_decay", "chirp", "impulse_left"),
            expected_family_gain={
                "passive_decay": {"predicted_marginal_value": 0.18, "value_per_probe_step": 0.26},
                "chirp": {"predicted_marginal_value": -0.03, "value_per_probe_step": -0.04},
                "impulse_left": {"predicted_marginal_value": -0.02, "value_per_probe_step": -0.01},
            },
            family_counts={"passive_decay": 0, "chirp": 0, "impulse_left": 0},
            global_family_counts={"passive_decay": 0, "chirp": 3, "impulse_left": 2},
        )
        self.assertFalse(require_seed)

    def test_three_probe_budget_only_forces_two_family_coverage_slots(self):
        budget = desired_family_coverage_budget(
            ("passive_decay", "chirp", "impulse_left", "boundary_push"),
            max_probe_episodes=3,
            min_seed_support=1,
        )
        self.assertEqual(budget, 2)

    def test_adaptive_probe_continues_when_future_gain_is_still_high(self):
        should_continue, reason = should_continue_probing_adaptive(
            probe_count=2,
            max_probe_episodes=4,
            uncertainty_scalar=0.05,
            uncertainty_probe_threshold=0.20,
            probe_surprise=0.10,
            surprise_probe_threshold=0.75,
            best_expected_gain=0.45,
            future_probe_error=0.15,
        )
        self.assertTrue(should_continue)
        self.assertEqual(reason, "adaptive_continue")

    def test_quota_probe_family_prioritizes_underused_specialist_family(self):
        chosen = choose_quota_probe_family(
            ("passive_decay", "chirp", "impulse_left", "boundary_push", "cart_brake"),
            expected_family_gain={
                "boundary_push": {"predicted_marginal_value": 0.06, "value_per_probe_step": 0.06},
                "cart_brake": {"predicted_marginal_value": 0.03, "value_per_probe_step": 0.03},
            },
            family_counts={"passive_decay": 1, "chirp": 0, "impulse_left": 0, "boundary_push": 0, "cart_brake": 0},
            global_family_counts={"passive_decay": 15, "chirp": 13, "impulse_left": 11, "boundary_push": 0, "cart_brake": 0},
            recent_families=("passive_decay", "chirp"),
        )
        self.assertEqual(chosen, "boundary_push")

    def test_choose_next_probe_family_penalizes_repeated_passive_decay(self):
        crawler_bundle = self._StubCrawlerBundle(("passive_decay", "chirp", "boundary_push"))
        chosen = choose_next_probe_family(
            crawler_bundle=crawler_bundle,
            expected_family_gain={
                "passive_decay": {
                    "predicted_marginal_value": 0.30,
                    "value_per_probe_step": 0.30,
                    "selection_score": 0.30,
                    "score": 0.30,
                },
                "chirp": {
                    "predicted_marginal_value": 0.18,
                    "value_per_probe_step": 0.18,
                    "selection_score": 0.18,
                    "score": 0.18,
                },
            },
            family_counts={"passive_decay": 1, "chirp": 0, "boundary_push": 0},
            global_family_counts={"passive_decay": 12, "chirp": 1, "boundary_push": 1},
            require_seed_family=False,
            family_realized_gain_history={"passive_decay": 0.0},
            recent_families=("passive_decay",),
        )
        self.assertEqual(chosen, "chirp")

    def test_choose_next_probe_family_returns_none_when_every_family_is_economically_bad(self):
        crawler_bundle = self._StubCrawlerBundle(("passive_decay", "chirp", "boundary_push"))
        chosen = choose_next_probe_family(
            crawler_bundle=crawler_bundle,
            expected_family_gain={
                "passive_decay": {
                    "predicted_marginal_value": -0.01,
                    "value_per_probe_step": -0.01,
                    "selection_score": 0.03,
                    "score": 0.04,
                },
                "chirp": {
                    "predicted_marginal_value": -0.10,
                    "value_per_probe_step": -0.05,
                    "selection_score": 0.05,
                    "score": 0.06,
                },
            },
            family_counts={"passive_decay": 0, "chirp": 0, "boundary_push": 0},
            global_family_counts={"passive_decay": 0, "chirp": 0, "boundary_push": 0},
            require_seed_family=True,
        )
        self.assertIsNone(chosen)

    def test_choose_next_probe_family_suppresses_repeat_negative_family(self):
        crawler_bundle = self._StubCrawlerBundle(("passive_decay", "chirp", "boundary_push"))
        chosen = choose_next_probe_family(
            crawler_bundle=crawler_bundle,
            expected_family_gain={
                "chirp": {
                    "predicted_marginal_value": 0.04,
                    "value_per_probe_step": 0.05,
                    "selection_score": 0.06,
                    "score": 0.06,
                },
                "boundary_push": {
                    "predicted_marginal_value": 0.03,
                    "value_per_probe_step": 0.04,
                    "selection_score": 0.05,
                    "score": 0.05,
                },
            },
            family_counts={"passive_decay": 0, "chirp": 1, "boundary_push": 0},
            global_family_counts={"passive_decay": 0, "chirp": 3, "boundary_push": 0},
            require_seed_family=False,
            family_realized_gain_history={"chirp": 0.0},
            family_bad_streaks={"chirp": 2},
            recent_families=("chirp",),
        )
        self.assertEqual(chosen, "boundary_push")

    def test_family_score_counters_average_cleanly(self):
        totals = default_family_score_counter(("passive_decay", "chirp"))
        counts = {"passive_decay": 0, "chirp": 0}
        update_family_score_counter(
            totals,
            counts,
            {
                "passive_decay": {
                "predicted_mechanics_reduction": 0.4,
                "predicted_future_error_reduction": 0.2,
                "predicted_split_reduction": 0.1,
                "predicted_entropy_reduction": 0.3,
                "predicted_hypothesis_separation": 0.25,
                "diversity_bonus": 1.0,
                "coverage_bonus": 0.1,
                "quota_bonus": 0.0,
                "future_error_estimate": 0.3,
                "signature_norm": 0.6,
                "estimated_probe_cost": 0.8,
                "repeat_penalty": 0.0,
                "global_repeat_penalty": 0.0,
                "realized_gain_calibration": 1.0,
                "realized_gain_bonus": 0.0,
                "predicted_marginal_value": 0.35,
                "value_per_probe_step": 0.4375,
                "selection_score": 0.45,
                    "score": 0.35,
                }
            },
        )
        update_family_score_counter(
            totals,
            counts,
            {
                "passive_decay": {
                "predicted_mechanics_reduction": 0.2,
                "predicted_future_error_reduction": 0.1,
                "predicted_split_reduction": 0.05,
                "predicted_entropy_reduction": 0.15,
                "predicted_hypothesis_separation": 0.10,
                "diversity_bonus": 0.5,
                "coverage_bonus": 0.0,
                "quota_bonus": 0.0,
                "future_error_estimate": 0.2,
                "signature_norm": 0.4,
                "estimated_probe_cost": 0.6,
                "repeat_penalty": 0.1,
                "global_repeat_penalty": 0.0,
                "realized_gain_calibration": 0.8,
                "realized_gain_bonus": 0.05,
                "predicted_marginal_value": 0.20,
                "value_per_probe_step": 0.3333333333,
                "selection_score": 0.25,
                    "score": 0.20,
                }
            },
        )
        averaged = average_family_score_counter(totals, counts)
        self.assertAlmostEqual(averaged["passive_decay"]["score"], 0.275, places=5)
        self.assertAlmostEqual(averaged["passive_decay"]["estimated_probe_cost"], 0.7, places=5)
        self.assertAlmostEqual(averaged["passive_decay"]["value_per_probe_step"], 0.3854166666, places=5)
        self.assertAlmostEqual(averaged["passive_decay"]["predicted_entropy_reduction"], 0.225, places=5)
        self.assertAlmostEqual(averaged["passive_decay"]["predicted_hypothesis_separation"], 0.175, places=5)

        scalar_totals = default_family_metric_counter(("passive_decay",))
        scalar_counts = {"passive_decay": 0}
        update_family_scalar_counter(scalar_totals, scalar_counts, {"passive_decay": 0.8})
        update_family_scalar_counter(scalar_totals, scalar_counts, {"passive_decay": 0.2})
        scalar_avg = average_family_scalar_counter(scalar_totals, scalar_counts)
        self.assertAlmostEqual(scalar_avg["passive_decay"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
