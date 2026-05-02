import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teenyreason.app.benchmark_diagnostics import build_latent_support_diagnostics
from teenyreason.viz.dashboard import (
    build_benchmark_payload,
    build_index_payload,
    build_latent_payload,
)
from teenyreason.viz.diagnostics import summarize_solve_array
from teenyreason.viz.payloads import build_support_validity_payload


class DashboardPayloadTests(unittest.TestCase):
    def test_dashboard_sim_fanout_delta_uses_same_sign_convention(self):
        template_path = Path(__file__).resolve().parents[2] / "teenyreason" / "viz" / "templates" / "dashboard.html"
        template = template_path.read_text(encoding="utf-8")

        self.assertIn(
            "summaries.sim_fanout_episode.median - summaries.full_system_episode.median",
            template,
        )
        self.assertNotIn(
            "summaries.full_system_episode.median - summaries.sim_fanout_episode.median",
            template,
        )

    def test_dashboard_does_not_call_protocol_delta_expression_when_strict_unused(self):
        template_path = Path(__file__).resolve().parents[2] / "teenyreason" / "viz" / "templates" / "dashboard.html"
        template = template_path.read_text(encoding="utf-8")

        self.assertIn("strictExpressionActive", template)
        self.assertIn("conditioned branch", template)
        self.assertIn("strict unused", template)
        self.assertIn("rather than proving learned message contribution", template)

    def test_live_dashboard_does_not_hide_curves_for_non_cartpole_envs(self):
        template_path = Path(__file__).resolve().parents[2] / "teenyreason" / "viz" / "templates" / "dashboard.html"
        template = template_path.read_text(encoding="utf-8")

        self.assertNotIn("if (!hasLivePayload || !isCartPole)", template)
        self.assertIn("The animated canvas is CartPole-only", template)

    def test_live_dashboard_has_direct_ppo_comparison_board(self):
        template_path = Path(__file__).resolve().parents[2] / "teenyreason" / "viz" / "templates" / "dashboard.html"
        template = template_path.read_text(encoding="utf-8")

        self.assertIn("PPO vs Probe-Conditioned PPO", template)
        self.assertIn('data-deck-target="comparison"', template)
        self.assertIn('id="deck-comparison"', template)
        self.assertIn("comparisonBoard", template)
        self.assertIn("paperFigureBoard", template)
        self.assertIn("Sample Efficiency", template)
        self.assertIn("Performance Ceiling", template)
        self.assertIn("Learning Dynamics", template)
        self.assertIn("renderPaperFigureBoard(rawPayload)", template)
        self.assertIn("renderComparisonBoard(rawPayload)", template)
        self.assertIn("comparison_suite_id", template)
        self.assertIn("rolling 100-episode average", template)
        self.assertIn("baseline_env_step_solves", template)
        self.assertIn("baseline_total_env_steps", template)
        self.assertIn("probe_env_step_solves_with_encoder", template)
        self.assertIn("probe_total_env_steps_with_encoder", template)
        self.assertIn("unsolved @", template)

    def test_summarize_solve_array_marks_skipped_variants_as_not_run(self):
        summary = summarize_solve_array(
            np.asarray([-1, -1], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
        )

        self.assertTrue(summary["not_run"])
        self.assertEqual(summary["count"], 0)
        self.assertEqual(summary["values"], [])

    def test_support_validity_rejects_pooled_support_that_exceeds_budget(self):
        payload = build_support_validity_payload(
            num_envs=8,
            num_windows=144,
            window_count_mean=18.0,
            support_count_mean=18.0,
            support_group_count_mean=9.0,
            support_group_ratio_mean=0.50,
            split_group_overlap_mean=1.0,
        )

        self.assertFalse(payload["is_valid"])
        self.assertIn("canonical support budget is being exceeded", payload["reasons"])

    def test_support_validity_accepts_strict_cross_family_splits(self):
        payload = build_support_validity_payload(
            num_envs=50,
            num_windows=300,
            window_count_mean=6.0,
            support_count_mean=4.0,
            support_group_count_mean=4.0,
            support_group_ratio_mean=1.0,
            split_group_overlap_mean=0.0,
        )

        self.assertTrue(payload["is_valid"])
        self.assertNotIn("split halves only partly overlap by probe family", payload["reasons"])

    def test_latent_support_diagnostics_use_canonical_support_modes_when_available(self):
        snapshot = {
            "env_belief_mean": np.asarray([[0.0, 0.0], [0.2, 0.1]], dtype=np.float32),
            "window_latent_mean": np.asarray(
                [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]],
                dtype=np.float32,
            ),
            "window_probe_mode": np.asarray(
                ["passive_decay", "impulse_left", "chirp", "cart_brake", "boundary_push", "cart_brake"],
                dtype="U",
            ),
            "window_is_support": np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int8),
            "env_window_count": np.asarray([3, 3], dtype=np.int32),
            "env_support_count": np.asarray([3, 3], dtype=np.int32),
        }

        diagnostics = build_latent_support_diagnostics(snapshot)

        self.assertAlmostEqual(diagnostics["mechanics_window_share"], 1.0)
        self.assertAlmostEqual(diagnostics["stress_window_share"], 0.0)
        self.assertAlmostEqual(diagnostics["available_stress_window_share"], 0.5)

    def test_latent_payload_flags_undercovered_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "toy_latent_snapshot.npz"
            np.savez(
                artifact_path,
                env_belief_mean=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                projection_2d=np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
                env_uncertainty=np.asarray([0.2, 0.3], dtype=np.float32),
                env_params=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                env_window_count=np.asarray([1, 1], dtype=np.int32),
                env_support_count=np.asarray([1, 1], dtype=np.int32),
                env_support_group_ratio=np.asarray([1.0, 1.0], dtype=np.float32),
                env_instance_id=np.asarray([0, 1], dtype=np.int32),
                env_view_spread=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
                window_env_instance_id=np.asarray([0, 1], dtype=np.int32),
                window_reward_sum=np.asarray([1.0, 1.5], dtype=np.float32),
                window_probe_mode=np.asarray(["impulse_left", "impulse_right"], dtype="U"),
                window_terminated=np.asarray([0, 0], dtype=np.int8),
                window_latent_mean=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                env_param_names=np.asarray(["gravity", "masscart"], dtype="U"),
                pca_explained=np.asarray([0.9, 0.1], dtype=np.float32),
            )

            payload = build_latent_payload(artifact_path)

            self.assertEqual(payload["summary"]["num_envs"], 2)
            self.assertEqual(payload["summary"]["num_windows"], 2)
            self.assertEqual(payload["support_validity"]["status"], "invalid")
            self.assertFalse(payload["support_validity"]["is_valid"])
            self.assertIn("undercovered", payload["support_validity"]["headline"].lower())

    def test_latent_payload_pads_one_dimensional_projection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "toy_one_dim_projection_snapshot.npz"
            np.savez(
                artifact_path,
                env_belief_mean=np.asarray([[0.1], [0.3]], dtype=np.float32),
                projection_2d=np.asarray([[0.0], [0.2]], dtype=np.float32),
                env_uncertainty=np.asarray([0.2, 0.3], dtype=np.float32),
                env_params=np.asarray([[1.0], [3.0]], dtype=np.float32),
                env_window_count=np.asarray([1, 1], dtype=np.int32),
                env_support_count=np.asarray([1, 1], dtype=np.int32),
                env_support_group_ratio=np.asarray([1.0, 1.0], dtype=np.float32),
                env_instance_id=np.asarray([0, 1], dtype=np.int32),
                env_view_spread=np.asarray([[0.0], [0.0]], dtype=np.float32),
                window_env_instance_id=np.asarray([0, 1], dtype=np.int32),
                window_reward_sum=np.asarray([1.0, 1.5], dtype=np.float32),
                window_probe_mode=np.asarray(["impulse_left", "impulse_right"], dtype="U"),
                window_terminated=np.asarray([0, 0], dtype=np.int8),
                window_latent_mean=np.asarray([[0.1], [0.3]], dtype=np.float32),
                env_param_names=np.asarray(["gravity"], dtype="U"),
                pca_explained=np.asarray([1.0], dtype=np.float32),
            )

            payload = build_latent_payload(artifact_path)

            self.assertEqual(payload["points"][0]["x"], 0.0)
            self.assertEqual(payload["points"][0]["y"], 0.0)
            self.assertAlmostEqual(payload["points"][1]["x"], 0.2)
            self.assertEqual(payload["points"][1]["y"], 0.0)

    def test_latent_payload_includes_system_id_block_without_dropping_latent_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "toy_particle_snapshot.npz"
            np.savez(
                artifact_path,
                env_belief_mean=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                projection_2d=np.asarray([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
                env_uncertainty=np.asarray([0.2, 0.3], dtype=np.float32),
                env_params=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                env_window_count=np.asarray([4, 4], dtype=np.int32),
                env_support_count=np.asarray([4, 4], dtype=np.int32),
                env_support_group_ratio=np.asarray([1.0, 1.0], dtype=np.float32),
                env_instance_id=np.asarray([0, 1], dtype=np.int32),
                env_view_spread=np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
                window_env_instance_id=np.asarray([0, 1], dtype=np.int32),
                window_reward_sum=np.asarray([1.0, 1.5], dtype=np.float32),
                window_probe_mode=np.asarray(["impulse_left", "impulse_right"], dtype="U"),
                window_terminated=np.asarray([0, 0], dtype=np.int8),
                window_latent_mean=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                env_param_names=np.asarray(["gravity", "masscart"], dtype="U"),
                pca_explained=np.asarray([0.9, 0.1], dtype=np.float32),
                particle_entropy=np.asarray([3.2, 2.8], dtype=np.float32),
                particle_ess_ratio=np.asarray([0.72, 0.66], dtype=np.float32),
                particle_leaveout_shift=np.asarray([0.08, 0.12], dtype=np.float32),
                sysid_validation_top1=np.asarray([0.61], dtype=np.float32),
                sysid_validation_margin=np.asarray([0.34], dtype=np.float32),
                sysid_validation_nll=np.asarray([0.27], dtype=np.float32),
                sysid_trusted=np.asarray([1.0], dtype=np.float32),
            )

            payload = build_latent_payload(artifact_path)

            self.assertEqual(payload["summary"]["num_envs"], 2)
            self.assertIn("points", payload)
            self.assertTrue(payload["system_id"]["available"])
            self.assertTrue(payload["system_id"]["trusted"])
            self.assertAlmostEqual(payload["system_id"]["validation_top1"], 0.61, places=5)
            self.assertAlmostEqual(payload["system_id"]["particle_ess_ratio_mean"], 0.69, places=5)

    def test_benchmark_payload_reads_family_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "toy_solve_benchmark.npz"
            np.savez(
                artifact_path,
                env_name=np.asarray("continuous_cartpole"),
                benchmark_profile=np.asarray("fast"),
                benchmark_mode=np.asarray("fair"),
                probe_budget_mode=np.asarray("fair_two_probe_handoff"),
                seeds=np.asarray([0], dtype=np.int64),
                baseline_solves=np.asarray([100], dtype=np.int64),
                probe_solves=np.asarray([80], dtype=np.int64),
                probe_shadow_solves=np.asarray([72], dtype=np.int64),
                probe_no_expression_solves=np.asarray([95], dtype=np.int64),
                full_system_episode_solves=np.asarray([61], dtype=np.int64),
                full_system_state_only_episode_solves=np.asarray([73], dtype=np.int64),
                full_system_oracle_episode_solves=np.asarray([44], dtype=np.int64),
                sim_fanout_episode_solves=np.asarray([39], dtype=np.int64),
                baseline_episode_solves=np.asarray([100], dtype=np.int64),
                probe_episode_solves=np.asarray([80], dtype=np.int64),
                probe_shadow_episode_solves=np.asarray([72], dtype=np.int64),
                probe_no_expression_episode_solves=np.asarray([95], dtype=np.int64),
                full_system_step_solves=np.asarray([10900], dtype=np.int64),
                full_system_state_only_step_solves=np.asarray([12600], dtype=np.int64),
                full_system_oracle_step_solves=np.asarray([9800], dtype=np.int64),
                sim_fanout_step_solves=np.asarray([9100], dtype=np.int64),
                baseline_step_solves=np.asarray([8000], dtype=np.int64),
                probe_step_solves=np.asarray([12000], dtype=np.int64),
                probe_shadow_step_solves=np.asarray([11800], dtype=np.int64),
                probe_no_expression_step_solves=np.asarray([14000], dtype=np.int64),
                baseline_total_env_steps=np.asarray([8000], dtype=np.int64),
                probe_total_env_steps=np.asarray([17000], dtype=np.int64),
                probe_shadow_total_env_steps=np.asarray([16500], dtype=np.int64),
                probe_no_expression_total_env_steps=np.asarray([19000], dtype=np.int64),
                full_system_total_env_steps=np.asarray([15400], dtype=np.int64),
                full_system_state_only_total_env_steps=np.asarray([17100], dtype=np.int64),
                full_system_oracle_total_env_steps=np.asarray([14800], dtype=np.int64),
                sim_fanout_total_env_steps=np.asarray([13100], dtype=np.int64),
                baseline_control_env_steps=np.asarray([8000], dtype=np.int64),
                probe_probe_env_steps=np.asarray([5000], dtype=np.int64),
                probe_control_env_steps=np.asarray([12000], dtype=np.int64),
                probe_shadow_probe_env_steps=np.asarray([5000], dtype=np.int64),
                probe_shadow_control_env_steps=np.asarray([11500], dtype=np.int64),
                probe_post_expression_env_steps=np.asarray([12000], dtype=np.int64),
                probe_post_expression_episodes=np.asarray([80], dtype=np.int64),
                probe_shadow_post_expression_env_steps=np.asarray([11500], dtype=np.int64),
                probe_shadow_post_expression_episodes=np.asarray([72], dtype=np.int64),
                probe_no_expression_probe_env_steps=np.asarray([5000], dtype=np.int64),
                probe_no_expression_control_env_steps=np.asarray([14000], dtype=np.int64),
                probe_no_expression_post_expression_env_steps=np.asarray([14000], dtype=np.int64),
                probe_no_expression_post_expression_episodes=np.asarray([95], dtype=np.int64),
                full_system_probe_env_steps=np.asarray([5100], dtype=np.int64),
                full_system_control_env_steps=np.asarray([10300], dtype=np.int64),
                full_system_post_context_env_steps=np.asarray([8800], dtype=np.int64),
                full_system_post_context_episodes=np.asarray([61], dtype=np.int64),
                full_system_oracle_probe_env_steps=np.asarray([5000], dtype=np.int64),
                full_system_oracle_control_env_steps=np.asarray([9800], dtype=np.int64),
                full_system_oracle_post_context_env_steps=np.asarray([7600], dtype=np.int64),
                full_system_oracle_post_context_episodes=np.asarray([44], dtype=np.int64),
                sim_fanout_probe_env_steps=np.asarray([5600], dtype=np.int64),
                sim_fanout_control_env_steps=np.asarray([7500], dtype=np.int64),
                sim_fanout_post_context_env_steps=np.asarray([7500], dtype=np.int64),
                sim_fanout_post_context_episodes=np.asarray([39], dtype=np.int64),
                baseline_completed_episodes=np.asarray([100], dtype=np.int64),
                probe_completed_episodes=np.asarray([80], dtype=np.int64),
                probe_shadow_completed_episodes=np.asarray([72], dtype=np.int64),
                probe_no_expression_completed_episodes=np.asarray([95], dtype=np.int64),
                full_system_completed_episodes=np.asarray([61], dtype=np.int64),
                full_system_state_only_completed_episodes=np.asarray([3], dtype=np.int64),
                full_system_oracle_completed_episodes=np.asarray([44], dtype=np.int64),
                sim_fanout_completed_episodes=np.asarray([39], dtype=np.int64),
                full_system_controller_style=np.asarray(["belief_controller_context_learned"], dtype="U"),
                full_system_oracle_controller_style=np.asarray(["belief_controller_context_oracle"], dtype="U"),
                sim_fanout_controller_style=np.asarray(["sim_fanout_state_only"], dtype="U"),
                probe_encoder_steps=np.asarray([5600], dtype=np.int64),
                probe_windows_total=np.asarray([27], dtype=np.int64),
                probe_expression_scale_median=np.asarray([0.42], dtype=np.float32),
                probe_expression_scale_active_fraction=np.asarray([0.80], dtype=np.float32),
                probe_fair_ready_handoff_fraction=np.asarray([0.25], dtype=np.float32),
                probe_fair_expression_enabled_fraction=np.asarray([0.20], dtype=np.float32),
                probe_fair_expression_force_muted_fraction=np.asarray([0.80], dtype=np.float32),
                probe_fair_ready_confidence_median=np.asarray([0.67], dtype=np.float32),
                probe_fair_muted_confidence_median=np.asarray([0.12], dtype=np.float32),
                probe_expression_ready_but_muted_fraction=np.asarray([0.05], dtype=np.float32),
                probe_shadow_expression_enabled_fraction=np.asarray([0.35], dtype=np.float32),
                probe_shadow_expression_scale_median=np.asarray([0.28], dtype=np.float32),
                probe_shadow_confidence_median=np.asarray([0.31], dtype=np.float32),
                probe_shadow_strict_miss_fraction=np.asarray([0.15], dtype=np.float32),
                probe_run_classification=np.asarray(["latent_win"], dtype="U"),
                belief_mode=np.asarray(["particle_sysid"], dtype="U"),
                belief_progress_index=np.asarray([0.61], dtype=np.float32),
                system_id_progress_index=np.asarray([0.73], dtype=np.float32),
                sysid_trusted=np.asarray([1.0], dtype=np.float32),
                sysid_validation_top1=np.asarray([0.82], dtype=np.float32),
                sysid_validation_margin=np.asarray([1.4], dtype=np.float32),
                sysid_validation_nll=np.asarray([10.5], dtype=np.float32),
                particle_entropy_mean=np.asarray([1.2], dtype=np.float32),
                particle_entropy_norm_mean=np.asarray([0.25], dtype=np.float32),
                particle_ess_ratio_mean=np.asarray([0.18], dtype=np.float32),
                particle_leaveout_shift_mean=np.asarray([0.28], dtype=np.float32),
                particle_subset_stability_mean=np.asarray([0.49], dtype=np.float32),
                latent_mechanics_fit=np.asarray([0.52], dtype=np.float32),
                latent_split_top1=np.asarray([0.24], dtype=np.float32),
                latent_neighbor_alignment=np.asarray([0.31], dtype=np.float32),
                latent_gap_ratio=np.asarray([0.82], dtype=np.float32),
                latent_heldout_probe_error=np.asarray([0.41], dtype=np.float32),
                latent_probe_leakage=np.asarray([0.11], dtype=np.float32),
                latent_uncert_error_corr=np.asarray([0.23], dtype=np.float32),
                latent_support_diagnostics_json=np.asarray(
                    [
                        json.dumps(
                            {
                                "center_window_share": 0.18,
                                "directional_window_share": 0.72,
                                "mechanics_window_share": 0.84,
                                "passive_window_share": 0.14,
                                "stress_window_share": 0.28,
                                "effective_window_families": 4.2,
                                "window_mode_leakage": 0.22,
                                "env_mode_leakage": 0.09,
                            }
                        )
                    ],
                    dtype="U",
                ),
                latent_win_gate_json=np.asarray(
                    json.dumps(
                        {
                            "pass": True,
                            "failure_reasons": [],
                            "checks": {"full_benchmark": True},
                            "metrics": {"mechanics_fit_median": 0.52},
                        }
                    ),
                    dtype="U",
                ),
                latent_win_gate_failure_reasons_json=np.asarray(json.dumps([]), dtype="U"),
                probe_stop_reasons_json=np.asarray([json.dumps({"low_uncertainty_low_gain": 1})], dtype="U"),
                probe_final_stop_reason=np.asarray(["fair_two_probe_handoff"], dtype="U"),
                probe_family_expected_gain_json=np.asarray(
                    [json.dumps({"passive_decay": {"score": 0.4, "predicted_mechanics_reduction": 0.3, "predicted_future_error_reduction": 0.2, "raw_predicted_future_error_reduction": 0.06, "future_gain_for_choice": 0.11, "raw_future_error_estimate": 0.1, "future_error_estimate": 0.2}})],
                    dtype="U",
                ),
                probe_family_realized_gain_json=np.asarray([json.dumps({"passive_decay": 0.15})], dtype="U"),
                probe_family_future_error_json=np.asarray([json.dumps({"passive_decay": 0.45})], dtype="U"),
                probe_family_selection_count_json=np.asarray([json.dumps({"passive_decay": 3})], dtype="U"),
                probe_readiness_reason_counts_json=np.asarray([json.dumps({"future_probe_quality": 2, "subset_stability": 1})], dtype="U"),
                probe_readiness_component_means_json=np.asarray([json.dumps({"future_probe_quality": 0.44, "subset_stability": 0.31})], dtype="U"),
                probe_fair_stop_blocker_counts_json=np.asarray([json.dumps({"online_subset_stability": 2})], dtype="U"),
                probe_shadow_blocker_counts_json=np.asarray([json.dumps({"confidence": 2})], dtype="U"),
                probe_second_probe_selection_count_json=np.asarray([json.dumps({"chirp": 2})], dtype="U"),
                probe_second_probe_raw_future_gain_mean=np.asarray([0.06], dtype=np.float32),
                probe_second_probe_future_estimate_mean=np.asarray([0.14], dtype=np.float32),
                probe_second_probe_choice_future_gain_mean=np.asarray([0.11], dtype=np.float32),
                probe_family_coverage_satisfied_fraction=np.asarray([1.0], dtype=np.float32),
                probe_second_probe_value_driven_fraction=np.asarray([0.75], dtype=np.float32),
                probe_uniformity_pressure_active_fraction=np.asarray([0.25], dtype=np.float32),
                probe_env_expression_delta=np.asarray([14.0], dtype=np.float32),
                probe_forced_env_expression_delta=np.asarray([5.5], dtype=np.float32),
                probe_forced_env_expression_scale=np.asarray([0.15], dtype=np.float32),
                probe_strict_usage_status=np.asarray(["intermittent"], dtype="U"),
                probe_fair_handoff_probe_families_json=np.asarray([json.dumps(["boundary_push", "chirp"])], dtype="U"),
                probe_readiness_component_timeline_json=np.asarray([json.dumps([{"future_probe_quality": 0.44, "subset_stability": 0.31}])], dtype="U"),
                probe_online_future_quality_trace_json=np.asarray([json.dumps([0.41, 0.58])], dtype="U"),
                probe_online_subset_stability_trace_json=np.asarray([json.dumps([0.22, 0.48])], dtype="U"),
                probe_online_offline_gap_trace_json=np.asarray([json.dumps([0.12, 0.08])], dtype="U"),
                probe_online_subset_stability_mean=np.asarray([0.35], dtype=np.float32),
                probe_online_offline_gap_mean=np.asarray([0.10], dtype=np.float32),
                probe_online_geometry_complete_fraction=np.asarray([0.75], dtype=np.float32),
                probe_online_split_latent_disagreement_mean=np.asarray([0.06], dtype=np.float32),
                probe_online_split_retrieval_margin_deficit_mean=np.asarray([0.14], dtype=np.float32),
                probe_online_leaveout_shift_mean=np.asarray([0.09], dtype=np.float32),
                probe_teacher_action_agreement=np.asarray([0.0], dtype=np.float32),
                full_system_state_only_eval_returns_json=np.asarray([json.dumps([410.0, 380.0, 500.0])], dtype="U"),
                full_system_learned_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [422.0, 391.0, 500.0], "episode_total_env_steps": [16200, 15100, 14900], "mean_return": 437.67, "mean_total_env_steps": 15400.0, "solved_count": 1, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_state_only_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [410.0, 380.0, 500.0], "episode_total_env_steps": [17100, 16800, 17400], "mean_return": 430.0, "mean_total_env_steps": 17100.0, "solved_count": 1, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_zero_context_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [360.0, 340.0, 470.0], "episode_total_env_steps": [18000, 18200, 17600], "mean_return": 390.0, "mean_total_env_steps": 17933.33, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_shuffled_context_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [382.0, 355.0, 444.0], "episode_total_env_steps": [17750, 17900, 17500], "mean_return": 393.67, "mean_total_env_steps": 17716.67, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_stale_context_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [398.0, 372.0, 489.0], "episode_total_env_steps": [17350, 17050, 16980], "mean_return": 419.67, "mean_total_env_steps": 17126.67, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_online_refinement_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [401.0, 376.0, 492.0], "episode_total_env_steps": [17220, 16980, 16840], "mean_return": 423.0, "mean_total_env_steps": 17013.33, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_frozen_context_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [351.0, 328.0, 430.0], "episode_total_env_steps": [18220, 18310, 18040], "mean_return": 369.67, "mean_total_env_steps": 18190.0, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_actor_only_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [120.0, 98.0, 129.0], "episode_total_env_steps": [21000, 22400, 21900], "mean_return": 115.67, "mean_total_env_steps": 21766.67, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_state_only_ablation_delta=np.asarray([12.5], dtype=np.float32),
                full_system_zero_context_ablation_delta=np.asarray([31.5], dtype=np.float32),
                full_system_shuffled_context_ablation_delta=np.asarray([19.0], dtype=np.float32),
                full_system_stale_context_ablation_delta=np.asarray([14.5], dtype=np.float32),
                full_system_online_refinement_ablation_delta=np.asarray([11.0], dtype=np.float32),
                full_system_frozen_context_ablation_delta=np.asarray([68.0], dtype=np.float32),
                full_system_oracle_zero_context_ablation_delta=np.asarray([41.0], dtype=np.float32),
                full_system_oracle_learned_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [450.0, 430.0, 500.0], "episode_total_env_steps": [14800, 14500, 14300], "mean_return": 460.0, "mean_total_env_steps": 14533.33, "solved_count": 1, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_oracle_zero_context_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [330.0, 310.0, 460.0], "episode_total_env_steps": [18900, 19100, 18300], "mean_return": 366.67, "mean_total_env_steps": 18766.67, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_oracle_online_refinement_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [438.0, 418.0, 492.0], "episode_total_env_steps": [15020, 14790, 14560], "mean_return": 449.33, "mean_total_env_steps": 14790.0, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_oracle_frozen_context_eval_summary_json=np.asarray(
                    [json.dumps({"returns": [348.0, 336.0, 452.0], "episode_total_env_steps": [18750, 18620, 18240], "mean_return": 378.67, "mean_total_env_steps": 18536.67, "solved_count": 0, "fixture_count": 3})],
                    dtype="U",
                ),
                full_system_oracle_shuffled_context_ablation_delta=np.asarray([27.5], dtype=np.float32),
                full_system_oracle_stale_context_ablation_delta=np.asarray([18.0], dtype=np.float32),
                full_system_oracle_online_refinement_ablation_delta=np.asarray([9.5], dtype=np.float32),
                full_system_oracle_frozen_context_ablation_delta=np.asarray([81.33], dtype=np.float32),
            )

            payload = build_benchmark_payload(artifact_path)

            self.assertEqual(payload["benchmark_mode"], "fair")
            self.assertEqual(payload["benchmark_profile"], "fast")
            self.assertEqual(payload["probe_budget_mode"], "fair_two_probe_handoff")
            self.assertEqual(payload["run_classification"], "latent_win")
            self.assertEqual(payload["probe_strict_usage_status"], "intermittent")
            self.assertEqual(payload["probe_honesty_headline"], "")
            self.assertTrue(payload["probe_shadow_available"])
            self.assertEqual(len(payload["rows"]), 1)
            row = payload["rows"][0]
            self.assertEqual(row["probe_stop_reasons"]["low_uncertainty_low_gain"], 1)
            self.assertEqual(row["probe_family_selection_count"]["passive_decay"], 3)
            self.assertAlmostEqual(row["probe_family_realized_gain"]["passive_decay"], 0.15)
            self.assertEqual(row["probe_run_classification"], "latent_win")
            self.assertEqual(row["probe_no_expression_episode_solve"], 95)
            self.assertEqual(row["probe_shadow_episode_solve"], 72)
            self.assertEqual(row["probe_fair_handoff_probe_families"], ["boundary_push", "chirp"])
            self.assertEqual(row["probe_online_future_quality_trace"], [0.41, 0.58])
            self.assertAlmostEqual(row["probe_online_offline_gap_mean"], 0.10)
            self.assertAlmostEqual(row["belief_progress_index"], 0.61)
            self.assertAlmostEqual(row["latent_mechanics_fit"], 0.52)
            self.assertAlmostEqual(row["latent_split_top1"], 0.24)
            self.assertAlmostEqual(row["latent_neighbor_alignment"], 0.31)
            self.assertAlmostEqual(row["latent_gap_ratio"], 0.82)
            self.assertAlmostEqual(row["latent_probe_leakage"], 0.11)
            self.assertAlmostEqual(row["latent_uncert_error_corr"], 0.23)
            self.assertAlmostEqual(row["latent_center_window_share"], 0.18)
            self.assertAlmostEqual(row["latent_mechanics_window_share"], 0.84)
            self.assertAlmostEqual(row["latent_passive_window_share"], 0.14)
            self.assertAlmostEqual(row["latent_stress_window_share"], 0.28)
            self.assertAlmostEqual(row["latent_window_mode_leakage"], 0.22)
            self.assertAlmostEqual(
                payload["summaries"]["latent_support_diagnostics"]["env_mode_leakage"],
                0.09,
            )
            self.assertTrue(payload["latent_win_gate"]["pass"])
            self.assertEqual(payload["latent_win_gate_failure_reasons"], [])
            self.assertEqual(
                payload["summaries"]["probe_fair_handoff_pair_count"]["boundary_push / chirp"],
                1.0,
            )
            self.assertEqual(row["full_system_episode_solve"], 61)
            self.assertEqual(row["full_system_state_only_episode_solve"], 73)
            self.assertEqual(row["full_system_step_solve"], 10900)
            self.assertEqual(row["full_system_state_only_step_solve"], 12600)
            self.assertEqual(row["full_system_post_context_env_steps"], 8800)
            self.assertEqual(row["full_system_oracle_episode_solve"], 44)
            self.assertEqual(row["full_system_oracle_step_solve"], 9800)
            self.assertEqual(row["full_system_oracle_post_context_env_steps"], 7600)
            self.assertEqual(row["sim_fanout_episode_solve"], 39)
            self.assertEqual(row["sim_fanout_step_solve"], 9100)
            self.assertEqual(row["sim_fanout_post_context_env_steps"], 7500)
            self.assertAlmostEqual(row["probe_fair_ready_handoff_fraction"], 0.25)
            self.assertAlmostEqual(row["probe_fair_expression_force_muted_fraction"], 0.80)
            self.assertAlmostEqual(row["probe_expression_ready_but_muted_fraction"], 0.05)
            self.assertAlmostEqual(row["probe_shadow_expression_enabled_fraction"], 0.35)
            self.assertAlmostEqual(row["probe_shadow_strict_miss_fraction"], 0.15)
            self.assertEqual(row["probe_final_stop_reason"], "fair_two_probe_handoff")
            self.assertEqual(row["belief_mode"], "particle_sysid")
            self.assertTrue(row["sysid_trusted"])
            self.assertAlmostEqual(row["sysid_validation_top1"], 0.82)
            self.assertAlmostEqual(row["particle_leaveout_shift_mean"], 0.28)
            self.assertTrue(payload["summaries"]["system_id"]["available"])
            self.assertEqual(payload["summaries"]["system_id"]["mode"], "particle_sysid")
            self.assertAlmostEqual(
                payload["summaries"]["system_id"]["validation_top1_median"],
                0.82,
                places=5,
            )
            self.assertEqual(row["probe_second_probe_selection_count"]["chirp"], 2)
            self.assertAlmostEqual(row["probe_second_probe_raw_future_gain_mean"], 0.06)
            self.assertAlmostEqual(row["probe_second_probe_future_estimate_mean"], 0.14)
            self.assertAlmostEqual(row["probe_second_probe_choice_future_gain_mean"], 0.11)
            self.assertAlmostEqual(row["probe_family_coverage_satisfied_fraction"], 1.0)
            self.assertAlmostEqual(row["probe_second_probe_value_driven_fraction"], 0.75)
            self.assertAlmostEqual(row["probe_uniformity_pressure_active_fraction"], 0.25)
            self.assertAlmostEqual(row["probe_env_expression_delta"], 14.0)
            self.assertAlmostEqual(row["probe_forced_env_expression_delta"], 5.5)
            self.assertAlmostEqual(row["probe_forced_env_expression_scale"], 0.15)
            self.assertEqual(row["probe_strict_usage_status"], "intermittent")
            self.assertEqual(row["probe_fair_stop_blocker_counts"]["online_subset_stability"], 2)
            self.assertEqual(row["probe_online_subset_stability_trace"], [0.22, 0.48])
            self.assertAlmostEqual(row["probe_online_subset_stability_mean"], 0.35)
            self.assertAlmostEqual(row["probe_online_geometry_complete_fraction"], 0.75)
            self.assertAlmostEqual(row["probe_online_split_latent_disagreement_mean"], 0.06)
            self.assertAlmostEqual(row["probe_online_split_retrieval_margin_deficit_mean"], 0.14)
            self.assertAlmostEqual(row["probe_online_leaveout_shift_mean"], 0.09)
            self.assertTrue(row["probe_shadow_available"])
            self.assertTrue(row["full_system_available"])
            self.assertTrue(row["full_system_state_only_available"])
            self.assertTrue(row["full_system_zero_context_available"])
            self.assertTrue(row["full_system_shuffled_context_available"])
            self.assertTrue(row["full_system_stale_context_available"])
            self.assertTrue(row["full_system_frozen_context_available"])
            self.assertAlmostEqual(row["full_system_learned_eval_summary"]["mean_return"], 437.67, places=2)
            self.assertAlmostEqual(row["full_system_state_only_eval_summary"]["mean_total_env_steps"], 17100.0, places=2)
            self.assertAlmostEqual(row["full_system_shuffled_context_eval_summary"]["mean_return"], 393.67, places=2)
            self.assertAlmostEqual(row["full_system_frozen_context_eval_summary"]["mean_return"], 369.67, places=2)
            self.assertEqual(row["full_system_state_only_eval_returns"], [410.0, 380.0, 500.0])
            self.assertAlmostEqual(row["full_system_state_only_ablation_delta"], 12.5)
            self.assertAlmostEqual(row["full_system_zero_context_ablation_delta"], 31.5)
            self.assertAlmostEqual(row["full_system_shuffled_context_ablation_delta"], 19.0)
            self.assertAlmostEqual(row["full_system_online_refinement_ablation_delta"], 11.0)
            self.assertAlmostEqual(row["full_system_frozen_context_ablation_delta"], 68.0)
            self.assertTrue(payload["full_system_available"])

    def test_benchmark_payload_flags_unused_strict_latent_honesty_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "unused_strict_benchmark.npz"
            np.savez(
                artifact_path,
                env_name=np.asarray("continuous_cartpole"),
                benchmark_profile=np.asarray("fast"),
                benchmark_mode=np.asarray("fair"),
                probe_budget_mode=np.asarray("fair_two_probe_handoff"),
                seeds=np.asarray([0], dtype=np.int64),
                baseline_episode_solves=np.asarray([120], dtype=np.int64),
                probe_episode_solves=np.asarray([80], dtype=np.int64),
                probe_shadow_episode_solves=np.asarray([-1], dtype=np.int64),
                probe_no_expression_episode_solves=np.asarray([100], dtype=np.int64),
                full_system_episode_solves=np.asarray([-1], dtype=np.int64),
                full_system_state_only_episode_solves=np.asarray([-1], dtype=np.int64),
                full_system_oracle_episode_solves=np.asarray([-1], dtype=np.int64),
                sim_fanout_episode_solves=np.asarray([-1], dtype=np.int64),
                baseline_step_solves=np.asarray([9000], dtype=np.int64),
                probe_step_solves=np.asarray([11000], dtype=np.int64),
                probe_shadow_step_solves=np.asarray([-1], dtype=np.int64),
                probe_no_expression_step_solves=np.asarray([13000], dtype=np.int64),
                full_system_step_solves=np.asarray([-1], dtype=np.int64),
                full_system_state_only_step_solves=np.asarray([-1], dtype=np.int64),
                full_system_oracle_step_solves=np.asarray([-1], dtype=np.int64),
                sim_fanout_step_solves=np.asarray([-1], dtype=np.int64),
                baseline_total_env_steps=np.asarray([9000], dtype=np.int64),
                probe_total_env_steps=np.asarray([11000], dtype=np.int64),
                probe_shadow_total_env_steps=np.asarray([-1], dtype=np.int64),
                probe_no_expression_total_env_steps=np.asarray([13000], dtype=np.int64),
                full_system_total_env_steps=np.asarray([-1], dtype=np.int64),
                full_system_state_only_total_env_steps=np.asarray([-1], dtype=np.int64),
                full_system_oracle_total_env_steps=np.asarray([-1], dtype=np.int64),
                sim_fanout_total_env_steps=np.asarray([-1], dtype=np.int64),
                baseline_control_env_steps=np.asarray([9000], dtype=np.int64),
                probe_probe_env_steps=np.asarray([4000], dtype=np.int64),
                probe_control_env_steps=np.asarray([7000], dtype=np.int64),
                probe_post_expression_env_steps=np.asarray([7000], dtype=np.int64),
                probe_post_expression_episodes=np.asarray([80], dtype=np.int64),
                probe_shadow_probe_env_steps=np.asarray([-1], dtype=np.int64),
                probe_shadow_control_env_steps=np.asarray([-1], dtype=np.int64),
                probe_shadow_post_expression_env_steps=np.asarray([-1], dtype=np.int64),
                probe_shadow_post_expression_episodes=np.asarray([-1], dtype=np.int64),
                probe_no_expression_probe_env_steps=np.asarray([4000], dtype=np.int64),
                probe_no_expression_control_env_steps=np.asarray([9000], dtype=np.int64),
                probe_no_expression_post_expression_env_steps=np.asarray([9000], dtype=np.int64),
                probe_no_expression_post_expression_episodes=np.asarray([100], dtype=np.int64),
                full_system_probe_env_steps=np.asarray([-1], dtype=np.int64),
                full_system_control_env_steps=np.asarray([-1], dtype=np.int64),
                full_system_post_context_env_steps=np.asarray([-1], dtype=np.int64),
                full_system_post_context_episodes=np.asarray([-1], dtype=np.int64),
                full_system_oracle_probe_env_steps=np.asarray([-1], dtype=np.int64),
                full_system_oracle_control_env_steps=np.asarray([-1], dtype=np.int64),
                full_system_oracle_post_context_env_steps=np.asarray([-1], dtype=np.int64),
                full_system_oracle_post_context_episodes=np.asarray([-1], dtype=np.int64),
                sim_fanout_probe_env_steps=np.asarray([-1], dtype=np.int64),
                sim_fanout_control_env_steps=np.asarray([-1], dtype=np.int64),
                sim_fanout_post_context_env_steps=np.asarray([-1], dtype=np.int64),
                sim_fanout_post_context_episodes=np.asarray([-1], dtype=np.int64),
                baseline_completed_episodes=np.asarray([120], dtype=np.int64),
                probe_completed_episodes=np.asarray([80], dtype=np.int64),
                probe_shadow_completed_episodes=np.asarray([0], dtype=np.int64),
                probe_no_expression_completed_episodes=np.asarray([100], dtype=np.int64),
                full_system_completed_episodes=np.asarray([0], dtype=np.int64),
                full_system_state_only_completed_episodes=np.asarray([0], dtype=np.int64),
                full_system_oracle_completed_episodes=np.asarray([0], dtype=np.int64),
                sim_fanout_completed_episodes=np.asarray([0], dtype=np.int64),
                full_system_controller_style=np.asarray([""], dtype="U"),
                full_system_oracle_controller_style=np.asarray([""], dtype="U"),
                sim_fanout_controller_style=np.asarray([""], dtype="U"),
                probe_encoder_steps=np.asarray([4000], dtype=np.int64),
                probe_windows_total=np.asarray([12], dtype=np.int64),
                probe_expression_scale_median=np.asarray([0.0], dtype=np.float32),
                probe_expression_scale_active_fraction=np.asarray([0.0], dtype=np.float32),
                probe_fair_ready_handoff_fraction=np.asarray([0.0], dtype=np.float32),
                probe_fair_expression_enabled_fraction=np.asarray([0.0], dtype=np.float32),
                probe_fair_expression_force_muted_fraction=np.asarray([1.0], dtype=np.float32),
                probe_fair_ready_confidence_median=np.asarray([0.0], dtype=np.float32),
                probe_fair_muted_confidence_median=np.asarray([0.1], dtype=np.float32),
                probe_expression_ready_but_muted_fraction=np.asarray([0.0], dtype=np.float32),
                probe_shadow_expression_enabled_fraction=np.asarray([0.0], dtype=np.float32),
                probe_shadow_expression_scale_median=np.asarray([0.0], dtype=np.float32),
                probe_shadow_confidence_median=np.asarray([0.0], dtype=np.float32),
                probe_shadow_strict_miss_fraction=np.asarray([0.0], dtype=np.float32),
                probe_run_classification=np.asarray(["protocol_win"], dtype="U"),
                belief_progress_index=np.asarray([0.30], dtype=np.float32),
                latent_mechanics_fit=np.asarray([0.25], dtype=np.float32),
                latent_split_top1=np.asarray([0.10], dtype=np.float32),
                latent_neighbor_alignment=np.asarray([0.08], dtype=np.float32),
                latent_gap_ratio=np.asarray([3.5], dtype=np.float32),
                latent_heldout_probe_error=np.asarray([0.70], dtype=np.float32),
                latent_probe_leakage=np.asarray([0.30], dtype=np.float32),
                latent_uncert_error_corr=np.asarray([0.0], dtype=np.float32),
                latent_win_gate_json=np.asarray(json.dumps({"pass": False, "failure_reasons": ["probe_ready_fraction_too_low"]}), dtype="U"),
                latent_win_gate_failure_reasons_json=np.asarray(json.dumps(["probe_ready_fraction_too_low"]), dtype="U"),
                probe_stop_reasons_json=np.asarray([json.dumps({"fair_two_probe_handoff": 1})], dtype="U"),
                probe_final_stop_reason=np.asarray(["fair_two_probe_handoff"], dtype="U"),
                probe_family_expected_gain_json=np.asarray([json.dumps({})], dtype="U"),
                probe_family_realized_gain_json=np.asarray([json.dumps({})], dtype="U"),
                probe_family_future_error_json=np.asarray([json.dumps({})], dtype="U"),
                probe_family_selection_count_json=np.asarray([json.dumps({})], dtype="U"),
                probe_readiness_reason_counts_json=np.asarray([json.dumps({"leaveout_stability": 4})], dtype="U"),
                probe_readiness_component_means_json=np.asarray([json.dumps({"leaveout_stability": 0.12})], dtype="U"),
                probe_fair_stop_blocker_counts_json=np.asarray([json.dumps({"message_mode": 4})], dtype="U"),
                probe_shadow_blocker_counts_json=np.asarray([json.dumps({})], dtype="U"),
                probe_second_probe_selection_count_json=np.asarray([json.dumps({})], dtype="U"),
                probe_second_probe_raw_future_gain_mean=np.asarray([0.0], dtype=np.float32),
                probe_second_probe_future_estimate_mean=np.asarray([0.0], dtype=np.float32),
                probe_second_probe_choice_future_gain_mean=np.asarray([0.0], dtype=np.float32),
                probe_family_coverage_satisfied_fraction=np.asarray([0.0], dtype=np.float32),
                probe_second_probe_value_driven_fraction=np.asarray([0.0], dtype=np.float32),
                probe_uniformity_pressure_active_fraction=np.asarray([0.0], dtype=np.float32),
                probe_env_expression_delta=np.asarray([-4.0], dtype=np.float32),
                probe_forced_env_expression_delta=np.asarray([-2.0], dtype=np.float32),
                probe_forced_env_expression_scale=np.asarray([0.15], dtype=np.float32),
                probe_strict_usage_status=np.asarray(["unused"], dtype="U"),
                probe_fair_handoff_probe_families_json=np.asarray([json.dumps([])], dtype="U"),
                probe_readiness_component_timeline_json=np.asarray([json.dumps([])], dtype="U"),
                probe_message_ablation_config_diff_json=np.asarray([json.dumps({})], dtype="U"),
                probe_online_future_quality_trace_json=np.asarray([json.dumps([])], dtype="U"),
                probe_online_subset_stability_trace_json=np.asarray([json.dumps([])], dtype="U"),
                probe_online_offline_gap_trace_json=np.asarray([json.dumps([])], dtype="U"),
                probe_message_input_delta_mean=np.asarray([0.0], dtype=np.float32),
                probe_message_input_delta_max=np.asarray([0.0], dtype=np.float32),
                probe_muted_message_input_delta_mean=np.asarray([0.0], dtype=np.float32),
                probe_muted_message_input_delta_max=np.asarray([0.0], dtype=np.float32),
                probe_actor_message_norm_mean=np.asarray([0.0], dtype=np.float32),
                probe_actor_message_nonzero_fraction=np.asarray([0.0], dtype=np.float32),
                probe_muted_actor_message_nonzero_fraction=np.asarray([0.0], dtype=np.float32),
                probe_matched_mute_parity_fraction=np.asarray([1.0], dtype=np.float32),
                probe_online_subset_stability_mean=np.asarray([0.0], dtype=np.float32),
                probe_online_offline_gap_mean=np.asarray([0.0], dtype=np.float32),
                probe_online_geometry_complete_fraction=np.asarray([0.0], dtype=np.float32),
                probe_online_split_latent_disagreement_mean=np.asarray([0.0], dtype=np.float32),
                probe_online_split_retrieval_margin_deficit_mean=np.asarray([0.0], dtype=np.float32),
                probe_online_leaveout_shift_mean=np.asarray([0.15], dtype=np.float32),
                probe_teacher_action_agreement=np.asarray([0.0], dtype=np.float32),
            )

            payload = build_benchmark_payload(artifact_path)
            row = payload["rows"][0]

            self.assertEqual(payload["probe_strict_usage_status"], "unused")
            self.assertEqual(payload["probe_honesty_headline"], "Episode win without strict latent usage")
            self.assertAlmostEqual(
                row["probe_env_expression_delta"],
                -4.0,
            )
            self.assertAlmostEqual(
                row["probe_forced_env_expression_delta"],
                -2.0,
            )
            self.assertAlmostEqual(
                row["probe_forced_env_expression_scale"],
                0.15,
            )
            self.assertEqual(row["probe_strict_usage_status"], "unused")
            self.assertEqual(
                max(
                    row["probe_readiness_reason_counts"].items(),
                    key=lambda item: (item[1], item[0]),
                )[0],
                "leaveout_stability",
            )
            self.assertEqual(
                payload["summaries"]["probe_strict_usage_counts"]["unused"],
                1.0,
            )
            self.assertAlmostEqual(
                payload["summaries"]["probe_env_expression_delta"]["mean"],
                -4.0,
            )
            self.assertAlmostEqual(
                payload["summaries"]["probe_forced_env_expression_delta"]["mean"],
                -2.0,
            )
            self.assertAlmostEqual(
                payload["summaries"]["probe_forced_env_expression_scale"]["mean"],
                0.15,
            )
            self.assertAlmostEqual(
                payload["summaries"]["probe_readiness_component_means"]["leaveout_stability"],
                0.12,
            )
            self.assertFalse(payload["probe_shadow_available"])
            self.assertFalse(payload["full_system_available"])
            self.assertFalse(payload["full_system_state_only_available"])
            self.assertFalse(payload["full_system_zero_context_available"])
            self.assertFalse(payload["full_system_shuffled_context_available"])
            self.assertFalse(payload["full_system_stale_context_available"])
            self.assertFalse(payload["full_system_frozen_context_available"])
            self.assertFalse(payload["full_system_oracle_available"])
            self.assertFalse(payload["full_system_oracle_frozen_context_available"])
            self.assertFalse(payload["sim_fanout_available"])

    def test_index_payload_prefers_non_archived_benchmark_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "dashboard_context.json").write_text(
                json.dumps(
                    {
                        "env_name": "ContinuousCartPole-v0",
                        "env_display_name": "Continuous CartPole",
                        "benchmark_tag": "toy",
                        "default_benchmark_summary": "toy_archived_planner_solve_benchmark.npz",
                        "default_latent_snapshot": "toy_seed_1_latent_snapshot.npz",
                        "seeds": [0, 1],
                        "benchmark_profile": "archived_planner",
                    }
                ),
                encoding="utf-8",
            )
            non_archived_path = artifact_dir / "toy_solve_benchmark.npz"
            archived_path = artifact_dir / "toy_archived_planner_solve_benchmark.npz"
            np.savez(
                non_archived_path,
                env_name=np.asarray("continuous_cartpole"),
                benchmark_profile=np.asarray("fast"),
                seeds=np.asarray([0], dtype=np.int64),
            )
            np.savez(
                archived_path,
                env_name=np.asarray("continuous_cartpole"),
                benchmark_profile=np.asarray("archived_planner"),
                seeds=np.asarray([0], dtype=np.int64),
            )

            payload = build_index_payload(artifact_dir)

            self.assertEqual(payload["run_context"]["default_benchmark_summary"], "toy_solve_benchmark.npz")
            self.assertEqual(payload["benchmark_summaries"][0], "toy_solve_benchmark.npz")
            self.assertEqual(payload["benchmark_summaries"][1], "toy_archived_planner_solve_benchmark.npz")


if __name__ == "__main__":
    unittest.main()
