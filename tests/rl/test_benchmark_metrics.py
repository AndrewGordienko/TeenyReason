import unittest

from teenyreason.app.benchmark_support import (
    compute_belief_progress_index,
    evaluate_latent_win_gate,
    probe_strict_usage_status,
)


class BenchmarkMetricTests(unittest.TestCase):
    def test_belief_progress_index_matches_weighted_formula(self):
        score = compute_belief_progress_index(
            mechanics_fit=0.50,
            neighbor_alignment=0.25,
            split_retrieval=0.20,
            heldout_probe_error=0.40,
            uncert_error_corr=0.30,
            probe_leakage=0.10,
        )

        expected = (
            0.30 * 0.50
            + 0.20 * 0.25
            + 0.15 * 0.20
            + 0.15 * (1.0 - 0.40)
            + 0.10 * 0.30
            + 0.10 * (1.0 - 0.10)
        )
        self.assertAlmostEqual(score, expected, places=6)

    def test_belief_progress_index_clamps_invalid_inputs(self):
        score = compute_belief_progress_index(
            mechanics_fit=2.0,
            neighbor_alignment=-1.0,
            split_retrieval=3.0,
            heldout_probe_error=-4.0,
            uncert_error_corr=-0.5,
            probe_leakage=9.0,
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_latent_win_gate_reports_specific_failure_reasons(self):
        gate = evaluate_latent_win_gate(
            benchmark_profile="full",
            seed_count=5,
            baseline_episode_solves=[100, 100, 100, 100, 100],
            baseline_completed_episodes=[100, 100, 100, 100, 100],
            probe_episode_solves=[150, 150, 150, None, None],
            probe_completed_episodes=[150, 150, 150, 150, 150],
            baseline_step_solves=[10000, 10000, 10000, 10000, 10000],
            baseline_total_env_steps=[10000, 10000, 10000, 10000, 10000],
            probe_step_solves=[15000, 15000, 15000, None, None],
            probe_total_env_steps=[15000, 15000, 15000, 15000, 15000],
            probe_env_expression_delta=[-5.0, -3.0, -1.0, -2.0, -4.0],
            probe_no_expression_available=False,
            probe_ready_fraction=[0.10] * 5,
            probe_muted_fraction=[0.90] * 5,
            latent_mechanics_fit=[0.30] * 5,
            latent_neighbor_alignment=[0.10] * 5,
            latent_split_retrieval=[0.15] * 5,
            latent_gap_ratio=[3.0] * 5,
            latent_probe_leakage=[0.30] * 5,
            latent_uncert_error_corr=[0.05] * 5,
            full_system_state_only_ablation_delta=[10.0] * 5,
            full_system_zero_context_ablation_delta=[8.0] * 5,
            full_system_shuffled_context_ablation_delta=[6.0] * 5,
            full_system_stale_context_ablation_delta=[4.0] * 5,
        )

        self.assertFalse(gate["pass"])
        self.assertIn("probe_success_rate_below_baseline", gate["failure_reasons"])
        self.assertIn("probe_no_expression_missing", gate["failure_reasons"])
        self.assertIn("env_expression_not_beating_noexpr", gate["failure_reasons"])
        self.assertIn("probe_ready_fraction_too_low", gate["failure_reasons"])
        self.assertIn("probe_muted_fraction_too_high", gate["failure_reasons"])
        self.assertIn("mechanics_fit_too_low", gate["failure_reasons"])
        self.assertIn("probe_leakage_too_high", gate["failure_reasons"])

    def test_latent_win_gate_passes_when_all_conditions_clear(self):
        gate = evaluate_latent_win_gate(
            benchmark_profile="full",
            seed_count=5,
            baseline_episode_solves=[200, 210, 205, 215, 208],
            baseline_completed_episodes=[200, 210, 205, 215, 208],
            probe_episode_solves=[140, 150, 145, 148, 146],
            probe_completed_episodes=[140, 150, 145, 148, 146],
            baseline_step_solves=[40000, 41000, 40500, 42000, 41500],
            baseline_total_env_steps=[40000, 41000, 40500, 42000, 41500],
            probe_step_solves=[25000, 25500, 24800, 25200, 25100],
            probe_total_env_steps=[25000, 25500, 24800, 25200, 25100],
            probe_env_expression_delta=[25.0, 31.0, 28.0, 30.0, 29.0],
            probe_no_expression_available=True,
            probe_ready_fraction=[0.60] * 5,
            probe_muted_fraction=[0.20] * 5,
            latent_mechanics_fit=[0.65] * 5,
            latent_neighbor_alignment=[0.30] * 5,
            latent_split_retrieval=[0.50] * 5,
            latent_gap_ratio=[0.75] * 5,
            latent_probe_leakage=[0.10] * 5,
            latent_uncert_error_corr=[0.35] * 5,
            full_system_state_only_ablation_delta=[60.0] * 5,
            full_system_zero_context_ablation_delta=[58.0] * 5,
            full_system_shuffled_context_ablation_delta=[54.0] * 5,
            full_system_stale_context_ablation_delta=[56.0] * 5,
        )

        self.assertTrue(gate["pass"])
        self.assertEqual(gate["failure_reasons"], [])

    def test_probe_strict_usage_status_buckets_enabled_fraction(self):
        self.assertEqual(probe_strict_usage_status(None), "unused")
        self.assertEqual(probe_strict_usage_status(0.0), "unused")
        self.assertEqual(probe_strict_usage_status(0.25), "intermittent")
        self.assertEqual(probe_strict_usage_status(0.50), "active")


if __name__ == "__main__":
    unittest.main()
