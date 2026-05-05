import unittest

from teenyreason.multidomain.diagnostics.sample_performance import (
    build_sample_performance_block,
    sample_performance_row,
)


class SamplePerformanceTests(unittest.TestCase):
    def test_language_reports_peak_samples_and_no_solve_threshold(self):
        domain = {
            "rows": [
                {"train_char_budget": 50_000, "baseline_bpc": 3.4, "belief_bpc": 3.3},
                {"train_char_budget": 100_000, "baseline_bpc": 3.2, "belief_bpc": 3.21},
            ]
        }

        block = build_sample_performance_block("language", domain)
        row = sample_performance_row("language", {"sample_performance": block})

        self.assertEqual(row["sample_axis"], "train_chars")
        self.assertEqual(row["baseline_samples_to_peak"], 100_000.0)
        self.assertEqual(row["crawler_samples_to_peak"], 100_000.0)
        self.assertAlmostEqual(row["peak_score_delta"], -0.01)
        self.assertAlmostEqual(row["best_solver_gain"], 0.1)
        self.assertEqual(row["best_solver_gain_sample"], 50_000.0)
        self.assertFalse(row["solve_available"])

    def test_image_reports_sample_savings_to_accuracy_threshold(self):
        domain = {
            "rows": [
                {"label_budget": 256, "baseline_accuracy": 0.87, "belief_accuracy": 0.91},
                {"label_budget": 1024, "baseline_accuracy": 0.92, "belief_accuracy": 0.93},
            ]
        }

        row = sample_performance_row(
            "image",
            {"sample_performance": build_sample_performance_block("image", domain)},
        )

        self.assertTrue(row["solve_available"])
        self.assertEqual(row["baseline_samples_to_solve"], 1024.0)
        self.assertEqual(row["crawler_samples_to_solve"], 256.0)
        self.assertEqual(row["solve_sample_savings"], 768.0)
        self.assertTrue(row["crawler_wins_solve_samples"])

    def test_cartpole_uses_research_metric_sample_counts(self):
        domain = {
            "metrics": {
                "baseline_solve_steps": 8000.0,
                "probe_solve_steps": 12000.0,
                "baseline_steps_to_peak": 7600.0,
                "probe_steps_to_peak": 11600.0,
                "baseline_best_return": 480.0,
                "probe_best_return": 500.0,
                "probe_step_savings_vs_baseline": -4000.0,
                "probe_step_savings_vs_no_expression": 2000.0,
            }
        }

        row = sample_performance_row(
            "cartpole",
            {"sample_performance": build_sample_performance_block("cartpole", domain)},
        )

        self.assertTrue(row["solve_available"])
        self.assertEqual(row["solve_sample_savings"], -4000.0)
        self.assertEqual(row["peak_sample_savings"], -4000.0)
        self.assertEqual(row["peak_score_delta"], 20.0)
        self.assertEqual(row["state"], "crawler_loses_solve_samples")


if __name__ == "__main__":
    unittest.main()
