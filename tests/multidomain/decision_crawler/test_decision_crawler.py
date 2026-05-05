import unittest

from teenyreason.multidomain.suite import MultidomainSuiteConfig, build_suite_payload
from teenyreason.multidomain.decision_crawler import (
    BoardDecisionLocalAdapter,
    DecisionLocalCrawlerConfig,
    LanguageDecisionLocalAdapter,
    decision_local_crawler_row,
    run_decision_local_crawler,
    run_decision_local_crawler_suite,
)


class DecisionLocalCrawlerTests(unittest.TestCase):
    def test_language_crawler_beats_zero_and_shuffled_belief(self):
        result = run_decision_local_crawler(
            LanguageDecisionLocalAdapter(),
            seeds=tuple(range(6)),
            config=DecisionLocalCrawlerConfig(max_interventions=2, cost_weight=0.01),
        )

        self.assertGreater(result["crawler_decision_score"], result["baseline_decision_score"])
        self.assertGreater(result["crawler_decision_score"], result["zero_score"])
        self.assertGreater(result["crawler_decision_score"], result["shuffled_score"])
        self.assertGreater(result["regret_reduction"], 0.0)
        self.assertGreater(result["content_lift"], 0.0)

    def test_board_adapter_runs_through_same_decision_contract(self):
        result = run_decision_local_crawler(
            BoardDecisionLocalAdapter(),
            seeds=(1, 3, 5),
            config=DecisionLocalCrawlerConfig(max_interventions=1, cost_weight=0.01),
        )

        self.assertEqual(result["domain"], "board")
        self.assertEqual(result["crawler_decision_score"], 1.0)
        self.assertEqual(result["zero_score"], 0.0)
        self.assertGreater(result["voi"], 0.0)
        self.assertGreater(result["belief_entropy_reduction"], 0.0)

    def test_suite_returns_dashboard_rows_for_all_domains(self):
        result = run_decision_local_crawler_suite()
        rows = result["summary_rows"]
        domains = {row["domain"] for row in rows}

        self.assertEqual(domains, {"cartpole", "language", "image", "board"})
        for row in rows:
            table_row = decision_local_crawler_row(row["domain"], result[row["domain"]])
            self.assertEqual(table_row["domain"], row["domain"])
            self.assertIn("verdict", table_row)
            self.assertGreaterEqual(table_row["crawler_decision_score"], table_row["baseline_decision_score"])

    def test_suite_payload_exposes_decision_local_crawler_rows(self):
        crawler_result = run_decision_local_crawler_suite()
        payload = build_suite_payload(
            config=MultidomainSuiteConfig(run_rl_benchmark=False),
            run_id="decision-crawler-test",
            started_at=1.0,
            results={"rl": {}, "decision_local_crawler": crawler_result},
            detail_paths={},
        )

        rows = payload["cross_domain"]["decision_local_crawler_rows"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["domain"], "cartpole")
        self.assertGreater(rows[1]["regret_reduction"], 0.0)


if __name__ == "__main__":
    unittest.main()
