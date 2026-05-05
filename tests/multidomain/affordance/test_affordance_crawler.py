import unittest

from teenyreason.multidomain.suite import MultidomainSuiteConfig, build_suite_payload
from teenyreason.multidomain.affordance import (
    AffordanceCrawlerSuiteConfig,
    PersistentAffordanceConfig,
    run_affordance_crawler_suite,
    run_persistent_affordance_crawler,
)
from teenyreason.multidomain.decision_crawler import (
    BoardDecisionLocalAdapter,
    LanguageDecisionLocalAdapter,
)


class PersistentAffordanceCrawlerTests(unittest.TestCase):
    def test_language_reuse_flips_probe_economics_positive(self):
        result = run_persistent_affordance_crawler(
            LanguageDecisionLocalAdapter(),
            seeds=(0, 1, 2),
            config=PersistentAffordanceConfig(reuse_horizon=16, max_expensive_probes=1),
        )

        self.assertGreater(result["regret_reduction"], 0.0)
        self.assertGreater(result["net_value_after_reuse"], 0.0)
        self.assertLess(result["amortized_probe_cost"], result["total_probe_cost"])
        self.assertLessEqual(result["break_even_reuse_count"], result["reuse_horizon"])
        self.assertEqual(result["verdict"], "persistent_affordance_economics_positive")

    def test_board_affordance_uses_passive_updates_and_one_probe(self):
        result = run_persistent_affordance_crawler(
            BoardDecisionLocalAdapter(),
            seeds=(1, 3),
            config=PersistentAffordanceConfig(reuse_horizon=9, max_expensive_probes=1),
        )

        self.assertEqual(result["affordance_decision_score"], 1.0)
        self.assertGreater(result["passive_update_fraction"], 0.8)
        self.assertLess(result["dedicated_probe_fraction"], 0.2)
        self.assertGreater(result["net_value_after_reuse"], 0.0)

    def test_suite_payload_exposes_affordance_rows(self):
        suite_result = run_affordance_crawler_suite(AffordanceCrawlerSuiteConfig())
        payload = build_suite_payload(
            config=MultidomainSuiteConfig(run_rl_benchmark=False),
            run_id="affordance-test",
            started_at=1.0,
            results={"rl": {}, "affordance_crawler": suite_result},
            detail_paths={},
        )

        rows = payload["cross_domain"]["affordance_crawler_rows"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["domain"], "cartpole")
        for row in rows:
            self.assertGreater(row["net_value_after_reuse"], 0.0)
            self.assertEqual(row["verdict"], "persistent_affordance_economics_positive")


if __name__ == "__main__":
    unittest.main()
