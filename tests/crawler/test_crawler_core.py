import unittest

import numpy as np

from teenyreason.crawler import (
    Crawler,
    LinearMessageProjector,
    RoundRobinQueryPolicy,
    ScriptedWorldAdapter,
    SupportLimitStopPolicy,
    VectorBeliefBackend,
    crawler_message_to_env_expression,
    env_expression_to_crawler_message,
)
from teenyreason.crawler.compat import legacy_step_result_to_crawler_step
from teenyreason.crawler.types import (
    ControllerBeliefContext,
    CrawlerMessage,
    EnvExpression,
    LegacyCrawlerStepResult,
    MetricBelief,
    PredictiveBelief,
    UncertaintyEstimate,
)


class CrawlerCoreTests(unittest.TestCase):
    def test_generic_crawler_collects_evidence_and_emits_message(self):
        crawler = Crawler(
            world=ScriptedWorldAdapter(
                query_payloads={
                    "scan": {"vector": np.asarray([1.0, 0.0], dtype=np.float32)},
                    "stress": {"vector": np.asarray([0.0, 1.0], dtype=np.float32)},
                },
                source_prefix="fake",
            ),
            belief_backend=VectorBeliefBackend(vector_key="vector"),
            query_policy=RoundRobinQueryPolicy(),
            stop_policy=SupportLimitStopPolicy(min_support=2),
            message_projector=LinearMessageProjector(),
            max_steps=3,
        )

        result = crawler.run(seed=7)

        self.assertEqual(result.stop_reason, "support_limit")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.final_belief_state.support_size, 2)
        np.testing.assert_allclose(
            result.final_belief_state.latent,
            np.asarray([0.5, 0.5], dtype=np.float32),
        )
        self.assertIn("query_names", result.final_message.metadata)

    def test_message_roundtrip_preserves_vector_and_confidence(self):
        message = CrawlerMessage(
            vector=np.asarray([0.2, -0.4, 0.8], dtype=np.float32),
            confidence=0.75,
            ready=True,
            uncertainty=0.12,
            metadata={"bits_per_dim": 8},
        )

        env_expression = crawler_message_to_env_expression(message, compressed=True)
        roundtrip = env_expression_to_crawler_message(env_expression)

        self.assertTrue(env_expression.compressed)
        self.assertEqual(roundtrip.ready, message.ready)
        self.assertAlmostEqual(roundtrip.confidence, message.confidence)
        np.testing.assert_allclose(roundtrip.vector, message.vector)

    def test_legacy_step_result_can_be_viewed_as_generic_step(self):
        predictive = PredictiveBelief(
            mean_raw=np.asarray([0.1, 0.2], dtype=np.float32),
            mean_unit=np.asarray([0.4, 0.6], dtype=np.float32),
            logvar=np.asarray([0.0, 0.0], dtype=np.float32),
            view_spread=np.asarray([0.1, 0.1], dtype=np.float32),
            env_param_mean=np.asarray([1.0], dtype=np.float32),
            env_param_std=np.asarray([0.2], dtype=np.float32),
            future_probe_error=0.3,
            support_count=2,
            support_diversity_ratio=0.5,
        )
        metric = MetricBelief(
            mean_raw=np.asarray([0.1, 0.2], dtype=np.float32),
            mean_unit=np.asarray([0.4, 0.6], dtype=np.float32),
            split_mean_a=np.asarray([0.0, 1.0], dtype=np.float32),
            split_mean_b=np.asarray([1.0, 0.0], dtype=np.float32),
            nearest_between_distance=0.2,
            gap_ratio=0.8,
        )
        uncertainty = UncertaintyEstimate(
            vector=np.asarray([0.3, 0.7], dtype=np.float32),
            scalar=0.5,
            feature_names=("a", "b"),
            feature_weights=np.asarray([0.6, 0.4], dtype=np.float32),
        )
        env_expression = EnvExpression(
            vector=np.asarray([0.2, 0.4], dtype=np.float32),
            confidence=0.9,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
        )
        legacy = LegacyCrawlerStepResult(
            predictive_belief=predictive,
            metric_belief=metric,
            uncertainty=uncertainty,
            env_expression=env_expression,
            controller_context=ControllerBeliefContext(
                mechanics_code=np.asarray([0.1], dtype=np.float32),
                affordance_code=np.asarray([0.2], dtype=np.float32),
                confidence=0.9,
                uncertainty_scalar=0.1,
            ),
            expected_family_gain={"scan": {"value": 0.5}},
            realized_family_gain={"scan": 0.2},
            stop_reason="support_limit",
        )

        step = legacy_step_result_to_crawler_step(
            legacy,
            query_name="scan",
            source_id="legacy:1",
        )

        self.assertEqual(step.query_name, "scan")
        self.assertEqual(step.belief_state.support_size, 2)
        self.assertEqual(step.stop_reason, "support_limit")
        self.assertIn("expected_family_gain", step.metadata)


if __name__ == "__main__":
    unittest.main()
