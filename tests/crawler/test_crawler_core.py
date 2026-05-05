import unittest

import numpy as np

from teenyreason.crawler import (
    Crawler,
    CrawlerMindMap,
    IntrinsicDrive,
    LinearMessageProjector,
    RoundRobinQueryPolicy,
    ScriptedWorldAdapter,
    SupportLimitStopPolicy,
    VectorBeliefBackend,
    crawler_message_to_env_expression,
    env_expression_to_crawler_message,
)
from teenyreason.cognition.imagination import Proposal, Target, TargetBank
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
from teenyreason.cognition.skills import SkillMemory
from teenyreason.cognition.skills.schema import IntrinsicGoal, SkillRecord


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

    def test_mindmap_residual_memory_corrects_similar_overestimates(self):
        mindmap = CrawlerMindMap()
        first = Proposal(
            proposal_id="p0",
            domain="control",
            context_id="ctx",
            context_latent=(0.1, 0.2),
            target=None,
            intervention=np.asarray([[0.5], [0.5]], dtype=np.float32),
            predicted_lift=40.0,
            predicted_utility=40.0,
            trust_score=0.8,
            support_confidence=1.0,
            horizon=2,
        )
        initial = mindmap.correction_for_proposal(first)
        mindmap.add_proposal(first, initial)
        mindmap.add_residual(
            first,
            real_lift=5.0,
            accepted=False,
            validation_cost=2.0,
            correction=initial,
        )

        second = Proposal(
            proposal_id="p1",
            domain="control",
            context_id="ctx",
            context_latent=(0.11, 0.19),
            target=None,
            intervention=np.asarray([[0.48], [0.52]], dtype=np.float32),
            predicted_lift=30.0,
            predicted_utility=30.0,
            trust_score=0.8,
            support_confidence=1.0,
            horizon=2,
        )
        correction = mindmap.correction_for_proposal(second)
        summary = mindmap.summary()

        self.assertGreater(correction.penalty, 20.0)
        self.assertLess(correction.corrected_predicted_lift, 10.0)
        self.assertEqual(summary["crawler_mindmap_residual_memory_count"], 1.0)
        self.assertGreater(summary["crawler_mindmap_node_count"], 0.0)

    def test_intrinsic_drive_penalizes_known_bad_imagination(self):
        mindmap = CrawlerMindMap()
        proposal = Proposal(
            proposal_id="bad",
            domain="control",
            context_id="ctx",
            context_latent=(0.0, 0.0),
            target=None,
            intervention=np.asarray([[1.0]], dtype=np.float32),
            predicted_lift=20.0,
            trust_score=0.9,
        )
        correction = mindmap.correction_for_proposal(proposal)
        mindmap.add_residual(
            proposal,
            real_lift=-5.0,
            accepted=False,
            validation_cost=3.0,
            correction=correction,
        )
        targets = TargetBank(
            [
                Target("bad", "high_return_state", latent=(0.0, 0.0), utility=20.0, stability=0.2),
                Target("stable", "high_return_state", latent=(4.0, 4.0), utility=5.0, stability=1.0),
            ]
        )

        ranked = IntrinsicDrive().refresh(targets, mindmap, SkillMemory(), None)

        bad = [target for target in ranked if target.target_id == "target:bad"][0]
        stable = [target for target in ranked if target.target_id == "target:stable"][0]
        self.assertGreater(bad.signal.residual_penalty, 0.5)
        self.assertGreater(stable.score, bad.score)

    def test_mindmap_promotes_skill_and_composition_relations(self):
        goal = IntrinsicGoal(
            goal_id=1,
            goal_kind="return_to_stable_island",
            target_delta=np.asarray([1.0, 0.0], dtype=np.float32),
            anchor_observation=np.asarray([0.0, 0.0], dtype=np.float32),
            priority=1.0,
            source="test",
        )
        first = SkillRecord(
            skill_id=1,
            goal=goal,
            initiation_observation=np.asarray([0.0, 0.0], dtype=np.float32),
            termination_observation=np.asarray([1.0, 0.0], dtype=np.float32),
            actions=np.asarray([[0.2], [0.2]], dtype=np.float32),
            outcome_delta=np.asarray([1.0, 0.0], dtype=np.float32),
            real_return_lift=2.0,
            survival_lift=8.0,
            terminal_avoid=1.0,
            reliability=0.8,
        )
        second = SkillRecord(
            skill_id=2,
            goal=goal,
            initiation_observation=np.asarray([1.02, 0.01], dtype=np.float32),
            termination_observation=np.asarray([2.0, 0.0], dtype=np.float32),
            actions=np.asarray([[0.1], [0.1]], dtype=np.float32),
            outcome_delta=np.asarray([1.0, 0.0], dtype=np.float32),
            real_return_lift=1.0,
            survival_lift=4.0,
            terminal_avoid=0.0,
            reliability=0.7,
        )
        mindmap = CrawlerMindMap()

        mindmap.add_skill_node(first)
        mindmap.add_skill_node(second)
        mindmap.add_skill_compositions([first, second])
        summary = mindmap.summary()

        self.assertEqual(summary["crawler_mindmap_skill_count"], 2.0)
        self.assertGreater(summary["crawler_mindmap_skill_composition_count"], 0.0)
        self.assertGreater(summary["crawler_mindmap_factor_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
