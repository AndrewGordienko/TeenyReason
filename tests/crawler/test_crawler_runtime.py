import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from teenyreason.crawler.runtime import (
    CrawlerExpressionResult,
    CrawlerRuntimeConfig,
    LatentCrawler,
    latest_crawler_checkpoint,
)
from teenyreason.crawler.types import (
    ControllerBeliefContext,
    EnvExpression,
    LegacyCrawlerStepResult,
)
from teenyreason.envs import CONTINUOUS_CARTPOLE_NAME, make_env


class _DummyBundle:
    def __init__(self):
        self.encoder = object()
        self.belief_aggregator = object()
        self.env_param_predictor = object()
        self.env_future_predictor = object()
        self.predictor = object()
        self.device = torch.device("cpu")
        self.action_vocab_size = 9
        self.family_names = ("passive_decay", "impulse_left")


class _ContextSink:
    def __init__(self):
        self.context = None

    def set_crawler_context(self, context):
        self.context = context


def _step_result() -> LegacyCrawlerStepResult:
    expression = EnvExpression(
        vector=np.asarray([0.1, 0.2], dtype=np.float32),
        confidence=0.7,
        ready=True,
        uncertainty_scalar=0.3,
        compressed=False,
        metadata={},
    )
    context = ControllerBeliefContext(
        mechanics_code=np.asarray([0.4], dtype=np.float32),
        affordance_code=np.asarray([0.5], dtype=np.float32),
        confidence=0.6,
        uncertainty_scalar=0.3,
        metadata={},
    )
    return LegacyCrawlerStepResult(
        predictive_belief=None,
        metric_belief=None,
        uncertainty=None,
        env_expression=expression,
        controller_context=context,
        expected_family_gain={},
        realized_family_gain={},
        stop_reason=None,
    )


class CrawlerRuntimeTests(unittest.TestCase):
    def test_runtime_config_uses_best_cartpole_defaults(self):
        config = CrawlerRuntimeConfig.from_env(CONTINUOUS_CARTPOLE_NAME)

        self.assertEqual(config.env_name, CONTINUOUS_CARTPOLE_NAME)
        self.assertEqual(config.window_size, 16)
        self.assertEqual(config.base_probe_episodes, 2)
        self.assertEqual(config.max_probe_episodes, 3)

    def test_latest_checkpoint_uses_newest_supported_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "continuous_cartpole_ppo_seed_0_probe_ppo_checkpoint.pt"
            newer = root / "continuous_cartpole_ppo_seed_1_probe_ppo_checkpoint.pt"
            older.write_text("old", encoding="utf-8")
            newer.write_text("new", encoding="utf-8")
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            self.assertEqual(
                latest_crawler_checkpoint(CONTINUOUS_CARTPOLE_NAME, artifacts_dir=root),
                newer,
            )

    def test_expression_result_feeds_explicit_context_hook(self):
        result = CrawlerExpressionResult(
            env_expression=_step_result().env_expression,
            controller_context=_step_result().controller_context,
            step_result=_step_result(),
            belief=np.asarray([1.0], dtype=np.float32),
            probe_windows=(),
            probe_count=2,
            probe_steps=32,
            metadata={"env_name": CONTINUOUS_CARTPOLE_NAME},
        )
        sink = _ContextSink()

        returned = result.feed(sink)

        self.assertIs(returned, sink)
        self.assertTrue(sink.context["ready"])
        self.assertEqual(sink.context["probe_count"], 2)

    def test_expression_result_has_compact_summary(self):
        result = CrawlerExpressionResult(
            env_expression=_step_result().env_expression,
            controller_context=_step_result().controller_context,
            step_result=_step_result(),
            belief=np.asarray([1.0], dtype=np.float32),
            probe_windows=(),
            probe_count=2,
            probe_steps=32,
            metadata={},
        )

        self.assertEqual(
            repr(result),
            "Belief(ready=True, confidence=0.700, uncertainty=0.300, dim=2, probe_steps=32)",
        )

    def test_latent_crawler_runs_support_collector_as_runtime_facade(self):
        captured = {}

        def fake_collect_support_context(**kwargs):
            captured.update(kwargs)
            return {
                "belief": np.asarray([0.1, 0.2], dtype=np.float32),
                "step_result": _step_result(),
                "probe_count": 2,
                "probe_steps_total": 32,
                "probe_windows_total": 2,
                "probe_windows": [{"probe_family": "passive_decay"}],
            }

        config = CrawlerRuntimeConfig.from_env(CONTINUOUS_CARTPOLE_NAME)
        config = CrawlerRuntimeConfig(
            env_name=config.env_name,
            action_bins=config.action_bins,
            window_size=config.window_size,
            base_probe_episodes=config.base_probe_episodes,
            max_probe_episodes=config.max_probe_episodes,
            probe_adaptive_budget=config.probe_adaptive_budget,
            uncertainty_probe_threshold=config.uncertainty_probe_threshold,
            surprise_probe_threshold=config.surprise_probe_threshold,
            belief_bits_per_dim=8,
            belief_use_residual_sketch=True,
        )
        crawler = LatentCrawler.from_bundle(_DummyBundle(), config=config)
        env = make_env(CONTINUOUS_CARTPOLE_NAME)
        try:
            with patch(
                "teenyreason.rl.full_system.context_support.collect_support_context",
                side_effect=fake_collect_support_context,
            ):
                result = crawler.run(env, seed=7)
        finally:
            env.close()

        self.assertTrue(result.ready)
        self.assertEqual(result.probe_steps, 32)
        self.assertEqual(captured["belief_bits_per_dim"], 8)
        self.assertTrue(captured["belief_use_residual_sketch"])
        self.assertEqual(captured["env_name"], CONTINUOUS_CARTPOLE_NAME)

    def test_latent_crawler_is_callable(self):
        crawler = LatentCrawler.from_bundle(
            _DummyBundle(),
            config=CrawlerRuntimeConfig.from_env(CONTINUOUS_CARTPOLE_NAME),
        )
        with patch.object(crawler, "run", return_value="belief") as run_mock:
            result = crawler("env", seed=3)

        self.assertEqual(result, "belief")
        run_mock.assert_called_once_with("env", seed=3)


if __name__ == "__main__":
    unittest.main()
