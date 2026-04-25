import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from teenyreason.app.benchmark import (
    benchmark_profile_flags,
    default_seeds_for_profile,
    print_solve_summary,
    resolve_benchmark_profile,
    solve_eval_episodes_for_profile,
)
from teenyreason.app.config import build_experiment_config
from teenyreason.envs import CONTINUOUS_CARTPOLE_NAME, make_env
from teenyreason.probe.probe_data import default_env_params
from teenyreason.rl.full_system.affordance import choose_affordance_action, generate_candidate_actions
from teenyreason.rl.full_system.affordance_eval import (
    _build_evaluation_fixtures,
    _checkpoint_selection_key,
)
from teenyreason.rl.full_system.simulator_fanout import (
    PersistentFanoutLabelCache,
    SimulatorFanoutAdapter,
)
from teenyreason.rl.probe_policy.types import MatchedEvalSummary


class SimulatorFanoutTests(unittest.TestCase):
    def test_snapshot_restore_keeps_cartpole_branching_deterministic(self):
        env = make_env(CONTINUOUS_CARTPOLE_NAME)
        try:
            env.reset(seed=7)
            adapter = SimulatorFanoutAdapter(CONTINUOUS_CARTPOLE_NAME)
            candidates = np.asarray([[-0.5], [0.0], [0.5]], dtype=np.float32)

            baseline = adapter.snapshot(env)
            first = adapter.evaluate_constant_action_candidates(
                env,
                candidate_actions=candidates,
                horizon=4,
                gamma=0.99,
            )
            after_first = adapter.snapshot(env)
            second = adapter.evaluate_constant_action_candidates(
                env,
                candidate_actions=candidates,
                horizon=4,
                gamma=0.99,
            )
            after_second = adapter.snapshot(env)

            self.assertTrue(np.allclose(first["returns"], second["returns"]))
            self.assertTrue(np.allclose(first["risks"], second["risks"]))
            self.assertTrue(np.allclose(first["recoverability"], second["recoverability"]))
            self.assertTrue(np.allclose(baseline.state, after_first.state))
            self.assertTrue(np.allclose(baseline.state, after_second.state))
            self.assertEqual(baseline.elapsed_steps, after_first.elapsed_steps)
            self.assertEqual(baseline.elapsed_steps, after_second.elapsed_steps)
        finally:
            env.close()

    def test_persistent_label_cache_reuses_identical_branch_labels(self):
        env = make_env(CONTINUOUS_CARTPOLE_NAME)
        try:
            env.reset(seed=11)
            adapter = SimulatorFanoutAdapter(CONTINUOUS_CARTPOLE_NAME)
            snapshot = adapter.snapshot(env)
            candidates = np.asarray([[-0.5], [0.0], [0.5]], dtype=np.float32)
            with TemporaryDirectory() as tmpdir:
                cache = PersistentFanoutLabelCache(
                    env_name=CONTINUOUS_CARTPOLE_NAME,
                    cache_dir=Path(tmpdir),
                )
                first = cache.get_or_compute(
                    env=env,
                    adapter=adapter,
                    candidate_actions=candidates,
                    horizon=4,
                    gamma=0.99,
                    snapshot=snapshot,
                )
                second = cache.get_or_compute(
                    env=env,
                    adapter=adapter,
                    candidate_actions=candidates,
                    horizon=4,
                    gamma=0.99,
                    snapshot=snapshot,
                )

                self.assertTrue(np.allclose(first.returns, second.returns))
                self.assertTrue(np.allclose(first.scores, second.scores))
                self.assertEqual(first.best_idx, second.best_idx)
                self.assertGreaterEqual(len(list((Path(tmpdir) / CONTINUOUS_CARTPOLE_NAME).glob("*.npz"))), 1)
        finally:
            env.close()


class CheapBeliefActionTests(unittest.TestCase):
    def test_generate_candidate_actions_respects_bounds(self):
        candidates = generate_candidate_actions(
            mean_action=np.asarray([1.5], dtype=np.float32),
            action_low=np.asarray([-1.0], dtype=np.float32),
            action_high=np.asarray([1.0], dtype=np.float32),
        )
        self.assertEqual(candidates.shape, (5, 1))
        self.assertTrue(np.all(candidates <= 1.0))
        self.assertTrue(np.all(candidates >= -1.0))

    def test_trust_gating_switches_between_actor_blend_and_best_candidate(self):
        class StubController:
            def __init__(self, trust: float):
                self.trust = float(trust)

            def forward_with_hidden(self, state_t, context_t, hidden_state=None):
                del state_t, context_t
                return (
                    torch.tensor([[0.0]], dtype=torch.float32),
                    torch.tensor([0.0], dtype=torch.float32),
                    torch.zeros((1, 1), dtype=torch.float32),
                    {
                        "trust": torch.tensor([self.trust], dtype=torch.float32),
                        "trunk": torch.zeros((1, 2), dtype=torch.float32),
                    },
                )

            def evaluate_candidates(self, trunk, candidate_actions):
                del trunk, candidate_actions
                returns = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0]], dtype=torch.float32)
                risks = torch.zeros_like(returns)
                recoverability = torch.zeros_like(returns)
                return returns, risks, recoverability

        state_t = torch.zeros((1, 4), dtype=torch.float32)
        context_t = torch.zeros((1, 6), dtype=torch.float32)
        action_low = np.asarray([-1.0], dtype=np.float32)
        action_high = np.asarray([1.0], dtype=np.float32)

        actor_only = choose_affordance_action(
            controller=StubController(0.10),
            state_t=state_t,
            context_t=context_t,
            action_low=action_low,
            action_high=action_high,
            hidden_state=None,
        )
        blended = choose_affordance_action(
            controller=StubController(0.20),
            state_t=state_t,
            context_t=context_t,
            action_low=action_low,
            action_high=action_high,
            hidden_state=None,
        )
        best = choose_affordance_action(
            controller=StubController(0.40),
            state_t=state_t,
            context_t=context_t,
            action_low=action_low,
            action_high=action_high,
            hidden_state=None,
        )

        self.assertAlmostEqual(float(actor_only.action[0]), 0.0, places=6)
        self.assertAlmostEqual(float(blended.action[0]), 0.5, places=6)
        self.assertAlmostEqual(float(best.action[0]), 1.0, places=6)

    def test_force_state_only_uses_state_scores_instead_of_belief_residual(self):
        class ResidualStubController:
            def forward_with_hidden(self, state_t, context_t, hidden_state=None):
                del state_t, context_t, hidden_state
                return (
                    torch.tensor([[0.0]], dtype=torch.float32),
                    torch.tensor([0.0], dtype=torch.float32),
                    torch.zeros((1, 1), dtype=torch.float32),
                    {
                        "trust": torch.tensor([0.90], dtype=torch.float32),
                        "state_trunk": torch.zeros((1, 2), dtype=torch.float32),
                        "context_features": torch.zeros((1, 2), dtype=torch.float32),
                        "confidence": torch.tensor([1.0], dtype=torch.float32),
                        "uncertainty": torch.tensor([0.0], dtype=torch.float32),
                    },
                )

            def evaluate_candidate_scores(
                self,
                *,
                state_trunk,
                context_features,
                candidate_actions,
                trust,
                confidence=None,
                uncertainty=None,
            ):
                del state_trunk, context_features, candidate_actions, trust, confidence, uncertainty
                state_scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
                final_scores = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]], dtype=torch.float32)
                zeros = torch.zeros_like(state_scores)
                return {
                    "return": state_scores,
                    "risk": zeros,
                    "recoverability": zeros,
                    "state_scores": state_scores,
                    "belief_residual": final_scores - state_scores,
                    "final_scores": final_scores,
                }

        state_t = torch.zeros((1, 4), dtype=torch.float32)
        context_t = torch.zeros((1, 6), dtype=torch.float32)
        action_low = np.asarray([-1.0], dtype=np.float32)
        action_high = np.asarray([1.0], dtype=np.float32)

        belief_adjusted = choose_affordance_action(
            controller=ResidualStubController(),
            state_t=state_t,
            context_t=context_t,
            action_low=action_low,
            action_high=action_high,
            hidden_state=None,
        )
        state_only = choose_affordance_action(
            controller=ResidualStubController(),
            state_t=state_t,
            context_t=context_t,
            action_low=action_low,
            action_high=action_high,
            hidden_state=None,
            force_state_only=True,
        )

        self.assertAlmostEqual(float(belief_adjusted.action[0]), 0.0, places=6)
        self.assertAlmostEqual(float(state_only.action[0]), 1.0, places=6)


class EvaluationFixtureTests(unittest.TestCase):
    def test_fixture_build_collects_support_once_per_episode(self):
        class DummyContext:
            def __init__(self, vector):
                self.vector = np.asarray(vector, dtype=np.float32)

        class DummyStepResult:
            def __init__(self, vector):
                self.controller_context = DummyContext(vector)

        class DummyBundle:
            full_system_controller_dim = 6

            def build_oracle_controller_context(self, env_params):
                del env_params
                return DummyContext(np.ones((6,), dtype=np.float32))

        env = make_env(CONTINUOUS_CARTPOLE_NAME)
        try:
            base_physics = default_env_params(CONTINUOUS_CARTPOLE_NAME, env)
        finally:
            env.close()

        call_counter = {"count": 0}

        def fake_collect_support_context(**kwargs):
            del kwargs
            call_counter["count"] += 1
            return {
                "step_result": DummyStepResult(np.arange(6, dtype=np.float32)),
                "belief": None,
                "belief_hidden": None,
                "belief_posteriors": [],
                "probe_steps_total": 8,
                "probe_windows_total": 2,
                "probe_count": 2,
            }

        with patch(
            "teenyreason.rl.full_system._affordance_train_impl._collect_support_context",
            side_effect=fake_collect_support_context,
        ):
            fixtures = _build_evaluation_fixtures(
                crawler_bundle=DummyBundle(),
                encoder=None,
                belief_aggregator=None,
                env_param_predictor=None,
                env_future_predictor=None,
                predictor=None,
                env_name=CONTINUOUS_CARTPOLE_NAME,
                action_values=np.asarray([[0.0]], dtype=np.float32),
                window_size=16,
                randomize_physics=False,
                base_physics=base_physics,
                base_probe_episodes=2,
                max_probe_episodes=3,
                probe_adaptive_budget=False,
                uncertainty_probe_threshold=0.2,
                surprise_probe_threshold=0.7,
                eval_episodes=3,
                seed=5,
                use_context=True,
            )

            self.assertEqual(call_counter["count"], 3)


class BeliefCheckpointSelectionTests(unittest.TestCase):
    def test_checkpoint_selection_prefers_better_learned_over_state_only_delta(self):
        stronger_attribution = _checkpoint_selection_key(
            learned_summary=MatchedEvalSummary(
                returns=[420.0, 410.0, 430.0],
                episode_total_env_steps=[15000, 15100, 14900],
                mean_return=420.0,
                mean_total_env_steps=15000.0,
                solved_count=0,
                fixture_count=3,
            ),
            state_only_summary=MatchedEvalSummary(
                returns=[395.0, 400.0, 405.0],
                episode_total_env_steps=[15300, 15400, 15200],
                mean_return=400.0,
                mean_total_env_steps=15300.0,
                solved_count=0,
                fixture_count=3,
            ),
            zero_context_summary=MatchedEvalSummary(
                returns=[360.0, 370.0, 365.0],
                episode_total_env_steps=[15800, 15900, 15700],
                mean_return=365.0,
                mean_total_env_steps=15800.0,
                solved_count=0,
                fixture_count=3,
            ),
            stale_context_summary=MatchedEvalSummary(
                returns=[372.0, 378.0, 375.0],
                episode_total_env_steps=[15600, 15700, 15500],
                mean_return=375.0,
                mean_total_env_steps=15600.0,
                solved_count=0,
                fixture_count=3,
            ),
            shuffled_context_summary=MatchedEvalSummary(
                returns=[368.0, 372.0, 370.0],
                episode_total_env_steps=[15750, 15850, 15650],
                mean_return=370.0,
                mean_total_env_steps=15750.0,
                solved_count=0,
                fixture_count=3,
            ),
            training_return=250.0,
        )
        higher_training_peak = _checkpoint_selection_key(
            learned_summary=MatchedEvalSummary(
                returns=[420.0, 410.0, 430.0],
                episode_total_env_steps=[15000, 15100, 14900],
                mean_return=420.0,
                mean_total_env_steps=15000.0,
                solved_count=0,
                fixture_count=3,
            ),
            state_only_summary=MatchedEvalSummary(
                returns=[417.0, 418.0, 419.0],
                episode_total_env_steps=[15100, 15200, 15050],
                mean_return=418.0,
                mean_total_env_steps=15116.67,
                solved_count=0,
                fixture_count=3,
            ),
            zero_context_summary=MatchedEvalSummary(
                returns=[419.0, 420.0, 418.0],
                episode_total_env_steps=[15150, 15250, 15100],
                mean_return=419.0,
                mean_total_env_steps=15166.67,
                solved_count=0,
                fixture_count=3,
            ),
            stale_context_summary=MatchedEvalSummary(
                returns=[418.0, 417.0, 419.0],
                episode_total_env_steps=[15125, 15225, 15075],
                mean_return=418.0,
                mean_total_env_steps=15141.67,
                solved_count=0,
                fixture_count=3,
            ),
            shuffled_context_summary=MatchedEvalSummary(
                returns=[416.0, 417.0, 418.0],
                episode_total_env_steps=[15100, 15200, 15050],
                mean_return=417.0,
                mean_total_env_steps=15116.67,
                solved_count=0,
                fixture_count=3,
            ),
            training_return=300.0,
        )

        self.assertGreater(stronger_attribution, higher_training_peak)


class BenchmarkProfileTests(unittest.TestCase):
    def test_cartpole_fast_profile_defaults_and_flags(self):
        config = build_experiment_config(CONTINUOUS_CARTPOLE_NAME)
        self.assertEqual(resolve_benchmark_profile(config), "fast")
        self.assertEqual(default_seeds_for_profile(config.benchmark_profile), [0, 1])
        self.assertEqual(solve_eval_episodes_for_profile(config), 1)
        self.assertEqual(config.encoder_belief_subset_count, 10)
        self.assertEqual(config.encoder_belief_subset_size, 6)
        self.assertEqual(config.probe_max_steps, 320)

        fast_flags = benchmark_profile_flags("fast")
        self.assertFalse(fast_flags["run_probe_shadow"])
        self.assertTrue(fast_flags["run_belief_controller"])
        self.assertFalse(fast_flags["run_belief_controller_oracle"])
        self.assertTrue(fast_flags["run_sim_fanout"])
        self.assertFalse(fast_flags["run_archived_planner"])

        full_config = replace(config, benchmark_profile="full")
        self.assertEqual(solve_eval_episodes_for_profile(full_config), full_config.solve_eval_episodes)
        self.assertTrue(benchmark_profile_flags("archived_planner")["run_archived_planner"])

    def test_not_run_solve_summary_prints_not_run(self):
        buffer = StringIO()
        with redirect_stdout(buffer):
            print_solve_summary("Shadow", [None, None], [0, 0])
        self.assertIn("not run", buffer.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
