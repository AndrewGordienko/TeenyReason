import unittest

import numpy as np

from teenyreason.multidomain.domains.cartpole import (
    _initial_state,
    nominal_world,
    world_for_seed,
)
from teenyreason.multidomain.planning import (
    AdvancedGymMPCConfig,
    CartPoleLatentDynamicsModel,
    CartPoleLatentMPCConfig,
    CartPolePlannerComparisonConfig,
    CartPolePersistentAffordanceMPCConfig,
    RandomShootingPlanner,
    ScenarioActorConfig,
    run_cartpole_latent_mpc_benchmark,
    run_cartpole_persistent_affordance_mpc,
    run_cartpole_planner_comparison,
    run_scenario_actor,
)
from teenyreason.multidomain.planning.gym_mpc import TransitionBatch
from teenyreason.multidomain.planning.generic.scenario_actor.actor_critic import ReplayBuffer


class CartPoleLatentMPCTests(unittest.TestCase):
    def test_replay_buffer_tracks_real_and_imagined_sources_after_interleaved_appends(self):
        config = AdvancedGymMPCConfig(env_name="ContinuousCartPole-v0")
        batch = TransitionBatch(
            observations=np.asarray([[0.0], [1.0]], dtype=np.float32),
            actions=np.asarray([[0.0], [0.1]], dtype=np.float32),
            rewards=np.asarray([0.0, 1.0], dtype=np.float32),
            next_observations=np.asarray([[0.1], [1.1]], dtype=np.float32),
            dones=np.asarray([0.0, 1.0], dtype=np.float32),
            episode_returns=np.asarray([1.0], dtype=np.float32),
        )
        replay = ReplayBuffer.from_batch(batch, config)
        replay.append(
            np.asarray([[2.0]], dtype=np.float32),
            np.asarray([[0.2]], dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
            np.asarray([[2.1]], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            imagined=True,
        )
        replay.append(
            np.asarray([[3.0]], dtype=np.float32),
            np.asarray([[0.3]], dtype=np.float32),
            np.asarray([2.0], dtype=np.float32),
            np.asarray([[3.1]], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            imagined=False,
        )

        self.assertEqual(replay.real_count, 3)
        self.assertEqual(replay.imagined_count, 1)
        self.assertEqual(replay.source_indices(real_only=True).tolist(), [0, 1, 3])
        self.assertEqual(replay.source_indices(imagined_only=True).tolist(), [2])

    def test_action_conditioned_model_predicts_true_world_better_than_nominal(self):
        truth = world_for_seed(3)
        model = CartPoleLatentDynamicsModel()
        state = _initial_state(3)
        actions = np.asarray([-1.0, 0.0, 1.0, 1.0, -1.0], dtype=np.float32)

        true_states = model.rollout(state, actions, belief=truth)
        nominal_states = model.rollout(state, actions, belief=nominal_world())

        self.assertEqual(true_states.shape, (6, 4))
        self.assertGreater(float(np.mean(np.square(true_states - nominal_states))), 0.0)

    def test_random_shooting_planner_returns_first_candidate_action(self):
        model = CartPoleLatentDynamicsModel()
        planner = RandomShootingPlanner(horizon=4, candidate_count=12)
        result = planner.choose_action(
            model,
            _initial_state(0),
            belief=world_for_seed(0),
            seed=0,
        )

        self.assertIn(result.action, (-1.0, 0.0, 1.0))
        self.assertEqual(result.actions.shape, (4,))
        self.assertEqual(result.predicted_states.shape, (5, 4))

    def test_cartpole_latent_mpc_reports_belief_and_ablation_rows(self):
        result = run_cartpole_latent_mpc_benchmark(
            CartPoleLatentMPCConfig(
                seeds=(0, 1, 2, 3),
                control_steps=16,
                horizon=4,
                candidate_count=16,
            )
        )
        row = result["rows"][0]

        self.assertEqual(result["hidden_target"], "cartpole_mechanics_action_conditioned_world_model")
        self.assertEqual(len(result["rows"]), 4)
        self.assertIn("belief_mpc_return", row)
        self.assertIn("no_belief_return", row)
        self.assertIn("oracle_mpc_return", row)
        self.assertIn("belief_action_match_oracle", row)
        self.assertIn("belief_k_step_prediction_mse", row)
        self.assertIn("belief_samples_to_peak_return", row)
        self.assertIn("net_env_sample_savings", row)
        self.assertGreaterEqual(result["decode_accuracy"], 0.5)
        self.assertGreaterEqual(result["belief_action_match_oracle"], 0.0)
        self.assertIn("belief_samples_to_solve", result)
        self.assertIn("net_samples_to_solve_savings", result)

    def test_cartpole_planner_comparison_reports_matched_arms_and_roi(self):
        result = run_cartpole_planner_comparison(
            CartPolePlannerComparisonConfig(
                seeds=(0, 1, 2, 3),
                matched_seeds=(0, 1, 2, 3),
                control_steps=16,
                horizon=4,
                candidate_count=16,
            )
        )
        arms = {row["arm"] for row in result["arms"]}
        row = result["rows"][0]

        self.assertIn("mpc_no_belief", arms)
        self.assertIn("mpc_crawler_belief", arms)
        self.assertIn("mpc_oracle", arms)
        self.assertIn("mpc_persistent_affordance", arms)
        self.assertIn("mpc_cheap_then_fallback", arms)
        self.assertIn("oracle_action", row)
        self.assertIn("action_regret_reduction", row)
        self.assertIn("probe_roi", row)
        self.assertIn("persistent_probe_roi", row)
        self.assertIn("crawler_vs_no_belief_mpc_sample_savings", result)
        self.assertIn("persistent_affordance_amortized_samples_to_solve", result)
        self.assertIn("persistent_affordance_amortized_vs_no_belief_mpc_sample_savings", result)
        self.assertGreaterEqual(result["fallback_wake_rate"], 0.0)

    def test_cartpole_persistent_affordance_mpc_reports_amortized_costs(self):
        result = run_cartpole_persistent_affordance_mpc(
            CartPolePersistentAffordanceMPCConfig(
                mpc=CartPoleLatentMPCConfig(
                    seeds=(0, 1, 2, 3),
                    control_steps=16,
                    horizon=4,
                    candidate_count=16,
                ),
                reuse_horizon=24,
            )
        )
        row = result["rows"][0]

        self.assertEqual(result["hidden_target"], "cartpole_mechanics_reused_affordance_world_model")
        self.assertEqual(len(result["rows"]), 4)
        self.assertIn("affordance_mpc_return", row)
        self.assertIn("selected_probe_cost", row)
        self.assertIn("amortized_probe_cost", row)
        self.assertIn("affordance_samples_to_solve_amortized", row)
        self.assertLessEqual(result["amortized_probe_cost"], result["probe_cost"])
        self.assertGreaterEqual(result["probe_future_adjusted_value"], 0.0)

    def test_scenario_actor_uses_memory_and_reports_soft_weighting(self):
        result = run_scenario_actor(
            ScenarioActorConfig(
                AdvancedGymMPCConfig(
                    env_name="ContinuousCartPole-v0",
                    seed=0,
                    probe_episodes=2,
                    probe_steps=8,
                    control_steps=4,
                    horizon=2,
                    candidate_count=8,
                    cem_iterations=1,
                    ensemble_size=2,
                    hidden_dim=16,
                    epochs=1,
                    batch_size=8,
                    scenario_window_count=3,
                    scenario_window_size=2,
                    scenario_variants_per_window=2,
                    scenario_variant_horizon=2,
                    skill_search_windows=1,
                    skill_candidate_count=4,
                    skill_halving_keep_count=2,
                    skill_real_validate_top=1,
                    skill_branch_steps=2,
                    skill_stable_island_count=2,
                    skill_intrinsic_goal_count=3,
                    skill_goal_actor_horizon=2,
                    solve_return=4.0,
                    collector="random",
                ),
                rounds=1,
            ),
            render_mode=None,
        )
        summary = result.summary()
        diagnostics = summary["diagnostics"]

        self.assertEqual(summary["model_family"], "ScenarioActor")
        self.assertEqual(summary["method"], "scenario_actor")
        self.assertIn("scenario_memory_real_count", diagnostics)
        self.assertIn("scenario_variant_count", diagnostics)
        self.assertIn("scenario_weight_mean", diagnostics)
        self.assertIn("scenario_prediction_gap", diagnostics)
        self.assertIn("skill_intrinsic_goal_count", diagnostics)
        self.assertIn("skill_real_validation_count", diagnostics)
        self.assertIn("skill_runtime_blend_mean", diagnostics)
        self.assertGreaterEqual(summary["total_samples"], summary["probe_samples"])
        self.assertEqual(result.action_low.shape, result.action_high.shape)


if __name__ == "__main__":
    unittest.main()
