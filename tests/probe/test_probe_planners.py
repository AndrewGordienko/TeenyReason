import unittest

import gymnasium as gym
import numpy as np

from teenyreason.envs import get_action_values
from teenyreason.probe.explorer import GenericProbeExplorer, build_probe_planner


class ProbePlannerTests(unittest.TestCase):
    def test_generic_builder_supports_discrete_envs_without_env_name_branching(self):
        env = gym.make("CartPole-v1")
        try:
            planner = build_probe_planner(
                action_space=env.action_space,
                action_values=None,
                rng=np.random.default_rng(0),
            )
        finally:
            env.close()

        self.assertIsInstance(planner, GenericProbeExplorer)
        self.assertEqual(planner.support_goal_sequence, ("action_0", "action_1"))

    def test_generic_builder_supports_multiaxis_box_envs(self):
        env = gym.make("LunarLanderContinuous-v3")
        try:
            action_values = get_action_values(env, 9)
            planner = build_probe_planner(
                action_space=env.action_space,
                action_values=action_values,
                rng=np.random.default_rng(0),
            )
        finally:
            env.close()

        self.assertIsInstance(planner, GenericProbeExplorer)
        self.assertIn("center", planner.support_goal_sequence)
        self.assertTrue(any(name.startswith("axis0_") for name in planner.support_goal_sequence))
        self.assertTrue(any(name.startswith("axis1_") for name in planner.support_goal_sequence))

    def test_generic_planner_scores_current_goal_actions_highest(self):
        action_space = gym.spaces.Box(
            low=np.asarray([-1.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        action_values = np.linspace(-1.0, 1.0, 5, dtype=np.float32).reshape(-1, 1)
        planner = GenericProbeExplorer(
            action_space=action_space,
            action_values=action_values,
            rng=np.random.default_rng(0),
        )
        planner.begin_env_instance()
        planner.begin_rollout(primary_goal="pos_2")
        scores = planner.action_prior_scores(
            np.asarray([0.0, 0.0], dtype=np.float32),
            recent_actions=[],
        )

        self.assertEqual(int(np.argmax(scores)), 4)

    def test_scalar_box_planner_raises_support_budget_for_cartpole_style_views(self):
        action_space = gym.spaces.Box(
            low=np.asarray([-1.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        action_values = np.linspace(-1.0, 1.0, 9, dtype=np.float32).reshape(-1, 1)
        planner = GenericProbeExplorer(
            action_space=action_space,
            action_values=action_values,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(planner.active_step_cap, 64)
        self.assertEqual(planner.active_window_stride, 6)
        self.assertEqual(planner.max_windows_per_rollout, 1)
        self.assertTrue(planner.emit_partial_terminal_window)
        self.assertGreaterEqual(planner.partial_window_min_steps, 1)
        self.assertEqual(planner.support_goal_sequence[-1], "center")
        self.assertLess(planner.center_recovery_bonus, 0.45)
        self.assertEqual(planner.min_windows_per_env, 2 * len(planner.support_goal_sequence))
        self.assertGreaterEqual(
            planner.max_support_retry_rollouts,
            len(planner.support_goal_sequence),
        )

    def test_generic_planner_tracks_boundary_and_surprise_events_from_transitions(self):
        action_space = gym.spaces.Box(
            low=np.asarray([-1.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        action_values = np.linspace(-1.0, 1.0, 5, dtype=np.float32).reshape(-1, 1)
        planner = GenericProbeExplorer(
            action_space=action_space,
            action_values=action_values,
            rng=np.random.default_rng(0),
        )
        planner.begin_env_instance()
        planner.begin_rollout(primary_goal="neg_2")
        planner.ensure_goal(np.asarray([0.0], dtype=np.float32))
        planner.observe_transition(
            prev_state=np.asarray([0.0], dtype=np.float32),
            action_idx=0,
            next_state=np.asarray([2.5], dtype=np.float32),
            terminated=True,
            truncated=False,
        )

        self.assertGreaterEqual(planner.coverage.boundary_events, 1)
        self.assertGreaterEqual(planner.coverage.termination_events, 1)


if __name__ == "__main__":
    unittest.main()
