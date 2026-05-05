from __future__ import annotations

import unittest

import numpy as np

from teenyreason.multidomain.planning.generic.collection.trajectory import make_trajectory
from teenyreason.cognition.skills import SkillMemory, build_discovery_context
from teenyreason.cognition.skills.reflex import feedback_repair_candidates
from teenyreason.cognition.skills.schema import IntrinsicGoal, SkillRecord


class FakeModel:
    def __init__(self):
        self.obs_mean = np.asarray([0.0, 0.0], dtype=np.float32)
        self.obs_std = np.asarray([1.0, 1.0], dtype=np.float32)


class GenericSkillLayerTest(unittest.TestCase):
    def test_discovery_mines_generic_goals_without_env_names(self):
        trajectories = [make_line_trajectory(seed=0, terminal=False), make_line_trajectory(seed=1, terminal=True)]

        context = build_discovery_context(
            trajectories,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            max_islands=3,
            max_goals=6,
            model=None,
            windows=[],
        )

        self.assertGreaterEqual(len(context.stable_islands), 1)
        self.assertGreaterEqual(len(context.intrinsic_goals), 1)
        goal_names = {goal.goal_kind for goal in context.intrinsic_goals}
        self.assertIn("return_to_stable_island", goal_names)
        self.assertTrue(all("biped" not in goal.goal_kind.lower() for goal in context.intrinsic_goals))

    def test_skill_memory_retrieves_validated_action_prior(self):
        goal = IntrinsicGoal(
            goal_id=0,
            goal_kind="extend_survival_from_frontier",
            target_delta=np.asarray([1.0, 0.0], dtype=np.float32),
            anchor_observation=np.asarray([0.0, 0.0], dtype=np.float32),
            priority=1.0,
            source="test",
        )
        record = SkillRecord(
            skill_id=1,
            goal=goal,
            initiation_observation=np.asarray([0.0, 0.0], dtype=np.float32),
            termination_observation=np.asarray([1.0, 0.0], dtype=np.float32),
            actions=np.asarray([[0.5], [0.25]], dtype=np.float32),
            outcome_delta=np.asarray([1.0, 0.0], dtype=np.float32),
            real_return_lift=2.0,
            survival_lift=4.0,
            terminal_avoid=1.0,
            reliability=0.9,
        )
        memory = SkillMemory([record])

        action, stats = memory.action_prior(
            np.asarray([0.05, 0.0], dtype=np.float32),
            FakeModel(),
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            step=0,
        )

        self.assertGreater(float(action[0]), 0.25)
        self.assertGreater(stats["skill_reliability"], 0.5)
        self.assertEqual(memory.summary()["skill_memory_count"], 1.0)

    def test_feedback_repair_candidate_is_closed_loop(self):
        trajectory = make_line_trajectory(seed=2, terminal=True)
        context = build_discovery_context(
            [trajectory],
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            max_islands=2,
            max_goals=3,
            model=None,
            windows=[],
        )
        goal = context.intrinsic_goals[0]
        window = type("Window", (), {"start": 1, "end": 4})()
        config = type("Config", (), {"skill_candidate_count": 8, "reflex_weight_scale": 0.35, "reflex_action_smoothing": 0.3})()

        candidates = feedback_repair_candidates(
            config,
            trajectory,
            window,
            [goal],
            context.factor_model,
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            seed=7,
        )

        policy, _goal = candidates[-1]
        left = policy.action(np.asarray([0.0, 0.0], dtype=np.float32), step=0, previous_action=np.asarray([0.0], dtype=np.float32))
        right = policy.action(np.asarray([4.0, 0.4], dtype=np.float32), step=0, previous_action=np.asarray([0.0], dtype=np.float32))
        self.assertTrue(candidates)
        self.assertEqual(left.shape, (1,))
        self.assertFalse(np.allclose(left, right))


def make_line_trajectory(*, seed: int, terminal: bool):
    observations = [np.asarray([float(i), 0.1 * i], dtype=np.float32) for i in range(5)]
    actions = [np.asarray([0.5], dtype=np.float32) for _ in range(5)]
    next_observations = [np.asarray([float(i + 1), 0.1 * (i + 1)], dtype=np.float32) for i in range(5)]
    rewards = [1.0, 1.0, 0.5, 0.25, -1.0 if terminal else 1.0]
    dones = [0.0, 0.0, 0.0, 0.0, 1.0 if terminal else 0.0]
    return make_trajectory(
        seed=seed,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
        discount=0.99,
    )


if __name__ == "__main__":
    unittest.main()
