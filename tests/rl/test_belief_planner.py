import unittest

import numpy as np
import torch

from teenyreason.models.belief.belief_losses import gaussian_moment_regularizer
from teenyreason.rl.full_system.objectives import (
    planner_prediction_surprise,
    surprise_z_score,
)
from teenyreason.rl.full_system.planner_eval import should_stop_belief_planner_plateau


class BeliefPlannerHelperTests(unittest.TestCase):
    def test_gaussian_moment_regularizer_prefers_spread_centered_batch(self):
        torch.manual_seed(0)
        gaussian_like = torch.randn(128, 6, dtype=torch.float32)
        collapsed = torch.ones((128, 6), dtype=torch.float32) * 0.25

        gaussian_loss = float(gaussian_moment_regularizer(gaussian_like).item())
        collapsed_loss = float(gaussian_moment_regularizer(collapsed).item())

        self.assertLess(gaussian_loss, collapsed_loss)

    def test_planner_prediction_surprise_grows_when_prediction_is_wrong(self):
        right_prediction = {
            "next_state_delta": torch.tensor([[0.10, -0.10]], dtype=torch.float32),
            "reward": torch.tensor([1.0], dtype=torch.float32),
            "term_logit": torch.tensor([-6.0], dtype=torch.float32),
            "disagreement": torch.tensor([0.05], dtype=torch.float32),
        }
        wrong_prediction = {
            "next_state_delta": torch.tensor([[0.70, 0.40]], dtype=torch.float32),
            "reward": torch.tensor([-0.5], dtype=torch.float32),
            "term_logit": torch.tensor([2.5], dtype=torch.float32),
            "disagreement": torch.tensor([0.60], dtype=torch.float32),
        }
        state = np.asarray([0.20, 0.10], dtype=np.float32)
        next_state = np.asarray([0.30, 0.00], dtype=np.float32)

        low_surprise = planner_prediction_surprise(
            normalized_state=state,
            prediction=right_prediction,
            normalized_next_state=next_state,
            reward=1.0,
            terminated=False,
            truncated=False,
        )
        high_surprise = planner_prediction_surprise(
            normalized_state=state,
            prediction=wrong_prediction,
            normalized_next_state=next_state,
            reward=1.0,
            terminated=False,
            truncated=False,
        )

        self.assertLess(low_surprise, high_surprise)

    def test_surprise_z_score_requires_history_then_flags_outlier(self):
        short_history = [0.2, 0.3, 0.25]
        long_history = [0.2, 0.3, 0.25, 0.22, 0.24, 0.21, 0.26, 0.23, 0.27]

        self.assertEqual(surprise_z_score(0.8, short_history), 0.0)
        self.assertGreater(surprise_z_score(0.8, long_history), 2.0)

    def test_plateau_stop_waits_for_warmup_and_patience(self):
        self.assertFalse(
            should_stop_belief_planner_plateau(
                current_episode=150,
                warmup_episodes=200,
                patience=50,
                last_meaningful_progress_episode=100,
            )
        )
        self.assertFalse(
            should_stop_belief_planner_plateau(
                current_episode=240,
                warmup_episodes=200,
                patience=50,
                last_meaningful_progress_episode=200,
            )
        )
        self.assertTrue(
            should_stop_belief_planner_plateau(
                current_episode=251,
                warmup_episodes=200,
                patience=50,
                last_meaningful_progress_episode=200,
            )
        )


if __name__ == "__main__":
    unittest.main()
