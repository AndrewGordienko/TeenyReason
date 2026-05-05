import unittest

import numpy as np

from teenyreason.multidomain.planning.generic.collection.trajectory import make_trajectory
from teenyreason.cognition.scenario import ScenarioMemory, generate_variants, retrieve_windows
from teenyreason.cognition.scenario.weights import score_variant_weights


class ScenarioMemoryTests(unittest.TestCase):
    def test_memory_retrieves_familiar_high_value_windows(self):
        trajectory = make_trajectory(
            seed=0,
            observations=[np.asarray([float(idx), 0.0], dtype=np.float32) for idx in range(6)],
            actions=[np.asarray([0.5], dtype=np.float32) for _idx in range(6)],
            rewards=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            next_observations=[np.asarray([float(idx + 1), 0.0], dtype=np.float32) for idx in range(6)],
            dones=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            discount=0.99,
        )
        memory = ScenarioMemory.from_trajectories([trajectory])
        memory.add_rows(
            (
                {
                    "observation": np.asarray([0.0, 0.0], dtype=np.float32),
                    "action": np.asarray([0.0], dtype=np.float32),
                    "reward": 0.0,
                    "next_observation": np.asarray([0.1, 0.0], dtype=np.float32),
                    "done": 0.0,
                },
            ),
            source="imagined",
            surprise=0.5,
        )

        windows = retrieve_windows(memory, count=2, window_size=3)
        focused = retrieve_windows(memory, count=1, window_size=3, focus_observations=[np.asarray([4.0, 0.0], dtype=np.float32)], focus_weight=2.0)

        self.assertEqual(memory.summary()["scenario_memory_real_count"], 6.0)
        self.assertEqual(memory.summary()["scenario_memory_imagined_count"], 1.0)
        self.assertTrue(windows)
        self.assertTrue(focused)
        self.assertGreaterEqual(windows[0].best_return_to_go, windows[-1].best_return_to_go)

    def test_soft_weights_drop_with_surprise_and_uncertainty(self):
        trace = make_trajectory(
            seed=0,
            observations=[np.asarray([0.0], dtype=np.float32)],
            actions=[np.asarray([0.0], dtype=np.float32)],
            rewards=[1.0],
            next_observations=[np.asarray([0.1], dtype=np.float32)],
            dones=[0.0],
            discount=0.99,
        )
        window = retrieve_windows(ScenarioMemory.from_trajectories([trace]), count=1, window_size=1)[0]
        good = score_variant_weights(window, predicted_lift=5.0, uncertainty=0.0, done_risk=0.0, variant_surprise=0.0, advantage_temperature=10.0, uncertainty_scale=1.0, surprise_scale=1.0)
        bad = score_variant_weights(window, predicted_lift=5.0, uncertainty=2.0, done_risk=0.5, variant_surprise=2.0, advantage_temperature=10.0, uncertainty_scale=1.0, surprise_scale=1.0)

        self.assertGreater(good.combined, bad.combined)


if __name__ == "__main__":
    unittest.main()
