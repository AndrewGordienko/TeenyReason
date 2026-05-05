import unittest

import numpy as np

from teenyreason.envs import CONTINUOUS_CARTPOLE_NAME
from teenyreason.models.belief import (
    build_future_summary_targets,
    build_training_tensors,
)


class BeliefTargetTests(unittest.TestCase):
    def test_cartpole_future_targets_are_family_conditioned(self):
        states = np.asarray(
            [
                [
                    [0.0, 0.0, 0.02, 0.0],
                    [0.1, 0.2, 0.03, 0.1],
                    [0.2, 0.3, 0.04, 0.2],
                ],
                [
                    [0.0, 0.0, 0.02, 0.0],
                    [0.1, 0.2, 0.03, 0.1],
                    [0.2, 0.3, 0.04, 0.2],
                ],
            ],
            dtype=np.float32,
        )
        actions = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
        rewards = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        terminated = np.asarray([False, False], dtype=np.bool_)
        truncated = np.asarray([False, False], dtype=np.bool_)

        targets = build_future_summary_targets(
            states=states,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            action_vocab_size=3,
            probe_mode=np.asarray(["chirp", "boundary_push"], dtype="U"),
            env_name=CONTINUOUS_CARTPOLE_NAME,
        )

        self.assertEqual(targets.shape[0], 2)
        self.assertGreater(targets.shape[1], 20)
        self.assertFalse(np.allclose(targets[0], targets[1]))

    def test_build_training_tensors_normalizes_cartpole_future_targets(self):
        windows = {
            "states": np.asarray(
                [
                    [
                        [0.0, 0.0, 0.02, 0.0],
                        [0.1, 0.2, 0.03, 0.1],
                        [0.2, 0.3, 0.04, 0.2],
                        [0.3, 0.4, 0.05, 0.3],
                    ],
                    [
                        [0.0, 0.0, -0.02, 0.0],
                        [-0.1, -0.2, -0.03, -0.1],
                        [-0.2, -0.3, -0.04, -0.2],
                        [-0.3, -0.4, -0.05, -0.3],
                    ],
                ],
                dtype=np.float32,
            ),
            "actions": np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64),
            "rewards": np.asarray([[1.0, 1.0, 1.0], [1.0, 0.8, 0.6]], dtype=np.float32),
            "env_params": np.asarray([[9.8, 1.0], [10.5, 0.9]], dtype=np.float32),
            "env_instance_id": np.asarray([0, 1], dtype=np.int32),
            "probe_mode": np.asarray(["chirp", "boundary_push"], dtype="U"),
            "terminated": np.asarray([False, False], dtype=np.bool_),
            "truncated": np.asarray([False, False], dtype=np.bool_),
        }

        tensors = build_training_tensors(
            windows=windows,
            action_vocab_size=3,
            intervention_horizon=2,
            env_name=CONTINUOUS_CARTPOLE_NAME,
        )

        self.assertEqual(tensors["target_future_summary"].shape[0], 2)
        self.assertTrue(np.isfinite(tensors["target_future_summary"]).all())


if __name__ == "__main__":
    unittest.main()
