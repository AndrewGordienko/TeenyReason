import unittest
from unittest.mock import patch

import numpy as np
import torch

from teenyreason.crawler.library import CrawlerModelBundle, build_evidence_batch, quantize_vector
from teenyreason.representation.analysis import compute_message_rate_distortion


class CrawlerLibraryTests(unittest.TestCase):
    def test_build_evidence_batch_preserves_public_window_contract(self):
        windows = {
            "states": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "actions": np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64),
            "rewards": np.asarray([[0.1, 0.2, 0.3], [0.0, -0.1, 0.4]], dtype=np.float32),
            "terminated": np.asarray([False, True], dtype=np.bool_),
            "truncated": np.asarray([False, False], dtype=np.bool_),
            "probe_mode": np.asarray(["passive_decay", "impulse_right"], dtype="U"),
            "env_instance_id": np.asarray([11, 11], dtype=np.int32),
        }

        batch = build_evidence_batch(
            windows=windows,
            env_name="continuous_cartpole",
            window_size=3,
            action_vocab_size=5,
        )

        self.assertEqual(batch.env_name, "continuous_cartpole")
        self.assertEqual(batch.window_size, 3)
        self.assertEqual(batch.action_vocab_size, 5)
        self.assertEqual(len(batch.windows), 2)
        self.assertEqual(batch.windows[0].probe_family, "passive_decay")
        self.assertTrue(batch.windows[1].terminated)
        np.testing.assert_array_equal(batch.windows[1].actions, np.asarray([2, 1, 0], dtype=np.int64))

    def test_quantize_vector_preserves_shape_and_residual_is_finite(self):
        vector = np.asarray([0.25, -0.5, 1.5, 0.75], dtype=np.float32)
        coarse, residual = quantize_vector(
            vector,
            bits_per_dim=4,
            use_residual_sketch=True,
        )

        self.assertEqual(coarse.shape, vector.shape)
        self.assertIsNotNone(residual)
        self.assertEqual(residual.shape, vector.shape)
        self.assertTrue(np.isfinite(coarse).all())
        self.assertTrue(np.isfinite(residual).all())

    def test_build_env_belief_forwards_probe_group_ids(self):
        captured = {}

        def fake_aggregate_env_belief(**kwargs):
            captured["probe_group_ids"] = kwargs.get("probe_group_ids")
            payload = {
                "belief": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "env_mean": np.asarray([0.1, 0.2], dtype=np.float32),
                "env_mean_raw": np.asarray([0.1, 0.2], dtype=np.float32),
                "env_logvar": np.zeros((2,), dtype=np.float32),
                "view_spread": np.zeros((2,), dtype=np.float32),
                "env_param_mean": np.zeros((1,), dtype=np.float32),
                "env_param_std": np.ones((1,), dtype=np.float32) * 0.2,
                "support_count": np.asarray([2], dtype=np.int32),
                "support_group_ratio": np.asarray([1.0], dtype=np.float32),
            }
            return payload["belief"], payload

        bundle = CrawlerModelBundle(
            encoder=None,
            predictor=None,
            belief_aggregator=None,
            env_param_predictor=None,
            env_future_predictor=None,
            env_family_future_predictor=None,
            family_value_predictor=None,
            env_metric_projector=None,
            belief_message_projector=None,
            controller_context_projector=None,
            device=torch.device("cpu"),
            z_dim=2,
            window_size=4,
            action_vocab_size=3,
            belief_message_dim=2,
            controller_context_dim=6,
            family_names=("passive_decay", "chirp"),
        )
        group_ids = np.asarray([0, 1], dtype=np.int64)

        with patch(
            "teenyreason.crawler.library.aggregate_env_belief",
            side_effect=fake_aggregate_env_belief,
        ):
            _belief, _payload = bundle.build_env_belief(
                posterior_views=[
                    (np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)),
                    (np.ones((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)),
                ],
                probe_group_ids=group_ids,
            )

        np.testing.assert_array_equal(captured["probe_group_ids"], group_ids)

    def test_message_rate_distortion_returns_one_row_per_bitrate(self):
        env_mean = np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.1, -0.2],
                [-0.3, 0.5, 0.2],
            ],
            dtype=np.float32,
        )
        split_a = env_mean + 0.01
        split_b = env_mean - 0.01
        env_uncertainty = np.asarray([0.2, 0.4, 0.1], dtype=np.float32)
        env_params = np.asarray(
            [
                [9.8, 1.0],
                [11.0, 1.2],
                [8.9, 0.9],
            ],
            dtype=np.float32,
        )

        summary = compute_message_rate_distortion(
            env_mean=env_mean,
            env_split_mean_a=split_a,
            env_split_mean_b=split_b,
            env_uncertainty=env_uncertainty,
            env_params=env_params,
            belief_message_projector=None,
            device=torch.device("cpu"),
            compression_bits=(0, 8, 4),
        )

        self.assertEqual(summary["compression_bits"].tolist(), [0, 8, 4])
        self.assertEqual(summary["compression_mechanics_fit_r2"].shape[0], 3)
        self.assertEqual(summary["compression_split_retrieval_top1"].shape[0], 3)
        self.assertEqual(summary["compression_split_retrieval_mrr"].shape[0], 3)
        self.assertTrue(np.isfinite(summary["compression_message_norm_mean"]).all())


if __name__ == "__main__":
    unittest.main()
