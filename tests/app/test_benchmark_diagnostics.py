import unittest

import numpy as np

from teenyreason.app.benchmark.diagnostics import build_latent_support_diagnostics


class BenchmarkDiagnosticsTests(unittest.TestCase):
    def test_tied_support_modes_do_not_create_fake_dominance(self):
        snapshot = {
            "env_belief_mean": np.asarray([[0.0, 0.0], [0.2, 0.1]], dtype=np.float32),
            "window_latent_mean": np.asarray(
                [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]],
                dtype=np.float32,
            ),
            "window_probe_mode": np.asarray(
                ["passive_decay", "impulse_left", "chirp", "cart_brake"],
                dtype="U",
            ),
            "window_is_support": np.asarray([1, 1, 1, 1], dtype=np.int8),
            "env_window_count": np.asarray([4, 4], dtype=np.int32),
            "env_support_count": np.asarray([4, 4], dtype=np.int32),
            "env_dominant_probe_mode": np.asarray(["mixed", "mixed"], dtype="U"),
            "env_support_top_family_share": np.asarray([0.25, 0.25], dtype=np.float32),
            "env_support_effective_family_count": np.asarray([4.0, 4.0], dtype=np.float32),
            "env_support_family_entropy": np.asarray([1.386, 1.386], dtype=np.float32),
            "env_support_tied_top_family_count": np.asarray([4.0, 4.0], dtype=np.float32),
        }

        diagnostics = build_latent_support_diagnostics(snapshot)

        self.assertEqual(diagnostics["dominant_window_mode"], "mixed")
        self.assertAlmostEqual(diagnostics["env_dominant_mode_share"], 0.0)
        self.assertAlmostEqual(diagnostics["support_top_family_share_mean"], 0.25)
        self.assertAlmostEqual(diagnostics["support_effective_family_count_mean"], 4.0)
        self.assertAlmostEqual(diagnostics["support_tied_top_family_count_mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
