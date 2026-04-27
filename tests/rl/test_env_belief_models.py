import unittest

import torch

from teenyreason.models.envbelief.env_belief_models import MechanicsPosteriorUpdater


class EnvBeliefModelTests(unittest.TestCase):
    def test_posterior_update_returns_finite_context_and_evidence(self):
        updater = MechanicsPosteriorUpdater(latent_dim=6, param_dim=2, num_families=3)
        window_mean = torch.randn((2, 4, 6), dtype=torch.float32)
        window_logvar = torch.zeros((2, 4, 6), dtype=torch.float32)
        mask = torch.tensor(
            [[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        family_ids = torch.tensor(
            [[0, 1, 2, 1], [2, 0, 1, 2]],
            dtype=torch.long,
        )

        stats = updater.update(window_mean, window_logvar, mask, family_ids)

        self.assertEqual(tuple(stats["latent_context"].shape), (2, 6))
        self.assertEqual(tuple(stats["posterior_mean"].shape), (2, 2))
        self.assertEqual(tuple(stats["evidence_mean"].shape), (2, 4, 2))
        self.assertTrue(torch.isfinite(stats["latent_context"]).all())
        self.assertTrue(torch.isfinite(stats["posterior_mean"]).all())
        self.assertTrue(torch.all(stats["posterior_std"] > 0))


if __name__ == "__main__":
    unittest.main()
