import unittest

import torch

from teenyreason.models.envbelief.env_belief_subsets import build_random_subset_masks


class EnvBeliefSubsetTests(unittest.TestCase):
    def test_random_subset_masks_are_not_canonical_repeats(self):
        mask = torch.ones((1, 8), dtype=torch.float32)
        group_ids = torch.arange(8, dtype=torch.int64).reshape(1, 8)

        subset_masks = build_random_subset_masks(
            mask=mask,
            subset_count=5,
            subset_size=3,
            group_ids=group_ids,
            generator=torch.Generator().manual_seed(7),
        )

        chosen_rows = [
            tuple(torch.nonzero(subset_masks[0, idx] > 0, as_tuple=False).squeeze(-1).tolist())
            for idx in range(subset_masks.shape[1])
        ]

        self.assertTrue(all(len(row) == 3 for row in chosen_rows))
        self.assertGreater(len(set(chosen_rows)), 1)


if __name__ == "__main__":
    unittest.main()
