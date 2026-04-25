import unittest

import torch

from teenyreason.app.multidomain_suite import summarize_rows
from teenyreason.multidomain.image_benchmark import stratified_subset_indices
from teenyreason.multidomain.language_benchmark import build_char_vocab, encode_text, split_corpus


class MultidomainBenchmarkTests(unittest.TestCase):
    def test_stratified_subset_indices_spreads_budget_across_classes(self):
        labels = torch.tensor([0] * 6 + [1] * 6 + [2] * 6, dtype=torch.long)
        indices = stratified_subset_indices(labels, budget=9, seed=7)
        chosen_labels = labels[torch.tensor(indices, dtype=torch.long)]
        counts = {
            int(label): int((chosen_labels == label).sum().item())
            for label in torch.unique(chosen_labels)
        }
        self.assertEqual(len(indices), 9)
        self.assertEqual(counts, {0: 3, 1: 3, 2: 3})

    def test_language_vocab_roundtrip_and_split(self):
        text = "to be or not to be"
        stoi, itos = build_char_vocab(text)
        encoded = encode_text(text, stoi)
        decoded = "".join(itos[int(idx)] for idx in encoded)
        train, validation = split_corpus(encoded, validation_chars=5)
        self.assertEqual(decoded, text)
        self.assertEqual(len(train) + len(validation), len(encoded))
        self.assertGreater(len(validation), 0)

    def test_summarize_rows_reports_probe_minus_baseline(self):
        summary = summarize_rows(
            [
                {"baseline_accuracy": 0.70, "probe_accuracy": 0.80},
                {"baseline_accuracy": 0.75, "probe_accuracy": 0.85},
            ],
            baseline_key="baseline_accuracy",
            probe_key="probe_accuracy",
        )
        self.assertAlmostEqual(summary["baseline_mean"], 0.725)
        self.assertAlmostEqual(summary["probe_mean"], 0.825)
        self.assertAlmostEqual(summary["probe_minus_baseline"], 0.10)


if __name__ == "__main__":
    unittest.main()
