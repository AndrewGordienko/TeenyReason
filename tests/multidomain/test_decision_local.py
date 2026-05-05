import unittest

from teenyreason.multidomain.contracts.decision_local import build_decision_local_block, decision_local_row


class DecisionLocalBeliefTests(unittest.TestCase):
    def test_decision_local_belief_accepts_compact_causal_board(self):
        domain = {
            "belief_contribution_margin": 0.44,
            "ablation_gap": 0.44,
            "evidence_cost": 2.0,
            "rows": [
                {
                    "baseline_move_accuracy": 0.56,
                    "belief_move_accuracy": 1.0,
                }
            ],
            "interface": {"belief": {"bitrate": 128}},
            "causal_ablation": {"solver_gain": 0.44, "content_lift": 0.44},
            "world_understanding": {"counterfactual": 1.0},
            "belief_handoff": {"contract": {"evidence_cost": 2.0}},
        }

        block = build_decision_local_block("board", domain)
        row = decision_local_row("board", {**domain, "decision_local_belief": block})

        self.assertEqual(block["contract"]["state"], "decision_local_belief_ready")
        self.assertEqual(block["contract"]["blocker"], "ready")
        self.assertGreater(block["contract"]["utility_per_1k_bits"], 0.0)
        self.assertTrue(block["gates"]["decision_local"])
        self.assertEqual(row["state"], "decision_local_belief_ready")

    def test_decision_local_belief_blocks_wide_visual_side_channel(self):
        domain = {
            "evidence_cost": 4096.0,
            "rows": [
                {
                    "handoff_gate_used_baseline": False,
                    "decision_gate_use_belief": True,
                    "accuracy_gain": 0.01,
                    "decision_delta_correct_vs_best_ablation": 0.002,
                    "belief_bitrate": 20544,
                }
            ],
            "interface": {"belief": {"bitrate": 20544}},
            "world_understanding": {"counterfactual": 0.9},
            "belief_handoff": {"contract": {"evidence_cost": 4096.0}},
        }

        block = build_decision_local_block("image", domain)

        self.assertEqual(block["contract"]["blocker"], "belief_too_wide_for_decision")
        self.assertFalse(block["gates"]["compact_enough"])
        self.assertGreater(block["contract"]["mean_decision_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
