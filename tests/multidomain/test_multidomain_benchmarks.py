import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from teenyreason.multidomain.suite import (
    MultidomainSuiteConfig,
    build_suite_payload,
    run_multidomain_suite,
    summarize_rows,
)
from teenyreason.multidomain.domains.image.benchmark import (
    BeliefConditionedMNISTCNN,
    compress_prototype_context,
    run_synthetic_vision_rule_smoke,
    stratified_subset_indices,
    zero_prototype_context,
)
from teenyreason.multidomain.domains.board import (
    BoardProbeBenchmarkConfig,
    RULE_MISERE,
    RULE_NORMAL,
    best_moves,
    run_board_probe_benchmark,
)
from teenyreason.multidomain.domains.cartpole import (
    CartPoleControllerBridgeConfig,
    CartPoleMechanicsConfig,
    run_cartpole_controller_bridge,
    run_cartpole_mechanics_benchmark,
)
from teenyreason.multidomain.domains.cartpole_handoff import (
    LatentControlHandoffConfig,
    run_latent_control_handoff,
)
from teenyreason.multidomain.adapters.crawler import run_crawler_adapter
from teenyreason.multidomain.adapters.bridges import (
    BoardRuleAdapter,
    CartPoleControllerAdapter,
    ImageBridgeAdapter,
    LanguageBridgeAdapter,
    run_cartpole_adapter_bridge,
    run_image_adapter_bridge,
    run_language_adapter_bridge,
)
from teenyreason.multidomain.adapters.family import (
    FamilyBridgeConfig,
    run_image_family_bridge,
    run_language_family_bridge,
)
from teenyreason.multidomain.domains.language.benchmark import (
    BeliefConditionedCharTransformer,
    _select_language_handoff,
    build_char_vocab,
    encode_text,
    language_row_economics,
    run_synthetic_grammar_smoke,
    split_corpus,
)
from teenyreason.multidomain.domains.image.benchmark import _select_image_handoff, image_row_economics
from teenyreason.multidomain.suite.reporting.cartpole import cartpole_domain_payload
from teenyreason.multidomain.diagnostics.world_understanding import build_world_understanding_block
from teenyreason.multidomain.contracts.decision_gate import DecisionGateInput, evaluate_decision_delta_gate
from teenyreason.multidomain.contracts.decision_utility import build_decision_utility_block
from teenyreason.multidomain.contracts.handoff import build_belief_handoff_block, build_rate_distortion_block
from teenyreason.multidomain.contracts.handoff_repair import build_handoff_repair_block
from teenyreason.multidomain.adapters.real_causal import (
    RealCartPoleCausalAdapter,
    RealCausalAdapterConfig,
    RealImageCausalAdapter,
    RealLanguageCausalAdapter,
)


class MultidomainBenchmarkTests(unittest.TestCase):
    def test_decision_delta_gate_requires_all_ablation_arms(self):
        result = evaluate_decision_delta_gate(
            DecisionGateInput(
                domain="language",
                mode="adapter",
                lower_is_better=True,
                baseline_value=3.0,
                correct_value=2.9,
                zero_value=2.95,
                shuffled_value=2.89,
                stale_value=2.96,
                solver_gain=0.1,
                content_lift=0.05,
                evidence_cost=8.0,
                bits=512,
            )
        )

        self.assertFalse(result.use_belief)
        self.assertEqual(result.reason, "correct_not_better_than_shuffled")

    def test_decision_delta_gate_reports_positive_cost_adjusted_utility(self):
        result = evaluate_decision_delta_gate(
            DecisionGateInput(
                domain="board",
                mode="rule_belief",
                lower_is_better=False,
                baseline_value=0.55,
                correct_value=1.0,
                zero_value=0.55,
                shuffled_value=0.55,
                stale_value=0.55,
                solver_gain=0.45,
                content_lift=0.45,
                evidence_cost=2.0,
                bits=128,
            )
        )

        self.assertTrue(result.use_belief)
        self.assertAlmostEqual(result.expected_gain_per_cost, 0.225)

    def test_board_minimax_rule_variant_changes_optimal_moves(self):
        board = (-1, 0, -1, 1, 1, 0, 0, 0, 0)

        normal_moves = best_moves(board, False, RULE_NORMAL)
        misere_moves = best_moves(board, False, RULE_MISERE)

        self.assertEqual(normal_moves, (1, 5))
        self.assertEqual(misere_moves, (6,))

    def test_board_probe_benchmark_decodes_hidden_rule_and_beats_normal_minimax(self):
        result = run_board_probe_benchmark(
            BoardProbeBenchmarkConfig(
                seeds=(1,),
                probe_budget=1,
                challenge_positions=8,
            )
        )
        row = result["rows"][0]

        self.assertEqual(row["hidden_rule"], RULE_MISERE)
        self.assertEqual(row["decoded_rule"], RULE_MISERE)
        self.assertEqual(row["rule_decode_accuracy"], 1.0)
        self.assertGreaterEqual(row["belief_move_accuracy"], row["baseline_move_accuracy"])
        self.assertGreater(row["belief_move_accuracy"], row["shuffled_belief_move_accuracy"])
        artifact = result["artifacts"][0]
        self.assertIn("crawler_message", artifact)
        self.assertIn("raw_evidence_windows", artifact)

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

    def test_controlled_language_rule_requires_crawler_message(self):
        result = run_synthetic_grammar_smoke(seed=0)

        self.assertEqual(result["hidden_rule"], "previous_token")
        self.assertEqual(result["decoded_rule"], "previous_token")
        self.assertEqual(result["hidden_rule_decode_accuracy"], 1.0)
        self.assertEqual(result["belief_next_token_accuracy"], 1.0)
        self.assertEqual(result["baseline_next_token_accuracy"], 0.0)
        self.assertGreaterEqual(result["content_lift"], 0.5)

    def test_controlled_vision_rule_requires_crawler_message(self):
        result = run_synthetic_vision_rule_smoke(seed=0)

        self.assertEqual(result["hidden_rule"], "swapped_semantics")
        self.assertEqual(result["decoded_rule"], "swapped_semantics")
        self.assertEqual(result["hidden_rule_decode_accuracy"], 1.0)
        self.assertEqual(result["belief_label_accuracy"], 1.0)
        self.assertEqual(result["baseline_label_accuracy"], 0.0)
        self.assertGreaterEqual(result["content_lift"], 0.5)

    def test_controlled_cartpole_mechanics_requires_crawler_message(self):
        result = run_cartpole_mechanics_benchmark(
            CartPoleMechanicsConfig(seeds=tuple(range(8)))
        )

        self.assertEqual(result["mechanics_decode_accuracy"], 1.0)
        self.assertEqual(result["mechanics_r2"], 1.0)
        self.assertEqual(result["subset_agreement"], 1.0)
        self.assertGreater(result["mean_content_lift"], 0.0)

    def test_family_real_bridges_report_transfer_signal(self):
        config = FamilyBridgeConfig(seeds=tuple(range(8)))
        language = run_language_adapter_bridge(config)
        image = run_image_adapter_bridge(config)
        cartpole = run_cartpole_adapter_bridge(
            CartPoleControllerBridgeConfig(seeds=tuple(range(16)))
        )

        self.assertEqual(language["decode_accuracy"], 1.0)
        self.assertEqual(image["decode_accuracy"], 1.0)
        self.assertEqual(cartpole["decode_accuracy"], 1.0)
        self.assertGreater(language["content_lift"], 0.4)
        self.assertGreater(image["content_lift"], 0.3)
        self.assertGreater(cartpole["content_lift"], 0.1)

    def test_generic_crawler_runner_handles_all_bridge_modalities(self):
        seeds = tuple(range(4))
        config = FamilyBridgeConfig(seeds=seeds)
        adapters = (
            LanguageBridgeAdapter(config),
            ImageBridgeAdapter(config),
            CartPoleControllerAdapter(CartPoleControllerBridgeConfig(seeds=seeds)),
            BoardRuleAdapter(challenge_count=12),
        )

        results = [
            run_crawler_adapter(adapter, seeds=seeds)
            for adapter in adapters
        ]

        self.assertEqual([result["domain"] for result in results], ["language", "image", "cartpole", "board"])
        for result in results:
            self.assertEqual(result["adapter_contract"]["runner"], "run_crawler_adapter")
            self.assertGreaterEqual(result["decode_accuracy"], 0.5)
            self.assertIn("content_lift", result)
            self.assertIn("causal_world_model", result)
            self.assertEqual(result["causal_world_model"]["runner"], "run_causal_crawler")
            self.assertGreaterEqual(result["causal_world_model"]["counterfactual_accuracy"], 0.5)
            self.assertIn("query_families", result["adapter_contract"])

    def test_latent_control_handoff_reduces_dedicated_probe_cost(self):
        result = run_latent_control_handoff(
            LatentControlHandoffConfig(seeds=tuple(range(16)))
        )

        self.assertEqual(result["expensive_decode_accuracy"], 1.0)
        self.assertGreater(result["cheap_decode_accuracy"], 0.5)
        self.assertLess(result["centroid_head_decode_accuracy"], result["cheap_decode_accuracy"])
        self.assertGreater(result["cheap_content_lift"], 0.1)
        self.assertGreater(result["action_change_fraction"], 0.01)
        self.assertEqual(result["cheap_dedicated_probe_steps"], 0.0)
        self.assertGreater(result["dedicated_probe_steps_saved"], 0.0)
        self.assertIn("cheap_decision_gate_accept_rate", result)
        self.assertIn("wake_expensive_probe_rate", result)
        self.assertIn("selected_context_arm", result["rows"][0])

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

    def test_language_prefix_conditioning_changes_logits(self):
        torch.manual_seed(3)
        model = BeliefConditionedCharTransformer(
            vocab_size=11,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_sequence_length=12,
            belief_dim=6,
            prefix_tokens=2,
            use_belief_prefix=True,
        )
        model.eval()
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        zero = torch.zeros((6,), dtype=torch.float32)
        belief = torch.ones((6,), dtype=torch.float32)

        zero_logits = model(tokens, zero)
        repeated_zero_logits = model(tokens, zero)
        belief_logits = model(tokens, belief)

        self.assertTrue(torch.allclose(zero_logits, repeated_zero_logits))
        delta = torch.mean(torch.abs(zero_logits - belief_logits)).detach()
        self.assertGreater(float(delta), 1e-6)

    def test_language_adapter_conditioning_changes_logits_without_prefix_tokens(self):
        torch.manual_seed(4)
        model = BeliefConditionedCharTransformer(
            vocab_size=11,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_sequence_length=12,
            belief_dim=6,
            prefix_tokens=2,
            use_belief_prefix=False,
            handoff_mode="adapter",
        )
        model.eval()
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        zero = torch.zeros((6,), dtype=torch.float32)
        belief = torch.ones((6,), dtype=torch.float32)

        zero_logits = model(tokens, zero)
        belief_logits = model(tokens, belief)

        self.assertEqual(model.prefix_tokens, 0)
        delta = torch.mean(torch.abs(zero_logits - belief_logits)).detach()
        self.assertGreater(float(delta), 1e-6)

    def test_language_row_economics_reports_budget_gate_and_bit_efficiency(self):
        economics = language_row_economics(
            baseline_bpc=3.0,
            belief_bpc=2.9,
            ablations={
                "zero": {"bpc": 2.95},
                "shuffled": {"bpc": 2.92},
                "stale": {"bpc": 2.96},
            },
            belief_bitrate=512,
            support_windows=4,
        )

        self.assertTrue(economics["budget_gate_uses_belief"])
        self.assertAlmostEqual(economics["content_lift"], 0.02)
        self.assertAlmostEqual(economics["bpc_gain_per_support_window"], 0.025)
        self.assertGreater(economics["bpc_gain_per_1k_bits"], 0.0)

    def test_language_handoff_selection_falls_back_when_belief_loses(self):
        selected = _select_language_handoff(
            baseline={"bpc": 3.0, "cloze_accuracy": 0.2, "continuation_accuracy": 0.5},
            raw_best={
                "handoff_mode": "adapter",
                "bpc": 3.1,
                "cloze_accuracy": 0.3,
                "continuation_accuracy": 0.55,
                "ablation_metrics": {},
            },
            handoff_rows=[
                {
                    "mode": "adapter",
                    "bpc": 3.1,
                    "bpc_gain": -0.1,
                    "content_lift": 0.02,
                }
            ],
        )

        self.assertEqual(selected["handoff_mode"], "baseline_fallback")
        self.assertTrue(selected["used_baseline_fallback"])
        self.assertEqual(selected["bpc"], 3.0)

    def test_language_handoff_selection_requires_causal_content_lift(self):
        selected = _select_language_handoff(
            baseline={"bpc": 3.0, "cloze_accuracy": 0.2, "continuation_accuracy": 0.5},
            raw_best={
                "handoff_mode": "prefix",
                "bpc": 2.95,
                "cloze_accuracy": 0.3,
                "continuation_accuracy": 0.55,
                "ablation_metrics": {},
            },
            handoff_rows=[
                {
                    "mode": "prefix",
                    "bpc": 2.95,
                    "bpc_gain": 0.05,
                    "content_lift": 0.0,
                }
            ],
        )

        self.assertEqual(selected["handoff_mode"], "baseline_fallback")
        self.assertEqual(selected["gate_reason"], "content_lift_not_positive")

    def test_language_handoff_selection_can_use_non_raw_best_arm(self):
        selected = _select_language_handoff(
            baseline={"bpc": 3.0, "cloze_accuracy": 0.2, "continuation_accuracy": 0.5},
            raw_best={
                "handoff_mode": "prefix",
                "bpc": 2.94,
                "cloze_accuracy": 0.3,
                "continuation_accuracy": 0.55,
                "ablation_metrics": {
                    "zero": {"bpc": 2.94},
                    "shuffled": {"bpc": 2.96},
                    "stale": {"bpc": 2.96},
                },
            },
            handoff_results=[
                {
                    "handoff_mode": "prefix",
                    "bpc": 2.94,
                    "cloze_accuracy": 0.3,
                    "continuation_accuracy": 0.55,
                    "ablation_metrics": {
                        "zero": {"bpc": 2.94},
                        "shuffled": {"bpc": 2.96},
                        "stale": {"bpc": 2.96},
                    },
                },
                {
                    "handoff_mode": "adapter",
                    "bpc": 2.95,
                    "cloze_accuracy": 0.31,
                    "continuation_accuracy": 0.57,
                    "ablation_metrics": {
                        "zero": {"bpc": 2.99},
                        "shuffled": {"bpc": 2.98},
                        "stale": {"bpc": 2.97},
                    },
                },
            ],
            handoff_rows=[
                {"mode": "prefix", "bpc": 2.94, "bpc_gain": 0.06, "content_lift": 0.0},
                {"mode": "adapter", "bpc": 2.95, "bpc_gain": 0.05, "content_lift": 0.02},
            ],
        )

        self.assertEqual(selected["handoff_mode"], "adapter")
        self.assertFalse(selected["used_baseline_fallback"])
        self.assertTrue(selected["decision_gate"]["use_belief"])

    def test_image_prototype_context_changes_classifier_inputs(self):
        torch.manual_seed(5)
        model = BeliefConditionedMNISTCNN(feature_dim=8)
        model.eval()
        images = torch.rand((2, 1, 28, 28), dtype=torch.float32)
        zero_context = zero_prototype_context(8)
        learned_context = {
            "prototypes": torch.ones((10, 8), dtype=torch.float32),
            "confidence": 1.0,
            "uncertainty": 0.1,
            "counts": torch.ones((10,), dtype=torch.float32),
        }

        zero_features = model.context_features(images, zero_context)
        learned_features = model.context_features(images, learned_context)

        self.assertEqual(zero_features.shape[-1], 20)
        delta = torch.mean(torch.abs(zero_features - learned_features)).detach()
        self.assertGreater(float(delta), 1e-6)

    def test_image_row_economics_reports_compression_target_signal(self):
        economics = image_row_economics(
            baseline_accuracy=0.70,
            belief_accuracy=0.75,
            ablations={
                "zero": {"accuracy": 0.72},
                "shuffled": {"accuracy": 0.71},
                "stale": {"accuracy": 0.69},
            },
            belief_bitrate=20480,
            label_budget=256,
        )

        self.assertTrue(economics["budget_gate_uses_belief"])
        self.assertAlmostEqual(economics["best_ablation_accuracy"], 0.72)
        self.assertAlmostEqual(economics["content_lift"], 0.03)
        self.assertGreater(economics["accuracy_gain_per_1k_bits"], 0.0)

    def test_image_handoff_selection_prefers_baseline_when_belief_loses(self):
        selected = _select_image_handoff(
            baseline_accuracy=0.90,
            full_belief_accuracy=0.88,
            full_content_lift=0.02,
            compression_rows=[
                {
                    "bits": 8,
                    "accuracy": 0.89,
                    "solver_gain": -0.01,
                    "content_lift": 0.03,
                }
            ],
        )

        self.assertEqual(selected["mode"], "baseline_fallback")
        self.assertEqual(selected["accuracy"], 0.90)
        self.assertEqual(selected["solver_gain"], 0.0)

    def test_image_handoff_selection_requires_causal_content_lift(self):
        selected = _select_image_handoff(
            baseline_accuracy=0.70,
            full_belief_accuracy=0.75,
            full_content_lift=0.0,
            compression_rows=[],
        )

        self.assertEqual(selected["mode"], "baseline_fallback")
        self.assertEqual(selected["accuracy"], 0.70)

    def test_image_handoff_selection_prefers_bit_efficient_compressed_gate(self):
        selected = _select_image_handoff(
            baseline_accuracy=0.70,
            full_belief_accuracy=0.78,
            full_content_lift=0.04,
            full_bits=20480,
            full_ablations={
                "zero": {"accuracy": 0.73},
                "shuffled": {"accuracy": 0.74},
                "stale": {"accuracy": 0.72},
            },
            compression_rows=[
                {
                    "bits": 128,
                    "accuracy": 0.76,
                    "solver_gain": 0.06,
                    "content_lift": 0.03,
                    "zero_accuracy": 0.72,
                    "shuffled_accuracy": 0.73,
                    "stale_accuracy": 0.71,
                }
            ],
        )

        self.assertEqual(selected["mode"], "compressed_128")
        self.assertTrue(selected["decision_gate"]["use_belief"])

    def test_decision_utility_reports_fallback_saved_harm(self):
        domain = {
            "belief_contribution_margin": 0.0,
            "evidence_cost": 4096.0,
            "rows": [
                {
                    "baseline_accuracy": 0.91,
                    "belief_accuracy": 0.91,
                    "raw_belief_accuracy": 0.887,
                    "accuracy_gain": 0.0,
                    "raw_accuracy_gain": -0.023,
                    "content_lift": 0.0,
                    "raw_content_lift": 0.002,
                    "handoff_gate_used_baseline": True,
                }
            ],
            "belief_handoff": {
                "contract": {"evidence_cost": 4096.0},
                "economics": {"net_sample_savings": 0.0},
            },
            "causal_ablation": {"solver_gain": 0.0, "content_lift": 0.0},
        }

        block = build_decision_utility_block("image", domain)
        contract = block["contract"]

        self.assertEqual(contract["decision"], "baseline_fallback_saved_harm")
        self.assertAlmostEqual(contract["raw_solver_gain"], -0.023)
        self.assertAlmostEqual(contract["gated_solver_gain"], 0.0)
        self.assertAlmostEqual(contract["avoided_harm"], 0.023)
        self.assertEqual(contract["next_action"], "train_residual_gate_on_compressed_context")

    def test_handoff_repair_reports_best_arm_and_blocker(self):
        domain = {
            "evidence_cost": 8.0,
            "rows": [
                {
                    "handoff_mode": "baseline_fallback",
                    "handoff_gate_used_baseline": True,
                    "prefix_bpc_gain": 0.03,
                    "prefix_content_lift": 0.002,
                    "adapter_bpc_gain": -0.01,
                    "adapter_content_lift": 0.001,
                    "raw_bpc_gain": 0.03,
                    "raw_content_lift": 0.002,
                    "belief_bitrate": 512,
                }
            ],
        }

        block = build_handoff_repair_block("language", domain)

        self.assertEqual(block["best_arm"]["mode"], "prefix")
        self.assertEqual(block["blocker"], "use_best_positive_arm")
        self.assertGreater(block["best_arm"]["gain_per_cost"], 0.0)

    def test_compress_prototype_context_keeps_solver_shape_and_reports_bits(self):
        context = {
            "prototypes": torch.arange(80, dtype=torch.float32).view(10, 8),
            "confidence": 1.0,
            "uncertainty": 0.2,
            "counts": torch.ones((10,), dtype=torch.float32),
        }

        compressed = compress_prototype_context(context, target_bits=96)

        self.assertEqual(compressed["prototypes"].shape, (10, 8))
        self.assertEqual(compressed["target_bits"], 96)
        self.assertEqual(compressed["retained_feature_dims"], 1)
        self.assertGreater(float(torch.count_nonzero(compressed["prototypes"]).item()), 0.0)

    def test_world_understanding_scores_board_counterfactual_handoff(self):
        domain = {
            "belief_contribution_margin": 0.44,
            "trust": 1.0,
            "rows": [
                {"belief_value_accuracy": 1.0},
                {"belief_value_accuracy": 1.0},
            ],
            "metrics": {
                "rule_decode_accuracy": 1.0,
                "belief_value_accuracy": 1.0,
                "belief_bitrate": 128,
            },
            "interface": {"belief": {"message_dim": 4, "bitrate": 128}},
            "causal_ablation": {"content_lift": 0.44},
            "transfer_gap": {
                "decode_accuracy": 1.0,
                "subset_agreement": 1.0,
                "bridge_content_lift": 0.18,
                "real_content_lift": 0.44,
            },
            "latent_utility": {"evidence_cost": 2.0},
        }

        block = build_world_understanding_block("board", domain)

        self.assertGreaterEqual(block["counterfactual"], 0.99)
        self.assertGreaterEqual(block["transfer"], 0.99)
        self.assertEqual(block["verdict"], "compact_world_model_candidate")

    def test_shared_belief_handoff_contract_blocks_weak_solver_use(self):
        domain = {
            "belief_contribution_margin": -0.03,
            "trust": 0.56,
            "evidence_cost": 8.0,
            "interface": {
                "input_contract": {"hidden_target": "local character statistics"},
                "hidden_targets": {"target": "local character statistics"},
                "belief": {
                    "vector_dim": 16,
                    "message_dim": 16,
                    "bitrate": 512,
                    "uncertainty": 0.4,
                    "trust": 0.56,
                },
            },
            "causal_ablation": {
                "lower_is_better": True,
                "learned_value": 3.18,
                "solver_gain": -0.03,
                "content_lift": 0.001,
                "ablation_values": {
                    "zero": 3.19,
                    "shuffled": 3.181,
                    "stale": 3.19,
                },
            },
            "latent_utility": {
                "real_gain": -0.03,
                "wake_up_gate": {
                    "wake_expensive_probe": True,
                    "fallback_probe_roi": 0.07,
                    "reason": "bridge_to_real_gap_high",
                },
            },
            "world_understanding": {
                "counterfactual": 0.82,
                "intervention_lift": 0.32,
                "compression": 0.82,
                "transfer": 0.002,
            },
        }

        block = build_belief_handoff_block("language", domain)
        rate = build_rate_distortion_block("language", {**domain, "belief_handoff": block})

        self.assertFalse(block["gates"]["claim_allowed"])
        self.assertTrue(block["gates"]["all_ablation_arms_pass"])
        self.assertIn("solver_gain_not_positive", block["gates"]["failure_reasons"])
        self.assertLess(block["economics"]["gain_per_sample"], 0.0)
        self.assertEqual(rate["target_retained_utility"], 0.8)
        self.assertEqual(rate["next_test"], "repair_solver_handoff")

    def test_rate_distortion_block_uses_measured_image_compression_curve(self):
        domain = {
            "belief_contribution_margin": 0.02,
            "evidence_cost": 256.0,
            "artifact": {
                "compression_curve": [
                    {
                        "bits": 64,
                        "content_lift": 0.008,
                        "solver_gain": 0.012,
                        "retained_utility": 0.8,
                        "lift_per_1k_bits": 0.125,
                        "gain_per_1k_bits": 0.1875,
                        "decision": "candidate",
                    }
                ]
            },
            "interface": {
                "belief": {
                    "vector_dim": 642,
                    "message_dim": 642,
                    "bitrate": 20544,
                    "trust": 0.9,
                    "uncertainty": 0.1,
                }
            },
            "causal_ablation": {
                "lower_is_better": False,
                "learned_value": 0.90,
                "solver_gain": 0.02,
                "content_lift": 0.01,
                "ablation_values": {"zero": 0.88, "shuffled": 0.89, "stale": 0.87},
            },
            "latent_utility": {"real_gain": 0.02},
            "world_understanding": {"transfer": 0.3},
        }

        block = build_belief_handoff_block("image", domain)
        rate = build_rate_distortion_block("image", {**domain, "belief_handoff": block})

        measured_bits = [row["bits"] for row in rate["rows"] if row["measured"]]
        self.assertIn(64, measured_bits)
        self.assertEqual(rate["best_measured_bits"], 64)

    def test_shared_belief_handoff_contract_allows_board_like_solver_use(self):
        domain = {
            "belief_contribution_margin": 0.44,
            "trust": 1.0,
            "evidence_cost": 2.0,
            "interface": {
                "input_contract": {"hidden_target": "normal versus misere rule"},
                "belief": {
                    "vector_dim": 4,
                    "message_dim": 4,
                    "bitrate": 128,
                    "uncertainty": 0.0,
                    "trust": 1.0,
                },
            },
            "causal_ablation": {
                "lower_is_better": False,
                "learned_value": 1.0,
                "solver_gain": 0.44,
                "content_lift": 0.44,
                "ablation_values": {
                    "zero": 0.56,
                    "shuffled": 0.56,
                    "stale": 0.56,
                },
            },
            "latent_utility": {"real_gain": 0.44, "wake_up_gate": {"wake_expensive_probe": False}},
            "world_understanding": {"counterfactual": 1.0, "transfer": 1.0},
        }

        block = build_belief_handoff_block("board", domain)

        self.assertTrue(block["gates"]["claim_allowed"])
        self.assertEqual(block["gates"]["failure_reasons"], [])
        self.assertGreater(block["economics"]["gain_per_sample"], 0.0)

    def test_real_causal_adapters_use_generic_runner_without_modality_branching(self):
        from teenyreason.crawler.causal import run_causal_crawler

        config = RealCausalAdapterConfig(
            seeds=(0, 1),
            language_spans=3,
            language_span_length=12,
            image_digits=(0, 1),
            image_examples_per_digit=2,
            cartpole_probe_steps=6,
        )
        text = "abcd abcd abcd\n" * 80
        language = run_causal_crawler(
            RealLanguageCausalAdapter(config, text=text),
            seeds=config.seeds,
        )
        images = torch.zeros((4, 1, 28, 28), dtype=torch.float32)
        images[0:2, :, 10:14, 10:14] = 1.0
        images[2:4, :, 16:20, 16:20] = 1.0
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        image = run_causal_crawler(
            RealImageCausalAdapter(config, tensors=(images, labels)),
            seeds=config.seeds,
        )
        cartpole = run_causal_crawler(
            RealCartPoleCausalAdapter(config),
            seeds=config.seeds,
        )

        for result in (language, image, cartpole):
            self.assertEqual(result["runner"], "run_causal_crawler")
            self.assertGreaterEqual(result["factor_decode_accuracy"], 0.5)
            self.assertGreater(result["counterfactual_accuracy"], 0.5)
            self.assertGreater(result["mean_total_cost"], 0.0)

    def test_suite_payload_uses_standard_domain_sections(self):
        payload = build_suite_payload(
            config=MultidomainSuiteConfig(run_rl_benchmark=False),
            run_id="suite-test",
            started_at=1.0,
            results={
                "rl": {},
                "language": {
                    "dataset": "Tiny Shakespeare",
                    "model_family": "BeliefConditionedCharTransformer",
                    "support_windows": 2,
                    "synthetic_grammar": {"hidden_rule_decode_accuracy": 1.0},
                    "rows": [
                        {
                            "baseline_bpc": 2.0,
                            "belief_bpc": 1.8,
                            "shuffled_belief_bpc": 1.9,
                            "continuation_accuracy": 0.7,
                        }
                    ],
                    "artifacts": [
                        {
                            "raw_evidence_windows": [{"query_family": "support_span"}],
                            "domain_belief": [0.1, 0.2],
                            "uncertainty_estimate": 0.4,
                            "ablation_metrics": {"zero_gap": 0.1},
                            "subset_agreement": 0.8,
                            "belief_bitrate": 512,
                        }
                    ],
                },
                "language_bridge": {
                    "dataset": "GeneratedGrammarFamily",
                    "model_family": "RuleMessage+NextTokenSolver",
                    "hidden_target": "grammar_family_next_token_rule",
                    "decode_accuracy": 1.0,
                    "subset_agreement": 1.0,
                    "baseline_accuracy": 0.3,
                    "belief_accuracy": 1.0,
                    "zero_accuracy": 0.3,
                    "shuffled_accuracy": 0.1,
                    "stale_accuracy": 0.2,
                    "content_lift": 0.7,
                },
                "image": {
                    "dataset": "MNIST",
                    "model_family": "BeliefConditionedMNISTCNN",
                    "rows": [
                        {
                            "baseline_accuracy": 0.7,
                            "belief_accuracy": 0.8,
                            "shuffled_belief_accuracy": 0.75,
                            "prototype_stability": 0.6,
                        }
                    ],
                    "artifacts": [
                        {
                            "raw_evidence_windows": [{"class": 7}],
                            "domain_belief": {"prototype_count": 10, "feature_dim": 16},
                            "uncertainty_estimate": 0.3,
                            "ablation_metrics": {"shuffled_gap": 0.05},
                            "subset_agreement": 0.6,
                            "belief_bitrate": 1024,
                        }
                    ],
                },
                "image_bridge": {
                    "dataset": "GeneratedShapeSemanticsFamily",
                    "model_family": "RuleMessage+TemplateClassifier",
                    "hidden_target": "shape_family_label_semantics",
                    "decode_accuracy": 1.0,
                    "subset_agreement": 1.0,
                    "baseline_accuracy": 0.25,
                    "belief_accuracy": 1.0,
                    "zero_accuracy": 0.25,
                    "shuffled_accuracy": 0.1,
                    "stale_accuracy": 0.2,
                    "content_lift": 0.75,
                },
                "board": {
                    "dataset": "TicTacToe hidden-rule positions",
                    "model_family": "CrawlerMessage+ExactMinimax",
                    "rows": [
                        {
                            "baseline_move_accuracy": 0.55,
                            "belief_move_accuracy": 1.0,
                            "shuffled_belief_move_accuracy": 0.55,
                            "rule_decode_accuracy": 1.0,
                            "message_confidence": 1.0,
                            "query_count": 1,
                        }
                    ],
                    "artifacts": [
                        {
                            "raw_evidence_windows": [{"query_family": "x_line_completion"}],
                            "domain_belief": [0.0, 1.0, 0.5, 0.0],
                            "uncertainty_estimate": 0.0,
                            "ablation_metrics": {"shuffled_gap": 0.45},
                            "subset_agreement": 1.0,
                            "belief_bitrate": 128,
                        }
                    ],
                },
                "cartpole_latent_mpc": {
                    "hidden_target": "cartpole_mechanics_action_conditioned_world_model",
                    "model_family": "CrawlerBelief+ActionConditionedLatentMPC",
                    "decode_accuracy": 1.0,
                    "no_belief_return": -7.7,
                    "belief_mpc_return": -6.4,
                    "oracle_mpc_return": -6.4,
                    "solver_gain": 1.3,
                    "content_lift": 0.2,
                    "oracle_gap": 0.0,
                    "belief_action_match_oracle": 1.0,
                    "no_belief_action_match_oracle": 0.76,
                    "belief_k_step_prediction_mse": 0.0,
                    "no_belief_k_step_prediction_mse": 0.77,
                    "belief_solve_rate": 0.98,
                    "no_belief_solve_rate": 1.0,
                    "belief_samples_to_peak_return": 152.0,
                    "no_belief_samples_to_peak_return": 80.0,
                    "belief_samples_to_solve": 152.0,
                    "no_belief_samples_to_solve": 80.0,
                    "net_samples_to_solve_savings": -72.0,
                    "net_env_sample_savings": -72.0,
                    "probe_steps": 72,
                    "horizon": 4,
                    "candidate_count": 32,
                },
                "cartpole_planner_comparison": {
                    "profile": "smoke",
                    "decode_accuracy": 1.0,
                    "cheap_decode_accuracy": 1.0,
                    "solver_gain": 1.3,
                    "content_lift": 0.2,
                    "belief_action_match_oracle": 1.0,
                    "no_belief_action_match_oracle": 0.76,
                    "oracle_gap": 0.0,
                    "action_regret_reduction": 1.3,
                    "probe_roi": 0.018,
                    "cheap_probe_roi": 0.02,
                    "fallback_probe_roi": 0.01,
                    "persistent_affordance_probe_roi": 0.05,
                    "fallback_wake_rate": 0.5,
                    "no_belief_mpc_samples_to_solve": 80.0,
                    "crawler_belief_mpc_samples_to_solve": 152.0,
                    "oracle_mpc_samples_to_solve": 80.0,
                    "cheap_fallback_samples_to_solve": 92.0,
                    "persistent_affordance_samples_to_solve": 98.0,
                    "persistent_affordance_amortized_samples_to_solve": 80.75,
                    "crawler_vs_no_belief_mpc_sample_savings": -72.0,
                    "cheap_fallback_vs_no_belief_mpc_sample_savings": -12.0,
                    "persistent_affordance_vs_no_belief_mpc_sample_savings": -18.0,
                    "persistent_affordance_amortized_vs_no_belief_mpc_sample_savings": -0.75,
                    "persistent_affordance_probe_cost": 18.0,
                    "persistent_affordance_amortized_probe_cost": 0.75,
                    "persistent_affordance_reuse_horizon": 24.0,
                    "persistent_affordance_regret_reduction": 1.2,
                    "persistent_affordance_probe_value": 8.0,
                    "probe_steps": 72.0,
                    "cheap_probe_steps": 12.0,
                    "control_steps": 80.0,
                    "horizon": 4.0,
                    "candidate_count": 32.0,
                    "diagnostic_state": "planner_belief_predictive_but_costly",
                    "rows": [],
                    "arms": [],
                },
            },
            detail_paths={},
        )

        self.assertEqual(payload["schema_version"], 1)
        self.assertIn("cartpole", payload["domains"])
        self.assertIn("language", payload["domains"])
        self.assertIn("image", payload["domains"])
        self.assertIn("board", payload["domains"])
        self.assertEqual(len(payload["cross_domain"]["metric_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["causal_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["mechanism_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["transfer_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["latent_utility_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["wake_up_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["world_understanding_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["belief_handoff_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["rate_distortion_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["handoff_repair_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["decision_utility_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["decision_local_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["sample_performance_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["predictive_planner_rows"]), 4)
        self.assertEqual(len(payload["cross_domain"]["planner_comparison_rows"]), 4)
        self.assertTrue(payload["cross_domain"]["acceptance"]["language_latent_claim_allowed"])
        self.assertTrue(payload["cross_domain"]["acceptance"]["image_latent_claim_allowed"])
        self.assertTrue(payload["cross_domain"]["acceptance"]["board_latent_claim_allowed"])
        self.assertTrue(payload["domains"]["language"]["claim_allowed"])
        self.assertTrue(payload["domains"]["image"]["claim_allowed"])
        self.assertTrue(payload["domains"]["board"]["claim_allowed"])
        for domain_name in ("language", "image", "board"):
            artifact = payload["domains"][domain_name]["artifact"]
            self.assertIn("raw_evidence_windows", artifact)
            self.assertIn("domain_belief", artifact)
            self.assertIn("uncertainty_estimate", artifact)
            self.assertIn("ablation_metrics", artifact)
            self.assertIn("interface", payload["domains"][domain_name])
            self.assertIn("causal_ablation", payload["domains"][domain_name])
            self.assertIn("belief_handoff", payload["domains"][domain_name])
            self.assertIn("rate_distortion", payload["domains"][domain_name])
            self.assertIn("handoff_repair", payload["domains"][domain_name])
            self.assertIn("decision_utility", payload["domains"][domain_name])
            self.assertIn("decision_local_belief", payload["domains"][domain_name])
            self.assertIn("sample_performance", payload["domains"][domain_name])
            self.assertIn("input_contract", payload["domains"][domain_name]["interface"])
            self.assertIn("content_lift", payload["domains"][domain_name]["causal_ablation"])
        board_causal = payload["domains"]["board"]["causal_ablation"]
        self.assertTrue(board_causal["content_causal"])
        self.assertAlmostEqual(board_causal["content_lift"], 0.45)
        language_transfer = payload["domains"]["language"]["transfer_gap"]
        self.assertAlmostEqual(language_transfer["bridge_content_lift"], 0.7)
        self.assertAlmostEqual(language_transfer["real_content_lift"], 0.1)
        language_utility = payload["domains"]["language"]["latent_utility"]
        self.assertAlmostEqual(language_utility["bridge_to_real_gap"], 0.6)
        self.assertEqual(language_utility["bottleneck"], "extend_scale_test")
        self.assertIn("world_understanding", payload["domains"]["language"])
        self.assertIn("counterfactual", payload["domains"]["board"]["world_understanding"])
        planner_row = payload["cross_domain"]["predictive_planner_rows"][0]
        self.assertEqual(planner_row["domain"], "cartpole")
        self.assertAlmostEqual(planner_row["solver_gain"], 1.3)
        self.assertAlmostEqual(planner_row["net_samples_to_solve_savings"], -72.0)
        comparison_row = payload["cross_domain"]["planner_comparison_rows"][0]
        self.assertEqual(comparison_row["domain"], "cartpole")
        self.assertEqual(comparison_row["profile"], "smoke")
        self.assertAlmostEqual(comparison_row["crawler_vs_no_belief_mpc_sample_savings"], -72.0)
        self.assertAlmostEqual(comparison_row["persistent_affordance_amortized_samples_to_solve"], 80.75)
        self.assertAlmostEqual(
            comparison_row["persistent_affordance_amortized_vs_no_belief_mpc_sample_savings"],
            -0.75,
        )
        self.assertIn(
            comparison_row["verdict"],
            {
                "planner_belief_predictive_but_costly",
                "planner_belief_wins_vs_ppo",
                "persistent_affordance_predictive_but_costly",
                "persistent_affordance_wins_vs_mpc",
                "persistent_affordance_wins_vs_ppo",
            },
        )

    def test_suite_acceptance_blocks_tiny_ablation_crumbs(self):
        payload = build_suite_payload(
            config=MultidomainSuiteConfig(run_rl_benchmark=False),
            run_id="suite-test",
            started_at=1.0,
            results={
                "rl": {},
                "language": {
                    "rows": [
                        {
                            "baseline_bpc": 3.15,
                            "belief_bpc": 3.18,
                            "shuffled_belief_bpc": 3.182,
                            "continuation_accuracy": 0.55,
                        }
                    ],
                    "artifacts": [{"subset_agreement": 0.8, "belief_bitrate": 512}],
                },
                "image": {
                    "rows": [
                        {
                            "baseline_accuracy": 0.91,
                            "belief_accuracy": 0.88,
                            "shuffled_belief_accuracy": 0.878,
                            "prototype_stability": 0.9,
                        }
                    ],
                    "artifacts": [{"subset_agreement": 0.9, "belief_bitrate": 1024}],
                },
                "board": {
                    "rows": [
                        {
                            "baseline_move_accuracy": 0.80,
                            "belief_move_accuracy": 0.81,
                            "shuffled_belief_move_accuracy": 0.80,
                            "message_confidence": 0.55,
                        }
                    ],
                    "artifacts": [{"subset_agreement": 0.55, "belief_bitrate": 128}],
                },
            },
            detail_paths={},
        )

        acceptance = payload["cross_domain"]["acceptance"]
        self.assertFalse(acceptance["language_latent_claim_allowed"])
        self.assertFalse(acceptance["image_latent_claim_allowed"])
        self.assertFalse(acceptance["board_latent_claim_allowed"])

    def test_cartpole_suite_payload_uses_rich_benchmark_diagnostics(self):
        payload = cartpole_domain_payload(
            {
                "env_name": "ContinuousCartPole-v0",
                "probe_strict_usage_status": "intermittent",
                "probe_honesty_headline": "Env expression harmful under matched eval",
                "rows": [
                    {"probe_probe_env_steps": 40, "probe_encoder_steps": 10, "probe_control_env_steps": 90},
                    {"probe_probe_env_steps": 20, "probe_encoder_steps": 8, "probe_control_env_steps": 80},
                    {"probe_probe_env_steps": 60, "probe_encoder_steps": 12, "probe_control_env_steps": 100},
                ],
                "summaries": {
                    "probe_episode": {"median": 199.0},
                    "probe_no_expression_episode": {"median": 432.0},
                    "probe_env_expression_delta": {"mean": -8.0},
                    "probe_forced_env_expression_delta": {"mean": -19.0},
                    "full_system_learned_eval": {"mean_return": {"mean": 150.33}},
                    "full_system_zero_context_eval": {"mean_return": {"mean": 150.44}},
                    "full_system_shuffled_context_eval": {"mean_return": {"mean": 150.33}},
                    "full_system_stale_context_eval": {"mean_return": {"mean": 150.33}},
                    "system_id": {
                        "trusted_fraction": 1.0,
                        "progress_median": 0.85,
                        "validation_top1_median": 0.867,
                        "validation_margin_median": 9.0,
                        "particle_entropy_median": 1.33,
                        "particle_ess_ratio_median": 0.025,
                        "particle_leaveout_shift_median": 0.294,
                        "particle_subset_stability_median": 0.223,
                    },
                },
                "research_metrics": {
                    "arms": {
                        "baseline": {"solve_steps_median": 8000.0},
                        "probe": {"solve_steps_median": 12000.0},
                        "probe_no_expression": {"solve_steps_median": 14000.0},
                        "full_system": {"solve_steps_median": 10900.0},
                        "sim_fanout": {"solve_steps_median": 9100.0},
                    },
                    "deltas": {
                        "probe_step_savings_vs_baseline": -4000.0,
                        "probe_step_savings_vs_no_expression": 2000.0,
                    },
                    "peak": {
                        "baseline_steps_to_peak_median": 7600.0,
                        "probe_steps_to_peak_median": 11600.0,
                        "probe_steps_to_peak_savings_vs_baseline": -4000.0,
                        "baseline_best_return_median": 480.0,
                        "probe_best_return_median": 500.0,
                    },
                },
            },
            None,
        )

        self.assertAlmostEqual(payload["trust"], 1.0)
        self.assertAlmostEqual(payload["evidence_cost"], 40.0)
        self.assertAlmostEqual(payload["belief_contribution_margin"], 233.0)
        self.assertLess(payload["ablation_gap"], 0.0)
        self.assertAlmostEqual(payload["metrics"]["sysid_validation_top1"], 0.867)
        self.assertAlmostEqual(payload["metrics"]["probe_expression_delta"], -8.0)
        self.assertEqual(payload["metrics"]["baseline_solve_steps"], 8000.0)
        self.assertEqual(payload["metrics"]["probe_solve_steps"], 12000.0)
        self.assertEqual(payload["metrics"]["probe_steps_to_peak"], 11600.0)
        self.assertEqual(payload["metrics"]["probe_best_return"], 500.0)
        self.assertEqual(set(payload["rows"][0]), {"probe_probe_env_steps", "probe_encoder_steps", "probe_control_env_steps"})

    def test_empty_suite_run_writes_dashboard_ready_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch("time.strftime", return_value="20260102_030405"):
            result = run_multidomain_suite(
                MultidomainSuiteConfig(
                    artifact_dir=Path(tmpdir),
                    run_rl_benchmark=False,
                    run_image_benchmark=False,
                    run_language_benchmark=False,
                )
            )

            self.assertIn("suite", result)
            self.assertTrue(Path(result["suite_path"]).exists())


if __name__ == "__main__":
    unittest.main()
