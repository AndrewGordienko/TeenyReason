"""Per-domain dashboard payload builders."""

from __future__ import annotations

from pathlib import Path

from ...envs import CONTINUOUS_CARTPOLE_NAME
from ...viz.payloads import build_benchmark_payload
from .config import MultidomainSuiteConfig


def _last_row(result: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(result, dict):
        return {}
    rows = result.get("rows")
    if isinstance(rows, list) and rows:
        row = rows[-1]
        if isinstance(row, dict):
            return row
    return {}


def _load_cartpole_benchmark_payload(
    config: MultidomainSuiteConfig,
    cartpole_result: object,
) -> dict[str, object] | None:
    benchmark_paths: list[Path] = []
    if isinstance(cartpole_result, dict):
        benchmark_tag = cartpole_result.get("benchmark_tag")
        if isinstance(benchmark_tag, str) and benchmark_tag:
            benchmark_paths.append(config.artifact_dir / f"{benchmark_tag}_solve_benchmark.npz")
    else:
        benchmark_paths.append(config.artifact_dir / "continuous_cartpole_ppo_solve_benchmark.npz")
    for benchmark_path in benchmark_paths:
        if not benchmark_path.exists():
            continue
        return build_benchmark_payload(benchmark_path)
    return None


def _language_domain_payload(result: dict[str, object] | None, detail_path: Path | None) -> dict[str, object]:
    row = _last_row(result)
    baseline = float(row.get("baseline_bpc", 0.0))
    belief = float(row.get("belief_bpc", row.get("probe_bpc", 0.0)))
    shuffled = float(row.get("shuffled_belief_bpc", 0.0))
    artifact = {}
    artifacts = result.get("artifacts") if isinstance(result, dict) else None
    if isinstance(artifacts, list) and artifacts and isinstance(artifacts[-1], dict):
        artifact = artifacts[-1]
    return {
        "domain": "language",
        "title": "Shakespeare LM",
        "dataset": None if result is None else result.get("dataset"),
        "model_family": None if result is None else result.get("model_family"),
        "headline_metric": {"name": "validation_bpc", "value": belief, "lower_is_better": True},
        "baseline_metric": {"name": "plain_transformer_bpc", "value": baseline},
        "belief_contribution_margin": baseline - belief,
        "ablation_gap": shuffled - belief,
        "readiness": "active" if belief > 0 else "idle",
        "trust": float(max(0.0, min(1.0, row.get("continuation_accuracy", 0.0)))),
        "evidence_cost": float(result.get("support_windows", 0) if isinstance(result, dict) else 0),
        "rows": [] if result is None else result.get("rows", []),
        "artifact_ref": None if detail_path is None else detail_path.name,
        "artifact": artifact,
        "metrics": {
            "continuation_accuracy": float(row.get("continuation_accuracy", 0.0)),
            "cloze_accuracy": float(row.get("cloze_accuracy", 0.0)),
            "prefix_sensitivity": float(row.get("prefix_sensitivity", 0.0)),
            "selected_handoff_mode": str(row.get("handoff_mode", "")),
            "raw_handoff_mode": str(row.get("raw_handoff_mode", "")),
            "raw_belief_bpc": float(row.get("raw_belief_bpc", row.get("belief_bpc", 0.0))),
            "raw_bpc_gain": float(row.get("raw_bpc_gain", row.get("bpc_gain", 0.0))),
            "raw_content_lift": float(row.get("raw_content_lift", row.get("content_lift", 0.0))),
            "handoff_gate_used_baseline": bool(row.get("handoff_gate_used_baseline", False)),
            "handoff_gate_reason": str(row.get("handoff_gate_reason", "")),
            "decision_gate_use_belief": bool(row.get("decision_gate_use_belief", False)),
            "decision_gate_reason": str(row.get("decision_gate_reason", "")),
            "decision_delta_correct_vs_best_ablation": float(
                row.get("decision_delta_correct_vs_best_ablation", 0.0)
            ),
            "adapter_bpc": float(row.get("adapter_bpc", 0.0)),
            "adapter_bpc_gain": float(row.get("adapter_bpc_gain", 0.0)),
            "adapter_content_lift": float(row.get("adapter_content_lift", 0.0)),
            "prefix_bpc": float(row.get("prefix_bpc", 0.0)),
            "prefix_bpc_gain": float(row.get("prefix_bpc_gain", 0.0)),
            "prefix_content_lift": float(row.get("prefix_content_lift", 0.0)),
            "hidden_rule_decode_quality": float(
                result.get("synthetic_grammar", {}).get("hidden_rule_decode_accuracy", 0.0)
            )
            if isinstance(result, dict)
            else 0.0,
            **_controlled_language_metrics(result),
            "subset_consistency": float(artifact.get("subset_agreement", 0.0)),
            "belief_bitrate": int(artifact.get("belief_bitrate", 0)),
        },
    }


def _image_domain_payload(result: dict[str, object] | None, detail_path: Path | None) -> dict[str, object]:
    row = _last_row(result)
    baseline = float(row.get("baseline_accuracy", 0.0))
    belief = float(row.get("belief_accuracy", row.get("probe_accuracy", 0.0)))
    shuffled = float(row.get("shuffled_belief_accuracy", 0.0))
    artifact = {}
    artifacts = result.get("artifacts") if isinstance(result, dict) else None
    if isinstance(artifacts, list) and artifacts and isinstance(artifacts[-1], dict):
        artifact = artifacts[-1]
    return {
        "domain": "image",
        "title": "MNIST Vision",
        "dataset": None if result is None else result.get("dataset"),
        "model_family": None if result is None else result.get("model_family"),
        "headline_metric": {"name": "accuracy", "value": belief, "lower_is_better": False},
        "baseline_metric": {"name": "plain_cnn_accuracy", "value": baseline},
        "belief_contribution_margin": belief - baseline,
        "ablation_gap": belief - shuffled,
        "readiness": "active" if belief > 0 else "idle",
        "trust": float(max(0.0, min(1.0, row.get("prototype_stability", 0.0)))),
        "evidence_cost": float(row.get("label_budget", 0.0)),
        "rows": [] if result is None else result.get("rows", []),
        "artifact_ref": None if detail_path is None else detail_path.name,
        "artifact": artifact,
        "metrics": {
            "prototype_stability": float(row.get("prototype_stability", 0.0)),
            "belief_nll": float(row.get("belief_nll", 0.0)),
            "zero_accuracy": float(row.get("zero_belief_accuracy", 0.0)),
            "shuffled_accuracy": shuffled,
            "stale_accuracy": float(row.get("stale_belief_accuracy", 0.0)),
            "selected_handoff_mode": str(row.get("handoff_mode", "")),
            "raw_belief_accuracy": float(row.get("raw_belief_accuracy", row.get("belief_accuracy", 0.0))),
            "raw_accuracy_gain": float(row.get("raw_accuracy_gain", row.get("accuracy_gain", 0.0))),
            "raw_content_lift": float(row.get("raw_content_lift", row.get("content_lift", 0.0))),
            "raw_zero_accuracy": float(row.get("raw_zero_belief_accuracy", row.get("zero_belief_accuracy", 0.0))),
            "raw_shuffled_accuracy": float(
                row.get("raw_shuffled_belief_accuracy", row.get("shuffled_belief_accuracy", 0.0))
            ),
            "raw_stale_accuracy": float(row.get("raw_stale_belief_accuracy", row.get("stale_belief_accuracy", 0.0))),
            "handoff_gate_used_baseline": bool(row.get("handoff_gate_used_baseline", False)),
            "handoff_gate_reason": str(row.get("handoff_gate_reason", "")),
            "decision_gate_use_belief": bool(row.get("decision_gate_use_belief", False)),
            "decision_gate_reason": str(row.get("decision_gate_reason", "")),
            "decision_delta_correct_vs_best_ablation": float(
                row.get("decision_delta_correct_vs_best_ablation", 0.0)
            ),
            "best_compressed_accuracy": float(row.get("best_compressed_accuracy", 0.0)),
            "best_compressed_bits": int(row.get("best_compressed_bits", 0)),
            "best_compressed_content_lift": float(row.get("best_compressed_content_lift", 0.0)),
            "residual_accuracy": float(row.get("residual_accuracy", 0.0)),
            "residual_accuracy_gain": float(row.get("residual_accuracy_gain", 0.0)),
            "residual_content_lift": float(row.get("residual_content_lift", 0.0)),
            "residual_generic_gain": float(row.get("residual_accuracy_gain", 0.0))
            - max(float(row.get("residual_content_lift", 0.0)), 0.0),
            **_controlled_image_metrics(result),
            "subset_consistency": float(artifact.get("subset_agreement", 0.0)),
            "belief_bitrate": int(artifact.get("belief_bitrate", 0)),
        },
    }


def _board_domain_payload(result: dict[str, object] | None, detail_path: Path | None) -> dict[str, object]:
    row = _last_row(result)
    baseline = float(row.get("baseline_move_accuracy", 0.0))
    belief = float(row.get("belief_move_accuracy", 0.0))
    shuffled = float(row.get("shuffled_belief_move_accuracy", 0.0))
    artifact = {}
    artifacts = result.get("artifacts") if isinstance(result, dict) else None
    if isinstance(artifacts, list) and artifacts and isinstance(artifacts[-1], dict):
        artifact = artifacts[-1]
    return {
        "domain": "board",
        "title": "Tic-Tac-Toe Rules",
        "dataset": None if result is None else result.get("dataset"),
        "model_family": None if result is None else result.get("model_family"),
        "headline_metric": {"name": "best_move_accuracy", "value": belief, "lower_is_better": False},
        "baseline_metric": {"name": "normal_minimax_accuracy", "value": baseline},
        "belief_contribution_margin": belief - baseline,
        "ablation_gap": belief - shuffled,
        "readiness": "active" if belief > 0 else "idle",
        "trust": float(max(0.0, min(1.0, row.get("message_confidence", 0.0)))),
        "evidence_cost": float(row.get("query_count", 0.0)),
        "rows": [] if result is None else result.get("rows", []),
        "artifact_ref": None if detail_path is None else detail_path.name,
        "artifact": artifact,
        "metrics": {
            "rule_decode_accuracy": float(row.get("rule_decode_accuracy", 0.0)),
            "baseline_value_accuracy": float(row.get("baseline_value_accuracy", 0.0)),
            "belief_value_accuracy": float(row.get("belief_value_accuracy", 0.0)),
            "zero_accuracy": float(row.get("zero_belief_move_accuracy", 0.0)),
            "shuffled_accuracy": shuffled,
            "stale_accuracy": float(row.get("stale_belief_move_accuracy", 0.0)),
            "subset_consistency": float(artifact.get("subset_agreement", 0.0)),
            "belief_bitrate": int(artifact.get("belief_bitrate", 0)),
        },
    }


def _controlled_language_metrics(result: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(result, dict):
        return {}
    controlled = result.get("synthetic_grammar", {})
    if not isinstance(controlled, dict):
        return {}
    return {
        "mechanism_hidden_target": "synthetic_next_token_rule",
        "mechanism_hidden_rule": controlled.get("hidden_rule", ""),
        "mechanism_decoded_rule": controlled.get("decoded_rule", ""),
        "mechanism_decode_accuracy": float(controlled.get("hidden_rule_decode_accuracy", 0.0)),
        "mechanism_baseline_accuracy": float(
            controlled.get("mean_baseline_next_token_accuracy", controlled.get("baseline_next_token_accuracy", 0.0))
        ),
        "mechanism_belief_accuracy": float(controlled.get("belief_next_token_accuracy", 0.0)),
        "mechanism_zero_accuracy": float(controlled.get("zero_next_token_accuracy", 0.0)),
        "mechanism_shuffled_accuracy": float(controlled.get("shuffled_next_token_accuracy", 0.0)),
        "mechanism_stale_accuracy": float(controlled.get("stale_next_token_accuracy", 0.0)),
        "mechanism_content_lift": float(controlled.get("content_lift", 0.0)),
        "mechanism_subset_agreement": float(controlled.get("subset_agreement", 0.0)),
    }


def _controlled_image_metrics(result: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(result, dict):
        return {}
    controlled = result.get("controlled_vision", {})
    if not isinstance(controlled, dict):
        return {}
    return {
        "mechanism_hidden_target": "synthetic_visual_label_semantics",
        "mechanism_hidden_rule": controlled.get("hidden_rule", ""),
        "mechanism_decoded_rule": controlled.get("decoded_rule", ""),
        "mechanism_decode_accuracy": float(controlled.get("hidden_rule_decode_accuracy", 0.0)),
        "mechanism_baseline_accuracy": float(
            controlled.get("mean_baseline_label_accuracy", controlled.get("baseline_label_accuracy", 0.0))
        ),
        "mechanism_belief_accuracy": float(controlled.get("belief_label_accuracy", 0.0)),
        "mechanism_zero_accuracy": float(controlled.get("zero_label_accuracy", 0.0)),
        "mechanism_shuffled_accuracy": float(controlled.get("shuffled_label_accuracy", 0.0)),
        "mechanism_stale_accuracy": float(controlled.get("stale_label_accuracy", 0.0)),
        "mechanism_content_lift": float(controlled.get("content_lift", 0.0)),
        "mechanism_subset_agreement": float(controlled.get("subset_agreement", 0.0)),
    }
