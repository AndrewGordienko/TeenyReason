"""Empirical handoff repair diagnostics for crawler belief consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .decision_gate import (
    DecisionGateInput,
    DecisionGateResult,
    decision_gate_payload,
    evaluate_decision_delta_gate,
)


@dataclass(frozen=True)
class HandoffArm:
    """One candidate way to hand belief to a solver."""

    mode: str
    solver_gain: float
    content_lift: float
    evidence_cost: float
    bits: int


def attach_handoff_repair_blocks(domains: dict[str, object]) -> None:
    """Attach best-arm and blocker diagnostics to every domain payload."""
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            domain["handoff_repair"] = build_handoff_repair_block(domain_name, domain)


def build_handoff_repair_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Build empirical arm-selection diagnostics for one domain."""
    arms = _candidate_arms(domain_name, domain)
    best = _best_useful_arm(arms)
    current = _current_arm(domain_name, domain, arms)
    decision_delta = _decision_delta(domain_name, domain, best, current)
    accepted_fraction = _accepted_fraction(domain_name, domain)
    fallback_roi = _fallback_roi(domain)
    best_gate = _gate_for_arm(domain_name, best, decision_delta, fallback_roi)
    current_gate = _gate_for_arm(domain_name, current, decision_delta, fallback_roi)
    blocker = _blocker(domain_name, domain, best, current, best_gate, fallback_roi)
    next_action = _next_action(domain_name, blocker)
    return {
        "schema_version": 1,
        "best_arm": _arm_payload(best),
        "current_arm": _arm_payload(current),
        "best_gate": decision_gate_payload(best_gate) if best_gate is not None else {},
        "current_gate": decision_gate_payload(current_gate) if current_gate is not None else {},
        "arm_count": len(arms),
        "accepted_fraction": accepted_fraction,
        "decision_delta_correct_vs_shuffled": decision_delta,
        "decision_delta_correct_vs_best_ablation": 0.0
        if best_gate is None
        else best_gate.decision_delta_correct_vs_best_ablation,
        "fallback_probe_roi": fallback_roi,
        "blocker": blocker,
        "next_action": next_action,
    }


def handoff_repair_row(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Return one dashboard/API row for handoff repair priorities."""
    block = domain.get("handoff_repair", {})
    if not isinstance(block, dict):
        block = build_handoff_repair_block(domain_name, domain)
    best = _dict(block.get("best_arm"))
    current = _dict(block.get("current_arm"))
    return {
        "domain": domain_name,
        "current_arm": current.get("mode", ""),
        "current_gain": current.get("solver_gain", 0.0),
        "best_arm": best.get("mode", ""),
        "best_gain": best.get("solver_gain", 0.0),
        "best_content_lift": best.get("content_lift", 0.0),
        "best_gain_per_cost": best.get("gain_per_cost", 0.0),
        "best_lift_per_1k_bits": best.get("lift_per_1k_bits", 0.0),
        "accepted_fraction": block.get("accepted_fraction", 0.0),
        "decision_delta_correct_vs_shuffled": block.get("decision_delta_correct_vs_shuffled", 0.0),
        "decision_delta_correct_vs_best_ablation": block.get("decision_delta_correct_vs_best_ablation", 0.0),
        "fallback_probe_roi": block.get("fallback_probe_roi", 0.0),
        "blocker": block.get("blocker", ""),
        "next_action": block.get("next_action", ""),
    }


def _candidate_arms(domain_name: str, domain: dict[str, Any]) -> list[HandoffArm]:
    if domain_name == "language":
        return _language_arms(domain)
    if domain_name == "image":
        return _image_arms(domain)
    if domain_name == "board":
        return _board_arms(domain)
    return _generic_arms(domain_name, domain)


def _language_arms(domain: dict[str, Any]) -> list[HandoffArm]:
    row = _last_row(domain)
    cost = max(_float(domain.get("evidence_cost", 0.0)), 1.0)
    bits = int(_float(row.get("belief_bitrate", _contract_bits(domain))))
    return [
        HandoffArm("prefix", _float(row.get("prefix_bpc_gain", 0.0)), _float(row.get("prefix_content_lift", 0.0)), cost, bits),
        HandoffArm("adapter", _float(row.get("adapter_bpc_gain", 0.0)), _float(row.get("adapter_content_lift", 0.0)), cost, bits),
        HandoffArm("raw_best", _float(row.get("raw_bpc_gain", 0.0)), _float(row.get("raw_content_lift", 0.0)), cost, bits),
    ]


def _image_arms(domain: dict[str, Any]) -> list[HandoffArm]:
    row = _last_row(domain)
    cost = max(_float(domain.get("evidence_cost", row.get("label_budget", 0.0))), 1.0)
    bits = int(_float(row.get("belief_bitrate", _contract_bits(domain))))
    arms = [
        HandoffArm(
            "full_belief",
            _float(row.get("raw_accuracy_gain", row.get("accuracy_gain", 0.0))),
            _float(row.get("raw_content_lift", row.get("content_lift", 0.0))),
            cost,
            bits,
        )
    ]
    artifact = _dict(domain.get("artifact"))
    curve = artifact.get("compression_curve", [])
    if isinstance(curve, list):
        for item in curve:
            if not isinstance(item, dict):
                continue
            arms.append(
                HandoffArm(
                    str(item.get("mode", f"compressed_{int(_float(item.get('bits', 0.0)))}")),
                    _float(item.get("solver_gain", item.get("accuracy_gain_vs_baseline", 0.0))),
                    _float(item.get("content_lift", 0.0)),
                    cost,
                    int(_float(item.get("bits", 0.0))),
                )
            )
    return arms


def _board_arms(domain: dict[str, Any]) -> list[HandoffArm]:
    causal = _dict(domain.get("causal_ablation"))
    return [
        HandoffArm(
            "rule_belief",
            _float(causal.get("solver_gain", domain.get("belief_contribution_margin", 0.0))),
            _float(causal.get("content_lift", domain.get("ablation_gap", 0.0))),
            max(_float(domain.get("evidence_cost", 0.0)), 1.0),
            _contract_bits(domain),
        )
    ]


def _generic_arms(domain_name: str, domain: dict[str, Any]) -> list[HandoffArm]:
    causal = _dict(domain.get("causal_ablation"))
    return [
        HandoffArm(
            f"{domain_name}_belief",
            _float(causal.get("solver_gain", domain.get("belief_contribution_margin", 0.0))),
            _float(causal.get("content_lift", domain.get("ablation_gap", 0.0))),
            max(_float(domain.get("evidence_cost", 0.0)), 1.0),
            _contract_bits(domain),
        )
    ]


def _best_useful_arm(arms: list[HandoffArm]) -> HandoffArm | None:
    useful = [arm for arm in arms if arm.solver_gain > 0.0 and arm.content_lift > 0.0]
    if not useful:
        return None
    return max(useful, key=lambda arm: (arm.solver_gain / max(arm.evidence_cost, 1.0), arm.content_lift))


def _current_arm(domain_name: str, domain: dict[str, Any], arms: list[HandoffArm]) -> HandoffArm | None:
    row = _last_row(domain)
    selected = str(row.get("handoff_mode", ""))
    if domain_name == "language" and selected:
        return _find_arm(arms, selected)
    if domain_name == "image" and selected:
        return _find_arm(arms, selected)
    best = _best_useful_arm(arms)
    return best if best is not None else (arms[0] if arms else None)


def _decision_delta(
    domain_name: str,
    domain: dict[str, Any],
    best: HandoffArm | None,
    current: HandoffArm | None,
) -> float:
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        return _float(metrics.get("learned_eval_return", 0.0)) - _float(metrics.get("shuffled_eval_return", 0.0))
    arm = best if best is not None else current
    return 0.0 if arm is None else arm.content_lift


def _accepted_fraction(domain_name: str, domain: dict[str, Any]) -> float:
    rows = domain.get("rows")
    if domain_name in {"language", "image"} and isinstance(rows, list) and rows:
        accepted = [
            row
            for row in rows
            if isinstance(row, dict) and not bool(row.get("handoff_gate_used_baseline", False))
        ]
        return float(len(accepted)) / float(len(rows))
    if domain_name == "board":
        return 1.0
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        return max(0.0, min(1.0, _float(metrics.get("probe_msg_seen_frac", 0.0))))
    return 0.0


def _fallback_roi(domain: dict[str, Any]) -> float:
    utility = _dict(domain.get("latent_utility"))
    gate = _dict(utility.get("wake_up_gate"))
    return _float(gate.get("fallback_probe_roi", 0.0))


def _blocker(
    domain_name: str,
    domain: dict[str, Any],
    best: HandoffArm | None,
    current: HandoffArm | None,
    best_gate: DecisionGateResult | None,
    fallback_roi: float,
) -> str:
    handoff = _dict(domain.get("belief_handoff"))
    economics = _dict(handoff.get("economics"))
    net_sample_savings = _float(economics.get("net_sample_savings", 0.0))
    if best is None:
        return "no_positive_causal_solver_arm"
    if best_gate is not None and not best_gate.use_belief:
        if best_gate.reason.startswith("correct_not_better"):
            return "belief_not_decision_causal"
        return best_gate.reason
    if domain_name == "cartpole" and net_sample_savings <= 0.0:
        return "probe_cost_exceeds_solver_gain"
    if current is None or current.mode == "baseline_fallback":
        return "use_best_positive_arm"
    if fallback_roi > 0.0 and current.solver_gain <= 0.0:
        return "fallback_saved_harm"
    return "ready_for_scale_test"


def _next_action(domain_name: str, blocker: str) -> str:
    if blocker == "probe_cost_exceeds_solver_gain":
        return "train_cheap_context_policy_then_probe_only_on_positive_delta"
    if blocker == "belief_not_decision_causal":
        return "learn_correct_vs_shuffled_decision_delta_gate"
    if blocker == "no_positive_causal_solver_arm":
        if domain_name == "language":
            return "freeze_baseline_and_train_residual_adapter_gate"
        if domain_name == "image":
            return "train_compressed_residual_gate_before_full_belief"
        return "repair_solver_handoff"
    if blocker == "use_best_positive_arm":
        return "promote_best_arm_with_ablation_gate"
    if blocker == "fallback_saved_harm":
        return "keep_baseline_fallback_and_collect_more_gate_labels"
    return "scale_harder_split"


def _gate_for_arm(
    domain_name: str,
    arm: HandoffArm | None,
    decision_delta: float,
    fallback_roi: float,
) -> DecisionGateResult | None:
    if arm is None:
        return None
    ablation_delta = decision_delta if domain_name == "cartpole" else arm.content_lift
    return evaluate_decision_delta_gate(
        DecisionGateInput(
            domain=domain_name,
            mode=arm.mode,
            lower_is_better=False,
            baseline_value=0.0,
            correct_value=ablation_delta,
            zero_value=0.0,
            shuffled_value=0.0,
            stale_value=0.0,
            solver_gain=arm.solver_gain,
            content_lift=arm.content_lift,
            evidence_cost=arm.evidence_cost,
            bits=arm.bits,
            fallback_roi=fallback_roi,
        )
    )


def _arm_payload(arm: HandoffArm | None) -> dict[str, Any]:
    if arm is None:
        return {
            "mode": "none",
            "solver_gain": 0.0,
            "content_lift": 0.0,
            "evidence_cost": 0.0,
            "bits": 0,
            "gain_per_cost": 0.0,
            "lift_per_cost": 0.0,
            "gain_per_1k_bits": 0.0,
            "lift_per_1k_bits": 0.0,
        }
    return {
        "mode": arm.mode,
        "solver_gain": arm.solver_gain,
        "content_lift": arm.content_lift,
        "evidence_cost": arm.evidence_cost,
        "bits": arm.bits,
        "gain_per_cost": arm.solver_gain / max(arm.evidence_cost, 1.0),
        "lift_per_cost": arm.content_lift / max(arm.evidence_cost, 1.0),
        "gain_per_1k_bits": arm.solver_gain * 1000.0 / max(float(arm.bits), 1.0),
        "lift_per_1k_bits": arm.content_lift * 1000.0 / max(float(arm.bits), 1.0),
    }


def _find_arm(arms: list[HandoffArm], mode: str) -> HandoffArm | None:
    for arm in arms:
        if arm.mode == mode:
            return arm
    return None


def _contract_bits(domain: dict[str, Any]) -> int:
    interface = _dict(domain.get("interface"))
    belief = _dict(interface.get("belief"))
    return int(_float(belief.get("bitrate", 0.0)))


def _last_row(domain: dict[str, Any]) -> dict[str, Any]:
    rows = domain.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[-1], dict):
        return rows[-1]
    return {}


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
