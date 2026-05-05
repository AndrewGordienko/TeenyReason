"""Decision-level utility accounting for crawler belief handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionUtilityContract:
    """One domain's decision-facing belief economics."""

    domain: str
    raw_solver_gain: float
    gated_solver_gain: float
    accepted_fraction: float
    gain_when_accepted: float
    cost_when_accepted: float
    decision_delta_correct_vs_shuffled: float
    decision_delta_correct_vs_best_ablation: float
    avoided_harm: float
    net_utility: float
    decision: str
    next_action: str


def attach_decision_utility_blocks(domains: dict[str, object]) -> None:
    """Attach decision-level utility diagnostics to every domain."""
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            domain["decision_utility"] = build_decision_utility_block(domain_name, domain)


def build_decision_utility_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Build normalized decision-utility fields for one domain."""
    contract = _contract_from_domain(domain_name, domain)
    return {
        "schema_version": 1,
        "contract": {
            "domain": contract.domain,
            "raw_solver_gain": contract.raw_solver_gain,
            "gated_solver_gain": contract.gated_solver_gain,
            "accepted_fraction": contract.accepted_fraction,
            "gain_when_accepted": contract.gain_when_accepted,
            "cost_when_accepted": contract.cost_when_accepted,
            "decision_delta_correct_vs_shuffled": contract.decision_delta_correct_vs_shuffled,
            "decision_delta_correct_vs_best_ablation": contract.decision_delta_correct_vs_best_ablation,
            "avoided_harm": contract.avoided_harm,
            "net_utility": contract.net_utility,
            "decision": contract.decision,
            "next_action": contract.next_action,
        },
        "gates": {
            "raw_belief_helpful": contract.raw_solver_gain > 0.0,
            "gated_belief_helpful": contract.gated_solver_gain > 0.0,
            "decision_causal": contract.decision_delta_correct_vs_shuffled > 0.0,
            "cost_positive": contract.cost_when_accepted > 0.0,
            "net_positive": contract.net_utility > 0.0,
            "use_belief": contract.decision == "use_belief",
        },
    }


def decision_utility_row(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Return one compact dashboard row for decision-level belief utility."""
    block = domain.get("decision_utility", {})
    if not isinstance(block, dict):
        block = build_decision_utility_block(domain_name, domain)
    contract = block.get("contract", {})
    if not isinstance(contract, dict):
        contract = {}
    return {
        "domain": domain_name,
        "raw_solver_gain": contract.get("raw_solver_gain", 0.0),
        "gated_solver_gain": contract.get("gated_solver_gain", 0.0),
        "accepted_fraction": contract.get("accepted_fraction", 0.0),
        "gain_when_accepted": contract.get("gain_when_accepted", 0.0),
        "cost_when_accepted": contract.get("cost_when_accepted", 0.0),
        "decision_delta_correct_vs_shuffled": contract.get("decision_delta_correct_vs_shuffled", 0.0),
        "decision_delta_correct_vs_best_ablation": contract.get("decision_delta_correct_vs_best_ablation", 0.0),
        "avoided_harm": contract.get("avoided_harm", 0.0),
        "net_utility": contract.get("net_utility", 0.0),
        "decision": contract.get("decision", ""),
        "next_action": contract.get("next_action", ""),
    }


def _contract_from_domain(domain_name: str, domain: dict[str, Any]) -> DecisionUtilityContract:
    raw_gain = _raw_solver_gain(domain_name, domain)
    gated_gain = _gated_solver_gain(domain_name, domain)
    accepted_fraction = _accepted_fraction(domain_name, domain)
    gain_when_accepted = _gain_when_accepted(domain_name, domain, gated_gain)
    cost_when_accepted = _cost_when_accepted(domain, accepted_fraction)
    decision_delta = _decision_delta(domain_name, domain)
    decision_delta_best = _decision_delta_best_ablation(domain_name, domain, decision_delta)
    avoided_harm = max(0.0, gated_gain - raw_gain)
    net_utility = _net_utility(domain_name, domain, gated_gain, cost_when_accepted)
    decision = _decision(
        raw_solver_gain=raw_gain,
        gated_solver_gain=gated_gain,
        decision_delta=decision_delta_best,
        accepted_fraction=accepted_fraction,
        net_utility=net_utility,
    )
    return DecisionUtilityContract(
        domain=domain_name,
        raw_solver_gain=raw_gain,
        gated_solver_gain=gated_gain,
        accepted_fraction=accepted_fraction,
        gain_when_accepted=gain_when_accepted,
        cost_when_accepted=cost_when_accepted,
        decision_delta_correct_vs_shuffled=decision_delta,
        decision_delta_correct_vs_best_ablation=decision_delta_best,
        avoided_harm=avoided_harm,
        net_utility=net_utility,
        decision=decision,
        next_action=_next_action(domain_name, decision, raw_gain, gated_gain, decision_delta),
    )


def _raw_solver_gain(domain_name: str, domain: dict[str, Any]) -> float:
    row = _last_row(domain)
    if domain_name == "language":
        return _float(row.get("raw_bpc_gain", row.get("bpc_gain", 0.0)))
    if domain_name == "image":
        return _float(row.get("raw_accuracy_gain", row.get("accuracy_gain", 0.0)))
    causal = _dict(domain.get("causal_ablation"))
    return _float(causal.get("solver_gain", domain.get("belief_contribution_margin", 0.0)))


def _gated_solver_gain(domain_name: str, domain: dict[str, Any]) -> float:
    row = _last_row(domain)
    if domain_name == "language":
        return _float(row.get("bpc_gain", domain.get("belief_contribution_margin", 0.0)))
    if domain_name == "image":
        return _float(row.get("accuracy_gain", domain.get("belief_contribution_margin", 0.0)))
    causal = _dict(domain.get("causal_ablation"))
    return _float(causal.get("solver_gain", domain.get("belief_contribution_margin", 0.0)))


def _accepted_fraction(domain_name: str, domain: dict[str, Any]) -> float:
    rows = domain.get("rows")
    if isinstance(rows, list) and rows:
        if domain_name in {"language", "image"}:
            accepted = [
                1.0
                for row in rows
                if isinstance(row, dict) and not bool(row.get("handoff_gate_used_baseline", False))
            ]
            return float(len(accepted) / max(len(rows), 1))
        if domain_name == "board":
            return 1.0
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        seen = _float(metrics.get("probe_msg_seen_frac", metrics.get("strict_env_expression_usage", 0.0)))
        return max(0.0, min(1.0, seen))
    return 0.0


def _gain_when_accepted(domain_name: str, domain: dict[str, Any], fallback_gain: float) -> float:
    rows = domain.get("rows")
    gains: list[float] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict) or bool(row.get("handoff_gate_used_baseline", False)):
                continue
            if domain_name == "language":
                gains.append(_float(row.get("bpc_gain", 0.0)))
            elif domain_name == "image":
                gains.append(_float(row.get("accuracy_gain", 0.0)))
    if gains:
        return sum(gains) / float(len(gains))
    if domain_name == "board":
        return fallback_gain
    return 0.0


def _cost_when_accepted(domain: dict[str, Any], accepted_fraction: float) -> float:
    if accepted_fraction <= 0.0:
        return 0.0
    handoff = _dict(domain.get("belief_handoff"))
    contract = _dict(handoff.get("contract"))
    evidence_cost = _float(contract.get("evidence_cost", domain.get("evidence_cost", 0.0)))
    return evidence_cost / max(accepted_fraction, 1e-9)


def _decision_delta(domain_name: str, domain: dict[str, Any]) -> float:
    row = _last_row(domain)
    if domain_name == "language":
        return _float(row.get("content_lift", row.get("raw_content_lift", 0.0)))
    if domain_name == "image":
        return _float(row.get("content_lift", row.get("raw_content_lift", 0.0)))
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        learned = _float(metrics.get("learned_eval_return", 0.0))
        shuffled = _float(metrics.get("shuffled_eval_return", 0.0))
        return learned - shuffled
    causal = _dict(domain.get("causal_ablation"))
    return _float(causal.get("content_lift", domain.get("ablation_gap", 0.0)))


def _decision_delta_best_ablation(domain_name: str, domain: dict[str, Any], fallback: float) -> float:
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        learned = _float(metrics.get("learned_eval_return", 0.0))
        ablations = [
            _float(metrics.get("zero_eval_return", 0.0)),
            _float(metrics.get("shuffled_eval_return", 0.0)),
            _float(metrics.get("stale_eval_return", 0.0)),
        ]
        if any(value > 0.0 for value in ablations):
            return learned - max(ablations)
    return fallback


def _net_utility(domain_name: str, domain: dict[str, Any], gated_gain: float, cost: float) -> float:
    handoff = _dict(domain.get("belief_handoff"))
    economics = _dict(handoff.get("economics"))
    if domain_name == "cartpole":
        return _float(economics.get("net_sample_savings", gated_gain - cost))
    return gated_gain


def _decision(
    *,
    raw_solver_gain: float,
    gated_solver_gain: float,
    decision_delta: float,
    accepted_fraction: float,
    net_utility: float,
) -> str:
    if gated_solver_gain > 0.0 and decision_delta > 0.0 and net_utility > 0.0:
        return "use_belief"
    if raw_solver_gain < 0.0 <= gated_solver_gain:
        return "baseline_fallback_saved_harm"
    if accepted_fraction <= 0.0:
        return "do_not_use_belief"
    if decision_delta <= 0.0:
        return "not_decision_causal"
    if net_utility <= 0.0:
        return "cost_exceeds_gain"
    return "needs_handoff_repair"


def _next_action(
    domain_name: str,
    decision: str,
    raw_gain: float,
    gated_gain: float,
    decision_delta: float,
) -> str:
    if decision == "use_belief":
        if domain_name == "board":
            return "scale_rule_space"
        return "tighten_gate_and_compress"
    if decision == "baseline_fallback_saved_harm":
        if domain_name == "image":
            return "train_residual_gate_on_compressed_context"
        if domain_name == "language":
            return "freeze_baseline_then_train_small_adapter_gate"
        return "keep_fallback_and_repair_solver_interface"
    if raw_gain > 0.0 and gated_gain <= 0.0:
        return "lower_conservative_gate_after_validation"
    if decision_delta <= 0.0:
        return "learn_decision_delta_predictor"
    if domain_name == "cartpole":
        return "cheap_context_ppo_with_positive_roi_gate"
    return "repair_solver_handoff"


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
