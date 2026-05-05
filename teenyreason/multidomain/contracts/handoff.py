"""Shared belief-to-solver handoff contract for multi-domain reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


COMPRESSION_TARGET_BITS = (8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class BeliefHandoffContract:
    """Solver-facing fields every domain must expose for claim accounting."""

    domain: str
    vector_dim: int
    confidence: float
    uncertainty: float
    evidence_cost: float
    hidden_target: str
    intervention_coverage: float
    counterfactual_score: float
    compression_bits: int
    expected_solver_utility: float


def attach_belief_handoff_blocks(domains: dict[str, object]) -> None:
    """Attach one shared handoff contract and rate rows per domain."""
    for domain_name, domain in domains.items():
        if not isinstance(domain, dict):
            continue
        domain["belief_handoff"] = build_belief_handoff_block(domain_name, domain)
        domain["rate_distortion"] = build_rate_distortion_block(domain_name, domain)


def build_belief_handoff_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Build the common solver contract and hard claim gates."""
    contract = _contract_from_domain(domain_name, domain)
    causal = _dict(domain.get("causal_ablation"))
    utility = _dict(domain.get("latent_utility"))
    world = _dict(domain.get("world_understanding"))
    metrics = _dict(domain.get("metrics"))

    lower_is_better = bool(causal.get("lower_is_better", False))
    learned = _float(causal.get("learned_value", 0.0))
    ablations = _dict(causal.get("ablation_values"))
    beats = {
        name: _beats(learned, _float(value), lower_is_better)
        for name, value in ablations.items()
    }
    required_arms = ("zero", "shuffled", "stale")
    missing_arms = [name for name in required_arms if name not in beats]
    all_ablation_arms_pass = bool(beats) and all(beats.values()) and not missing_arms

    solver_gain = _float(causal.get("solver_gain", domain.get("belief_contribution_margin", 0.0)))
    content_lift = _float(causal.get("content_lift", domain.get("ablation_gap", 0.0)))
    gain_per_sample = _ratio(solver_gain, contract.evidence_cost)
    lift_per_sample = _ratio(content_lift, contract.evidence_cost)
    gain_per_1k_bits = _ratio(solver_gain * 1000.0, float(contract.compression_bits))
    lift_per_1k_bits = _ratio(content_lift * 1000.0, float(contract.compression_bits))
    net_sample_savings = _net_sample_savings(domain_name, domain, solver_gain, contract.evidence_cost)

    positive_gain = solver_gain > 0.0
    positive_gain_per_sample = gain_per_sample > 0.0
    fallback = _fallback_block(domain, utility, metrics, net_sample_savings)
    claim_allowed = bool(
        all_ablation_arms_pass
        and positive_gain
        and positive_gain_per_sample
        and content_lift > 0.0
    )
    failure_reasons = _failure_reasons(
        all_ablation_arms_pass=all_ablation_arms_pass,
        missing_arms=missing_arms,
        positive_gain=positive_gain,
        positive_gain_per_sample=positive_gain_per_sample,
        content_lift=content_lift,
        net_sample_savings=net_sample_savings,
    )

    return {
        "schema_version": 1,
        "contract": {
            "domain": contract.domain,
            "vector_dim": contract.vector_dim,
            "confidence": contract.confidence,
            "uncertainty": contract.uncertainty,
            "evidence_cost": contract.evidence_cost,
            "hidden_target": contract.hidden_target,
            "intervention_coverage": contract.intervention_coverage,
            "counterfactual_score": contract.counterfactual_score,
            "compression_bits": contract.compression_bits,
            "expected_solver_utility": contract.expected_solver_utility,
        },
        "evaluation_modes": {
            "correct": learned,
            "zero": ablations.get("zero", 0.0),
            "shuffled": ablations.get("shuffled", 0.0),
            "stale": ablations.get("stale", 0.0),
            "compressed": _float(world.get("compression", 0.0)),
            "cheap": _float(metrics.get("handoff_cheap_content_lift", 0.0)),
            "cheap_plus_expensive_fallback": fallback["expected_net_utility"],
        },
        "economics": {
            "solver_gain": solver_gain,
            "content_lift": content_lift,
            "gain_per_sample": gain_per_sample,
            "lift_per_sample": lift_per_sample,
            "gain_per_1k_bits": gain_per_1k_bits,
            "lift_per_1k_bits": lift_per_1k_bits,
            "net_sample_savings": net_sample_savings,
        },
        "fallback": fallback,
        "gates": {
            "correct_beats_zero": bool(beats.get("zero", False)),
            "correct_beats_shuffled": bool(beats.get("shuffled", False)),
            "correct_beats_stale": bool(beats.get("stale", False)),
            "all_ablation_arms_pass": all_ablation_arms_pass,
            "positive_gain": positive_gain,
            "positive_gain_per_sample": positive_gain_per_sample,
            "claim_allowed": claim_allowed,
            "failure_reasons": failure_reasons,
        },
    }


def build_rate_distortion_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Report current measured bitrate and explicit compression targets."""
    handoff = _dict(domain.get("belief_handoff"))
    if not handoff:
        handoff = build_belief_handoff_block(domain_name, domain)
    contract = _dict(handoff.get("contract"))
    economics = _dict(handoff.get("economics"))
    world = _dict(domain.get("world_understanding"))
    current_bits = int(max(_float(contract.get("compression_bits", 0.0)), 0.0))
    content_lift = _float(economics.get("content_lift", 0.0))
    solver_gain = _float(economics.get("solver_gain", 0.0))
    transfer = _float(world.get("transfer", 0.0))

    rows: list[dict[str, Any]] = _measured_compression_rows(domain, content_lift)
    if current_bits > 0:
        rows.append(
            {
                "bits": current_bits,
                "measured": True,
                "solver_gain": solver_gain,
                "content_lift": content_lift,
                "transfer_retained": 1.0 if content_lift > 0.0 else 0.0,
                "lift_per_1k_bits": _ratio(content_lift * 1000.0, float(current_bits)),
                "decision": _rate_decision(content_lift, solver_gain, transfer, measured=True),
            }
        )

    for bits in COMPRESSION_TARGET_BITS:
        if bits == current_bits:
            continue
        if current_bits > 0 and bits > current_bits:
            continue
        target_lift = 0.8 * content_lift
        rows.append(
            {
                "bits": bits,
                "measured": False,
                "solver_gain": 0.0,
                "content_lift": 0.0,
                "target_content_lift": target_lift,
                "target_solver_gain": 0.8 * solver_gain,
                "transfer_retained": 0.0,
                "lift_per_1k_bits": 0.0,
                "decision": "run_compression_arm",
            }
        )

    rows = sorted(rows, key=lambda row: (int(row["bits"]), not bool(row["measured"])))
    best_measured = _best_measured_rate(rows)
    return {
        "schema_version": 1,
        "target_retained_utility": 0.8,
        "rows": rows,
        "best_measured_bits": int(best_measured.get("bits", 0)),
        "best_measured_lift_per_1k_bits": _float(best_measured.get("lift_per_1k_bits", 0.0)),
        "next_test": _rate_next_test(domain_name, current_bits, content_lift, solver_gain, transfer),
    }


def belief_handoff_row(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Compact dashboard row for shared handoff gates."""
    block = _dict(domain.get("belief_handoff"))
    contract = _dict(block.get("contract"))
    economics = _dict(block.get("economics"))
    gates = _dict(block.get("gates"))
    failure_reasons = gates.get("failure_reasons", [])
    if not isinstance(failure_reasons, list):
        failure_reasons = []
    return {
        "domain": domain_name,
        "claim_allowed": bool(gates.get("claim_allowed", False)),
        "all_ablation_arms_pass": bool(gates.get("all_ablation_arms_pass", False)),
        "positive_gain_per_sample": bool(gates.get("positive_gain_per_sample", False)),
        "gain_per_sample": economics.get("gain_per_sample", 0.0),
        "lift_per_sample": economics.get("lift_per_sample", 0.0),
        "net_sample_savings": economics.get("net_sample_savings", 0.0),
        "compression_bits": contract.get("compression_bits", 0),
        "expected_solver_utility": contract.get("expected_solver_utility", 0.0),
        "counterfactual_score": contract.get("counterfactual_score", 0.0),
        "failure_reasons": ",".join(str(item) for item in failure_reasons),
    }


def rate_distortion_row(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Compact dashboard row for the current compression state."""
    block = _dict(domain.get("rate_distortion"))
    rows = block.get("rows", [])
    measured = [row for row in rows if isinstance(row, dict) and row.get("measured")] if isinstance(rows, list) else []
    current = measured[-1] if measured else {}
    target_rows = [row for row in rows if isinstance(row, dict) and not row.get("measured")] if isinstance(rows, list) else []
    smallest_target = target_rows[0] if target_rows else {}
    return {
        "domain": domain_name,
        "current_bits": current.get("bits", 0),
        "current_content_lift": current.get("content_lift", 0.0),
        "current_lift_per_1k_bits": current.get("lift_per_1k_bits", 0.0),
        "current_transfer_retained": current.get("transfer_retained", 0.0),
        "target_bits": smallest_target.get("bits", 0),
        "target_content_lift": smallest_target.get("target_content_lift", 0.0),
        "best_measured_bits": block.get("best_measured_bits", 0),
        "best_measured_lift_per_1k_bits": block.get("best_measured_lift_per_1k_bits", 0.0),
        "next_test": block.get("next_test", ""),
    }


def _contract_from_domain(domain_name: str, domain: dict[str, Any]) -> BeliefHandoffContract:
    interface = _dict(domain.get("interface"))
    belief = _dict(interface.get("belief"))
    metrics = _dict(domain.get("metrics"))
    utility = _dict(domain.get("latent_utility"))
    world = _dict(domain.get("world_understanding"))
    causal = _dict(domain.get("causal_ablation"))
    hidden_targets = _dict(interface.get("hidden_targets"))
    input_contract = _dict(interface.get("input_contract"))

    hidden_target = str(
        metrics.get(
            "real_causal_hidden_target",
            hidden_targets.get("target", input_contract.get("hidden_target", "")),
        )
    )
    return BeliefHandoffContract(
        domain=domain_name,
        vector_dim=int(_float(belief.get("vector_dim", belief.get("message_dim", 0)))),
        confidence=_float(belief.get("trust", domain.get("trust", 0.0))),
        uncertainty=_float(belief.get("uncertainty", 0.0)),
        evidence_cost=max(_float(domain.get("evidence_cost", utility.get("evidence_cost", 0.0))), 0.0),
        hidden_target=hidden_target,
        intervention_coverage=_float(metrics.get("real_causal_intervention_coverage", world.get("intervention_lift", 0.0))),
        counterfactual_score=_float(metrics.get("real_causal_counterfactual_accuracy", world.get("counterfactual", 0.0))),
        compression_bits=int(_float(belief.get("bitrate", metrics.get("belief_bitrate", 0.0)))),
        expected_solver_utility=_float(utility.get("real_gain", causal.get("solver_gain", 0.0))),
    )


def _fallback_block(
    domain: dict[str, Any],
    utility: dict[str, Any],
    metrics: dict[str, Any],
    net_sample_savings: float,
) -> dict[str, Any]:
    gate = _dict(utility.get("wake_up_gate"))
    fallback_count = 1 if bool(gate.get("wake_expensive_probe", False)) else 0
    expensive_steps = _float(metrics.get("handoff_expensive_dedicated_probe_steps", 0.0))
    cheap_steps = _float(metrics.get("handoff_cheap_dedicated_probe_steps", 0.0))
    avoided_steps = _float(metrics.get("handoff_dedicated_probe_steps_saved", max(0.0, expensive_steps - cheap_steps)))
    fallback_roi = _float(gate.get("fallback_probe_roi", 0.0))
    return {
        "wake_expensive_probe": bool(gate.get("wake_expensive_probe", False)),
        "fallback_probe_roi": fallback_roi,
        "expensive_fallback_count": fallback_count,
        "avoided_dedicated_probe_steps": avoided_steps,
        "cheap_context_return_lift": _float(metrics.get("handoff_cheap_content_lift", 0.0)),
        "expected_net_utility": fallback_roi * max(avoided_steps, 1.0),
        "net_sample_savings_vs_baseline": net_sample_savings,
        "reason": str(gate.get("reason", "")),
    }


def _net_sample_savings(
    domain_name: str,
    domain: dict[str, Any],
    solver_gain: float,
    evidence_cost: float,
) -> float:
    metrics = _dict(domain.get("metrics"))
    if domain_name == "cartpole":
        probe_steps = _float(metrics.get("probe_total_probe_steps", 0.0))
        control_steps = _float(metrics.get("probe_control_steps", 0.0))
        if probe_steps > 0.0 or control_steps > 0.0:
            return solver_gain - probe_steps
    return solver_gain


def _failure_reasons(
    *,
    all_ablation_arms_pass: bool,
    missing_arms: list[str],
    positive_gain: bool,
    positive_gain_per_sample: bool,
    content_lift: float,
    net_sample_savings: float,
) -> list[str]:
    reasons: list[str] = []
    if missing_arms:
        reasons.append("missing_" + "_".join(missing_arms))
    if not all_ablation_arms_pass:
        reasons.append("belief_not_ablation_causal")
    if not positive_gain:
        reasons.append("solver_gain_not_positive")
    if not positive_gain_per_sample:
        reasons.append("gain_per_sample_not_positive")
    if content_lift <= 0.0:
        reasons.append("content_lift_not_positive")
    if net_sample_savings < 0.0:
        reasons.append("negative_net_sample_savings")
    return reasons


def _rate_decision(content_lift: float, solver_gain: float, transfer: float, *, measured: bool) -> str:
    if not measured:
        return "run_compression_arm"
    if content_lift <= 0.0:
        return "repair_handoff_first"
    if solver_gain <= 0.0:
        return "belief_not_helping_solver"
    if transfer < 0.2:
        return "transfer_bottleneck"
    return "compress_until_80pct_retained"


def _rate_next_test(
    domain_name: str,
    current_bits: int,
    content_lift: float,
    solver_gain: float,
    transfer: float,
) -> str:
    if content_lift <= 0.0 or solver_gain <= 0.0:
        return "repair_solver_handoff"
    if domain_name == "image" or current_bits > 4096:
        return "compression_curve"
    if transfer < 0.2:
        return "adapter_handoff_ablation"
    return "smaller_bitrate_arm"


def _best_measured_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if row.get("measured")]
    if not measured:
        return {}
    return max(measured, key=lambda row: _float(row.get("lift_per_1k_bits", 0.0)))


def _measured_compression_rows(domain: dict[str, Any], full_content_lift: float) -> list[dict[str, Any]]:
    artifact = _dict(domain.get("artifact"))
    raw_rows = artifact.get("compression_curve", [])
    if not isinstance(raw_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        bits = int(_float(raw.get("bits", 0.0)))
        content_lift = _float(raw.get("content_lift", 0.0))
        retained = _float(raw.get("retained_utility", 0.0))
        if retained <= 0.0 and full_content_lift > 0.0:
            retained = max(0.0, content_lift / full_content_lift)
        rows.append(
            {
                "bits": bits,
                "measured": True,
                "solver_gain": _float(raw.get("solver_gain", raw.get("accuracy_gain_vs_baseline", 0.0))),
                "content_lift": content_lift,
                "transfer_retained": retained,
                "lift_per_1k_bits": _float(raw.get("lift_per_1k_bits", 0.0)),
                "gain_per_1k_bits": _float(raw.get("gain_per_1k_bits", 0.0)),
                "decision": str(raw.get("decision", _rate_decision(content_lift, 0.0, retained, measured=True))),
            }
        )
    return rows


def _beats(learned: float, ablation: float, lower_is_better: bool) -> bool:
    if lower_is_better:
        return learned < ablation
    return learned > ablation


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)
