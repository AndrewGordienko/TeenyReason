"""Latent utility economics for the multi-domain crawler suite."""

from __future__ import annotations

from typing import Any


TARGET_REAL_CONTENT_LIFT = {
    "cartpole": 0.10,
    "language": 0.02,
    "image": 0.02,
    "board": 0.05,
}

EXPENSIVE_WAKE_GAP_FLOOR = {
    "cartpole": 0.50,
    "language": 0.20,
    "image": 0.20,
    "board": 0.05,
}

EXPENSIVE_WAKE_ROI_FLOOR = {
    "cartpole": 0.0,
    "language": 0.01,
    "image": 0.001,
    "board": 0.0,
}


def attach_latent_utility_blocks(domains: dict[str, object]) -> None:
    """Attach one empirical utility block to every domain payload."""
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            domain["latent_utility"] = build_latent_utility_block(domain_name, domain)


def latent_utility_row(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    """Return the compact dashboard/API row for latent utility economics."""
    utility = domain.get("latent_utility", {})
    if not isinstance(utility, dict):
        utility = {}
    return {
        "domain": domain_name,
        "real_gain": utility.get("real_gain", 0.0),
        "real_content_lift": utility.get("real_content_lift", 0.0),
        "bridge_content_lift": utility.get("bridge_content_lift", 0.0),
        "bridge_to_real_gap": utility.get("bridge_to_real_gap", 0.0),
        "low_budget_gain": utility.get("low_budget_gain", 0.0),
        "high_budget_gain": utility.get("high_budget_gain", 0.0),
        "budget_gate_mean_gain": utility.get("budget_gate_mean_gain", 0.0),
        "gate_activation_fraction": utility.get("gate_activation_fraction", 0.0),
        "gain_per_1k_bits": utility.get("gain_per_1k_bits", 0.0),
        "lift_per_evidence": utility.get("lift_per_evidence", 0.0),
        "bottleneck": utility.get("bottleneck", ""),
    }


def wake_up_row(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    """Return expensive-probe wake-up decisions with empirical triggers."""
    utility = domain.get("latent_utility", {})
    if not isinstance(utility, dict):
        utility = {}
    gate = utility.get("wake_up_gate", {})
    if not isinstance(gate, dict):
        gate = {}
    return {
        "domain": domain_name,
        "wake_expensive_probe": bool(gate.get("wake_expensive_probe", False)),
        "reason": gate.get("reason", ""),
        "confidence_low": bool(gate.get("confidence_low", False)),
        "bridge_gap_high": bool(gate.get("bridge_gap_high", False)),
        "high_budget_regression": bool(gate.get("high_budget_regression", False)),
        "real_lift_below_target": bool(gate.get("real_lift_below_target", False)),
        "fallback_probe_roi": gate.get("fallback_probe_roi", 0.0),
        "fallback_roi_floor": gate.get("fallback_roi_floor", 0.0),
        "fallback_roi_positive": bool(gate.get("fallback_roi_positive", False)),
        "target_real_content_lift": gate.get("target_real_content_lift", 0.0),
    }


def build_latent_utility_block(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    """Summarize whether the latent is paying for itself on the real solver."""
    gains = _budget_gains(domain_name, domain)
    real_gain = _float(domain.get("belief_contribution_margin", 0.0))
    causal = _dict(domain.get("causal_ablation"))
    transfer = _dict(domain.get("transfer_gap"))
    metrics = _dict(domain.get("metrics"))
    interface = _dict(domain.get("interface"))
    belief = _dict(interface.get("belief"))

    real_lift = _float(causal.get("content_lift", 0.0))
    bridge_lift = _float(transfer.get("bridge_content_lift", metrics.get("bridge_content_lift", 0.0)))
    bridge_gap = _float(transfer.get("bridge_to_real_gap", bridge_lift - real_lift))
    mechanism_lift = _float(
        transfer.get("mechanism_content_lift", metrics.get("mechanism_content_lift", 0.0))
    )
    bitrate = _float(belief.get("bitrate", metrics.get("belief_bitrate", 0.0)))
    evidence_cost = max(_float(domain.get("evidence_cost", 0.0)), 0.0)
    target_lift = TARGET_REAL_CONTENT_LIFT.get(domain_name, 0.0)

    positive_gains = [gain for gain in gains if gain > 0.0]
    negative_gains = [gain for gain in gains if gain < 0.0]
    budget_gate_mean = _mean([max(0.0, gain) for gain in gains])
    active_gain = _mean(positive_gains)
    gate_activation = len(positive_gains) / max(float(len(gains)), 1.0)
    high_gain = gains[-1] if gains else real_gain
    low_gain = gains[0] if gains else real_gain

    utility = {
        "schema_version": 1,
        "real_gain": real_gain,
        "real_content_lift": real_lift,
        "bridge_content_lift": bridge_lift,
        "mechanism_content_lift": mechanism_lift,
        "bridge_to_real_gap": bridge_gap,
        "target_real_content_lift": target_lift,
        "target_lift_shortfall": max(0.0, target_lift - real_lift),
        "low_budget_gain": low_gain,
        "high_budget_gain": high_gain,
        "mean_budget_gain": _mean(gains),
        "budget_gate_mean_gain": budget_gate_mean,
        "active_budget_mean_gain": active_gain,
        "gate_activation_fraction": gate_activation,
        "positive_budget_count": len(positive_gains),
        "negative_budget_count": len(negative_gains),
        "high_budget_regression": bool(high_gain < 0.0),
        "belief_bitrate": bitrate,
        "message_dim": _float(belief.get("message_dim", 0.0)),
        "evidence_cost": evidence_cost,
        "gain_per_1k_bits": _ratio(real_gain * 1000.0, bitrate),
        "content_lift_per_1k_bits": _ratio(real_lift * 1000.0, bitrate),
        "lift_per_evidence": _ratio(real_lift, evidence_cost),
        "bridge_gap_per_evidence": _ratio(max(0.0, bridge_gap), evidence_cost),
        "bottleneck": _bottleneck(domain_name, real_gain, real_lift, bridge_lift, bridge_gap, bitrate, high_gain),
    }
    utility["wake_up_gate"] = _wake_up_gate(
        domain_name=domain_name,
        domain=domain,
        utility=utility,
        target_lift=target_lift,
    )
    return utility


def _budget_gains(domain_name: str, domain: dict[str, object]) -> list[float]:
    rows = domain.get("rows")
    if not isinstance(rows, list):
        return []
    gains: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if domain_name == "language":
            if "bpc_gain" in row:
                gains.append(_float(row.get("bpc_gain", 0.0)))
                continue
            gains.append(_float(row.get("baseline_bpc", 0.0)) - _float(row.get("belief_bpc", 0.0)))
        elif domain_name == "image":
            if "accuracy_gain" in row:
                gains.append(_float(row.get("accuracy_gain", 0.0)))
                continue
            gains.append(_float(row.get("belief_accuracy", 0.0)) - _float(row.get("baseline_accuracy", 0.0)))
        elif domain_name == "board":
            gains.append(_float(row.get("belief_move_accuracy", 0.0)) - _float(row.get("baseline_move_accuracy", 0.0)))
    return gains


def _wake_up_gate(
    *,
    domain_name: str,
    domain: dict[str, object],
    utility: dict[str, object],
    target_lift: float,
) -> dict[str, object]:
    metrics = _dict(domain.get("metrics"))
    bridge_gap = _float(utility.get("bridge_to_real_gap", 0.0))
    real_lift = _float(utility.get("real_content_lift", 0.0))
    confidence = _domain_confidence(domain_name, domain, metrics)
    confidence_low = confidence < _confidence_floor(domain_name)
    bridge_gap_high = bridge_gap >= EXPENSIVE_WAKE_GAP_FLOOR.get(domain_name, 0.0)
    high_budget_regression = bool(utility.get("high_budget_regression", False))
    real_lift_below_target = real_lift < target_lift
    content_causal = bool(_dict(domain.get("causal_ablation")).get("content_causal", False))
    evidence_cost = max(_float(utility.get("evidence_cost", 0.0)), 0.0)
    fallback_roi = _ratio(max(0.0, bridge_gap), evidence_cost)
    roi_floor = EXPENSIVE_WAKE_ROI_FLOOR.get(domain_name, 0.0)
    roi_positive = fallback_roi >= roi_floor

    wake = False
    reasons: list[str] = []
    if domain_name == "board" and content_causal:
        reasons.append("rule_handoff_causal")
    else:
        if confidence_low:
            wake = True
            reasons.append("cheap_confidence_low")
        if bridge_gap_high and roi_positive:
            wake = True
            reasons.append("bridge_to_real_gap_high")
        elif bridge_gap_high:
            reasons.append("bridge_gap_roi_too_low")
        if high_budget_regression:
            wake = True
            reasons.append("high_budget_regression")
        if (
            real_lift_below_target
            and _float(utility.get("bridge_content_lift", 0.0)) > target_lift
            and roi_positive
        ):
            wake = True
            reasons.append("real_lift_below_target")
        elif real_lift_below_target and _float(utility.get("bridge_content_lift", 0.0)) > target_lift:
            reasons.append("real_lift_roi_too_low")

    return {
        "wake_expensive_probe": wake,
        "reason": ",".join(reasons) if reasons else "no_trigger",
        "confidence": confidence,
        "confidence_low": confidence_low,
        "bridge_gap_high": bridge_gap_high,
        "high_budget_regression": high_budget_regression,
        "real_lift_below_target": real_lift_below_target,
        "fallback_probe_roi": fallback_roi,
        "fallback_roi_floor": roi_floor,
        "fallback_roi_positive": roi_positive,
        "target_real_content_lift": target_lift,
    }


def _domain_confidence(domain_name: str, domain: dict[str, object], metrics: dict[str, object]) -> float:
    if domain_name == "cartpole" and "handoff_cheap_decode_accuracy" in metrics:
        return _float(metrics.get("handoff_cheap_decode_accuracy", 0.0))
    if domain_name == "language":
        return _float(metrics.get("continuation_accuracy", domain.get("trust", 0.0)))
    if domain_name == "image":
        return _float(metrics.get("prototype_stability", domain.get("trust", 0.0)))
    return _float(domain.get("trust", 0.0))


def _confidence_floor(domain_name: str) -> float:
    if domain_name == "cartpole":
        return 0.70
    if domain_name == "language":
        return 0.60
    if domain_name == "image":
        return 0.95
    if domain_name == "board":
        return 0.90
    return 0.50


def _bottleneck(
    domain_name: str,
    real_gain: float,
    real_lift: float,
    bridge_lift: float,
    bridge_gap: float,
    bitrate: float,
    high_gain: float,
) -> str:
    if domain_name == "board" and real_lift >= TARGET_REAL_CONTENT_LIFT["board"]:
        return "scale_rule_space"
    if bridge_lift > 0.0 and real_lift < TARGET_REAL_CONTENT_LIFT.get(domain_name, 0.0):
        if domain_name == "image" and bitrate > 4096.0:
            return "compress_or_gate_message"
        if high_gain < 0.0:
            return "budget_gate_or_adapter"
        if bridge_gap > 0.0:
            return "solver_handoff_interface"
    if real_gain < 0.0:
        return "negative_real_solver_gain"
    if real_lift <= 0.0:
        return "causal_ablation_not_yet_visible"
    return "extend_scale_test"


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)
