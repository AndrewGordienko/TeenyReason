"""Shared empirical gate for handing crawler belief to solvers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionGateInput:
    """Metric-normalized evidence for one candidate belief arm."""

    domain: str
    mode: str
    lower_is_better: bool
    baseline_value: float
    correct_value: float
    zero_value: float
    shuffled_value: float
    stale_value: float
    solver_gain: float
    content_lift: float
    evidence_cost: float
    bits: int
    confidence: float = 0.0
    uncertainty: float = 0.0
    expensive: bool = False
    fallback_roi: float = 0.0
    fallback_roi_floor: float = 0.0


@dataclass(frozen=True)
class DecisionGateResult:
    """Decision-facing utility and rejection reason for one belief arm."""

    domain: str
    mode: str
    use_belief: bool
    fallback_to_baseline: bool
    reason: str
    solver_gain: float
    content_lift: float
    decision_delta_correct_vs_zero: float
    decision_delta_correct_vs_shuffled: float
    decision_delta_correct_vs_stale: float
    decision_delta_correct_vs_best_ablation: float
    expected_gain: float
    expected_gain_per_cost: float
    expected_gain_per_1k_bits: float
    content_lift_per_cost: float
    content_lift_per_1k_bits: float
    evidence_cost: float
    bits: int
    confidence: float
    uncertainty: float
    fallback_roi: float
    fallback_roi_floor: float


def evaluate_decision_delta_gate(item: DecisionGateInput) -> DecisionGateResult:
    """Return the empirical solver-handoff decision for one belief arm."""
    zero_delta = _delta(item.correct_value, item.zero_value, item.lower_is_better)
    shuffled_delta = _delta(item.correct_value, item.shuffled_value, item.lower_is_better)
    stale_delta = _delta(item.correct_value, item.stale_value, item.lower_is_better)
    best_delta = min(zero_delta, shuffled_delta, stale_delta)
    gain_per_cost = _ratio(item.solver_gain, item.evidence_cost)
    gain_per_bits = _ratio(item.solver_gain * 1000.0, float(item.bits))
    lift_per_cost = _ratio(item.content_lift, item.evidence_cost)
    lift_per_bits = _ratio(item.content_lift * 1000.0, float(item.bits))
    reason = _gate_reason(
        solver_gain=item.solver_gain,
        content_lift=item.content_lift,
        zero_delta=zero_delta,
        shuffled_delta=shuffled_delta,
        stale_delta=stale_delta,
        gain_per_cost=gain_per_cost,
        expensive=item.expensive,
        fallback_roi=item.fallback_roi,
        fallback_roi_floor=item.fallback_roi_floor,
    )
    use_belief = reason == "use_belief"
    return DecisionGateResult(
        domain=item.domain,
        mode=item.mode,
        use_belief=use_belief,
        fallback_to_baseline=not use_belief,
        reason=reason,
        solver_gain=item.solver_gain,
        content_lift=item.content_lift,
        decision_delta_correct_vs_zero=zero_delta,
        decision_delta_correct_vs_shuffled=shuffled_delta,
        decision_delta_correct_vs_stale=stale_delta,
        decision_delta_correct_vs_best_ablation=best_delta,
        expected_gain=item.solver_gain if use_belief else 0.0,
        expected_gain_per_cost=gain_per_cost if use_belief else 0.0,
        expected_gain_per_1k_bits=gain_per_bits if use_belief else 0.0,
        content_lift_per_cost=lift_per_cost if use_belief else 0.0,
        content_lift_per_1k_bits=lift_per_bits if use_belief else 0.0,
        evidence_cost=item.evidence_cost,
        bits=item.bits,
        confidence=item.confidence,
        uncertainty=item.uncertainty,
        fallback_roi=item.fallback_roi,
        fallback_roi_floor=item.fallback_roi_floor,
    )


def decision_gate_payload(result: DecisionGateResult) -> dict[str, object]:
    """Return JSON-safe gate fields for artifacts and dashboards."""
    return {
        "domain": result.domain,
        "mode": result.mode,
        "use_belief": result.use_belief,
        "fallback_to_baseline": result.fallback_to_baseline,
        "reason": result.reason,
        "solver_gain": result.solver_gain,
        "content_lift": result.content_lift,
        "decision_delta_correct_vs_zero": result.decision_delta_correct_vs_zero,
        "decision_delta_correct_vs_shuffled": result.decision_delta_correct_vs_shuffled,
        "decision_delta_correct_vs_stale": result.decision_delta_correct_vs_stale,
        "decision_delta_correct_vs_best_ablation": result.decision_delta_correct_vs_best_ablation,
        "expected_gain": result.expected_gain,
        "expected_gain_per_cost": result.expected_gain_per_cost,
        "expected_gain_per_1k_bits": result.expected_gain_per_1k_bits,
        "content_lift_per_cost": result.content_lift_per_cost,
        "content_lift_per_1k_bits": result.content_lift_per_1k_bits,
        "evidence_cost": result.evidence_cost,
        "bits": result.bits,
        "confidence": result.confidence,
        "uncertainty": result.uncertainty,
        "fallback_roi": result.fallback_roi,
        "fallback_roi_floor": result.fallback_roi_floor,
    }


def _gate_reason(
    *,
    solver_gain: float,
    content_lift: float,
    zero_delta: float,
    shuffled_delta: float,
    stale_delta: float,
    gain_per_cost: float,
    expensive: bool,
    fallback_roi: float,
    fallback_roi_floor: float,
) -> str:
    if solver_gain <= 0.0:
        return "solver_gain_not_positive"
    if content_lift <= 0.0:
        return "content_lift_not_positive"
    if zero_delta <= 0.0:
        return "correct_not_better_than_zero"
    if shuffled_delta <= 0.0:
        return "correct_not_better_than_shuffled"
    if stale_delta <= 0.0:
        return "correct_not_better_than_stale"
    if gain_per_cost <= 0.0:
        return "gain_per_cost_not_positive"
    if expensive and fallback_roi < fallback_roi_floor:
        return "expensive_roi_below_floor"
    return "use_belief"


def _delta(correct: float, ablated: float, lower_is_better: bool) -> float:
    if lower_is_better:
        return ablated - correct
    return correct - ablated


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator
