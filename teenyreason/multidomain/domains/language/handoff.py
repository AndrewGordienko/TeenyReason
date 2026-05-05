"""Language solver handoff arm selection."""

from __future__ import annotations

from ...contracts.decision_gate import DecisionGateInput, decision_gate_payload, evaluate_decision_delta_gate


def best_language_handoff(
    baseline: dict[str, float],
    handoff_results: list[dict[str, float | dict[str, float]]],
) -> dict[str, float | dict[str, float]]:
    """Return the raw lowest-BPC language arm."""
    if not handoff_results:
        raise ValueError("At least one language handoff mode is required.")
    return min(handoff_results, key=lambda item: float(item.get("bpc", baseline["bpc"])))


def select_language_handoff(
    *,
    baseline: dict[str, float],
    raw_best: dict[str, float | dict[str, float]],
    handoff_rows: list[dict[str, float | int | str | bool | dict[str, object]]],
    handoff_results: list[dict[str, float | dict[str, float]]] | None = None,
) -> dict[str, float | dict[str, float] | str | bool]:
    """Select a solver-facing language arm without hiding raw belief metrics."""
    candidates = list(handoff_results) if handoff_results else [raw_best]
    gated: list[dict[str, float | dict[str, float] | str | bool]] = []
    for result in candidates:
        mode = str(result.get("handoff_mode", ""))
        row = handoff_row_by_mode(handoff_rows, mode)
        gain = float(row.get("bpc_gain", float(baseline["bpc"]) - float(result["bpc"])))
        lift = float(row.get("content_lift", 0.0))
        gate = evaluate_decision_delta_gate(
            language_gate_input(
                baseline=baseline,
                raw_best=result,
                mode=mode,
                solver_gain=gain,
                content_lift=lift,
            )
        )
        gated.append(
            {
                **result,
                "used_baseline_fallback": False,
                "gate_reason": gate.reason,
                "decision_gate": decision_gate_payload(gate),
            }
        )
    viable = [
        candidate
        for candidate in gated
        if bool(_dict(candidate.get("decision_gate")).get("use_belief", False))
    ]
    if viable:
        return min(viable, key=lambda item: float(item.get("bpc", baseline["bpc"])))
    fallback_metric = {
        "bpc": float(baseline["bpc"]),
        "cloze_accuracy": float(baseline.get("cloze_accuracy", 0.0)),
        "continuation_accuracy": float(baseline.get("continuation_accuracy", 0.0)),
    }
    best_rejected = _best_rejected(gated)
    return {
        **fallback_metric,
        "handoff_mode": "baseline_fallback",
        "ablation_metrics": {
            "zero": dict(fallback_metric),
            "shuffled": dict(fallback_metric),
            "stale": dict(fallback_metric),
        },
        "used_baseline_fallback": True,
        "gate_reason": str(best_rejected.get("gate_reason", "solver_gain_not_positive")),
        "decision_gate": _dict(best_rejected.get("decision_gate")),
    }


def language_handoff_rows(
    *,
    baseline: dict[str, float],
    handoff_results: list[dict[str, float | dict[str, float]]],
    belief_bitrate: int,
    support_windows: int,
) -> list[dict[str, float | int | str | bool | dict[str, object]]]:
    """Build per-arm language handoff rows with shared gate diagnostics."""
    rows: list[dict[str, float | int | str | bool | dict[str, object]]] = []
    for result in handoff_results:
        ablations = result.get("ablation_metrics", {})
        if not isinstance(ablations, dict):
            continue
        economics = language_row_economics(
            baseline_bpc=float(baseline["bpc"]),
            belief_bpc=float(result["bpc"]),
            ablations=ablations,
            belief_bitrate=belief_bitrate,
            support_windows=support_windows,
        )
        mode = str(result.get("handoff_mode", ""))
        gate = evaluate_decision_delta_gate(
            language_gate_input(
                baseline=baseline,
                raw_best=result,
                mode=mode,
                solver_gain=float(economics["bpc_gain"]),
                content_lift=float(economics["content_lift"]),
            )
        )
        rows.append(
            {
                "mode": mode,
                "bpc": float(result["bpc"]),
                "bpc_gain": float(economics["bpc_gain"]),
                "content_lift": float(economics["content_lift"]),
                "bpc_gain_per_1k_bits": float(economics["bpc_gain_per_1k_bits"]),
                "content_lift_per_1k_bits": float(economics["content_lift_per_1k_bits"]),
                "decision_gate_use_belief": bool(gate.use_belief),
                "decision_gate_reason": gate.reason,
                "decision_delta_correct_vs_best_ablation": gate.decision_delta_correct_vs_best_ablation,
                "decision_gate": decision_gate_payload(gate),
                "cloze_accuracy": float(result["cloze_accuracy"]),
                "continuation_accuracy": float(result["continuation_accuracy"]),
                "budget_gate_uses_belief": bool(economics["budget_gate_uses_belief"]),
            }
        )
    return rows


def language_row_economics(
    *,
    baseline_bpc: float,
    belief_bpc: float,
    ablations: dict[str, dict[str, float]],
    belief_bitrate: int,
    support_windows: int,
) -> dict[str, float | int | bool]:
    """Compute per-budget language handoff economics."""
    zero_bpc = float(ablations["zero"]["bpc"])
    shuffled_bpc = float(ablations["shuffled"]["bpc"])
    stale_bpc = float(ablations["stale"]["bpc"])
    best_ablation_bpc = min(zero_bpc, shuffled_bpc, stale_bpc)
    bpc_gain = float(baseline_bpc) - float(belief_bpc)
    content_lift = best_ablation_bpc - float(belief_bpc)
    windows = max(float(support_windows), 1.0)
    bits = max(float(belief_bitrate), 1.0)
    return {
        "best_ablation_bpc": best_ablation_bpc,
        "bpc_gain": bpc_gain,
        "content_lift": content_lift,
        "budget_gate_uses_belief": bool(bpc_gain > 0.0 and content_lift >= 0.0),
        "bpc_gain_per_support_window": bpc_gain / windows,
        "content_lift_per_support_window": content_lift / windows,
        "bpc_gain_per_1k_bits": bpc_gain * 1000.0 / bits,
        "content_lift_per_1k_bits": content_lift * 1000.0 / bits,
    }


def language_gate_input(
    *,
    baseline: dict[str, float],
    raw_best: dict[str, float | dict[str, float]],
    mode: str,
    solver_gain: float,
    content_lift: float,
) -> DecisionGateInput:
    """Convert one language arm into the shared decision-gate contract."""
    correct = float(raw_best.get("bpc", baseline["bpc"]))
    ablations = raw_best.get("ablation_metrics", {})
    if not isinstance(ablations, dict):
        ablations = {}
    fallback_ablation = correct + max(float(content_lift), 0.0)
    return DecisionGateInput(
        domain="language",
        mode=mode or "belief",
        lower_is_better=True,
        baseline_value=float(baseline["bpc"]),
        correct_value=correct,
        zero_value=ablation_bpc(ablations, "zero", fallback_ablation),
        shuffled_value=ablation_bpc(ablations, "shuffled", fallback_ablation),
        stale_value=ablation_bpc(ablations, "stale", fallback_ablation),
        solver_gain=float(solver_gain),
        content_lift=float(content_lift),
        evidence_cost=1.0,
        bits=1,
    )


def ablation_bpc(ablations: dict[str, object], name: str, fallback: float) -> float:
    """Read one BPC ablation value with a conservative fallback."""
    value = ablations.get(name, {})
    if isinstance(value, dict):
        return float(value.get("bpc", fallback))
    return fallback


def handoff_row_by_mode(
    rows: list[dict[str, float | int | str | bool | dict[str, object]]],
    mode: str,
) -> dict[str, float | int | str | bool | dict[str, object]]:
    """Find one handoff summary row by mode."""
    for row in rows:
        if row.get("mode") == mode:
            return row
    return {}


def handoff_metric(
    rows: list[dict[str, float | int | str | bool | dict[str, object]]],
    mode: str,
    key: str,
) -> float:
    """Read one numeric handoff metric by mode."""
    for row in rows:
        if row.get("mode") == mode:
            return float(row.get(key, 0.0))
    return 0.0


def _best_rejected(
    candidates: list[dict[str, float | dict[str, float] | str | bool]],
) -> dict[str, float | dict[str, float] | str | bool]:
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda item: (
            float(_dict(item.get("decision_gate")).get("solver_gain", 0.0)),
            float(_dict(item.get("decision_gate")).get("content_lift", 0.0)),
        ),
    )


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
