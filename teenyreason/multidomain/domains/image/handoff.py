"""Image solver handoff selection and compressed-belief gates."""

from __future__ import annotations

from ...contracts.decision_gate import DecisionGateInput, decision_gate_payload, evaluate_decision_delta_gate


def image_compression_rows(
    *,
    compression_metrics: object,
    baseline_accuracy: float,
    full_belief_accuracy: float,
    full_content_lift: float,
) -> list[dict[str, float | int | bool | str | dict[str, object]]]:
    """Normalize measured compression arms into dashboard/gate rows."""
    if not isinstance(compression_metrics, list):
        return []
    rows: list[dict[str, float | int | bool | str | dict[str, object]]] = []
    for item in compression_metrics:
        if not isinstance(item, dict):
            continue
        bits = int(item.get("bits", 0))
        source = str(item.get("source", "belief"))
        mode = f"residual_compressed_{bits}" if source == "residual_adapter" else f"compressed_{bits}"
        accuracy = float(item.get("accuracy", 0.0))
        zero = float(item.get("zero_accuracy", accuracy - max(float(item.get("content_lift", 0.0)), 0.0)))
        shuffled = float(item.get("shuffled_accuracy", zero))
        stale = float(item.get("stale_accuracy", zero))
        best_ablation = max(zero, shuffled, stale)
        solver_gain = accuracy - float(baseline_accuracy)
        content_lift = accuracy - best_ablation
        retained = content_lift / max(float(full_content_lift), 1e-9) if full_content_lift > 0.0 else 0.0
        gate = evaluate_decision_delta_gate(
            image_gate_input(
                baseline_accuracy=baseline_accuracy,
                candidate={
                    "mode": mode,
                    "bits": bits,
                    "accuracy": accuracy,
                    "solver_gain": solver_gain,
                    "content_lift": content_lift,
                    "zero_accuracy": zero,
                    "shuffled_accuracy": shuffled,
                    "stale_accuracy": stale,
                },
            )
        )
        rows.append(
            {
                "bits": bits,
                "mode": mode,
                "source": source,
                "measured": True,
                "accuracy": accuracy,
                "solver_gain": solver_gain,
                "content_lift": content_lift,
                "zero_accuracy": zero,
                "shuffled_accuracy": shuffled,
                "stale_accuracy": stale,
                "best_ablation_accuracy": best_ablation,
                "lift_per_1k_bits": content_lift * 1000.0 / max(float(bits), 1.0),
                "gain_per_1k_bits": solver_gain * 1000.0 / max(float(bits), 1.0),
                "retained_utility": max(0.0, retained),
                "retained_accuracy": accuracy / max(float(full_belief_accuracy), 1e-9),
                "accuracy_gap_vs_full": float(full_belief_accuracy) - accuracy,
                "accuracy_gain_vs_baseline": solver_gain,
                "retained_feature_dims": int(item.get("retained_feature_dims", 0)),
                "decision_gate": decision_gate_payload(gate),
                "decision": "candidate" if gate.use_belief and retained >= 0.8 else gate.reason,
            }
        )
    return sorted(rows, key=lambda row: int(row["bits"]))


def best_compressed_row(
    rows: list[dict[str, float | int | bool | str | dict[str, object]]],
) -> dict[str, float | int | bool | str | dict[str, object]]:
    """Return the most bit-efficient measured compressed arm."""
    if not rows:
        return {}
    return max(rows, key=lambda row: float(row.get("lift_per_1k_bits", 0.0)))


def select_image_handoff(
    *,
    baseline_accuracy: float,
    full_belief_accuracy: float,
    full_content_lift: float,
    compression_rows: list[dict[str, float | int | bool | str | dict[str, object]]],
    full_bits: int = 0,
    full_ablations: dict[str, dict[str, float]] | None = None,
) -> dict[str, float | int | str | dict[str, object]]:
    """Pick a solver-facing image context with empirical baseline fallback."""
    full_candidate = {
        "mode": "full_belief",
        "bits": max(int(full_bits), 1),
        "accuracy": float(full_belief_accuracy),
        "solver_gain": float(full_belief_accuracy) - float(baseline_accuracy),
        "content_lift": float(full_content_lift),
        "reason": "full_belief_positive",
    }
    if full_ablations:
        full_candidate.update(
            {
                "zero_accuracy": float(full_ablations.get("zero", {}).get("accuracy", 0.0)),
                "shuffled_accuracy": float(full_ablations.get("shuffled", {}).get("accuracy", 0.0)),
                "stale_accuracy": float(full_ablations.get("stale", {}).get("accuracy", 0.0)),
            }
        )
    candidates: list[dict[str, float | int | str]] = [full_candidate]
    for row in compression_rows:
        mode = str(row.get("mode", f"compressed_{int(row.get('bits', 0))}"))
        candidates.append(
            {
                "mode": mode,
                "bits": int(row.get("bits", 0)),
                "accuracy": float(row.get("accuracy", 0.0)),
                "solver_gain": float(row.get("solver_gain", 0.0)),
                "content_lift": float(row.get("content_lift", 0.0)),
                "zero_accuracy": float(row.get("zero_accuracy", 0.0)),
                "shuffled_accuracy": float(row.get("shuffled_accuracy", 0.0)),
                "stale_accuracy": float(row.get("stale_accuracy", 0.0)),
                "reason": "compressed_belief_positive",
            }
        )
    gated: list[dict[str, float | int | str | dict[str, object]]] = []
    for candidate in candidates:
        gate = evaluate_decision_delta_gate(
            image_gate_input(
                baseline_accuracy=baseline_accuracy,
                candidate=candidate,
            )
        )
        gated.append({**candidate, "decision_gate": decision_gate_payload(gate), "reason": gate.reason})
    viable = [candidate for candidate in gated if bool(candidate["decision_gate"]["use_belief"])]
    if viable:
        return max(
            viable,
            key=lambda candidate: (
                float(candidate["decision_gate"].get("expected_gain_per_1k_bits", 0.0)),
                float(candidate["accuracy"]),
            ),
        )
    return {
        "mode": "baseline_fallback",
        "bits": 0,
        "accuracy": float(baseline_accuracy),
        "solver_gain": 0.0,
        "content_lift": 0.0,
        "reason": best_image_rejection_reason(gated),
        "decision_gate": gated[0]["decision_gate"] if gated else {},
    }


def image_gate_input(
    *,
    baseline_accuracy: float,
    candidate: dict[str, float | int | str],
) -> DecisionGateInput:
    """Convert one image candidate into the shared decision-gate contract."""
    correct = float(candidate.get("accuracy", baseline_accuracy))
    content_lift = float(candidate.get("content_lift", 0.0))
    fallback_ablation = correct - max(content_lift, 0.0)
    return DecisionGateInput(
        domain="image",
        mode=str(candidate.get("mode", "belief")),
        lower_is_better=False,
        baseline_value=float(baseline_accuracy),
        correct_value=correct,
        zero_value=float(candidate.get("zero_accuracy", fallback_ablation)),
        shuffled_value=float(candidate.get("shuffled_accuracy", fallback_ablation)),
        stale_value=float(candidate.get("stale_accuracy", fallback_ablation)),
        solver_gain=float(candidate.get("solver_gain", 0.0)),
        content_lift=content_lift,
        evidence_cost=1.0,
        bits=max(int(candidate.get("bits", 0)), 1),
    )


def best_image_rejection_reason(
    candidates: list[dict[str, float | int | str | dict[str, object]]],
) -> str:
    """Explain why no image arm survived the shared gate."""
    if not candidates:
        return "raw_belief_failed_gain_or_content_lift"
    best = max(
        candidates,
        key=lambda item: (
            float(item.get("solver_gain", 0.0)),
            float(item.get("content_lift", 0.0)),
        ),
    )
    gate = best.get("decision_gate", {})
    if isinstance(gate, dict):
        return str(gate.get("reason", "raw_belief_failed_gain_or_content_lift"))
    return "raw_belief_failed_gain_or_content_lift"
