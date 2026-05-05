"""Decision-local counterfactual belief diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionLocalBelief:
    """One domain's decision-local value of crawler belief."""

    domain: str
    decision_queries: int
    accepted_fraction: float
    mean_decision_delta: float
    positive_delta_fraction: float
    counterfactual_score: float
    action_sensitivity: float
    useful_bits: int
    utility_per_1k_bits: float
    utility_per_sample: float
    value_of_information: float
    decision_locality_score: float
    state: str
    blocker: str
    next_action: str


def attach_decision_local_blocks(domains: dict[str, object]) -> None:
    """Attach decision-local belief diagnostics to every domain."""
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            domain["decision_local_belief"] = build_decision_local_block(domain_name, domain)


def build_decision_local_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Build the decision-local counterfactual value block."""
    contract = _contract_from_domain(domain_name, domain)
    return {
        "schema_version": 1,
        "contract": {
            "domain": contract.domain,
            "decision_queries": contract.decision_queries,
            "accepted_fraction": contract.accepted_fraction,
            "mean_decision_delta": contract.mean_decision_delta,
            "positive_delta_fraction": contract.positive_delta_fraction,
            "counterfactual_score": contract.counterfactual_score,
            "action_sensitivity": contract.action_sensitivity,
            "useful_bits": contract.useful_bits,
            "utility_per_1k_bits": contract.utility_per_1k_bits,
            "utility_per_sample": contract.utility_per_sample,
            "value_of_information": contract.value_of_information,
            "decision_locality_score": contract.decision_locality_score,
            "state": contract.state,
            "blocker": contract.blocker,
            "next_action": contract.next_action,
        },
        "gates": {
            "decision_delta_positive": contract.mean_decision_delta > 0.0,
            "counterfactual_ready": contract.counterfactual_score >= 0.7,
            "compact_enough": contract.useful_bits <= _target_bits(domain_name),
            "accepted_by_solver": contract.accepted_fraction > 0.0,
            "economics_positive": contract.value_of_information > 0.0,
            "decision_local": contract.state == "decision_local_belief_ready",
        },
    }


def decision_local_row(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Return one compact dashboard/API row."""
    block = domain.get("decision_local_belief", {})
    if not isinstance(block, dict):
        block = build_decision_local_block(domain_name, domain)
    contract = _dict(block.get("contract"))
    return {
        "domain": domain_name,
        "state": contract.get("state", ""),
        "score": contract.get("decision_locality_score", 0.0),
        "queries": contract.get("decision_queries", 0),
        "accepted_fraction": contract.get("accepted_fraction", 0.0),
        "mean_decision_delta": contract.get("mean_decision_delta", 0.0),
        "positive_delta_fraction": contract.get("positive_delta_fraction", 0.0),
        "counterfactual_score": contract.get("counterfactual_score", 0.0),
        "action_sensitivity": contract.get("action_sensitivity", 0.0),
        "useful_bits": contract.get("useful_bits", 0),
        "utility_per_1k_bits": contract.get("utility_per_1k_bits", 0.0),
        "utility_per_sample": contract.get("utility_per_sample", 0.0),
        "value_of_information": contract.get("value_of_information", 0.0),
        "blocker": contract.get("blocker", ""),
        "next_action": contract.get("next_action", ""),
    }


def _contract_from_domain(domain_name: str, domain: dict[str, Any]) -> DecisionLocalBelief:
    rows = [row for row in domain.get("rows", []) if isinstance(row, dict)]
    decision_queries = _decision_queries(domain_name, rows)
    accepted_fraction = _accepted_fraction(domain_name, domain, rows)
    mean_delta = _mean_decision_delta(domain_name, domain, rows)
    positive_delta_fraction = _positive_delta_fraction(domain_name, domain, rows)
    counterfactual_score = _counterfactual_score(domain)
    action_sensitivity = _action_sensitivity(domain_name, domain, rows)
    useful_bits = _useful_bits(domain_name, domain, rows)
    utility = _utility(domain_name, domain, rows)
    cost = _sample_cost(domain)
    utility_per_1k_bits = utility * 1000.0 / max(float(useful_bits), 1.0)
    utility_per_sample = utility / max(cost, 1.0)
    value_of_information = _value_of_information(domain_name, domain, utility, cost)
    score = _decision_locality_score(
        domain_name=domain_name,
        mean_delta=mean_delta,
        accepted_fraction=accepted_fraction,
        counterfactual_score=counterfactual_score,
        action_sensitivity=action_sensitivity,
        utility_per_1k_bits=utility_per_1k_bits,
        value_of_information=value_of_information,
    )
    blocker = _blocker(
        domain_name=domain_name,
        mean_delta=mean_delta,
        positive_delta_fraction=positive_delta_fraction,
        counterfactual_score=counterfactual_score,
        accepted_fraction=accepted_fraction,
        useful_bits=useful_bits,
        value_of_information=value_of_information,
    )
    state = _state_for_blocker(blocker)
    return DecisionLocalBelief(
        domain=domain_name,
        decision_queries=decision_queries,
        accepted_fraction=accepted_fraction,
        mean_decision_delta=mean_delta,
        positive_delta_fraction=positive_delta_fraction,
        counterfactual_score=counterfactual_score,
        action_sensitivity=action_sensitivity,
        useful_bits=useful_bits,
        utility_per_1k_bits=utility_per_1k_bits,
        utility_per_sample=utility_per_sample,
        value_of_information=value_of_information,
        decision_locality_score=score,
        state=state,
        blocker=blocker,
        next_action=_next_action(domain_name, blocker),
    )


def _decision_queries(domain_name: str, rows: list[dict[str, Any]]) -> int:
    if rows:
        return len(rows)
    if domain_name == "cartpole":
        return 1
    return 0


def _accepted_fraction(domain_name: str, domain: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        cheap = _float(metrics.get("handoff_selected_cheap_context_fraction", 0.0))
        expensive = _float(metrics.get("handoff_selected_expensive_context_fraction", 0.0))
        return _clip01(cheap + expensive)
    if domain_name == "board":
        return 1.0 if rows or _float(domain.get("belief_contribution_margin", 0.0)) > 0.0 else 0.0
    if not rows:
        return 0.0
    accepted = [
        row
        for row in rows
        if bool(row.get("decision_gate_use_belief", False))
        or not bool(row.get("handoff_gate_used_baseline", False))
    ]
    return float(len(accepted)) / float(len(rows))


def _mean_decision_delta(domain_name: str, domain: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        return _float(metrics.get("handoff_value_delta_correct_vs_shuffled", 0.0))
    if domain_name == "board":
        causal = _dict(domain.get("causal_ablation"))
        return _float(causal.get("content_lift", domain.get("ablation_gap", 0.0)))
    values = [
        _float(row.get("decision_delta_correct_vs_best_ablation", row.get("content_lift", 0.0)))
        for row in rows
    ]
    return _mean(values)


def _positive_delta_fraction(domain_name: str, domain: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    if domain_name in {"cartpole", "board"}:
        return 1.0 if _mean_decision_delta(domain_name, domain, rows) > 0.0 else 0.0
    if not rows:
        return 0.0
    positives = [
        1.0
        for row in rows
        if _float(row.get("decision_delta_correct_vs_best_ablation", row.get("content_lift", 0.0))) > 0.0
    ]
    return float(len(positives)) / float(len(rows))


def _counterfactual_score(domain: dict[str, Any]) -> float:
    world = _dict(domain.get("world_understanding"))
    metrics = _dict(domain.get("metrics"))
    return _clip01(
        max(
            _float(world.get("counterfactual", 0.0)),
            _float(metrics.get("real_causal_counterfactual_accuracy", 0.0)),
        )
    )


def _action_sensitivity(domain_name: str, domain: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        return _clip01(
            max(
                _float(metrics.get("handoff_action_change_fraction", 0.0)),
                _float(metrics.get("handoff_value_delta_correct_vs_shuffled", 0.0)) / 10.0,
            )
        )
    if domain_name == "board":
        return _clip01(abs(_mean_decision_delta(domain_name, domain, rows)) / 0.25)
    target = 0.02 if domain_name in {"language", "image"} else 0.1
    return _clip01(abs(_mean_decision_delta(domain_name, domain, rows)) / target)


def _useful_bits(domain_name: str, domain: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    accepted_bits = [
        int(_float(row.get("belief_bitrate", 0.0)))
        for row in rows
        if bool(row.get("decision_gate_use_belief", False))
        or not bool(row.get("handoff_gate_used_baseline", False))
    ]
    accepted_bits = [bits for bits in accepted_bits if bits > 0]
    if accepted_bits:
        return int(round(_mean([float(bits) for bits in accepted_bits])))
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        if _float(metrics.get("handoff_selected_cheap_context_fraction", 0.0)) > 0.0:
            return 128
    interface = _dict(domain.get("interface"))
    belief = _dict(interface.get("belief"))
    metrics = _dict(domain.get("metrics"))
    return int(_float(belief.get("bitrate", metrics.get("belief_bitrate", 0.0))))


def _utility(domain_name: str, domain: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    accepted_values: list[float] = []
    for row in rows:
        accepted = bool(row.get("decision_gate_use_belief", False)) or not bool(
            row.get("handoff_gate_used_baseline", False)
        )
        if not accepted:
            continue
        if domain_name == "language":
            accepted_values.append(_float(row.get("bpc_gain", 0.0)))
        elif domain_name == "image":
            accepted_values.append(_float(row.get("accuracy_gain", 0.0)))
        elif domain_name == "board":
            accepted_values.append(_float(row.get("belief_move_accuracy", 0.0)) - _float(row.get("baseline_move_accuracy", 0.0)))
    if accepted_values:
        return _mean(accepted_values)
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        return _float(metrics.get("handoff_cheap_content_lift", 0.0)) * _float(
            metrics.get("handoff_selected_cheap_context_fraction", 0.0)
        )
    causal = _dict(domain.get("causal_ablation"))
    return _float(causal.get("solver_gain", domain.get("belief_contribution_margin", 0.0)))


def _sample_cost(domain: dict[str, Any]) -> float:
    handoff = _dict(domain.get("belief_handoff"))
    contract = _dict(handoff.get("contract"))
    return max(_float(contract.get("evidence_cost", domain.get("evidence_cost", 0.0))), 1.0)


def _value_of_information(domain_name: str, domain: dict[str, Any], utility: float, cost: float) -> float:
    if domain_name == "cartpole":
        metrics = _dict(domain.get("metrics"))
        saved = _float(metrics.get("handoff_dedicated_probe_steps_saved", 0.0))
        fallback_count = _float(metrics.get("handoff_expected_expensive_fallback_count", 0.0))
        expensive_steps = _float(metrics.get("handoff_expensive_dedicated_probe_steps", 0.0))
        return utility + max(saved - fallback_count * expensive_steps, 0.0) / max(expensive_steps, 1.0)
    return utility / max(cost, 1.0)


def _decision_locality_score(
    *,
    domain_name: str,
    mean_delta: float,
    accepted_fraction: float,
    counterfactual_score: float,
    action_sensitivity: float,
    utility_per_1k_bits: float,
    value_of_information: float,
) -> float:
    delta_target = _delta_target(domain_name)
    bit_target = _bit_utility_target(domain_name)
    voi_target = _voi_target(domain_name)
    return _weighted_mean(
        [
            (_clip01(mean_delta / delta_target), 0.24),
            (accepted_fraction, 0.16),
            (counterfactual_score, 0.22),
            (action_sensitivity, 0.16),
            (_clip01(utility_per_1k_bits / bit_target), 0.12),
            (_clip01(value_of_information / voi_target), 0.10),
        ]
    )


def _blocker(
    *,
    domain_name: str,
    mean_delta: float,
    positive_delta_fraction: float,
    counterfactual_score: float,
    accepted_fraction: float,
    useful_bits: int,
    value_of_information: float,
) -> str:
    if counterfactual_score < 0.7:
        return "counterfactual_model_too_weak"
    if mean_delta <= 0.0 or positive_delta_fraction <= 0.0:
        return "belief_not_decision_causal"
    if accepted_fraction <= 0.0:
        return "solver_rejects_belief"
    if useful_bits > _target_bits(domain_name):
        return "belief_too_wide_for_decision"
    if value_of_information <= 0.0:
        return "negative_value_of_information"
    if accepted_fraction < 0.75:
        return "partial_solver_acceptance"
    return "ready"


def _state_for_blocker(blocker: str) -> str:
    if blocker == "ready":
        return "decision_local_belief_ready"
    if blocker == "partial_solver_acceptance":
        return "partial_decision_local_belief"
    return "blocked"


def _next_action(domain_name: str, blocker: str) -> str:
    if blocker == "ready":
        if domain_name == "board":
            return "scale_harder_rule_space"
        return "scale_with_same_gate"
    if blocker == "counterfactual_model_too_weak":
        return "train_action_conditioned_counterfactual_head"
    if blocker == "belief_not_decision_causal":
        return "train_decision_delta_objective_against_ablations"
    if blocker == "solver_rejects_belief":
        if domain_name == "language":
            return "freeze_lm_and_train_residual_adapter_gate"
        if domain_name == "image":
            return "train_compressed_residual_visual_gate"
        if domain_name == "cartpole":
            return "route_cheap_belief_into_controller_policy"
        return "repair_solver_handoff"
    if blocker == "belief_too_wide_for_decision":
        return "distill_decision_factors_to_smaller_bitrate"
    if blocker == "partial_solver_acceptance":
        if domain_name == "language":
            return "stabilize_residual_adapter_across_train_budgets"
        if domain_name == "image":
            return "extend_compressed_gate_to_mid_high_label_budgets"
        if domain_name == "cartpole":
            return "route_cheap_belief_into_policy_training_loop"
        return "increase_solver_acceptance_coverage"
    return "raise_probe_roi_or_sleep_expensive_probe"


def _target_bits(domain_name: str) -> int:
    if domain_name == "image":
        return 512
    if domain_name == "language":
        return 512
    return 128


def _delta_target(domain_name: str) -> float:
    if domain_name == "cartpole":
        return 1.0
    if domain_name in {"language", "image"}:
        return 0.01
    if domain_name == "board":
        return 0.1
    return 0.05


def _bit_utility_target(domain_name: str) -> float:
    if domain_name == "cartpole":
        return 1.0
    if domain_name in {"language", "image"}:
        return 0.05
    return 1.0


def _voi_target(domain_name: str) -> float:
    if domain_name == "cartpole":
        return 0.5
    if domain_name in {"language", "image"}:
        return 0.0001
    return 0.1


def _dict(value: object) -> dict[str, Any]:
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


def _weighted_mean(items: list[tuple[float, float]]) -> float:
    total = sum(weight for _value, weight in items)
    if total <= 0.0:
        return 0.0
    return _clip01(sum(_clip01(value) * weight for value, weight in items) / total)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
