"""CartPole summary shaping for tri-domain suite artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...contracts.decision_gate import DecisionGateInput, decision_gate_payload, evaluate_decision_delta_gate


def _valid_numbers(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    numbers: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            numbers.append(number)
    return numbers


def _mean(values: object) -> float:
    numbers = _valid_numbers(values)
    return sum(numbers) / max(len(numbers), 1)


def _row_median(rows: object, key: str) -> float:
    if not isinstance(rows, list):
        return 0.0
    values = _valid_numbers([row.get(key) for row in rows if isinstance(row, dict)])
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _summary_value(summary: dict[str, Any], key: str, stat: str = "median") -> float:
    value = summary.get(key, {})
    if not isinstance(value, dict):
        return 0.0
    try:
        return float(value.get(stat, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _eval_return(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, {})
    if not isinstance(value, dict):
        return 0.0
    mean_return = value.get("mean_return", {})
    if not isinstance(mean_return, dict):
        return 0.0
    try:
        return float(mean_return.get("mean", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _research_number(result: dict[str, Any], *keys: str) -> float | None:
    current: Any = result.get("research_metrics", {})
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None:
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _cartpole_research_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_solve_steps": _research_number(result, "arms", "baseline", "solve_steps_median"),
        "probe_solve_steps": _research_number(result, "arms", "probe", "solve_steps_median"),
        "probe_no_expression_solve_steps": _research_number(
            result,
            "arms",
            "probe_no_expression",
            "solve_steps_median",
        ),
        "full_system_solve_steps": _research_number(result, "arms", "full_system", "solve_steps_median"),
        "sim_fanout_solve_steps": _research_number(result, "arms", "sim_fanout", "solve_steps_median"),
        "probe_step_savings_vs_baseline": _research_number(
            result,
            "deltas",
            "probe_step_savings_vs_baseline",
        ),
        "probe_step_savings_vs_no_expression": _research_number(
            result,
            "deltas",
            "probe_step_savings_vs_no_expression",
        ),
        "baseline_steps_to_peak": _research_number(
            result,
            "peak",
            "baseline_steps_to_peak_median",
        ),
        "probe_steps_to_peak": _research_number(
            result,
            "peak",
            "probe_steps_to_peak_median",
        ),
        "probe_steps_to_peak_savings_vs_baseline": _research_number(
            result,
            "peak",
            "probe_steps_to_peak_savings_vs_baseline",
        ),
        "baseline_best_return": _research_number(result, "peak", "baseline_best_return_median"),
        "probe_best_return": _research_number(result, "peak", "probe_best_return_median"),
    }


def _strict_status(result: dict[str, Any]) -> str:
    status = result.get("probe_strict_usage_status", "unknown")
    if isinstance(status, str):
        return status
    if isinstance(status, list):
        return ",".join(str(item) for item in status)
    return str(status)


def _slim_rows(rows: object) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    keep = (
        "seed",
        "baseline_episode_solve",
        "probe_episode_solve",
        "probe_no_expression_episode_solve",
        "full_system_episode_solve",
        "sim_fanout_episode_solve",
        "probe_probe_env_steps",
        "probe_encoder_steps",
        "probe_control_env_steps",
        "probe_env_expression_delta",
        "probe_forced_env_expression_delta",
        "probe_strict_usage_status",
        "sysid_trusted",
        "sysid_validation_top1",
        "particle_subset_stability_mean",
        "belief_source",
    )
    slim: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            slim.append({key: row.get(key) for key in keep if key in row})
    return slim


def cartpole_domain_payload(
    result: dict[str, Any] | None,
    detail_path: Path | None,
) -> dict[str, Any]:
    """Build a dashboard lane summary from compact or rich CartPole payloads."""
    if not isinstance(result, dict):
        return {
            "domain": "cartpole",
            "title": "CartPole RL",
            "headline_metric": {
                "name": "probe_solve_episode",
                "value": 0.0,
                "lower_is_better": True,
            },
            "rows": [],
            "metrics": {},
        }
    if isinstance(result.get("summaries"), dict):
        return _cartpole_from_dashboard_payload(result, detail_path)
    return _cartpole_from_compact_result(result, detail_path)


def _cartpole_from_dashboard_payload(
    result: dict[str, Any],
    detail_path: Path | None,
) -> dict[str, Any]:
    summaries = result.get("summaries", {})
    rows = result.get("rows", [])
    system_id = summaries.get("system_id", {})
    learned_return = _eval_return(summaries, "full_system_learned_eval")
    zero_return = _eval_return(summaries, "full_system_zero_context_eval")
    shuffled_return = _eval_return(summaries, "full_system_shuffled_context_eval")
    stale_return = _eval_return(summaries, "full_system_stale_context_eval")
    ablation_return = max(zero_return, shuffled_return, stale_return)
    probe_solve = _summary_value(summaries, "probe_episode", "median")
    noexpr_solve = _summary_value(summaries, "probe_no_expression_episode", "median")
    solver_gain = noexpr_solve - probe_solve
    content_lift = learned_return - ablation_return
    decision_gate = evaluate_decision_delta_gate(
        DecisionGateInput(
            domain="cartpole",
            mode="cartpole_belief",
            lower_is_better=False,
            baseline_value=0.0,
            correct_value=learned_return,
            zero_value=zero_return,
            shuffled_value=shuffled_return,
            stale_value=stale_return,
            solver_gain=solver_gain,
            content_lift=content_lift,
            evidence_cost=_row_median(rows, "probe_probe_env_steps"),
            bits=128,
        )
    )
    return {
        "domain": "cartpole",
        "title": "CartPole RL",
        "dataset": result.get("env_name"),
        "model_family": "Probe-conditioned PPO",
        "headline_metric": {
            "name": "probe_solve_episode",
            "value": probe_solve,
            "lower_is_better": True,
        },
        "baseline_metric": {
            "name": "probe_noexpr_solve_episode",
            "value": noexpr_solve,
        },
        "belief_contribution_margin": solver_gain,
        "ablation_gap": content_lift,
        "readiness": _strict_status(result),
        "trust": float(system_id.get("trusted_fraction", 0.0)) if isinstance(system_id, dict) else 0.0,
        "evidence_cost": _row_median(rows, "probe_probe_env_steps"),
        "rows": _slim_rows(rows),
        "artifact_ref": None if detail_path is None else detail_path.name,
        "metrics": {
            "belief_source": system_id.get("belief_source_counts", {}) if isinstance(system_id, dict) else {},
            "strict_expression_usage": _strict_status(result),
            "honesty_headline": result.get("probe_honesty_headline", ""),
            "probe_expression_delta": _summary_value(summaries, "probe_env_expression_delta", "mean"),
            "probe_forced_expression_delta": _summary_value(
                summaries,
                "probe_forced_env_expression_delta",
                "mean",
            ),
            "sysid_progress": float(system_id.get("progress_median", 0.0)) if isinstance(system_id, dict) else 0.0,
            "sysid_validation_top1": float(system_id.get("validation_top1_median", 0.0)) if isinstance(system_id, dict) else 0.0,
            "sysid_validation_margin": float(system_id.get("validation_margin_median", 0.0)) if isinstance(system_id, dict) else 0.0,
            "particle_entropy": float(system_id.get("particle_entropy_median", 0.0)) if isinstance(system_id, dict) else 0.0,
            "particle_ess_ratio": float(system_id.get("particle_ess_ratio_median", 0.0)) if isinstance(system_id, dict) else 0.0,
            "particle_leaveout_shift": float(system_id.get("particle_leaveout_shift_median", 0.0)) if isinstance(system_id, dict) else 0.0,
            "particle_subset_stability": float(system_id.get("particle_subset_stability_median", 0.0)) if isinstance(system_id, dict) else 0.0,
            "learned_eval_return": learned_return,
            "zero_eval_return": zero_return,
            "shuffled_eval_return": shuffled_return,
            "stale_eval_return": stale_return,
            "decision_gate_use_belief": decision_gate.use_belief,
            "decision_gate_reason": decision_gate.reason,
            "decision_delta_correct_vs_best_ablation": decision_gate.decision_delta_correct_vs_best_ablation,
            "decision_gate": decision_gate_payload(decision_gate),
            "probe_encoder_steps": _row_median(rows, "probe_encoder_steps"),
            "probe_total_probe_steps": _row_median(rows, "probe_probe_env_steps"),
            "probe_control_steps": _row_median(rows, "probe_control_env_steps"),
            **_cartpole_research_metrics(result),
        },
    }


def _cartpole_from_compact_result(
    result: dict[str, Any],
    detail_path: Path | None,
) -> dict[str, Any]:
    probe_solve = _mean(result.get("probe_episode_solves"))
    probe_noexpr = _mean(result.get("probe_no_expression_episode_solves"))
    return {
        "domain": "cartpole",
        "title": "CartPole RL",
        "dataset": result.get("env_name"),
        "model_family": "Probe-conditioned PPO",
        "headline_metric": {
            "name": "probe_solve_episode",
            "value": probe_solve,
            "lower_is_better": True,
        },
        "baseline_metric": {
            "name": "probe_noexpr_solve_episode",
            "value": probe_noexpr,
        },
        "belief_contribution_margin": probe_noexpr - probe_solve,
        "ablation_gap": 0.0,
        "readiness": _strict_status(result),
        "trust": 0.0,
        "evidence_cost": 0.0,
        "rows": [],
        "artifact_ref": None if detail_path is None else detail_path.name,
        "metrics": {
            "belief_source": result.get("belief_source", "unknown"),
            "strict_expression_usage": _strict_status(result),
            "probe_expression_delta": _mean(result.get("probe_env_expression_delta")),
            "probe_forced_expression_delta": _mean(result.get("probe_forced_env_expression_delta")),
        },
    }
