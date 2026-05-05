"""Metric attachment helpers for suite domain payloads."""

from __future__ import annotations


def _attach_cartpole_mechanism(
    domain: dict[str, object],
    result: dict[str, object] | None,
) -> None:
    if not isinstance(result, dict):
        return
    rows = result.get("rows", [])
    row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    artifact = {}
    artifacts = result.get("artifacts", [])
    if isinstance(artifacts, list) and artifacts and isinstance(artifacts[-1], dict):
        artifact = artifacts[-1]
    metrics = domain.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        domain["metrics"] = metrics
    domain.setdefault("dataset", result.get("dataset"))
    domain.setdefault("model_family", result.get("model_family"))
    domain.setdefault("artifact", artifact)
    if not domain.get("readiness"):
        domain["readiness"] = "mechanics-check"
    if not domain.get("trust"):
        domain["trust"] = float(result.get("mechanics_decode_accuracy", 0.0))
    if not domain.get("evidence_cost"):
        domain["evidence_cost"] = float(len(result.get("support_families", [])))
    belief_score = _transition_score(result.get("mean_belief_transition_mse", 0.0))
    zero_score = _transition_score(result.get("mean_zero_transition_mse", 0.0))
    shuffled_score = _transition_score(result.get("mean_shuffled_transition_mse", 0.0))
    stale_score = _transition_score(result.get("mean_stale_transition_mse", 0.0))
    baseline_score = _transition_score(result.get("mean_baseline_transition_mse", 0.0))
    metrics.update(
        {
            "mechanism_hidden_target": "controlled_cartpole_mechanics",
            "mechanism_hidden_rule": row.get("hidden_rule", ""),
            "mechanism_decoded_rule": row.get("decoded_rule", ""),
            "mechanism_decode_accuracy": float(result.get("mechanics_decode_accuracy", 0.0)),
            "mechanism_baseline_accuracy": baseline_score,
            "mechanism_belief_accuracy": belief_score,
            "mechanism_zero_accuracy": zero_score,
            "mechanism_shuffled_accuracy": shuffled_score,
            "mechanism_stale_accuracy": stale_score,
            "mechanism_content_lift": belief_score - max(zero_score, shuffled_score, stale_score),
            "mechanism_subset_agreement": float(result.get("subset_agreement", 0.0)),
            "mechanics_r2": float(result.get("mechanics_r2", 0.0)),
            "mechanics_subset_agreement": float(result.get("subset_agreement", 0.0)),
            "mechanics_uncertainty_error_corr": float(result.get("uncertainty_error_corr", 0.0)),
            "belief_bitrate": int(artifact.get("belief_bitrate", metrics.get("belief_bitrate", 0))),
            "subset_consistency": float(artifact.get("subset_agreement", metrics.get("subset_consistency", 0.0))),
        }
    )


def _attach_family_bridge(
    domain: dict[str, object],
    result: dict[str, object] | None,
    *,
    metric_name: str,
) -> None:
    if not isinstance(result, dict):
        return
    metrics = domain.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        domain["metrics"] = metrics
    if not domain.get("readiness"):
        domain["readiness"] = "bridge-check"
    contract = result.get("adapter_contract", {})
    if not isinstance(contract, dict):
        contract = {}
    causal_world = result.get("causal_world_model", {})
    if not isinstance(causal_world, dict):
        causal_world = {}
    metrics.update(
        {
            "bridge_hidden_target": str(result.get("hidden_target", "")),
            "bridge_dataset": str(result.get("dataset", "")),
            "bridge_model_family": str(result.get("model_family", "")),
            "bridge_metric_name": metric_name,
            "bridge_adapter_runner": str(contract.get("runner", "")),
            "bridge_query_family_count": int(len(contract.get("query_families", []))),
            "bridge_decode_accuracy": float(result.get("decode_accuracy", 0.0)),
            "bridge_subset_agreement": float(result.get("subset_agreement", 0.0)),
            "bridge_baseline_value": float(result.get("baseline_accuracy", result.get("baseline_return", 0.0))),
            "bridge_belief_value": float(result.get("belief_accuracy", result.get("belief_return", 0.0))),
            "bridge_zero_value": float(result.get("zero_accuracy", result.get("zero_return", 0.0))),
            "bridge_shuffled_value": float(result.get("shuffled_accuracy", result.get("shuffled_return", 0.0))),
            "bridge_stale_value": float(result.get("stale_accuracy", result.get("stale_return", 0.0))),
            "bridge_solver_gain": float(result.get("solver_gain", 0.0)),
            "bridge_content_lift": float(result.get("content_lift", 0.0)),
            "bridge_causal_factor_decode": float(causal_world.get("factor_decode_accuracy", 0.0)),
            "bridge_causal_counterfactual_accuracy": float(
                causal_world.get("counterfactual_accuracy", 0.0)
            ),
            "bridge_causal_understanding_score": float(
                causal_world.get("understanding_score", 0.0)
            ),
            "bridge_causal_mean_total_cost": float(causal_world.get("mean_total_cost", 0.0)),
        }
    )


def _attach_latent_handoff(
    domain: dict[str, object],
    result: dict[str, object] | None,
) -> None:
    if not isinstance(result, dict):
        return
    metrics = domain.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        domain["metrics"] = metrics
    metrics.update(
        {
            "handoff_hidden_target": str(result.get("hidden_target", "")),
            "handoff_model_family": str(result.get("model_family", "")),
            "handoff_expensive_decode_accuracy": float(result.get("expensive_decode_accuracy", 0.0)),
            "handoff_cheap_decode_accuracy": float(result.get("cheap_decode_accuracy", 0.0)),
            "handoff_centroid_head_decode_accuracy": float(
                result.get("centroid_head_decode_accuracy", 0.0)
            ),
            "handoff_cheap_confidence": float(result.get("cheap_confidence", 0.0)),
            "handoff_centroid_head_confidence": float(
                result.get("centroid_head_confidence", 0.0)
            ),
            "handoff_baseline_return": float(result.get("baseline_return", 0.0)),
            "handoff_expensive_return": float(result.get("expensive_return", 0.0)),
            "handoff_cheap_return": float(result.get("cheap_return", 0.0)),
            "handoff_cheap_content_lift": float(result.get("cheap_content_lift", 0.0)),
            "handoff_action_change_fraction": float(result.get("action_change_fraction", 0.0)),
            "handoff_mean_abs_action_delta": float(result.get("mean_abs_action_delta", 0.0)),
            "handoff_value_delta_correct_vs_shuffled": float(
                result.get("value_delta_correct_vs_shuffled", 0.0)
            ),
            "handoff_expensive_dedicated_probe_steps": float(
                result.get("expensive_dedicated_probe_steps", 0.0)
            ),
            "handoff_cheap_dedicated_probe_steps": float(
                result.get("cheap_dedicated_probe_steps", 0.0)
            ),
            "handoff_dual_use_probe_fraction": float(result.get("dual_use_probe_fraction", 0.0)),
            "handoff_dedicated_probe_steps_saved": float(
                result.get("dedicated_probe_steps_saved", 0.0)
            ),
            "handoff_cheap_decision_gate_accept_rate": float(
                result.get("cheap_decision_gate_accept_rate", 0.0)
            ),
            "handoff_expensive_decision_gate_accept_rate": float(
                result.get("expensive_decision_gate_accept_rate", 0.0)
            ),
            "handoff_wake_expensive_probe_rate": float(
                result.get("wake_expensive_probe_rate", 0.0)
            ),
            "handoff_fallback_probe_roi": float(result.get("fallback_probe_roi", 0.0)),
            "handoff_fallback_roi_floor": float(result.get("fallback_roi_floor", 0.0)),
            "handoff_expected_expensive_fallback_count": float(
                result.get("expected_expensive_fallback_count", 0.0)
            ),
            "handoff_selected_cheap_context_fraction": float(
                result.get("selected_cheap_context_fraction", 0.0)
            ),
            "handoff_selected_expensive_context_fraction": float(
                result.get("selected_expensive_context_fraction", 0.0)
            ),
            "handoff_selected_baseline_fraction": float(
                result.get("selected_baseline_fraction", 0.0)
            ),
        }
    )


def _attach_real_causal_understanding(
    domain: dict[str, object],
    result: dict[str, object] | None,
) -> None:
    if not isinstance(result, dict):
        return
    metrics = domain.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        domain["metrics"] = metrics
    metrics.update(
        {
            "real_causal_hidden_target": str(result.get("hidden_target", "")),
            "real_causal_factor_decode": float(result.get("factor_decode_accuracy", 0.0)),
            "real_causal_counterfactual_accuracy": float(result.get("counterfactual_accuracy", 0.0)),
            "real_causal_understanding_score": float(result.get("understanding_score", 0.0)),
            "real_causal_intervention_coverage": float(result.get("intervention_coverage", 0.0)),
            "real_causal_mean_total_cost": float(result.get("mean_total_cost", 0.0)),
        }
    )


def _attach_predictive_planner(
    domain: dict[str, object],
    result: dict[str, object] | None,
) -> None:
    if not isinstance(result, dict):
        return
    metrics = domain.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        domain["metrics"] = metrics
    metrics.update(
        {
            "planner_hidden_target": str(result.get("hidden_target", "")),
            "planner_model_family": str(result.get("model_family", "")),
            "planner_decode_accuracy": float(result.get("decode_accuracy", 0.0)),
            "planner_confidence": float(result.get("confidence", 0.0)),
            "planner_no_belief_return": float(result.get("no_belief_return", 0.0)),
            "planner_belief_return": float(result.get("belief_mpc_return", 0.0)),
            "planner_oracle_return": float(result.get("oracle_mpc_return", 0.0)),
            "planner_shuffled_return": float(result.get("shuffled_mpc_return", 0.0)),
            "planner_stale_return": float(result.get("stale_mpc_return", 0.0)),
            "planner_solver_gain": float(result.get("solver_gain", 0.0)),
            "planner_content_lift": float(result.get("content_lift", 0.0)),
            "planner_oracle_gap": float(result.get("oracle_gap", 0.0)),
            "planner_action_match_oracle": float(result.get("belief_action_match_oracle", 0.0)),
            "planner_no_belief_action_match_oracle": float(
                result.get("no_belief_action_match_oracle", 0.0)
            ),
            "planner_one_step_prediction_mse": float(
                result.get("belief_one_step_prediction_mse", 0.0)
            ),
            "planner_k_step_prediction_mse": float(
                result.get("belief_k_step_prediction_mse", 0.0)
            ),
            "planner_no_belief_k_step_prediction_mse": float(
                result.get("no_belief_k_step_prediction_mse", 0.0)
            ),
            "planner_belief_solve_rate": float(result.get("belief_solve_rate", 0.0)),
            "planner_no_belief_solve_rate": float(result.get("no_belief_solve_rate", 0.0)),
            "planner_oracle_solve_rate": float(result.get("oracle_solve_rate", 0.0)),
            "planner_belief_samples_to_peak": float(
                result.get("belief_samples_to_peak_return", 0.0)
            ),
            "planner_no_belief_samples_to_peak": float(
                result.get("no_belief_samples_to_peak_return", 0.0)
            ),
            "planner_belief_samples_to_solve": _optional_float(
                result.get("belief_samples_to_solve")
            ),
            "planner_no_belief_samples_to_solve": _optional_float(
                result.get("no_belief_samples_to_solve")
            ),
            "planner_net_samples_to_solve_savings": _optional_float(
                result.get("net_samples_to_solve_savings")
            ),
            "planner_net_env_sample_savings": float(result.get("net_env_sample_savings", 0.0)),
            "planner_probe_steps": float(result.get("probe_steps", 0.0)),
            "planner_control_steps": float(result.get("control_steps", 0.0)),
            "planner_horizon": float(result.get("horizon", 0.0)),
            "planner_candidate_count": float(result.get("candidate_count", 0.0)),
        }
    )


def _attach_planner_comparison(
    domain: dict[str, object],
    result: dict[str, object] | None,
) -> None:
    if not isinstance(result, dict):
        return
    metrics = domain.setdefault("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
        domain["metrics"] = metrics
    metrics.update(
        {
            "planner_comparison_profile": str(result.get("profile", "")),
            "planner_comparison_decode_accuracy": float(result.get("decode_accuracy", 0.0)),
            "planner_comparison_cheap_decode_accuracy": float(
                result.get("cheap_decode_accuracy", 0.0)
            ),
            "planner_comparison_solver_gain": float(result.get("solver_gain", 0.0)),
            "planner_comparison_content_lift": float(result.get("content_lift", 0.0)),
            "planner_comparison_action_match_oracle": float(
                result.get("belief_action_match_oracle", 0.0)
            ),
            "planner_comparison_no_belief_action_match_oracle": float(
                result.get("no_belief_action_match_oracle", 0.0)
            ),
            "planner_comparison_oracle_gap": float(result.get("oracle_gap", 0.0)),
            "planner_comparison_action_regret_reduction": float(
                result.get("action_regret_reduction", 0.0)
            ),
            "planner_comparison_probe_roi": float(result.get("probe_roi", 0.0)),
            "planner_comparison_cheap_probe_roi": float(result.get("cheap_probe_roi", 0.0)),
            "planner_comparison_fallback_probe_roi": float(
                result.get("fallback_probe_roi", 0.0)
            ),
            "planner_comparison_persistent_affordance_probe_roi": float(
                result.get("persistent_affordance_probe_roi", 0.0)
            ),
            "planner_comparison_fallback_wake_rate": float(
                result.get("fallback_wake_rate", 0.0)
            ),
            "planner_comparison_belief_beats_no_belief_fraction": float(
                result.get("belief_beats_no_belief_fraction", 0.0)
            ),
            "planner_comparison_belief_beats_all_ablation_fraction": float(
                result.get("belief_beats_all_ablation_fraction", 0.0)
            ),
            "planner_comparison_no_belief_samples_to_solve": _optional_float(
                result.get("no_belief_mpc_samples_to_solve")
            ),
            "planner_comparison_crawler_samples_to_solve": _optional_float(
                result.get("crawler_belief_mpc_samples_to_solve")
            ),
            "planner_comparison_oracle_samples_to_solve": _optional_float(
                result.get("oracle_mpc_samples_to_solve")
            ),
            "planner_comparison_fallback_samples_to_solve": _optional_float(
                result.get("cheap_fallback_samples_to_solve")
            ),
            "planner_comparison_persistent_affordance_samples_to_solve": _optional_float(
                result.get("persistent_affordance_samples_to_solve")
            ),
            "planner_comparison_persistent_affordance_amortized_samples_to_solve": (
                _optional_float(result.get("persistent_affordance_amortized_samples_to_solve"))
            ),
            "planner_comparison_vs_no_belief_savings": _optional_float(
                result.get("crawler_vs_no_belief_mpc_sample_savings")
            ),
            "planner_comparison_fallback_vs_no_belief_savings": _optional_float(
                result.get("cheap_fallback_vs_no_belief_mpc_sample_savings")
            ),
            "planner_comparison_persistent_affordance_vs_no_belief_savings": _optional_float(
                result.get("persistent_affordance_vs_no_belief_mpc_sample_savings")
            ),
            "planner_comparison_persistent_affordance_amortized_vs_no_belief_savings": (
                _optional_float(
                    result.get(
                        "persistent_affordance_amortized_vs_no_belief_mpc_sample_savings"
                    )
                )
            ),
            "planner_comparison_persistent_affordance_probe_cost": float(
                result.get("persistent_affordance_probe_cost", 0.0)
            ),
            "planner_comparison_persistent_affordance_amortized_probe_cost": float(
                result.get("persistent_affordance_amortized_probe_cost", 0.0)
            ),
            "planner_comparison_persistent_affordance_reuse_horizon": float(
                result.get("persistent_affordance_reuse_horizon", 0.0)
            ),
            "planner_comparison_persistent_affordance_regret_reduction": float(
                result.get("persistent_affordance_regret_reduction", 0.0)
            ),
            "planner_comparison_persistent_affordance_probe_value": float(
                result.get("persistent_affordance_probe_value", 0.0)
            ),
            "planner_comparison_diagnostic_state": str(result.get("diagnostic_state", "")),
            "planner_comparison_probe_steps": float(result.get("probe_steps", 0.0)),
            "planner_comparison_cheap_probe_steps": float(result.get("cheap_probe_steps", 0.0)),
            "planner_comparison_control_steps": float(result.get("control_steps", 0.0)),
            "planner_comparison_horizon": float(result.get("horizon", 0.0)),
            "planner_comparison_candidate_count": float(result.get("candidate_count", 0.0)),
        }
    )


def _transition_score(error: object) -> float:
    try:
        value = float(error)
    except (TypeError, ValueError):
        value = 0.0
    return float(1.0 / (1.0 + 1000.0 * max(value, 0.0)))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
