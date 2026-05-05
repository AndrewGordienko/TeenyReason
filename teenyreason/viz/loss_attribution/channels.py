"""Solver handoff channels and diagnostic decisions."""

from __future__ import annotations

import numpy as np

from .common import _counter_rows, _mean, _median, _row_values


def _expression_channel(rows: list[dict]) -> dict[str, object]:
    return {
        "strict_usage_counts": _counter_rows(rows, "probe_strict_usage_status"),
        "readiness_reason_counts": _counter_rows(rows, "probe_readiness_reason_counts"),
        "fair_stop_blocker_counts": _counter_rows(rows, "probe_fair_stop_blocker_counts"),
        "ready_handoff_fraction_mean": _mean(_row_values(rows, "probe_fair_ready_handoff_fraction")),
        "expression_enabled_fraction_mean": _mean(_row_values(rows, "probe_fair_expression_enabled_fraction")),
        "force_muted_fraction_mean": _mean(_row_values(rows, "probe_fair_expression_force_muted_fraction")),
        "expression_scale_median": _median(_row_values(rows, "probe_expression_scale_median")),
        "expression_delta_median": _median(_row_values(rows, "probe_env_expression_delta")),
        "forced_expression_delta_median": _median(_row_values(rows, "probe_forced_env_expression_delta")),
        "message_input_delta_mean": _mean(_row_values(rows, "probe_message_input_delta_mean")),
        "teacher_action_agreement_median": _median(_row_values(rows, "probe_teacher_action_agreement")),
    }

def _matched_eval_return(row: dict, key: str) -> float | None:
    summary = row.get(key)
    if not isinstance(summary, dict):
        return None
    value = summary.get("mean_return")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result

def _full_system_context_channel(rows: list[dict]) -> dict[str, object]:
    eval_keys = {
        "learned": "full_system_learned_eval_summary",
        "state_only": "full_system_state_only_eval_summary",
        "zero": "full_system_zero_context_eval_summary",
        "shuffled": "full_system_shuffled_context_eval_summary",
        "stale": "full_system_stale_context_eval_summary",
        "no_refresh": "full_system_online_refinement_eval_summary",
        "frozen": "full_system_frozen_context_eval_summary",
        "actor_only": "full_system_actor_only_eval_summary",
    }
    return_medians: dict[str, float] = {}
    return_means: dict[str, float] = {}
    for label, key in eval_keys.items():
        values = [
            value
            for value in (_matched_eval_return(row, key) for row in rows)
            if value is not None
        ]
        return_medians[label] = _median(values)
        return_means[label] = _mean(values)

    learned = return_means.get("learned", 0.0)
    state_only = return_means.get("state_only", 0.0)
    zero = return_means.get("zero", 0.0)
    shuffled = return_means.get("shuffled", 0.0)
    stale = return_means.get("stale", 0.0)
    no_refresh = return_means.get("no_refresh", 0.0)
    frozen = return_means.get("frozen", 0.0)
    context_controls = max(zero, shuffled, stale)
    refresh_controls = max(no_refresh, frozen)
    return {
        "return_mean": return_means,
        "return_median": return_medians,
        "state_channel_lift_mean": float(learned - state_only),
        "context_specific_lift_mean": float(learned - context_controls),
        "zero_context_delta_mean": float(learned - zero),
        "shuffled_context_delta_mean": float(learned - shuffled),
        "stale_context_delta_mean": float(learned - stale),
        "no_refresh_delta_mean": float(learned - no_refresh),
        "frozen_context_delta_mean": float(learned - frozen),
        "refresh_penalty_mean": float(max(0.0, refresh_controls - learned)),
        "context_content_causal": bool(learned > context_controls + 25.0),
        "state_channel_causal": bool(learned > state_only + 25.0),
        "online_refresh_helpful": bool(learned > refresh_controls + 25.0),
    }

def _decision_rows(
    *,
    sample_economics: dict[str, object],
    expression_channel: dict[str, object],
    full_system_context: dict[str, object],
    representation_gate: dict[str, object],
    latent_win_gate: dict[str, object],
    family_economics: dict[str, object],
) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    probe_step_savings = float(
        sample_economics.get(
            "probe_capped_step_savings_vs_baseline",
            sample_economics.get("probe_step_savings_vs_baseline", 0.0),
        )
    )
    if probe_step_savings <= 0.0:
        decisions.append(
            {
                "priority": 1,
                "decision": "Do not tune PPO for claims until probe total env steps beat baseline.",
                "evidence": {
                    "probe_capped_step_savings_vs_baseline": probe_step_savings,
                    "probe_solved_only_step_savings_vs_baseline": sample_economics.get("probe_step_savings_vs_baseline"),
                    "encoder_fraction_of_probe_total": sample_economics.get("encoder_fraction_of_probe_total"),
                    "online_probe_fraction_of_probe_total": sample_economics.get("online_probe_fraction_of_probe_total"),
                },
            }
        )
    if float(expression_channel.get("expression_enabled_fraction_mean", 0.0)) < 0.20:
        decisions.append(
            {
                "priority": 2,
                "decision": "Fix readiness and expression handoff before increasing controller capacity.",
                "evidence": {
                    "expression_enabled_fraction_mean": expression_channel.get("expression_enabled_fraction_mean"),
                    "readiness_reason_counts": expression_channel.get("readiness_reason_counts"),
                    "fair_stop_blocker_counts": expression_channel.get("fair_stop_blocker_counts"),
                },
            }
        )
    if (
        full_system_context
        and bool(full_system_context.get("state_channel_causal", False))
        and not bool(full_system_context.get("context_content_causal", False))
    ):
        decisions.append(
            {
                "priority": 3,
                "decision": "Separate controller/state-channel gains from belief-content gains.",
                "evidence": {
                    "state_channel_lift_mean": full_system_context.get("state_channel_lift_mean"),
                    "context_specific_lift_mean": full_system_context.get("context_specific_lift_mean"),
                    "return_mean": full_system_context.get("return_mean"),
                },
            }
        )
    if representation_gate.get("available") and float(representation_gate.get("latent_pass_fraction", 0.0)) < 1.0:
        decisions.append(
            {
                "priority": 4,
                "decision": "Prioritize latent subset stability and retrieval geometry.",
                "evidence": {
                    "latent_pass_fraction": representation_gate.get("latent_pass_fraction"),
                    "failure_reasons": representation_gate.get("failure_reasons"),
                },
            }
        )
    if int(family_economics.get("families_clearing_second_probe_floor", 0)) <= 0:
        decisions.append(
            {
                "priority": 5,
                "decision": "Keep the second-probe floor and improve family value estimates until at least one active family clears it.",
                "evidence": {
                    "families_clearing_second_probe_floor": family_economics.get("families_clearing_second_probe_floor"),
                    "best_family": family_economics.get("best_family"),
                },
            }
        )
    if not bool(latent_win_gate.get("pass", False)):
        decisions.append(
            {
                "priority": 6,
                "decision": "Treat current benchmark as diagnostic evidence, not a latent-win claim.",
                "evidence": {
                    "failure_reasons": latent_win_gate.get("failure_reasons", []),
                },
            }
        )
    return sorted(decisions, key=lambda row: int(row["priority"]))

