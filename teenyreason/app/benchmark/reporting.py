"""Benchmark result aggregation, reporting, and artifact persistence."""

import json

import numpy as np

from .artifacts import save_benchmark_results
from .health import (
    format_status,
    geometry_status,
    signal_status,
    system_id_geometry_status,
    utility_status,
)
from .support import (
    evaluate_latent_win_gate,
    print_solve_summary,
    probe_strict_usage_status,
)
from ..config import ExperimentConfig
from ...viz.live import LiveTrainingTraceWriter


def _diag_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _diag_median(rows: list[dict], key: str) -> float:
    values = _diag_values(rows, key)
    return float(np.median(np.asarray(values, dtype=np.float32))) if values else 0.0


def finalize_benchmark_run(
    *,
    config: ExperimentConfig,
    benchmark_profile: str,
    seeds: list[int],
    results: list[dict],
    live_trace: LiveTrainingTraceWriter,
) -> dict:
    """Aggregate a finished benchmark, print the summary, and persist artifacts."""
    baseline_episode_solves = [item["baseline_solve_episode"] for item in results]
    probe_episode_solves = [item["probe_solve_episode"] for item in results]
    probe_shadow_episode_solves = [item["probe_shadow_solve_episode"] for item in results]
    probe_no_expression_episode_solves = [item["probe_no_expression_solve_episode"] for item in results]
    full_system_episode_solves = [item["full_system_solve_episode"] for item in results]
    full_system_state_only_episode_solves = [
        item["full_system_state_only_solve_episode"] for item in results
    ]
    full_system_oracle_episode_solves = [
        item["full_system_oracle_solve_episode"] for item in results
    ]
    sim_fanout_episode_solves = [item["sim_fanout_solve_episode"] for item in results]
    baseline_step_solves = [item["baseline_solve_env_steps"] for item in results]
    probe_step_solves = [item["probe_solve_env_steps"] for item in results]
    probe_shadow_step_solves = [item["probe_shadow_solve_env_steps"] for item in results]
    probe_no_expression_step_solves = [item["probe_no_expression_solve_env_steps"] for item in results]
    full_system_step_solves = [item["full_system_solve_env_steps"] for item in results]
    full_system_state_only_step_solves = [
        item["full_system_state_only_solve_env_steps"] for item in results
    ]
    full_system_oracle_step_solves = [
        item["full_system_oracle_solve_env_steps"] for item in results
    ]
    sim_fanout_step_solves = [item["sim_fanout_solve_env_steps"] for item in results]
    baseline_best_returns = [float(item.get("baseline_best_return", 0.0)) for item in results]
    probe_best_returns = [float(item.get("probe_best_return", 0.0)) for item in results]
    baseline_peak_env_steps = [
        item.get("baseline_peak_env_steps")
        for item in results
    ]
    probe_peak_env_steps_with_encoder = [
        item.get("probe_peak_env_steps_with_encoder")
        for item in results
    ]
    baseline_total_env_steps = [item["baseline_total_env_steps"] for item in results]
    probe_total_env_steps = [item["probe_total_env_steps"] for item in results]
    probe_shadow_total_env_steps = [item["probe_shadow_total_env_steps"] for item in results]
    probe_no_expression_total_env_steps = [item["probe_no_expression_total_env_steps"] for item in results]
    full_system_total_env_steps = [item["full_system_total_env_steps"] for item in results]
    full_system_state_only_total_env_steps = [
        item["full_system_state_only_total_env_steps"] for item in results
    ]
    full_system_oracle_total_env_steps = [
        item["full_system_oracle_total_env_steps"] for item in results
    ]
    sim_fanout_total_env_steps = [item["sim_fanout_total_env_steps"] for item in results]
    baseline_control_env_steps = [item["baseline_control_env_steps"] for item in results]
    probe_probe_env_steps = [item["probe_probe_env_steps"] for item in results]
    probe_control_env_steps = [item["probe_control_env_steps"] for item in results]
    probe_shadow_probe_env_steps = [item["probe_shadow_probe_env_steps"] for item in results]
    probe_shadow_control_env_steps = [item["probe_shadow_control_env_steps"] for item in results]
    probe_no_expression_probe_env_steps = [item["probe_no_expression_probe_env_steps"] for item in results]
    probe_no_expression_control_env_steps = [item["probe_no_expression_control_env_steps"] for item in results]
    full_system_probe_env_steps = [item["full_system_probe_env_steps"] for item in results]
    full_system_control_env_steps = [item["full_system_control_env_steps"] for item in results]
    full_system_oracle_probe_env_steps = [
        item["full_system_oracle_probe_env_steps"] for item in results
    ]
    full_system_oracle_control_env_steps = [
        item["full_system_oracle_control_env_steps"] for item in results
    ]
    sim_fanout_probe_env_steps = [item["sim_fanout_probe_env_steps"] for item in results]
    sim_fanout_control_env_steps = [item["sim_fanout_control_env_steps"] for item in results]
    probe_post_expression_env_steps = [item["probe_post_expression_env_steps"] for item in results]
    probe_post_expression_episodes = [item["probe_post_expression_episodes"] for item in results]
    probe_shadow_post_expression_env_steps = [
        item["probe_shadow_post_expression_env_steps"] for item in results
    ]
    probe_shadow_post_expression_episodes = [
        item["probe_shadow_post_expression_episodes"] for item in results
    ]
    probe_no_expression_post_expression_env_steps = [
        item["probe_no_expression_post_expression_env_steps"]
        for item in results
    ]
    probe_no_expression_post_expression_episodes = [
        item["probe_no_expression_post_expression_episodes"]
        for item in results
    ]
    full_system_post_context_env_steps = [
        item["full_system_post_context_env_steps"] for item in results
    ]
    full_system_post_context_episodes = [
        item["full_system_post_context_episodes"] for item in results
    ]
    full_system_oracle_post_context_env_steps = [
        item["full_system_oracle_post_context_env_steps"] for item in results
    ]
    full_system_oracle_post_context_episodes = [
        item["full_system_oracle_post_context_episodes"] for item in results
    ]
    sim_fanout_post_context_env_steps = [
        item["sim_fanout_post_context_env_steps"] for item in results
    ]
    sim_fanout_post_context_episodes = [
        item["sim_fanout_post_context_episodes"] for item in results
    ]
    baseline_completed_episodes = [item["baseline_completed_episodes"] for item in results]
    probe_completed_episodes = [item["probe_completed_episodes"] for item in results]
    probe_shadow_completed_episodes = [
        item["probe_shadow_completed_episodes"] for item in results
    ]
    probe_no_expression_completed_episodes = [
        item["probe_no_expression_completed_episodes"]
        for item in results
    ]
    full_system_completed_episodes = [
        item["full_system_completed_episodes"] for item in results
    ]
    full_system_state_only_completed_episodes = [
        item["full_system_state_only_completed_episodes"] for item in results
    ]
    full_system_oracle_completed_episodes = [
        item["full_system_oracle_completed_episodes"] for item in results
    ]
    sim_fanout_completed_episodes = [
        item["sim_fanout_completed_episodes"] for item in results
    ]
    full_system_controller_style = [
        item["full_system_controller_style"] for item in results
    ]
    full_system_oracle_controller_style = [
        item["full_system_oracle_controller_style"] for item in results
    ]
    sim_fanout_controller_style = [
        item["sim_fanout_controller_style"] for item in results
    ]
    probe_encoder_steps = [item["probe_encoder_steps"] for item in results]
    probe_windows_total = [item["probe_windows_total"] for item in results]
    probe_expression_scale_median = [
        float(item["probe_expression_scale_median"] or 0.0)
        for item in results
    ]
    probe_expression_scale_active_fraction = [
        float(item["probe_expression_scale_active_fraction"] or 0.0)
        for item in results
    ]
    probe_fair_ready_handoff_fraction = [
        float(item["probe_fair_ready_handoff_fraction"] or 0.0)
        for item in results
    ]
    probe_fair_expression_enabled_fraction = [
        float(item["probe_fair_expression_enabled_fraction"] or 0.0)
        for item in results
    ]
    probe_fair_expression_force_muted_fraction = [
        float(item["probe_fair_expression_force_muted_fraction"] or 0.0)
        for item in results
    ]
    probe_fair_ready_confidence_median = [
        float(item["probe_fair_ready_confidence_median"] or 0.0)
        for item in results
    ]
    probe_fair_muted_confidence_median = [
        float(item["probe_fair_muted_confidence_median"] or 0.0)
        for item in results
    ]
    probe_expression_ready_but_muted_fraction = [
        float(item["probe_expression_ready_but_muted_fraction"] or 0.0)
        for item in results
    ]
    probe_shadow_expression_enabled_fraction = [
        float(item["probe_shadow_expression_enabled_fraction"] or 0.0)
        for item in results
    ]
    probe_shadow_expression_scale_median = [
        float(item["probe_shadow_expression_scale_median"] or 0.0)
        for item in results
    ]
    probe_shadow_confidence_median = [
        float(item["probe_shadow_confidence_median"] or 0.0)
        for item in results
    ]
    probe_shadow_strict_miss_fraction = [
        float(item["probe_shadow_strict_miss_fraction"] or 0.0)
        for item in results
    ]
    probe_second_probe_raw_future_gain_mean = [
        float(item["probe_second_probe_raw_future_gain_mean"] or 0.0)
        for item in results
    ]
    probe_second_probe_future_estimate_mean = [
        float(item["probe_second_probe_future_estimate_mean"] or 0.0)
        for item in results
    ]
    probe_second_probe_choice_future_gain_mean = [
        float(item["probe_second_probe_choice_future_gain_mean"] or 0.0)
        for item in results
    ]
    probe_family_coverage_satisfied_fraction = [
        float(item["probe_family_coverage_satisfied_fraction"] or 0.0)
        for item in results
    ]
    probe_second_probe_value_driven_fraction = [
        float(item["probe_second_probe_value_driven_fraction"] or 0.0)
        for item in results
    ]
    probe_uniformity_pressure_active_fraction = [
        float(item["probe_uniformity_pressure_active_fraction"] or 0.0)
        for item in results
    ]
    probe_online_offline_gap_mean = [
        float(item["probe_online_offline_gap_mean"] or 0.0)
        for item in results
    ]
    probe_message_input_delta_mean = [
        float(item["probe_message_input_delta_mean"] or 0.0)
        for item in results
    ]
    probe_message_input_delta_max = [
        float(item["probe_message_input_delta_max"] or 0.0)
        for item in results
    ]
    probe_muted_message_input_delta_mean = [
        float(item["probe_muted_message_input_delta_mean"] or 0.0)
        for item in results
    ]
    probe_muted_message_input_delta_max = [
        float(item["probe_muted_message_input_delta_max"] or 0.0)
        for item in results
    ]
    probe_actor_message_norm_mean = [
        float(item["probe_actor_message_norm_mean"] or 0.0)
        for item in results
    ]
    probe_actor_message_nonzero_fraction = [
        float(item["probe_actor_message_nonzero_fraction"] or 0.0)
        for item in results
    ]
    probe_muted_actor_message_nonzero_fraction = [
        float(item["probe_muted_actor_message_nonzero_fraction"] or 0.0)
        for item in results
    ]
    probe_matched_mute_parity_fraction = [
        float(item["probe_matched_mute_parity_fraction"] or 0.0)
        for item in results
    ]
    probe_online_subset_stability_mean = [
        float(item["probe_online_subset_stability_mean"] or 0.0)
        for item in results
    ]
    probe_online_geometry_complete_fraction = [
        float(item["probe_online_geometry_complete_fraction"] or 0.0)
        for item in results
    ]
    probe_online_split_latent_disagreement_mean = [
        float(item["probe_online_split_latent_disagreement_mean"] or 0.0)
        for item in results
    ]
    probe_online_split_retrieval_margin_deficit_mean = [
        float(item["probe_online_split_retrieval_margin_deficit_mean"] or 0.0)
        for item in results
    ]
    probe_online_leaveout_shift_mean = [
        float(item["probe_online_leaveout_shift_mean"] or 0.0)
        for item in results
    ]
    probe_teacher_action_agreement = [
        float(item["probe_teacher_action_agreement"] or 0.0)
        for item in results
    ]
    probe_env_expression_delta = [
        None if item["probe_env_expression_delta"] is None else float(item["probe_env_expression_delta"])
        for item in results
    ]
    probe_forced_env_expression_delta = [
        None
        if item["probe_forced_env_expression_delta"] is None
        else float(item["probe_forced_env_expression_delta"])
        for item in results
    ]
    probe_forced_env_expression_scale = [
        None
        if item["probe_forced_env_expression_scale"] is None
        else float(item["probe_forced_env_expression_scale"])
        for item in results
    ]
    probe_strict_usage_statuses = [
        str(
            item.get(
                "probe_strict_usage_status",
                probe_strict_usage_status(item.get("probe_fair_expression_enabled_fraction")),
            )
        )
        for item in results
    ]
    belief_progress_index = [
        float(item["belief_progress_index"] or 0.0)
        for item in results
    ]
    belief_mode = [
        str(item.get("belief_mode", config.belief_mode))
        for item in results
    ]
    belief_source = [
        str(item.get("belief_source", "sysid" if mode == "particle_sysid" else "learned"))
        for item, mode in zip(results, belief_mode)
    ]
    representation_repair_mode = [
        1.0 if bool(item.get("representation_repair_mode", config.representation_repair_mode)) else 0.0
        for item in results
    ]
    system_id_progress_index = [
        float(item.get("system_id_progress_index", 0.0) or 0.0)
        for item in results
    ]
    sysid_trusted = [
        1.0 if bool(item.get("sysid_trusted", False)) else 0.0
        for item in results
    ]
    sysid_validation_top1 = [
        float(item.get("sysid_validation_top1", 0.0) or 0.0)
        for item in results
    ]
    sysid_validation_margin = [
        float(item.get("sysid_validation_margin", 0.0) or 0.0)
        for item in results
    ]
    sysid_validation_nll = [
        float(item.get("sysid_validation_nll", 0.0) or 0.0)
        for item in results
    ]
    particle_entropy_mean = [
        float(item.get("particle_entropy_mean", 0.0) or 0.0)
        for item in results
    ]
    particle_entropy_norm_mean = [
        float(item.get("particle_entropy_norm_mean", 0.0) or 0.0)
        for item in results
    ]
    particle_ess_ratio_mean = [
        float(item.get("particle_ess_ratio_mean", 0.0) or 0.0)
        for item in results
    ]
    particle_leaveout_shift_mean = [
        float(item.get("particle_leaveout_shift_mean", 0.0) or 0.0)
        for item in results
    ]
    particle_subset_stability_mean = [
        float(item.get("particle_subset_stability_mean", 0.0) or 0.0)
        for item in results
    ]
    latent_mechanics_fit = [
        float(item["latent_mechanics_fit"] or 0.0)
        for item in results
    ]
    latent_split_top1 = [
        float(item["latent_split_top1"] or 0.0)
        for item in results
    ]
    latent_cross_split_top1 = [
        float(item.get("latent_cross_split_top1", item["latent_split_top1"]) or 0.0)
        for item in results
    ]
    latent_paired_split_top1 = [
        float(item.get("latent_paired_split_top1", 0.0) or 0.0)
        for item in results
    ]
    latent_cross_split_mrr = [
        float(item.get("latent_cross_split_mrr", item.get("latent_split_mrr", 0.0)) or 0.0)
        for item in results
    ]
    latent_paired_split_mrr = [
        float(item.get("latent_paired_split_mrr", 0.0) or 0.0)
        for item in results
    ]
    latent_neighbor_alignment = [
        float(item["latent_neighbor_alignment"] or 0.0)
        for item in results
    ]
    latent_gap_ratio = [
        float(item["latent_gap_ratio"] or 0.0)
        for item in results
    ]
    latent_heldout_probe_error = [
        float(item["latent_heldout_probe_error"] or 0.0)
        for item in results
    ]
    latent_probe_leakage = [
        float(item["latent_probe_leakage"] or 0.0)
        for item in results
    ]
    latent_uncert_error_corr = [
        float(item["latent_uncert_error_corr"] or 0.0)
        for item in results
    ]
    latent_support_diagnostics = [
        item.get("latent_support_diagnostics") or {}
        for item in results
    ]
    latent_support_diagnostics_json = [
        json.dumps(item, sort_keys=True)
        for item in latent_support_diagnostics
    ]
    representation_gate_json = [
        json.dumps(item.get("representation_gate") or {}, sort_keys=True)
        for item in results
    ]
    full_system_zero_context_ablation_delta = [
        float(item["full_system_zero_context_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_shuffled_context_ablation_delta = [
        float(item["full_system_shuffled_context_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_stale_context_ablation_delta = [
        float(item["full_system_stale_context_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_online_refinement_ablation_delta = [
        float(item["full_system_online_refinement_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_frozen_context_ablation_delta = [
        float(item["full_system_frozen_context_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_actor_only_ablation_delta = [
        float(item["full_system_actor_only_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_state_only_ablation_delta = [
        float(item["full_system_state_only_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_oracle_zero_context_ablation_delta = [
        float(item["full_system_oracle_zero_context_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_learned_eval_summary_json = [
        json.dumps(item["full_system_learned_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_state_only_eval_summary_json = [
        json.dumps(item["full_system_state_only_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_zero_context_eval_summary_json = [
        json.dumps(item["full_system_zero_context_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_shuffled_context_eval_summary_json = [
        json.dumps(item["full_system_shuffled_context_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_stale_context_eval_summary_json = [
        json.dumps(item["full_system_stale_context_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_online_refinement_eval_summary_json = [
        json.dumps(item["full_system_online_refinement_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_frozen_context_eval_summary_json = [
        json.dumps(item["full_system_frozen_context_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_actor_only_eval_summary_json = [
        json.dumps(item["full_system_actor_only_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_state_only_eval_returns_json = [
        json.dumps(item["full_system_state_only_eval_returns"] or [])
        for item in results
    ]
    full_system_oracle_shuffled_context_ablation_delta = [
        float(item["full_system_oracle_shuffled_context_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_oracle_stale_context_ablation_delta = [
        float(item["full_system_oracle_stale_context_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_oracle_online_refinement_ablation_delta = [
        float(item["full_system_oracle_online_refinement_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_oracle_frozen_context_ablation_delta = [
        float(item["full_system_oracle_frozen_context_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_oracle_actor_only_ablation_delta = [
        float(item["full_system_oracle_actor_only_ablation_delta"] or 0.0)
        for item in results
    ]
    full_system_oracle_learned_eval_summary_json = [
        json.dumps(item["full_system_oracle_learned_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_oracle_zero_context_eval_summary_json = [
        json.dumps(item["full_system_oracle_zero_context_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_oracle_shuffled_context_eval_summary_json = [
        json.dumps(item["full_system_oracle_shuffled_context_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_oracle_stale_context_eval_summary_json = [
        json.dumps(item["full_system_oracle_stale_context_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_oracle_online_refinement_eval_summary_json = [
        json.dumps(item["full_system_oracle_online_refinement_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_oracle_frozen_context_eval_summary_json = [
        json.dumps(item["full_system_oracle_frozen_context_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    full_system_oracle_actor_only_eval_summary_json = [
        json.dumps(item["full_system_oracle_actor_only_eval_summary"] or {}, sort_keys=True)
        for item in results
    ]
    probe_run_classification = [
        str(item["probe_run_classification"])
        for item in results
    ]
    probe_stop_reasons_json = [
        json.dumps(item["probe_stop_reasons"] or {}, sort_keys=True)
        for item in results
    ]
    probe_final_stop_reason = [
        str(item["probe_final_stop_reason"] or "")
        for item in results
    ]
    probe_family_expected_gain_json = [
        json.dumps(item["probe_family_expected_gain"] or {}, sort_keys=True)
        for item in results
    ]
    probe_family_realized_gain_json = [
        json.dumps(item["probe_family_realized_gain"] or {}, sort_keys=True)
        for item in results
    ]
    probe_family_future_error_json = [
        json.dumps(item["probe_family_future_error"] or {}, sort_keys=True)
        for item in results
    ]
    probe_family_selection_count_json = [
        json.dumps(item["probe_family_selection_count"] or {}, sort_keys=True)
        for item in results
    ]
    probe_readiness_reason_counts_json = [
        json.dumps(item["probe_readiness_reason_counts"] or {}, sort_keys=True)
        for item in results
    ]
    probe_readiness_component_means_json = [
        json.dumps(item["probe_readiness_component_means"] or {}, sort_keys=True)
        for item in results
    ]
    probe_fair_stop_blocker_counts_json = [
        json.dumps(item["probe_fair_stop_blocker_counts"] or {}, sort_keys=True)
        for item in results
    ]
    probe_shadow_blocker_counts_json = [
        json.dumps(item["probe_shadow_blocker_counts"] or {}, sort_keys=True)
        for item in results
    ]
    probe_second_probe_selection_count_json = [
        json.dumps(item["probe_second_probe_selection_count"] or {}, sort_keys=True)
        for item in results
    ]
    probe_fair_handoff_probe_families_json = [
        json.dumps(item["probe_fair_handoff_probe_families"] or [])
        for item in results
    ]
    probe_readiness_component_timeline_json = [
        json.dumps(item["probe_readiness_component_timeline"] or [])
        for item in results
    ]
    probe_message_ablation_config_diff_json = [
        json.dumps(item["probe_message_ablation_config_diff"] or {}, sort_keys=True)
        for item in results
    ]
    probe_online_future_quality_trace_json = [
        json.dumps(item["probe_online_future_quality_trace"] or [])
        for item in results
    ]
    probe_online_subset_stability_trace_json = [
        json.dumps(item["probe_online_subset_stability_trace"] or [])
        for item in results
    ]
    probe_online_offline_gap_trace_json = [
        json.dumps(item["probe_online_offline_gap_trace"] or [])
        for item in results
    ]
    latent_win_gate = evaluate_latent_win_gate(
        benchmark_profile=benchmark_profile,
        seed_count=len(seeds),
        baseline_episode_solves=baseline_episode_solves,
        baseline_completed_episodes=baseline_completed_episodes,
        probe_episode_solves=probe_episode_solves,
        probe_completed_episodes=probe_completed_episodes,
        baseline_step_solves=baseline_step_solves,
        baseline_total_env_steps=baseline_total_env_steps,
        probe_step_solves=probe_step_solves,
        probe_total_env_steps=probe_total_env_steps,
        probe_env_expression_delta=probe_env_expression_delta,
        probe_no_expression_available=(
            bool(probe_no_expression_completed_episodes)
            and all(int(value) > 0 for value in probe_no_expression_completed_episodes)
        ),
        probe_ready_fraction=probe_fair_ready_handoff_fraction,
        probe_muted_fraction=probe_fair_expression_force_muted_fraction,
        probe_expression_enabled_fraction=probe_fair_expression_enabled_fraction,
        latent_mechanics_fit=latent_mechanics_fit,
        latent_neighbor_alignment=latent_neighbor_alignment,
        latent_split_retrieval=latent_cross_split_top1,
        latent_gap_ratio=latent_gap_ratio,
        latent_probe_leakage=latent_probe_leakage,
        latent_uncert_error_corr=latent_uncert_error_corr,
        full_system_state_only_ablation_delta=full_system_state_only_ablation_delta,
        full_system_zero_context_ablation_delta=full_system_zero_context_ablation_delta,
        full_system_shuffled_context_ablation_delta=full_system_shuffled_context_ablation_delta,
        full_system_stale_context_ablation_delta=full_system_stale_context_ablation_delta,
    )
    belief_progress_index_mean = float(
        np.mean(np.asarray(belief_progress_index, dtype=np.float32))
    ) if belief_progress_index else 0.0
    belief_progress_index_median = float(
        np.median(np.asarray(belief_progress_index, dtype=np.float32))
    ) if belief_progress_index else 0.0
    latent_win_gate_failure_reasons = [
        str(reason) for reason in latent_win_gate["failure_reasons"]
    ]

    print("\n=== Benchmark Summary ===")
    for item in results:
        dominant_readiness_blocker = max(
            (item["probe_readiness_reason_counts"] or {}).items(),
            key=lambda pair: (int(pair[1]), str(pair[0])),
            default=("none", 0),
        )[0]
        leaveout_stability_mean = float(
            (item["probe_readiness_component_means"] or {}).get("leaveout_stability", 0.0)
        )
        print(
            f"seed={item['seed']} | "
            f"profile={item['benchmark_profile']} | "
            f"baseline_ep={item['baseline_solve_episode']} | "
            f"baseline_steps={item['baseline_solve_env_steps']} | "
            f"probe_ep={item['probe_solve_episode']} | "
            f"probe_steps={item['probe_solve_env_steps']} | "
            f"probe_shadow_ep={item['probe_shadow_solve_episode']} | "
            f"probe_shadow_steps={item['probe_shadow_solve_env_steps']} | "
            f"probe_noexpr_ep={item['probe_no_expression_solve_episode']} | "
            f"probe_noexpr_steps={item['probe_no_expression_solve_env_steps']} | "
            f"full_system_ep={item['full_system_solve_episode']} | "
            f"full_system_steps={item['full_system_solve_env_steps']} | "
            f"full_system_eval={None if item['full_system_learned_eval_summary'] is None else round(float(item['full_system_learned_eval_summary']['mean_return']), 2)} | "
            f"state_only_eval={None if item['full_system_state_only_eval_summary'] is None else round(float(item['full_system_state_only_eval_summary']['mean_return']), 2)} | "
            f"full_system_oracle_ep={item['full_system_oracle_solve_episode']} | "
            f"full_system_oracle_steps={item['full_system_oracle_solve_env_steps']} | "
            f"sim_fanout_ep={item['sim_fanout_solve_episode']} | "
            f"sim_fanout_steps={item['sim_fanout_solve_env_steps']} | "
            f"probe_encoder_steps={item['probe_encoder_steps']} | "
            f"probe_online_steps={item['probe_probe_env_steps'] - item['probe_encoder_steps']} | "
            f"probe_control_steps={item['probe_control_env_steps']} | "
            f"probe_post_expr_steps={item['probe_post_expression_env_steps']} | "
            f"probe_ready_frac={item['probe_fair_ready_handoff_fraction']} | "
            f"probe_muted_frac={item['probe_fair_expression_force_muted_fraction']} | "
            f"probe_shadow_enabled_frac={item['probe_shadow_expression_enabled_fraction']} | "
            f"probe_shadow_strict_miss_frac={item['probe_shadow_strict_miss_fraction']} | "
            f"probe_second_probe_future_gain={item['probe_second_probe_choice_future_gain_mean']} | "
            f"probe_coverage_satisfied={item['probe_family_coverage_satisfied_fraction']} | "
            f"probe_value_driven={item['probe_second_probe_value_driven_fraction']} | "
            f"probe_uniformity_pressure={item['probe_uniformity_pressure_active_fraction']} | "
            f"probe_online_gap={item['probe_online_offline_gap_mean']} | "
            f"probe_msg_delta={item['probe_message_input_delta_mean']} | "
            f"probe_muted_parity={item['probe_matched_mute_parity_fraction']} | "
            f"probe_msg_seen_frac={item['probe_actor_message_nonzero_fraction']} | "
            f"strict_usage={item.get('probe_strict_usage_status', probe_strict_usage_status(item.get('probe_fair_expression_enabled_fraction')))} | "
            f"probe_expr_delta={item['probe_env_expression_delta']} | "
            f"probe_forced_expr_delta={item['probe_forced_env_expression_delta']} | "
            f"probe_forced_expr_scale={item['probe_forced_env_expression_scale']} | "
            f"probe_blocker={dominant_readiness_blocker} | "
            f"probe_leaveout={leaveout_stability_mean:.3f} | "
            f"support_center={float((item.get('latent_support_diagnostics') or {}).get('center_window_share', 0.0)):.3f} | "
            f"support_dir={float((item.get('latent_support_diagnostics') or {}).get('directional_window_share', 0.0)):.3f} | "
            f"support_mech={float((item.get('latent_support_diagnostics') or {}).get('mechanics_window_share', 0.0)):.3f} | "
            f"support_passive={float((item.get('latent_support_diagnostics') or {}).get('passive_window_share', 0.0)):.3f} | "
            f"support_stress={float((item.get('latent_support_diagnostics') or {}).get('stress_window_share', 0.0)):.3f} | "
            f"support_eff={float((item.get('latent_support_diagnostics') or {}).get('effective_window_families', 0.0)):.2f} | "
            f"window_leak={float((item.get('latent_support_diagnostics') or {}).get('window_mode_leakage', 0.0)):.3f} | "
            f"bpi={item['belief_progress_index']:.3f} | "
            f"class={item['probe_run_classification']}"
        )
        if str(item.get("belief_mode", config.belief_mode)) == "particle_sysid":
            print(
                "  sysid | "
                f"trusted={bool(item.get('sysid_trusted', False))} | "
                f"top1={float(item.get('sysid_validation_top1', 0.0)):.3f} | "
                f"margin={float(item.get('sysid_validation_margin', 0.0)):.3f} | "
                f"ess={float(item.get('particle_ess_ratio_mean', 0.0)):.3f} | "
                f"leaveout={float(item.get('particle_leaveout_shift_mean', 0.0)):.3f} | "
                f"score={float(item.get('system_id_progress_index', 0.0)):.3f}"
            )
        if str(item.get("belief_mode", config.belief_mode)) == "particle_sysid":
            geometry_label = system_id_geometry_status(
                trusted=bool(item.get("sysid_trusted", False)),
                validation_top1=float(item.get("sysid_validation_top1", 0.0)),
                validation_margin=float(item.get("sysid_validation_margin", 0.0)),
                leaveout_shift=float(item.get("particle_leaveout_shift_mean", 0.0)),
                subset_stability=float(item.get("particle_subset_stability_mean", 0.0)),
            )
        else:
            geometry_label = geometry_status(
                split_mrr=item["latent_split_mrr"],
                neighbor_alignment=item["latent_neighbor_alignment"],
            )
        print(
            "  health | "
            f"{format_status('signal', signal_status(message_on_fraction=item['probe_message_on_fraction'], message_diag_fraction=item['probe_message_diag_fraction'], probe_expr_delta=item['probe_env_expression_delta']))} | "
            f"{format_status('geometry', geometry_label)} | "
            f"{format_status('utility', utility_status(baseline_episode=item['baseline_solve_episode'], probe_episode=item['probe_solve_episode'], no_message_episode=item['probe_no_expression_solve_episode']))}"
        )
        if (
            str(item.get("probe_strict_usage_status", "unused")) == "unused"
            and item["probe_solve_episode"] is not None
            and item["baseline_solve_episode"] is not None
            and int(item["probe_solve_episode"]) < int(item["baseline_solve_episode"])
        ):
            print("  honesty | Episode win without strict latent usage")
        if (
            item["probe_env_expression_delta"] is not None
            and float(item["probe_env_expression_delta"]) <= 0.0
        ):
            print("  honesty | Env expression harmful under matched eval")
    print(
        "Belief Progress Index: "
        f"median={belief_progress_index_median:.3f} | "
        f"mean={belief_progress_index_mean:.3f}"
    )
    print(
        "Latent Support Diagnostics: "
        f"center={_diag_median(latent_support_diagnostics, 'center_window_share'):.3f} | "
        f"directional={_diag_median(latent_support_diagnostics, 'directional_window_share'):.3f} | "
        f"mechanics={_diag_median(latent_support_diagnostics, 'mechanics_window_share'):.3f} | "
        f"passive={_diag_median(latent_support_diagnostics, 'passive_window_share'):.3f} | "
        f"stress={_diag_median(latent_support_diagnostics, 'stress_window_share'):.3f} | "
        f"eff-family={_diag_median(latent_support_diagnostics, 'effective_window_families'):.2f} | "
        f"support/env={_diag_median(latent_support_diagnostics, 'support_count_mean'):.1f} | "
        f"split-overlap={_diag_median(latent_support_diagnostics, 'split_group_overlap_mean'):.3f} | "
        f"cross-overlap={_diag_median(latent_support_diagnostics, 'cross_family_split_group_overlap_mean'):.3f} | "
        f"window-leak={_diag_median(latent_support_diagnostics, 'window_mode_leakage'):.3f} | "
        f"env-leak={_diag_median(latent_support_diagnostics, 'env_mode_leakage'):.3f} | "
        f"nearest={_diag_median(latent_support_diagnostics, 'nearest_between_median'):.4f}"
    )
    if any(str(mode) == "particle_sysid" for mode in belief_mode):
        print(
            "System-ID Diagnostics: "
            f"trusted={float(np.mean(np.asarray(sysid_trusted, dtype=np.float32))):.2f} | "
            f"top1={float(np.median(np.asarray(sysid_validation_top1, dtype=np.float32))):.3f} | "
            f"margin={float(np.median(np.asarray(sysid_validation_margin, dtype=np.float32))):.3f} | "
            f"ess={float(np.median(np.asarray(particle_ess_ratio_mean, dtype=np.float32))):.3f} | "
            f"leaveout={float(np.median(np.asarray(particle_leaveout_shift_mean, dtype=np.float32))):.3f} | "
            f"score={float(np.median(np.asarray(system_id_progress_index, dtype=np.float32))):.3f}"
        )
    print(
        "Latent Win Gate: "
        f"{'PASS' if bool(latent_win_gate['pass']) else 'BLOCKED'} | "
        f"reasons={latent_win_gate_failure_reasons if latent_win_gate_failure_reasons else ['none']}"
    )

    print_solve_summary(
        "Baseline PPO solve episode",
        baseline_episode_solves,
        baseline_completed_episodes,
    )
    print_solve_summary(
        "Probe-conditioned PPO solve episode",
        probe_episode_solves,
        probe_completed_episodes,
    )
    print_solve_summary(
        "Baseline PPO solve env steps",
        baseline_step_solves,
        baseline_total_env_steps,
    )
    print_solve_summary(
        "Probe + shadow env expression solve episode",
        probe_shadow_episode_solves,
        probe_shadow_completed_episodes,
    )
    print_solve_summary(
        "Probe + shadow env expression solve env steps",
        probe_shadow_step_solves,
        probe_shadow_total_env_steps,
    )
    print_solve_summary(
        "Probe-conditioned PPO solve env steps",
        probe_step_solves,
        probe_total_env_steps,
    )
    print_solve_summary(
        "Baseline PPO steps to peak return",
        baseline_peak_env_steps,
        baseline_total_env_steps,
    )
    print_solve_summary(
        "Probe-conditioned PPO steps to peak return",
        probe_peak_env_steps_with_encoder,
        probe_total_env_steps,
    )
    print_solve_summary(
        "Probe + no env expression solve episode",
        probe_no_expression_episode_solves,
        probe_no_expression_completed_episodes,
    )
    print_solve_summary(
        "Probe + no env expression solve env steps",
        probe_no_expression_step_solves,
        probe_no_expression_total_env_steps,
    )
    save_benchmark_results(
        config.env_name,
        config.benchmark_tag,
        benchmark_profile,
        config.benchmark_mode,
        config.probe_budget_mode,
        seeds,
        [value if value is not None else -1 for value in baseline_episode_solves],
        [value if value is not None else -1 for value in probe_episode_solves],
        [value if value is not None else -1 for value in probe_shadow_episode_solves],
        [value if value is not None else -1 for value in probe_no_expression_episode_solves],
        [value if value is not None else -1 for value in full_system_episode_solves],
        [value if value is not None else -1 for value in full_system_state_only_episode_solves],
        [value if value is not None else -1 for value in full_system_oracle_episode_solves],
        [value if value is not None else -1 for value in baseline_step_solves],
        [value if value is not None else -1 for value in probe_step_solves],
        [value if value is not None else -1 for value in probe_shadow_step_solves],
        [value if value is not None else -1 for value in probe_no_expression_step_solves],
        [value if value is not None else -1 for value in full_system_step_solves],
        [value if value is not None else -1 for value in full_system_state_only_step_solves],
        [value if value is not None else -1 for value in full_system_oracle_step_solves],
        baseline_total_env_steps,
        probe_total_env_steps,
        probe_shadow_total_env_steps,
        probe_no_expression_total_env_steps,
        full_system_total_env_steps,
        full_system_state_only_total_env_steps,
        full_system_oracle_total_env_steps,
        baseline_control_env_steps,
        probe_probe_env_steps,
        probe_control_env_steps,
        probe_post_expression_env_steps,
        [value if value is not None else -1 for value in probe_post_expression_episodes],
        probe_shadow_probe_env_steps,
        probe_shadow_control_env_steps,
        probe_shadow_post_expression_env_steps,
        [value if value is not None else -1 for value in probe_shadow_post_expression_episodes],
        probe_no_expression_probe_env_steps,
        probe_no_expression_control_env_steps,
        probe_no_expression_post_expression_env_steps,
        [value if value is not None else -1 for value in probe_no_expression_post_expression_episodes],
        full_system_probe_env_steps,
        full_system_control_env_steps,
        full_system_post_context_env_steps,
        full_system_post_context_episodes,
        full_system_oracle_probe_env_steps,
        full_system_oracle_control_env_steps,
        full_system_oracle_post_context_env_steps,
        full_system_oracle_post_context_episodes,
        baseline_completed_episodes,
        probe_completed_episodes,
        probe_shadow_completed_episodes,
        probe_no_expression_completed_episodes,
        full_system_completed_episodes,
        full_system_state_only_completed_episodes,
        full_system_oracle_completed_episodes,
        probe_encoder_steps,
        probe_windows_total,
        probe_expression_scale_median,
        probe_expression_scale_active_fraction,
        probe_fair_ready_handoff_fraction,
        probe_fair_expression_enabled_fraction,
        probe_fair_expression_force_muted_fraction,
        probe_fair_ready_confidence_median,
        probe_fair_muted_confidence_median,
        probe_expression_ready_but_muted_fraction,
        probe_shadow_expression_enabled_fraction,
        probe_shadow_expression_scale_median,
        probe_shadow_confidence_median,
        probe_shadow_strict_miss_fraction,
        probe_run_classification,
        belief_progress_index,
        latent_mechanics_fit,
        latent_split_top1,
        latent_neighbor_alignment,
        latent_gap_ratio,
        latent_heldout_probe_error,
        latent_probe_leakage,
        latent_uncert_error_corr,
        latent_support_diagnostics_json,
        json.dumps(latent_win_gate, sort_keys=True),
        json.dumps(latent_win_gate_failure_reasons),
        probe_stop_reasons_json,
        probe_final_stop_reason,
        probe_family_expected_gain_json,
        probe_family_realized_gain_json,
        probe_family_future_error_json,
        probe_family_selection_count_json,
        probe_readiness_reason_counts_json,
        probe_readiness_component_means_json,
        probe_fair_stop_blocker_counts_json,
        probe_shadow_blocker_counts_json,
        probe_second_probe_selection_count_json,
        probe_second_probe_raw_future_gain_mean,
        probe_second_probe_future_estimate_mean,
        probe_second_probe_choice_future_gain_mean,
        probe_family_coverage_satisfied_fraction,
        probe_second_probe_value_driven_fraction,
        probe_uniformity_pressure_active_fraction,
        probe_env_expression_delta,
        probe_forced_env_expression_delta,
        probe_forced_env_expression_scale,
        probe_strict_usage_statuses,
        probe_fair_handoff_probe_families_json,
        probe_readiness_component_timeline_json,
        probe_message_ablation_config_diff_json,
        probe_online_future_quality_trace_json,
        probe_online_subset_stability_trace_json,
        probe_online_offline_gap_trace_json,
        probe_message_input_delta_mean,
        probe_message_input_delta_max,
        probe_muted_message_input_delta_mean,
        probe_muted_message_input_delta_max,
        probe_actor_message_norm_mean,
        probe_actor_message_nonzero_fraction,
        probe_muted_actor_message_nonzero_fraction,
        probe_matched_mute_parity_fraction,
        probe_online_subset_stability_mean,
        probe_online_offline_gap_mean,
        probe_online_geometry_complete_fraction,
        probe_online_split_latent_disagreement_mean,
        probe_online_split_retrieval_margin_deficit_mean,
        probe_online_leaveout_shift_mean,
        probe_teacher_action_agreement,
        full_system_state_only_eval_returns_json,
        full_system_learned_eval_summary_json,
        full_system_state_only_eval_summary_json,
        full_system_zero_context_eval_summary_json,
        full_system_shuffled_context_eval_summary_json,
        full_system_stale_context_eval_summary_json,
        full_system_online_refinement_eval_summary_json,
        full_system_frozen_context_eval_summary_json,
        full_system_actor_only_eval_summary_json,
        full_system_state_only_ablation_delta,
        full_system_zero_context_ablation_delta,
        full_system_shuffled_context_ablation_delta,
        full_system_stale_context_ablation_delta,
        full_system_online_refinement_ablation_delta,
        full_system_frozen_context_ablation_delta,
        full_system_actor_only_ablation_delta,
        full_system_oracle_learned_eval_summary_json,
        full_system_oracle_zero_context_eval_summary_json,
        full_system_oracle_shuffled_context_eval_summary_json,
        full_system_oracle_stale_context_eval_summary_json,
        full_system_oracle_online_refinement_eval_summary_json,
        full_system_oracle_frozen_context_eval_summary_json,
        full_system_oracle_actor_only_eval_summary_json,
        full_system_oracle_zero_context_ablation_delta,
        full_system_oracle_shuffled_context_ablation_delta,
        full_system_oracle_stale_context_ablation_delta,
        full_system_oracle_online_refinement_ablation_delta,
        full_system_oracle_frozen_context_ablation_delta,
        full_system_oracle_actor_only_ablation_delta,
        [value if value is not None else -1 for value in sim_fanout_episode_solves],
        [value if value is not None else -1 for value in sim_fanout_step_solves],
        sim_fanout_total_env_steps,
        sim_fanout_probe_env_steps,
        sim_fanout_control_env_steps,
        sim_fanout_post_context_env_steps,
        [value if value is not None else -1 for value in sim_fanout_post_context_episodes],
        sim_fanout_completed_episodes,
        ["" if value is None else str(value) for value in full_system_controller_style],
        ["" if value is None else str(value) for value in full_system_oracle_controller_style],
        ["" if value is None else str(value) for value in sim_fanout_controller_style],
        extra_summary_fields={
            "baseline_best_returns": np.asarray(baseline_best_returns, dtype=np.float32),
            "probe_best_returns": np.asarray(probe_best_returns, dtype=np.float32),
            "baseline_peak_env_steps": np.asarray(
                [value if value is not None else -1 for value in baseline_peak_env_steps],
                dtype=np.int64,
            ),
            "probe_peak_env_steps_with_encoder": np.asarray(
                [
                    value if value is not None else -1
                    for value in probe_peak_env_steps_with_encoder
                ],
                dtype=np.int64,
            ),
            "belief_mode": np.asarray(belief_mode, dtype="U"),
            "belief_source": np.asarray(belief_source, dtype="U"),
            "representation_repair_mode": np.asarray(representation_repair_mode, dtype=np.float32),
            "system_id_progress_index": np.asarray(system_id_progress_index, dtype=np.float32),
            "sysid_trusted": np.asarray(sysid_trusted, dtype=np.float32),
            "sysid_validation_top1": np.asarray(sysid_validation_top1, dtype=np.float32),
            "sysid_validation_margin": np.asarray(sysid_validation_margin, dtype=np.float32),
            "sysid_validation_nll": np.asarray(sysid_validation_nll, dtype=np.float32),
            "particle_entropy_mean": np.asarray(particle_entropy_mean, dtype=np.float32),
            "particle_entropy_norm_mean": np.asarray(particle_entropy_norm_mean, dtype=np.float32),
            "particle_ess_ratio_mean": np.asarray(particle_ess_ratio_mean, dtype=np.float32),
            "particle_leaveout_shift_mean": np.asarray(particle_leaveout_shift_mean, dtype=np.float32),
            "particle_subset_stability_mean": np.asarray(particle_subset_stability_mean, dtype=np.float32),
            "representation_gate_json": np.asarray(representation_gate_json, dtype="U"),
            "latent_paired_split_top1": np.asarray(latent_paired_split_top1, dtype=np.float32),
            "latent_paired_split_mrr": np.asarray(latent_paired_split_mrr, dtype=np.float32),
            "latent_cross_split_top1": np.asarray(latent_cross_split_top1, dtype=np.float32),
            "latent_cross_split_mrr": np.asarray(latent_cross_split_mrr, dtype=np.float32),
        },
    )
    live_trace.finish(
        summary={
            "benchmark_profile": benchmark_profile,
            "baseline_episode_solves": baseline_episode_solves,
            "probe_episode_solves": probe_episode_solves,
            "probe_shadow_episode_solves": probe_shadow_episode_solves,
            "probe_no_expression_episode_solves": probe_no_expression_episode_solves,
            "full_system_episode_solves": full_system_episode_solves,
            "full_system_state_only_episode_solves": full_system_state_only_episode_solves,
            "full_system_oracle_episode_solves": full_system_oracle_episode_solves,
            "sim_fanout_episode_solves": sim_fanout_episode_solves,
            "full_system_state_only_eval_mean_returns": [
                None
                if item["full_system_state_only_eval_summary"] is None
                else float(item["full_system_state_only_eval_summary"]["mean_return"])
                for item in results
            ],
            "baseline_step_solves": baseline_step_solves,
            "probe_step_solves": probe_step_solves,
            "probe_shadow_step_solves": probe_shadow_step_solves,
            "probe_no_expression_step_solves": probe_no_expression_step_solves,
            "baseline_best_returns": baseline_best_returns,
            "probe_best_returns": probe_best_returns,
            "baseline_peak_env_steps": baseline_peak_env_steps,
            "probe_peak_env_steps_with_encoder": probe_peak_env_steps_with_encoder,
            "probe_env_expression_delta": probe_env_expression_delta,
            "probe_forced_env_expression_delta": probe_forced_env_expression_delta,
            "probe_strict_usage_status": probe_strict_usage_statuses,
            "full_system_step_solves": full_system_step_solves,
            "full_system_state_only_step_solves": full_system_state_only_step_solves,
            "full_system_oracle_step_solves": full_system_oracle_step_solves,
            "sim_fanout_step_solves": sim_fanout_step_solves,
            "full_system_state_only_ablation_delta": full_system_state_only_ablation_delta,
        }
    )
    return {
        "env_name": config.env_name,
        "benchmark_tag": config.benchmark_tag,
        "benchmark_profile": benchmark_profile,
        "benchmark_mode": config.benchmark_mode,
        "probe_budget_mode": config.probe_budget_mode,
        "representation_repair_mode": bool(config.representation_repair_mode),
        "seeds": list(seeds),
        "baseline_episode_solves": baseline_episode_solves,
        "probe_episode_solves": probe_episode_solves,
        "probe_shadow_episode_solves": probe_shadow_episode_solves,
        "probe_no_expression_episode_solves": probe_no_expression_episode_solves,
        "full_system_episode_solves": full_system_episode_solves,
        "full_system_state_only_episode_solves": full_system_state_only_episode_solves,
        "full_system_oracle_episode_solves": full_system_oracle_episode_solves,
        "sim_fanout_episode_solves": sim_fanout_episode_solves,
        "baseline_step_solves": baseline_step_solves,
        "probe_step_solves": probe_step_solves,
        "baseline_best_returns": baseline_best_returns,
        "probe_best_returns": probe_best_returns,
        "baseline_peak_env_steps": baseline_peak_env_steps,
        "probe_peak_env_steps_with_encoder": probe_peak_env_steps_with_encoder,
        "probe_shadow_step_solves": probe_shadow_step_solves,
        "probe_no_expression_step_solves": probe_no_expression_step_solves,
        "full_system_step_solves": full_system_step_solves,
        "full_system_state_only_step_solves": full_system_state_only_step_solves,
        "full_system_oracle_step_solves": full_system_oracle_step_solves,
        "sim_fanout_step_solves": sim_fanout_step_solves,
        "baseline_total_env_steps": baseline_total_env_steps,
        "probe_total_env_steps": probe_total_env_steps,
        "probe_shadow_total_env_steps": probe_shadow_total_env_steps,
        "probe_no_expression_total_env_steps": probe_no_expression_total_env_steps,
        "probe_env_expression_delta": probe_env_expression_delta,
        "probe_forced_env_expression_delta": probe_forced_env_expression_delta,
        "probe_forced_env_expression_scale": probe_forced_env_expression_scale,
        "probe_strict_usage_status": probe_strict_usage_statuses,
        "belief_source": belief_source,
        "representation_gate_json": representation_gate_json,
        "latent_paired_split_top1": latent_paired_split_top1,
        "latent_cross_split_top1": latent_cross_split_top1,
        "full_system_total_env_steps": full_system_total_env_steps,
        "full_system_state_only_total_env_steps": full_system_state_only_total_env_steps,
        "full_system_oracle_total_env_steps": full_system_oracle_total_env_steps,
        "sim_fanout_total_env_steps": sim_fanout_total_env_steps,
        "baseline_completed_episodes": baseline_completed_episodes,
        "probe_completed_episodes": probe_completed_episodes,
        "probe_shadow_completed_episodes": probe_shadow_completed_episodes,
        "probe_no_expression_completed_episodes": probe_no_expression_completed_episodes,
        "full_system_completed_episodes": full_system_completed_episodes,
        "full_system_state_only_completed_episodes": full_system_state_only_completed_episodes,
        "full_system_oracle_completed_episodes": full_system_oracle_completed_episodes,
        "sim_fanout_completed_episodes": sim_fanout_completed_episodes,
    }
