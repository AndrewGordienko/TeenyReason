"""Benchmark summary artifact payload builder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...envs import get_env_display_name
from ...models.sysid import PARTICLE_READINESS_LEAVEOUT_SCALE
from ..loss_attribution import build_loss_attribution_metrics
from ..research_metrics import build_benchmark_research_metrics, sample_savings_or_none
from ..diagnostics import summarize_solve_array
from .common import (
    aggregate_json_counter_rows,
    aggregate_json_list_rows,
    average_json_metric_rows,
    load_array_with_fallback,
    load_benchmark_summary,
    load_optional_json_rows,
    load_optional_string,
    normalize_matched_eval_summary,
    summarize_matched_eval_rows,
)


def build_benchmark_payload(path: Path) -> dict:
    """Convert one benchmark summary artifact into a JSON-friendly payload."""
    summary = load_benchmark_summary(path)
    env_name = load_optional_string(summary, "env_name")
    benchmark_profile = load_optional_string(summary, "benchmark_profile")
    seeds = summary["seeds"].astype(np.int64).tolist()
    baseline_episode_solves = load_array_with_fallback(
        summary,
        "baseline_episode_solves",
        "baseline_solves",
    ).astype(np.int64)
    probe_episode_solves = load_array_with_fallback(
        summary,
        "probe_episode_solves",
        "probe_solves",
    ).astype(np.int64)
    probe_shadow_episode_solves = summary.get(
        "probe_shadow_episode_solves",
        summary.get("probe_shadow_solves", probe_episode_solves),
    ).astype(np.int64)
    probe_no_expression_episode_solves = summary.get(
        "probe_no_expression_episode_solves",
        summary.get(
            "probe_no_expression_solves",
            np.full_like(probe_episode_solves, -1),
        ),
    ).astype(np.int64)
    full_system_episode_solves = summary.get(
        "full_system_episode_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    full_system_state_only_episode_solves = summary.get(
        "full_system_state_only_episode_solves",
        np.full_like(full_system_episode_solves, -1),
    ).astype(np.int64)
    full_system_oracle_episode_solves = summary.get(
        "full_system_oracle_episode_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    sim_fanout_episode_solves = summary.get(
        "sim_fanout_episode_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    baseline_step_solves = summary.get(
        "baseline_step_solves",
        np.full_like(baseline_episode_solves, -1),
    ).astype(np.int64)
    probe_step_solves = summary.get(
        "probe_step_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    probe_shadow_step_solves = summary.get(
        "probe_shadow_step_solves",
        np.full_like(probe_shadow_episode_solves, -1),
    ).astype(np.int64)
    probe_no_expression_step_solves = summary.get(
        "probe_no_expression_step_solves",
        np.full_like(probe_no_expression_episode_solves, -1),
    ).astype(np.int64)
    full_system_step_solves = summary.get(
        "full_system_step_solves",
        np.full_like(full_system_episode_solves, -1),
    ).astype(np.int64)
    full_system_state_only_step_solves = summary.get(
        "full_system_state_only_step_solves",
        np.full_like(full_system_state_only_episode_solves, -1),
    ).astype(np.int64)
    full_system_oracle_step_solves = summary.get(
        "full_system_oracle_step_solves",
        np.full_like(full_system_oracle_episode_solves, -1),
    ).astype(np.int64)
    sim_fanout_step_solves = summary.get(
        "sim_fanout_step_solves",
        np.full_like(sim_fanout_episode_solves, -1),
    ).astype(np.int64)
    baseline_total_env_steps = summary.get(
        "baseline_total_env_steps",
        np.full_like(baseline_episode_solves, 0),
    ).astype(np.int64)
    probe_total_env_steps = summary.get(
        "probe_total_env_steps",
        np.full_like(probe_episode_solves, 0),
    ).astype(np.int64)
    probe_shadow_total_env_steps = summary.get(
        "probe_shadow_total_env_steps",
        np.full_like(probe_shadow_episode_solves, 0),
    ).astype(np.int64)
    probe_no_expression_total_env_steps = summary.get(
        "probe_no_expression_total_env_steps",
        np.full_like(probe_no_expression_episode_solves, 0),
    ).astype(np.int64)
    full_system_total_env_steps = summary.get(
        "full_system_total_env_steps",
        np.full_like(full_system_episode_solves, 0),
    ).astype(np.int64)
    full_system_state_only_total_env_steps = summary.get(
        "full_system_state_only_total_env_steps",
        np.full_like(full_system_state_only_episode_solves, 0),
    ).astype(np.int64)
    full_system_oracle_total_env_steps = summary.get(
        "full_system_oracle_total_env_steps",
        np.full_like(full_system_oracle_episode_solves, 0),
    ).astype(np.int64)
    sim_fanout_total_env_steps = summary.get(
        "sim_fanout_total_env_steps",
        np.full_like(sim_fanout_episode_solves, 0),
    ).astype(np.int64)
    baseline_completed_episodes = summary.get(
        "baseline_completed_episodes",
        np.full_like(baseline_episode_solves, 0),
    ).astype(np.int64)
    probe_completed_episodes = summary.get(
        "probe_completed_episodes",
        np.full_like(probe_episode_solves, 0),
    ).astype(np.int64)
    probe_shadow_completed_episodes = summary.get(
        "probe_shadow_completed_episodes",
        np.full_like(probe_shadow_episode_solves, 0),
    ).astype(np.int64)
    probe_no_expression_completed_episodes = summary.get(
        "probe_no_expression_completed_episodes",
        np.full_like(probe_no_expression_episode_solves, 0),
    ).astype(np.int64)
    probe_no_expression_available = summary.get(
        "probe_no_expression_available",
        (probe_no_expression_completed_episodes > 0).astype(np.int8),
    ).astype(np.int8)
    latent_claim_valid = summary.get(
        "latent_claim_valid",
        np.zeros_like(probe_episode_solves, dtype=np.int8),
    ).astype(np.int8)
    full_system_completed_episodes = summary.get(
        "full_system_completed_episodes",
        np.full_like(full_system_episode_solves, 0),
    ).astype(np.int64)
    full_system_state_only_completed_episodes = summary.get(
        "full_system_state_only_completed_episodes",
        np.full_like(full_system_state_only_episode_solves, 0),
    ).astype(np.int64)
    full_system_oracle_completed_episodes = summary.get(
        "full_system_oracle_completed_episodes",
        np.full_like(full_system_oracle_episode_solves, 0),
    ).astype(np.int64)
    sim_fanout_completed_episodes = summary.get(
        "sim_fanout_completed_episodes",
        np.full_like(sim_fanout_episode_solves, 0),
    ).astype(np.int64)
    full_system_controller_style = summary.get(
        "full_system_controller_style",
        np.asarray([""] * len(full_system_episode_solves), dtype="U"),
    ).astype("U")
    full_system_oracle_controller_style = summary.get(
        "full_system_oracle_controller_style",
        np.asarray([""] * len(full_system_oracle_episode_solves), dtype="U"),
    ).astype("U")
    sim_fanout_controller_style = summary.get(
        "sim_fanout_controller_style",
        np.asarray([""] * len(sim_fanout_episode_solves), dtype="U"),
    ).astype("U")
    baseline_best_returns = summary.get(
        "baseline_best_returns",
        np.full(baseline_episode_solves.shape, np.nan, dtype=np.float32),
    ).astype(np.float32)
    probe_best_returns = summary.get(
        "probe_best_returns",
        np.full(probe_episode_solves.shape, np.nan, dtype=np.float32),
    ).astype(np.float32)
    baseline_peak_env_steps = summary.get(
        "baseline_peak_env_steps",
        np.full(baseline_episode_solves.shape, -1, dtype=np.int64),
    ).astype(np.int64)
    probe_peak_env_steps_with_encoder = summary.get(
        "probe_peak_env_steps_with_encoder",
        np.full(probe_episode_solves.shape, -1, dtype=np.int64),
    ).astype(np.int64)
    probe_encoder_steps = summary.get(
        "probe_encoder_steps",
        np.zeros_like(probe_episode_solves),
    ).astype(np.int64)
    baseline_control_env_steps = summary.get(
        "baseline_control_env_steps",
        baseline_total_env_steps,
    ).astype(np.int64)
    probe_probe_env_steps = summary.get(
        "probe_probe_env_steps",
        probe_total_env_steps,
    ).astype(np.int64)
    probe_control_env_steps = summary.get(
        "probe_control_env_steps",
        probe_total_env_steps,
    ).astype(np.int64)
    probe_shadow_probe_env_steps = summary.get(
        "probe_shadow_probe_env_steps",
        probe_shadow_total_env_steps,
    ).astype(np.int64)
    probe_shadow_control_env_steps = summary.get(
        "probe_shadow_control_env_steps",
        probe_shadow_total_env_steps,
    ).astype(np.int64)
    probe_post_expression_env_steps = summary.get(
        "probe_post_expression_env_steps",
        probe_control_env_steps,
    ).astype(np.int64)
    probe_shadow_post_expression_env_steps = summary.get(
        "probe_shadow_post_expression_env_steps",
        probe_shadow_control_env_steps,
    ).astype(np.int64)
    probe_post_expression_episodes = summary.get(
        "probe_post_expression_episodes",
        probe_episode_solves,
    ).astype(np.int64)
    probe_shadow_post_expression_episodes = summary.get(
        "probe_shadow_post_expression_episodes",
        probe_shadow_episode_solves,
    ).astype(np.int64)
    probe_no_expression_probe_env_steps = summary.get(
        "probe_no_expression_probe_env_steps",
        probe_no_expression_total_env_steps,
    ).astype(np.int64)
    probe_no_expression_control_env_steps = summary.get(
        "probe_no_expression_control_env_steps",
        probe_no_expression_total_env_steps,
    ).astype(np.int64)
    probe_no_expression_post_expression_env_steps = summary.get(
        "probe_no_expression_post_expression_env_steps",
        probe_no_expression_control_env_steps,
    ).astype(np.int64)
    probe_no_expression_post_expression_episodes = summary.get(
        "probe_no_expression_post_expression_episodes",
        probe_no_expression_episode_solves,
    ).astype(np.int64)
    full_system_probe_env_steps = summary.get(
        "full_system_probe_env_steps",
        full_system_total_env_steps,
    ).astype(np.int64)
    full_system_control_env_steps = summary.get(
        "full_system_control_env_steps",
        full_system_total_env_steps,
    ).astype(np.int64)
    full_system_post_context_env_steps = summary.get(
        "full_system_post_context_env_steps",
        full_system_control_env_steps,
    ).astype(np.int64)
    full_system_post_context_episodes = summary.get(
        "full_system_post_context_episodes",
        full_system_episode_solves,
    ).astype(np.int64)
    full_system_oracle_probe_env_steps = summary.get(
        "full_system_oracle_probe_env_steps",
        full_system_oracle_total_env_steps,
    ).astype(np.int64)
    full_system_oracle_control_env_steps = summary.get(
        "full_system_oracle_control_env_steps",
        full_system_oracle_total_env_steps,
    ).astype(np.int64)
    full_system_oracle_post_context_env_steps = summary.get(
        "full_system_oracle_post_context_env_steps",
        full_system_oracle_control_env_steps,
    ).astype(np.int64)
    full_system_oracle_post_context_episodes = summary.get(
        "full_system_oracle_post_context_episodes",
        full_system_oracle_episode_solves,
    ).astype(np.int64)
    sim_fanout_probe_env_steps = summary.get(
        "sim_fanout_probe_env_steps",
        sim_fanout_total_env_steps,
    ).astype(np.int64)
    sim_fanout_control_env_steps = summary.get(
        "sim_fanout_control_env_steps",
        sim_fanout_total_env_steps,
    ).astype(np.int64)
    sim_fanout_post_context_env_steps = summary.get(
        "sim_fanout_post_context_env_steps",
        sim_fanout_control_env_steps,
    ).astype(np.int64)
    sim_fanout_post_context_episodes = summary.get(
        "sim_fanout_post_context_episodes",
        sim_fanout_episode_solves,
    ).astype(np.int64)
    probe_windows_total = summary.get(
        "probe_windows_total",
        np.zeros_like(probe_episode_solves),
    ).astype(np.int64)
    probe_expression_scale_median = summary.get(
        "probe_expression_scale_median",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_expression_scale_active_fraction = summary.get(
        "probe_expression_scale_active_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_ready_handoff_fraction = summary.get(
        "probe_fair_ready_handoff_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_expression_enabled_fraction = summary.get(
        "probe_fair_expression_enabled_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_expression_force_muted_fraction = summary.get(
        "probe_fair_expression_force_muted_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_ready_confidence_median = summary.get(
        "probe_fair_ready_confidence_median",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_fair_muted_confidence_median = summary.get(
        "probe_fair_muted_confidence_median",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_expression_ready_but_muted_fraction = summary.get(
        "probe_expression_ready_but_muted_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_shadow_expression_enabled_fraction = summary.get(
        "probe_shadow_expression_enabled_fraction",
        np.zeros_like(probe_shadow_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_shadow_expression_scale_median = summary.get(
        "probe_shadow_expression_scale_median",
        np.zeros_like(probe_shadow_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_shadow_confidence_median = summary.get(
        "probe_shadow_confidence_median",
        np.zeros_like(probe_shadow_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_shadow_strict_miss_fraction = summary.get(
        "probe_shadow_strict_miss_fraction",
        np.zeros_like(probe_shadow_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_second_probe_raw_future_gain_mean = summary.get(
        "probe_second_probe_raw_future_gain_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_second_probe_future_estimate_mean = summary.get(
        "probe_second_probe_future_estimate_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_second_probe_choice_future_gain_mean = summary.get(
        "probe_second_probe_choice_future_gain_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_family_coverage_satisfied_fraction = summary.get(
        "probe_family_coverage_satisfied_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_second_probe_value_driven_fraction = summary.get(
        "probe_second_probe_value_driven_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_uniformity_pressure_active_fraction = summary.get(
        "probe_uniformity_pressure_active_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_offline_gap_mean = summary.get(
        "probe_online_offline_gap_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_subset_stability_mean = summary.get(
        "probe_online_subset_stability_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_geometry_complete_fraction = summary.get(
        "probe_online_geometry_complete_fraction",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_split_latent_disagreement_mean = summary.get(
        "probe_online_split_latent_disagreement_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_split_retrieval_margin_deficit_mean = summary.get(
        "probe_online_split_retrieval_margin_deficit_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_online_leaveout_shift_mean = summary.get(
        "probe_online_leaveout_shift_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_teacher_action_agreement = summary.get(
        "probe_teacher_action_agreement",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_env_expression_delta = summary.get(
        "probe_env_expression_delta",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_forced_env_expression_delta = summary.get(
        "probe_forced_env_expression_delta",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_forced_env_expression_scale = summary.get(
        "probe_forced_env_expression_scale",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_strict_usage_status = summary.get(
        "probe_strict_usage_status",
        np.asarray(["unused"] * len(probe_episode_solves), dtype="U"),
    ).astype("U")
    full_system_zero_context_ablation_delta = summary.get(
        "full_system_zero_context_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_shuffled_context_ablation_delta = summary.get(
        "full_system_shuffled_context_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_stale_context_ablation_delta = summary.get(
        "full_system_stale_context_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_online_refinement_ablation_delta = summary.get(
        "full_system_online_refinement_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_frozen_context_ablation_delta = summary.get(
        "full_system_frozen_context_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_actor_only_ablation_delta = summary.get(
        "full_system_actor_only_ablation_delta",
        np.zeros_like(full_system_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_state_only_ablation_delta = summary.get(
        "full_system_state_only_ablation_delta",
        np.zeros_like(full_system_state_only_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_zero_context_ablation_delta = summary.get(
        "full_system_oracle_zero_context_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_shuffled_context_ablation_delta = summary.get(
        "full_system_oracle_shuffled_context_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_stale_context_ablation_delta = summary.get(
        "full_system_oracle_stale_context_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_online_refinement_ablation_delta = summary.get(
        "full_system_oracle_online_refinement_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_frozen_context_ablation_delta = summary.get(
        "full_system_oracle_frozen_context_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    full_system_oracle_actor_only_ablation_delta = summary.get(
        "full_system_oracle_actor_only_ablation_delta",
        np.zeros_like(full_system_oracle_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    probe_run_classification = summary.get(
        "probe_run_classification",
        np.asarray(["protocol_win"] * len(probe_episode_solves), dtype="U"),
    ).astype("U")
    belief_mode = summary.get(
        "belief_mode",
        np.asarray(["latent_pool"] * len(probe_episode_solves), dtype="U"),
    ).astype("U")
    belief_source = summary.get(
        "belief_source",
        np.asarray(
            ["sysid" if mode == "particle_sysid" else "learned" for mode in belief_mode],
            dtype="U",
        ),
    ).astype("U")
    belief_progress_index = summary.get(
        "belief_progress_index",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    system_id_progress_index = summary.get(
        "system_id_progress_index",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    sysid_trusted = summary.get(
        "sysid_trusted",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    sysid_validation_top1 = summary.get(
        "sysid_validation_top1",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    sysid_validation_margin = summary.get(
        "sysid_validation_margin",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    sysid_validation_nll = summary.get(
        "sysid_validation_nll",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_entropy_mean = summary.get(
        "particle_entropy_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_entropy_norm_mean = summary.get(
        "particle_entropy_norm_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_ess_ratio_mean = summary.get(
        "particle_ess_ratio_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_leaveout_shift_mean = summary.get(
        "particle_leaveout_shift_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    particle_subset_stability_mean = summary.get(
        "particle_subset_stability_mean",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    derived_particle_subset_stability = np.clip(
        1.0 - particle_leaveout_shift_mean / PARTICLE_READINESS_LEAVEOUT_SCALE,
        0.0,
        1.0,
    ).astype(np.float32)
    particle_subset_stability_mean = np.maximum(
        particle_subset_stability_mean,
        derived_particle_subset_stability,
    ).astype(np.float32)
    latent_mechanics_fit = summary.get(
        "latent_mechanics_fit",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_split_top1 = summary.get(
        "latent_split_top1",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_cross_split_top1 = summary.get(
        "latent_cross_split_top1",
        latent_split_top1,
    ).astype(np.float32)
    latent_paired_split_top1 = summary.get(
        "latent_paired_split_top1",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_cross_split_mrr = summary.get(
        "latent_cross_split_mrr",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_paired_split_mrr = summary.get(
        "latent_paired_split_mrr",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_neighbor_alignment = summary.get(
        "latent_neighbor_alignment",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_gap_ratio = summary.get(
        "latent_gap_ratio",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_heldout_probe_error = summary.get(
        "latent_heldout_probe_error",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_probe_leakage = summary.get(
        "latent_probe_leakage",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_uncert_error_corr = summary.get(
        "latent_uncert_error_corr",
        np.zeros_like(probe_episode_solves, dtype=np.float32),
    ).astype(np.float32)
    latent_win_gate = {}
    latent_win_gate_raw = summary.get("latent_win_gate_json")
    if latent_win_gate_raw is not None:
        try:
            latent_win_gate = json.loads(
                str(
                    latent_win_gate_raw.item()
                    if getattr(latent_win_gate_raw, "shape", None) == ()
                    else latent_win_gate_raw
                )
            )
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            latent_win_gate = {}
    latent_win_gate_failure_reasons = []
    latent_win_gate_failure_reasons_raw = summary.get("latent_win_gate_failure_reasons_json")
    if latent_win_gate_failure_reasons_raw is not None:
        try:
            latent_win_gate_failure_reasons = json.loads(
                str(
                    latent_win_gate_failure_reasons_raw.item()
                    if latent_win_gate_failure_reasons_raw.shape == ()
                    else latent_win_gate_failure_reasons_raw
                )
            )
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            latent_win_gate_failure_reasons = []
    probe_stop_reasons_rows = load_optional_json_rows(summary.get("probe_stop_reasons_json"))
    probe_final_stop_reason = summary.get(
        "probe_final_stop_reason",
        np.asarray([""] * len(probe_episode_solves), dtype="U"),
    ).astype("U")
    probe_family_expected_gain_rows = load_optional_json_rows(summary.get("probe_family_expected_gain_json"))
    probe_family_realized_gain_rows = load_optional_json_rows(summary.get("probe_family_realized_gain_json"))
    probe_family_future_error_rows = load_optional_json_rows(summary.get("probe_family_future_error_json"))
    probe_family_selection_count_rows = load_optional_json_rows(summary.get("probe_family_selection_count_json"))
    probe_readiness_reason_rows = load_optional_json_rows(summary.get("probe_readiness_reason_counts_json"))
    probe_readiness_component_rows = load_optional_json_rows(summary.get("probe_readiness_component_means_json"))
    probe_fair_stop_blocker_rows = load_optional_json_rows(summary.get("probe_fair_stop_blocker_counts_json"))
    probe_shadow_blocker_rows = load_optional_json_rows(summary.get("probe_shadow_blocker_counts_json"))
    probe_second_probe_selection_rows = load_optional_json_rows(summary.get("probe_second_probe_selection_count_json"))
    probe_fair_handoff_probe_families_rows = load_optional_json_rows(
        summary.get("probe_fair_handoff_probe_families_json")
    )
    probe_readiness_component_timeline_rows = load_optional_json_rows(
        summary.get("probe_readiness_component_timeline_json")
    )
    probe_online_future_quality_trace_rows = load_optional_json_rows(
        summary.get("probe_online_future_quality_trace_json")
    )
    probe_online_subset_stability_trace_rows = load_optional_json_rows(
        summary.get("probe_online_subset_stability_trace_json")
    )
    probe_online_offline_gap_trace_rows = load_optional_json_rows(
        summary.get("probe_online_offline_gap_trace_json")
    )
    latent_support_diagnostic_rows = load_optional_json_rows(
        summary.get("latent_support_diagnostics_json")
    )
    representation_gate_rows = load_optional_json_rows(
        summary.get("representation_gate_json")
    )
    latent_claim_rejection_rows = load_optional_json_rows(
        summary.get("latent_claim_rejection_reasons_json")
    )
    full_system_state_only_eval_returns_rows = load_optional_json_rows(
        summary.get("full_system_state_only_eval_returns_json")
    )
    full_system_learned_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_learned_eval_summary_json"))
    ]
    full_system_state_only_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_state_only_eval_summary_json"))
    ]
    full_system_zero_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_zero_context_eval_summary_json"))
    ]
    full_system_shuffled_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_shuffled_context_eval_summary_json"))
    ]
    full_system_stale_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_stale_context_eval_summary_json"))
    ]
    full_system_online_refinement_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_online_refinement_eval_summary_json"))
    ]
    full_system_frozen_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_frozen_context_eval_summary_json"))
    ]
    full_system_actor_only_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_actor_only_eval_summary_json"))
    ]
    full_system_oracle_learned_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_learned_eval_summary_json"))
    ]
    full_system_oracle_zero_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_zero_context_eval_summary_json"))
    ]
    full_system_oracle_shuffled_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_shuffled_context_eval_summary_json"))
    ]
    full_system_oracle_stale_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_stale_context_eval_summary_json"))
    ]
    full_system_oracle_online_refinement_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_online_refinement_eval_summary_json"))
    ]
    full_system_oracle_frozen_context_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_frozen_context_eval_summary_json"))
    ]
    full_system_oracle_actor_only_eval_summary_rows = [
        normalize_matched_eval_summary(row)
        for row in load_optional_json_rows(summary.get("full_system_oracle_actor_only_eval_summary_json"))
    ]
    while len(full_system_state_only_eval_summary_rows) < len(seeds):
        idx = len(full_system_state_only_eval_summary_rows)
        legacy_returns = (
            full_system_state_only_eval_returns_rows[idx]
            if idx < len(full_system_state_only_eval_returns_rows)
            and isinstance(full_system_state_only_eval_returns_rows[idx], list)
            else []
        )
        completed = int(full_system_state_only_completed_episodes[idx])
        total_steps = int(full_system_state_only_total_env_steps[idx])
        mean_total_steps = (
            float(total_steps) / float(max(completed, 1))
            if completed > 0
            else 0.0
        )
        full_system_state_only_eval_summary_rows.append(
            normalize_matched_eval_summary(
                {
                    "returns": legacy_returns,
                    "episode_total_env_steps": [mean_total_steps] * max(len(legacy_returns), completed),
                    "mean_return": float(np.mean(np.asarray(legacy_returns, dtype=np.float32))) if legacy_returns else 0.0,
                    "mean_total_env_steps": mean_total_steps,
                    "solved_count": 0,
                    "fixture_count": max(len(legacy_returns), completed),
                }
            )
        )

    rows = []
    for idx, seed in enumerate(seeds):
        learned_eval_summary = (
            full_system_learned_eval_summary_rows[idx]
            if idx < len(full_system_learned_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        state_only_eval_summary = (
            full_system_state_only_eval_summary_rows[idx]
            if idx < len(full_system_state_only_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        zero_context_eval_summary = (
            full_system_zero_context_eval_summary_rows[idx]
            if idx < len(full_system_zero_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        shuffled_context_eval_summary = (
            full_system_shuffled_context_eval_summary_rows[idx]
            if idx < len(full_system_shuffled_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        stale_context_eval_summary = (
            full_system_stale_context_eval_summary_rows[idx]
            if idx < len(full_system_stale_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        online_refinement_eval_summary = (
            full_system_online_refinement_eval_summary_rows[idx]
            if idx < len(full_system_online_refinement_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        frozen_context_eval_summary = (
            full_system_frozen_context_eval_summary_rows[idx]
            if idx < len(full_system_frozen_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        actor_only_eval_summary = (
            full_system_actor_only_eval_summary_rows[idx]
            if idx < len(full_system_actor_only_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_learned_eval_summary = (
            full_system_oracle_learned_eval_summary_rows[idx]
            if idx < len(full_system_oracle_learned_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_zero_context_eval_summary = (
            full_system_oracle_zero_context_eval_summary_rows[idx]
            if idx < len(full_system_oracle_zero_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_shuffled_context_eval_summary = (
            full_system_oracle_shuffled_context_eval_summary_rows[idx]
            if idx < len(full_system_oracle_shuffled_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_stale_context_eval_summary = (
            full_system_oracle_stale_context_eval_summary_rows[idx]
            if idx < len(full_system_oracle_stale_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_online_refinement_eval_summary = (
            full_system_oracle_online_refinement_eval_summary_rows[idx]
            if idx < len(full_system_oracle_online_refinement_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_frozen_context_eval_summary = (
            full_system_oracle_frozen_context_eval_summary_rows[idx]
            if idx < len(full_system_oracle_frozen_context_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        oracle_actor_only_eval_summary = (
            full_system_oracle_actor_only_eval_summary_rows[idx]
            if idx < len(full_system_oracle_actor_only_eval_summary_rows)
            else normalize_matched_eval_summary({})
        )
        support_diag = (
            latent_support_diagnostic_rows[idx]
            if idx < len(latent_support_diagnostic_rows)
            else {}
        )
        latent_claim_rejections = (
            latent_claim_rejection_rows[idx]
            if idx < len(latent_claim_rejection_rows)
            else []
        )
        if not isinstance(latent_claim_rejections, list):
            latent_claim_rejections = []
        rows.append(
            {
                "seed": int(seed),
                "baseline_episode_solve": int(baseline_episode_solves[idx]),
                "probe_episode_solve": int(probe_episode_solves[idx]),
                "probe_shadow_episode_solve": int(probe_shadow_episode_solves[idx]),
                "probe_no_expression_episode_solve": int(probe_no_expression_episode_solves[idx]),
                "full_system_episode_solve": int(full_system_episode_solves[idx]),
                "full_system_state_only_episode_solve": int(full_system_state_only_episode_solves[idx]),
                "full_system_oracle_episode_solve": int(full_system_oracle_episode_solves[idx]),
                "sim_fanout_episode_solve": int(sim_fanout_episode_solves[idx]),
                "baseline_step_solve": int(baseline_step_solves[idx]),
                "probe_step_solve": int(probe_step_solves[idx]),
                "probe_shadow_step_solve": int(probe_shadow_step_solves[idx]),
                "probe_no_expression_step_solve": int(probe_no_expression_step_solves[idx]),
                "full_system_step_solve": int(full_system_step_solves[idx]),
                "full_system_state_only_step_solve": int(full_system_state_only_step_solves[idx]),
                "full_system_oracle_step_solve": int(full_system_oracle_step_solves[idx]),
                "sim_fanout_step_solve": int(sim_fanout_step_solves[idx]),
                "probe_episode_savings_vs_baseline": sample_savings_or_none(
                    int(baseline_episode_solves[idx]),
                    int(probe_episode_solves[idx]),
                ),
                "probe_step_savings_vs_baseline": sample_savings_or_none(
                    int(baseline_step_solves[idx]),
                    int(probe_step_solves[idx]),
                ),
                "probe_episode_savings_vs_no_expression": sample_savings_or_none(
                    int(probe_no_expression_episode_solves[idx]),
                    int(probe_episode_solves[idx]),
                ),
                "probe_step_savings_vs_no_expression": sample_savings_or_none(
                    int(probe_no_expression_step_solves[idx]),
                    int(probe_step_solves[idx]),
                ),
                "full_system_step_regret_vs_sim_fanout": (
                    None
                    if int(full_system_step_solves[idx]) < 0 or int(sim_fanout_step_solves[idx]) < 0
                    else int(full_system_step_solves[idx]) - int(sim_fanout_step_solves[idx])
                ),
                "baseline_total_env_steps": int(baseline_total_env_steps[idx]),
                "probe_total_env_steps": int(probe_total_env_steps[idx]),
                "probe_shadow_total_env_steps": int(probe_shadow_total_env_steps[idx]),
                "probe_no_expression_total_env_steps": int(probe_no_expression_total_env_steps[idx]),
                "full_system_total_env_steps": int(full_system_total_env_steps[idx]),
                "full_system_state_only_total_env_steps": int(full_system_state_only_total_env_steps[idx]),
                "full_system_oracle_total_env_steps": int(full_system_oracle_total_env_steps[idx]),
                "sim_fanout_total_env_steps": int(sim_fanout_total_env_steps[idx]),
                "baseline_control_env_steps": int(baseline_control_env_steps[idx]),
                "probe_probe_env_steps": int(probe_probe_env_steps[idx]),
                "probe_control_env_steps": int(probe_control_env_steps[idx]),
                "probe_post_expression_env_steps": int(probe_post_expression_env_steps[idx]),
                "probe_post_expression_episodes": int(probe_post_expression_episodes[idx]),
                "probe_shadow_probe_env_steps": int(probe_shadow_probe_env_steps[idx]),
                "probe_shadow_control_env_steps": int(probe_shadow_control_env_steps[idx]),
                "probe_shadow_post_expression_env_steps": int(probe_shadow_post_expression_env_steps[idx]),
                "probe_shadow_post_expression_episodes": int(probe_shadow_post_expression_episodes[idx]),
                "probe_no_expression_probe_env_steps": int(probe_no_expression_probe_env_steps[idx]),
                "probe_no_expression_control_env_steps": int(probe_no_expression_control_env_steps[idx]),
                "probe_no_expression_post_expression_env_steps": int(probe_no_expression_post_expression_env_steps[idx]),
                "probe_no_expression_post_expression_episodes": int(probe_no_expression_post_expression_episodes[idx]),
                "full_system_probe_env_steps": int(full_system_probe_env_steps[idx]),
                "full_system_control_env_steps": int(full_system_control_env_steps[idx]),
                "full_system_post_context_env_steps": int(full_system_post_context_env_steps[idx]),
                "full_system_post_context_episodes": int(full_system_post_context_episodes[idx]),
                "full_system_oracle_probe_env_steps": int(full_system_oracle_probe_env_steps[idx]),
                "full_system_oracle_control_env_steps": int(full_system_oracle_control_env_steps[idx]),
                "full_system_oracle_post_context_env_steps": int(full_system_oracle_post_context_env_steps[idx]),
                "full_system_oracle_post_context_episodes": int(full_system_oracle_post_context_episodes[idx]),
                "sim_fanout_probe_env_steps": int(sim_fanout_probe_env_steps[idx]),
                "sim_fanout_control_env_steps": int(sim_fanout_control_env_steps[idx]),
                "sim_fanout_post_context_env_steps": int(sim_fanout_post_context_env_steps[idx]),
                "sim_fanout_post_context_episodes": int(sim_fanout_post_context_episodes[idx]),
                "probe_encoder_steps": int(probe_encoder_steps[idx]),
                "probe_windows_total": int(probe_windows_total[idx]),
                "baseline_completed_episodes": int(baseline_completed_episodes[idx]),
                "probe_completed_episodes": int(probe_completed_episodes[idx]),
                "probe_shadow_completed_episodes": int(probe_shadow_completed_episodes[idx]),
                "probe_no_expression_completed_episodes": int(probe_no_expression_completed_episodes[idx]),
                "full_system_completed_episodes": int(full_system_completed_episodes[idx]),
                "full_system_state_only_completed_episodes": int(full_system_state_only_completed_episodes[idx]),
                "full_system_oracle_completed_episodes": int(full_system_oracle_completed_episodes[idx]),
                "sim_fanout_completed_episodes": int(sim_fanout_completed_episodes[idx]),
                "probe_no_expression_available": bool(probe_no_expression_available[idx]),
                "latent_claim_valid": bool(latent_claim_valid[idx]),
                "latent_claim_rejection_reasons": [
                    str(reason) for reason in latent_claim_rejections
                ],
                "full_system_controller_style": str(full_system_controller_style[idx]),
                "full_system_oracle_controller_style": str(full_system_oracle_controller_style[idx]),
                "sim_fanout_controller_style": str(sim_fanout_controller_style[idx]),
                "probe_expression_scale_median": float(probe_expression_scale_median[idx]),
                "probe_expression_scale_active_fraction": float(probe_expression_scale_active_fraction[idx]),
                "probe_fair_ready_handoff_fraction": float(probe_fair_ready_handoff_fraction[idx]),
                "probe_fair_expression_enabled_fraction": float(probe_fair_expression_enabled_fraction[idx]),
                "probe_fair_expression_force_muted_fraction": float(probe_fair_expression_force_muted_fraction[idx]),
                "probe_fair_ready_confidence_median": float(probe_fair_ready_confidence_median[idx]),
                "probe_fair_muted_confidence_median": float(probe_fair_muted_confidence_median[idx]),
                "probe_expression_ready_but_muted_fraction": float(probe_expression_ready_but_muted_fraction[idx]),
                "probe_shadow_expression_enabled_fraction": float(probe_shadow_expression_enabled_fraction[idx]),
                "probe_shadow_expression_scale_median": float(probe_shadow_expression_scale_median[idx]),
                "probe_shadow_confidence_median": float(probe_shadow_confidence_median[idx]),
                "probe_shadow_strict_miss_fraction": float(probe_shadow_strict_miss_fraction[idx]),
                "probe_run_classification": str(probe_run_classification[idx]),
                "belief_mode": str(belief_mode[idx]),
                "belief_source": str(belief_source[idx]),
                "belief_progress_index": float(belief_progress_index[idx]),
                "system_id_progress_index": float(system_id_progress_index[idx]),
                "sysid_trusted": bool(sysid_trusted[idx] > 0.5),
                "sysid_validation_top1": float(sysid_validation_top1[idx]),
                "sysid_validation_margin": float(sysid_validation_margin[idx]),
                "sysid_validation_nll": float(sysid_validation_nll[idx]),
                "particle_entropy_mean": float(particle_entropy_mean[idx]),
                "particle_entropy_norm_mean": float(particle_entropy_norm_mean[idx]),
                "particle_ess_ratio_mean": float(particle_ess_ratio_mean[idx]),
                "particle_leaveout_shift_mean": float(particle_leaveout_shift_mean[idx]),
                "particle_subset_stability_mean": float(particle_subset_stability_mean[idx]),
                "latent_mechanics_fit": float(latent_mechanics_fit[idx]),
                "latent_split_top1": float(latent_split_top1[idx]),
                "latent_cross_split_top1": float(latent_cross_split_top1[idx]),
                "latent_paired_split_top1": float(latent_paired_split_top1[idx]),
                "latent_cross_split_mrr": float(latent_cross_split_mrr[idx]),
                "latent_paired_split_mrr": float(latent_paired_split_mrr[idx]),
                "latent_neighbor_alignment": float(latent_neighbor_alignment[idx]),
                "latent_gap_ratio": float(latent_gap_ratio[idx]),
                "latent_heldout_probe_error": float(latent_heldout_probe_error[idx]),
                "latent_probe_leakage": float(latent_probe_leakage[idx]),
                "latent_uncert_error_corr": float(latent_uncert_error_corr[idx]),
                "latent_support_diagnostics": support_diag,
                "representation_gate": (
                    representation_gate_rows[idx]
                    if idx < len(representation_gate_rows)
                    else {}
                ),
                "latent_center_window_share": float(support_diag.get("center_window_share", 0.0)),
                "latent_directional_window_share": float(support_diag.get("directional_window_share", 0.0)),
                "latent_mechanics_window_share": float(support_diag.get("mechanics_window_share", 0.0)),
                "latent_passive_window_share": float(support_diag.get("passive_window_share", 0.0)),
                "latent_stress_window_share": float(support_diag.get("stress_window_share", 0.0)),
                "latent_window_mode_leakage": float(support_diag.get("window_mode_leakage", 0.0)),
                "latent_env_mode_leakage": float(support_diag.get("env_mode_leakage", 0.0)),
                "probe_stop_reasons": probe_stop_reasons_rows[idx] if idx < len(probe_stop_reasons_rows) else {},
                "probe_final_stop_reason": str(probe_final_stop_reason[idx]),
                "probe_family_expected_gain": probe_family_expected_gain_rows[idx] if idx < len(probe_family_expected_gain_rows) else {},
                "probe_family_realized_gain": probe_family_realized_gain_rows[idx] if idx < len(probe_family_realized_gain_rows) else {},
                "probe_family_future_error": probe_family_future_error_rows[idx] if idx < len(probe_family_future_error_rows) else {},
                "probe_family_selection_count": probe_family_selection_count_rows[idx] if idx < len(probe_family_selection_count_rows) else {},
                "probe_readiness_reason_counts": probe_readiness_reason_rows[idx] if idx < len(probe_readiness_reason_rows) else {},
                "probe_readiness_component_means": probe_readiness_component_rows[idx] if idx < len(probe_readiness_component_rows) else {},
                "probe_fair_stop_blocker_counts": probe_fair_stop_blocker_rows[idx] if idx < len(probe_fair_stop_blocker_rows) else {},
                "probe_shadow_blocker_counts": probe_shadow_blocker_rows[idx] if idx < len(probe_shadow_blocker_rows) else {},
                "probe_second_probe_selection_count": probe_second_probe_selection_rows[idx] if idx < len(probe_second_probe_selection_rows) else {},
                "probe_second_probe_raw_future_gain_mean": float(probe_second_probe_raw_future_gain_mean[idx]),
                "probe_second_probe_future_estimate_mean": float(probe_second_probe_future_estimate_mean[idx]),
                "probe_second_probe_choice_future_gain_mean": float(probe_second_probe_choice_future_gain_mean[idx]),
                "probe_family_coverage_satisfied_fraction": float(probe_family_coverage_satisfied_fraction[idx]),
                "probe_second_probe_value_driven_fraction": float(probe_second_probe_value_driven_fraction[idx]),
                "probe_uniformity_pressure_active_fraction": float(probe_uniformity_pressure_active_fraction[idx]),
                "probe_env_expression_delta": float(probe_env_expression_delta[idx]),
                "probe_forced_env_expression_delta": float(probe_forced_env_expression_delta[idx]),
                "probe_forced_env_expression_scale": float(probe_forced_env_expression_scale[idx]),
                "probe_strict_usage_status": str(probe_strict_usage_status[idx]),
                "probe_fair_handoff_probe_families": (
                    probe_fair_handoff_probe_families_rows[idx]
                    if idx < len(probe_fair_handoff_probe_families_rows)
                    else []
                ),
                "probe_readiness_component_timeline": (
                    probe_readiness_component_timeline_rows[idx]
                    if idx < len(probe_readiness_component_timeline_rows)
                    else []
                ),
                "probe_online_future_quality_trace": (
                    probe_online_future_quality_trace_rows[idx]
                    if idx < len(probe_online_future_quality_trace_rows)
                    else []
                ),
                "probe_online_subset_stability_trace": (
                    probe_online_subset_stability_trace_rows[idx]
                    if idx < len(probe_online_subset_stability_trace_rows)
                    else []
                ),
                "probe_online_offline_gap_trace": (
                    probe_online_offline_gap_trace_rows[idx]
                    if idx < len(probe_online_offline_gap_trace_rows)
                    else []
                ),
                "probe_online_subset_stability_mean": float(probe_online_subset_stability_mean[idx]),
                "probe_online_offline_gap_mean": float(probe_online_offline_gap_mean[idx]),
                "probe_online_geometry_complete_fraction": float(probe_online_geometry_complete_fraction[idx]),
                "probe_online_split_latent_disagreement_mean": float(probe_online_split_latent_disagreement_mean[idx]),
                "probe_online_split_retrieval_margin_deficit_mean": float(probe_online_split_retrieval_margin_deficit_mean[idx]),
                "probe_online_leaveout_shift_mean": float(probe_online_leaveout_shift_mean[idx]),
                "probe_teacher_action_agreement": float(probe_teacher_action_agreement[idx]),
                "full_system_learned_eval_summary": learned_eval_summary,
                "full_system_state_only_eval_summary": state_only_eval_summary,
                "full_system_zero_context_eval_summary": zero_context_eval_summary,
                "full_system_shuffled_context_eval_summary": shuffled_context_eval_summary,
                "full_system_stale_context_eval_summary": stale_context_eval_summary,
                "full_system_online_refinement_eval_summary": online_refinement_eval_summary,
                "full_system_frozen_context_eval_summary": frozen_context_eval_summary,
                "full_system_actor_only_eval_summary": actor_only_eval_summary,
                "full_system_state_only_eval_returns": (
                    full_system_state_only_eval_returns_rows[idx]
                    if idx < len(full_system_state_only_eval_returns_rows)
                    else []
                ),
                "full_system_state_only_ablation_delta": float(full_system_state_only_ablation_delta[idx]),
                "full_system_zero_context_ablation_delta": float(full_system_zero_context_ablation_delta[idx]),
                "full_system_shuffled_context_ablation_delta": float(full_system_shuffled_context_ablation_delta[idx]),
                "full_system_stale_context_ablation_delta": float(full_system_stale_context_ablation_delta[idx]),
                "full_system_online_refinement_ablation_delta": float(full_system_online_refinement_ablation_delta[idx]),
                "full_system_frozen_context_ablation_delta": float(full_system_frozen_context_ablation_delta[idx]),
                "full_system_actor_only_ablation_delta": float(full_system_actor_only_ablation_delta[idx]),
                "full_system_oracle_zero_context_ablation_delta": float(full_system_oracle_zero_context_ablation_delta[idx]),
                "full_system_oracle_shuffled_context_ablation_delta": float(full_system_oracle_shuffled_context_ablation_delta[idx]),
                "full_system_oracle_stale_context_ablation_delta": float(full_system_oracle_stale_context_ablation_delta[idx]),
                "full_system_oracle_online_refinement_ablation_delta": float(full_system_oracle_online_refinement_ablation_delta[idx]),
                "full_system_oracle_frozen_context_ablation_delta": float(full_system_oracle_frozen_context_ablation_delta[idx]),
                "full_system_oracle_actor_only_ablation_delta": float(full_system_oracle_actor_only_ablation_delta[idx]),
                "full_system_oracle_learned_eval_summary": oracle_learned_eval_summary,
                "full_system_oracle_zero_context_eval_summary": oracle_zero_context_eval_summary,
                "full_system_oracle_shuffled_context_eval_summary": oracle_shuffled_context_eval_summary,
                "full_system_oracle_stale_context_eval_summary": oracle_stale_context_eval_summary,
                "full_system_oracle_online_refinement_eval_summary": oracle_online_refinement_eval_summary,
                "full_system_oracle_frozen_context_eval_summary": oracle_frozen_context_eval_summary,
                "full_system_oracle_actor_only_eval_summary": oracle_actor_only_eval_summary,
                "probe_strictly_muted_but_shadow_eligible": bool(
                    float(probe_fair_expression_enabled_fraction[idx]) <= 0.0
                    and float(probe_shadow_expression_enabled_fraction[idx]) > 0.0
                ),
                "probe_shadow_available": bool(
                    int(probe_shadow_completed_episodes[idx]) > 0
                    or int(probe_shadow_episode_solves[idx]) >= 0
                ),
                "full_system_available": bool(
                    int(full_system_completed_episodes[idx]) > 0
                    or int(full_system_episode_solves[idx]) >= 0
                ),
                "full_system_state_only_available": bool(state_only_eval_summary["available"]),
                "full_system_zero_context_available": bool(zero_context_eval_summary["available"]),
                "full_system_shuffled_context_available": bool(shuffled_context_eval_summary["available"]),
                "full_system_stale_context_available": bool(stale_context_eval_summary["available"]),
                "full_system_frozen_context_available": bool(frozen_context_eval_summary["available"]),
                "full_system_oracle_available": bool(
                    int(full_system_oracle_completed_episodes[idx]) > 0
                    or int(full_system_oracle_episode_solves[idx]) >= 0
                ),
                "full_system_oracle_frozen_context_available": bool(
                    oracle_frozen_context_eval_summary["available"]
                ),
                "sim_fanout_available": bool(
                    int(sim_fanout_completed_episodes[idx]) > 0
                    or int(sim_fanout_episode_solves[idx]) >= 0
                ),
            }
        )

    classification_counts: dict[str, int] = {}
    for label in np.asarray(probe_run_classification, dtype="U").tolist():
        key = str(label)
        classification_counts[key] = classification_counts.get(key, 0) + 1
    dominant_classification = max(
        classification_counts.items(),
        key=lambda item: (item[1], item[0]),
        default=("protocol_win", 0),
    )[0]
    readiness_reason_totals = aggregate_json_counter_rows(probe_readiness_reason_rows)
    readiness_component_means = average_json_metric_rows(probe_readiness_component_rows)
    fair_stop_blocker_totals = aggregate_json_counter_rows(probe_fair_stop_blocker_rows)
    shadow_blocker_totals = aggregate_json_counter_rows(probe_shadow_blocker_rows)
    second_probe_selection_totals = aggregate_json_counter_rows(probe_second_probe_selection_rows)
    fair_handoff_pair_totals = aggregate_json_list_rows(probe_fair_handoff_probe_families_rows)
    strict_usage_counts: dict[str, int] = {}
    for label in np.asarray(probe_strict_usage_status, dtype="U").tolist():
        key = str(label)
        strict_usage_counts[key] = strict_usage_counts.get(key, 0) + 1
    dominant_strict_usage_status = max(
        strict_usage_counts.items(),
        key=lambda item: (item[1], item[0]),
        default=("unused", 0),
    )[0]
    probe_env_expression_delta_summary = summarize_solve_array(
        probe_episode_solves,
        probe_completed_episodes,
    )
    baseline_episode_summary = summarize_solve_array(
        baseline_episode_solves,
        baseline_completed_episodes,
    )
    probe_env_expression_delta_mean = (
        float(np.mean(probe_env_expression_delta))
        if probe_env_expression_delta.size
        else 0.0
    )
    honesty_headline = ""
    if (
        dominant_strict_usage_status == "unused"
        and probe_env_expression_delta_summary["median"] < baseline_episode_summary["median"]
    ):
        honesty_headline = "Episode win without strict latent usage"
    elif probe_env_expression_delta_mean <= 0.0:
        honesty_headline = "Env expression harmful under matched eval"

    research_metrics = build_benchmark_research_metrics(
        baseline_episode_solves=baseline_episode_solves,
        baseline_step_solves=baseline_step_solves,
        baseline_total_env_steps=baseline_total_env_steps,
        baseline_completed_episodes=baseline_completed_episodes,
        probe_episode_solves=probe_episode_solves,
        probe_step_solves=probe_step_solves,
        probe_total_env_steps=probe_total_env_steps,
        probe_completed_episodes=probe_completed_episodes,
        probe_no_expression_episode_solves=probe_no_expression_episode_solves,
        probe_no_expression_step_solves=probe_no_expression_step_solves,
        probe_no_expression_total_env_steps=probe_no_expression_total_env_steps,
        probe_no_expression_completed_episodes=probe_no_expression_completed_episodes,
        full_system_episode_solves=full_system_episode_solves,
        full_system_step_solves=full_system_step_solves,
        full_system_total_env_steps=full_system_total_env_steps,
        full_system_completed_episodes=full_system_completed_episodes,
        sim_fanout_episode_solves=sim_fanout_episode_solves,
        sim_fanout_step_solves=sim_fanout_step_solves,
        sim_fanout_total_env_steps=sim_fanout_total_env_steps,
        sim_fanout_completed_episodes=sim_fanout_completed_episodes,
        probe_probe_env_steps=probe_probe_env_steps,
        probe_encoder_steps=probe_encoder_steps,
        probe_control_env_steps=probe_control_env_steps,
        probe_post_expression_env_steps=probe_post_expression_env_steps,
        baseline_best_returns=baseline_best_returns,
        probe_best_returns=probe_best_returns,
        baseline_peak_env_steps=baseline_peak_env_steps,
        probe_peak_env_steps_with_encoder=probe_peak_env_steps_with_encoder,
    )
    loss_attribution = build_loss_attribution_metrics(
        benchmark_profile=benchmark_profile,
        rows=rows,
        latent_win_gate=latent_win_gate,
    )

    return {
        "name": path.name,
        "artifact_mtime": float(path.stat().st_mtime),
        "env_name": env_name,
        "env_display_name": None if env_name is None else get_env_display_name(env_name),
        "benchmark_profile": benchmark_profile,
        "benchmark_mode": load_optional_string(summary, "benchmark_mode"),
        "probe_budget_mode": load_optional_string(summary, "probe_budget_mode"),
        "full_system_controller_style": next(
            (str(value) for value in full_system_controller_style.tolist() if str(value)),
            "",
        ),
        "full_system_oracle_controller_style": next(
            (str(value) for value in full_system_oracle_controller_style.tolist() if str(value)),
            "",
        ),
        "sim_fanout_controller_style": next(
            (str(value) for value in sim_fanout_controller_style.tolist() if str(value)),
            "",
        ),
        "run_classification": dominant_classification,
        "probe_strict_usage_status": dominant_strict_usage_status,
        "probe_honesty_headline": honesty_headline,
        "latent_win_gate": latent_win_gate,
        "latent_win_gate_failure_reasons": latent_win_gate_failure_reasons,
        "probe_no_expression_available": bool(
            probe_no_expression_available.size > 0
            and np.all(probe_no_expression_available > 0)
        ),
        "latent_claim_valid": bool(
            latent_claim_valid.size > 0
            and np.all(latent_claim_valid > 0)
        ),
        "probe_shadow_available": bool(
            np.any(probe_shadow_completed_episodes > 0)
            or np.any(probe_shadow_episode_solves >= 0)
        ),
        "full_system_available": bool(
            np.any(full_system_completed_episodes > 0)
            or np.any(full_system_episode_solves >= 0)
        ),
        "full_system_state_only_available": bool(
            any(row["available"] for row in full_system_state_only_eval_summary_rows)
        ),
        "full_system_zero_context_available": bool(
            any(row["available"] for row in full_system_zero_context_eval_summary_rows)
        ),
        "full_system_shuffled_context_available": bool(
            any(row["available"] for row in full_system_shuffled_context_eval_summary_rows)
        ),
        "full_system_stale_context_available": bool(
            any(row["available"] for row in full_system_stale_context_eval_summary_rows)
        ),
        "full_system_frozen_context_available": bool(
            any(row["available"] for row in full_system_frozen_context_eval_summary_rows)
        ),
        "full_system_oracle_available": bool(
            np.any(full_system_oracle_completed_episodes > 0)
            or np.any(full_system_oracle_episode_solves >= 0)
        ),
        "full_system_oracle_frozen_context_available": bool(
            any(row["available"] for row in full_system_oracle_frozen_context_eval_summary_rows)
        ),
        "sim_fanout_available": bool(
            np.any(sim_fanout_completed_episodes > 0)
            or np.any(sim_fanout_episode_solves >= 0)
        ),
        "rows": rows,
        "research_metrics": research_metrics,
        "loss_attribution": loss_attribution,
        "summaries": {
            "baseline_episode": summarize_solve_array(
                baseline_episode_solves,
                baseline_completed_episodes,
            ),
            "probe_episode": summarize_solve_array(
                probe_episode_solves,
                probe_completed_episodes,
            ),
            "belief_progress_index": {
                "median": float(np.median(belief_progress_index)) if belief_progress_index.size else 0.0,
                "mean": float(np.mean(belief_progress_index)) if belief_progress_index.size else 0.0,
                "count": int(belief_progress_index.size),
            },
            "system_id": {
                "available": bool(
                    belief_mode.size
                    and (
                        np.any(belief_mode == "particle_sysid")
                        or np.any(sysid_validation_top1 > 0.0)
                    )
                ),
                "mode": (
                    "particle_sysid"
                    if bool(np.any(belief_mode == "particle_sysid"))
                    else "latent_pool"
                ),
                "belief_source_counts": {
                    str(source): int(np.sum(belief_source == source))
                    for source in sorted(set(str(value) for value in belief_source.tolist()))
                },
                "progress_median": float(np.median(system_id_progress_index)) if system_id_progress_index.size else 0.0,
                "progress_mean": float(np.mean(system_id_progress_index)) if system_id_progress_index.size else 0.0,
                "trusted_fraction": float(np.mean(sysid_trusted)) if sysid_trusted.size else 0.0,
                "validation_top1_median": float(np.median(sysid_validation_top1)) if sysid_validation_top1.size else 0.0,
                "validation_margin_median": float(np.median(sysid_validation_margin)) if sysid_validation_margin.size else 0.0,
                "validation_nll_median": float(np.median(sysid_validation_nll)) if sysid_validation_nll.size else 0.0,
                "particle_entropy_median": float(np.median(particle_entropy_mean)) if particle_entropy_mean.size else 0.0,
                "particle_ess_ratio_median": float(np.median(particle_ess_ratio_mean)) if particle_ess_ratio_mean.size else 0.0,
                "particle_leaveout_shift_median": float(np.median(particle_leaveout_shift_mean)) if particle_leaveout_shift_mean.size else 0.0,
                "particle_subset_stability_median": float(np.median(particle_subset_stability_mean)) if particle_subset_stability_mean.size else 0.0,
            },
            "latent_mechanics_fit": {
                "median": float(np.median(latent_mechanics_fit)) if latent_mechanics_fit.size else 0.0,
                "mean": float(np.mean(latent_mechanics_fit)) if latent_mechanics_fit.size else 0.0,
                "count": int(latent_mechanics_fit.size),
            },
            "latent_split_top1": {
                "median": float(np.median(latent_split_top1)) if latent_split_top1.size else 0.0,
                "mean": float(np.mean(latent_split_top1)) if latent_split_top1.size else 0.0,
                "count": int(latent_split_top1.size),
            },
            "latent_cross_split_top1": {
                "median": float(np.median(latent_cross_split_top1)) if latent_cross_split_top1.size else 0.0,
                "mean": float(np.mean(latent_cross_split_top1)) if latent_cross_split_top1.size else 0.0,
                "count": int(latent_cross_split_top1.size),
            },
            "latent_paired_split_top1": {
                "median": float(np.median(latent_paired_split_top1)) if latent_paired_split_top1.size else 0.0,
                "mean": float(np.mean(latent_paired_split_top1)) if latent_paired_split_top1.size else 0.0,
                "count": int(latent_paired_split_top1.size),
            },
            "latent_neighbor_alignment": {
                "median": float(np.median(latent_neighbor_alignment)) if latent_neighbor_alignment.size else 0.0,
                "mean": float(np.mean(latent_neighbor_alignment)) if latent_neighbor_alignment.size else 0.0,
                "count": int(latent_neighbor_alignment.size),
            },
            "latent_gap_ratio": {
                "median": float(np.median(latent_gap_ratio)) if latent_gap_ratio.size else 0.0,
                "mean": float(np.mean(latent_gap_ratio)) if latent_gap_ratio.size else 0.0,
                "count": int(latent_gap_ratio.size),
            },
            "latent_heldout_probe_error": {
                "median": float(np.median(latent_heldout_probe_error)) if latent_heldout_probe_error.size else 0.0,
                "mean": float(np.mean(latent_heldout_probe_error)) if latent_heldout_probe_error.size else 0.0,
                "count": int(latent_heldout_probe_error.size),
            },
            "latent_probe_leakage": {
                "median": float(np.median(latent_probe_leakage)) if latent_probe_leakage.size else 0.0,
                "mean": float(np.mean(latent_probe_leakage)) if latent_probe_leakage.size else 0.0,
                "count": int(latent_probe_leakage.size),
            },
            "latent_uncert_error_corr": {
                "median": float(np.median(latent_uncert_error_corr)) if latent_uncert_error_corr.size else 0.0,
                "mean": float(np.mean(latent_uncert_error_corr)) if latent_uncert_error_corr.size else 0.0,
                "count": int(latent_uncert_error_corr.size),
            },
            "latent_support_diagnostics": average_json_metric_rows(latent_support_diagnostic_rows),
            "probe_shadow_episode": summarize_solve_array(
                probe_shadow_episode_solves,
                probe_shadow_completed_episodes,
            ),
            "probe_no_expression_episode": summarize_solve_array(
                probe_no_expression_episode_solves,
                probe_no_expression_completed_episodes,
            ),
            "full_system_episode": summarize_solve_array(
                full_system_episode_solves,
                full_system_completed_episodes,
            ),
            "full_system_oracle_episode": summarize_solve_array(
                full_system_oracle_episode_solves,
                full_system_oracle_completed_episodes,
            ),
            "sim_fanout_episode": summarize_solve_array(
                sim_fanout_episode_solves,
                sim_fanout_completed_episodes,
            ),
            "baseline_steps": summarize_solve_array(
                baseline_step_solves,
                baseline_total_env_steps,
            ),
            "probe_steps": summarize_solve_array(
                probe_step_solves,
                probe_total_env_steps,
            ),
            "probe_shadow_steps": summarize_solve_array(
                probe_shadow_step_solves,
                probe_shadow_total_env_steps,
            ),
            "probe_no_expression_steps": summarize_solve_array(
                probe_no_expression_step_solves,
                probe_no_expression_total_env_steps,
            ),
            "full_system_steps": summarize_solve_array(
                full_system_step_solves,
                full_system_total_env_steps,
            ),
            "full_system_oracle_steps": summarize_solve_array(
                full_system_oracle_step_solves,
                full_system_oracle_total_env_steps,
            ),
            "sim_fanout_steps": summarize_solve_array(
                sim_fanout_step_solves,
                sim_fanout_total_env_steps,
            ),
            "probe_post_expression_steps": summarize_solve_array(
                probe_post_expression_env_steps,
                probe_total_env_steps,
            ),
            "probe_shadow_post_expression_steps": summarize_solve_array(
                probe_shadow_post_expression_env_steps,
                probe_shadow_total_env_steps,
            ),
            "full_system_post_context_steps": summarize_solve_array(
                full_system_post_context_env_steps,
                full_system_total_env_steps,
            ),
            "full_system_oracle_post_context_steps": summarize_solve_array(
                full_system_oracle_post_context_env_steps,
                full_system_oracle_total_env_steps,
            ),
            "sim_fanout_post_context_steps": summarize_solve_array(
                sim_fanout_post_context_env_steps,
                sim_fanout_total_env_steps,
            ),
            "probe_post_expression_episodes": summarize_solve_array(
                probe_post_expression_episodes,
                probe_completed_episodes,
            ),
            "probe_shadow_post_expression_episodes": summarize_solve_array(
                probe_shadow_post_expression_episodes,
                probe_shadow_completed_episodes,
            ),
            "full_system_post_context_episodes": summarize_solve_array(
                full_system_post_context_episodes,
                full_system_completed_episodes,
            ),
            "full_system_oracle_post_context_episodes": summarize_solve_array(
                full_system_oracle_post_context_episodes,
                full_system_oracle_completed_episodes,
            ),
            "sim_fanout_post_context_episodes": summarize_solve_array(
                sim_fanout_post_context_episodes,
                sim_fanout_completed_episodes,
            ),
            "probe_expression_scale_median": {
                "median": float(np.median(probe_expression_scale_median)) if probe_expression_scale_median.size else 0.0,
                "mean": float(np.mean(probe_expression_scale_median)) if probe_expression_scale_median.size else 0.0,
                "count": int(probe_expression_scale_median.size),
            },
            "probe_expression_scale_active_fraction": {
                "median": float(np.median(probe_expression_scale_active_fraction)) if probe_expression_scale_active_fraction.size else 0.0,
                "mean": float(np.mean(probe_expression_scale_active_fraction)) if probe_expression_scale_active_fraction.size else 0.0,
                "count": int(probe_expression_scale_active_fraction.size),
            },
            "probe_fair_ready_handoff_fraction": {
                "median": float(np.median(probe_fair_ready_handoff_fraction)) if probe_fair_ready_handoff_fraction.size else 0.0,
                "mean": float(np.mean(probe_fair_ready_handoff_fraction)) if probe_fair_ready_handoff_fraction.size else 0.0,
                "count": int(probe_fair_ready_handoff_fraction.size),
            },
            "probe_fair_expression_enabled_fraction": {
                "median": float(np.median(probe_fair_expression_enabled_fraction)) if probe_fair_expression_enabled_fraction.size else 0.0,
                "mean": float(np.mean(probe_fair_expression_enabled_fraction)) if probe_fair_expression_enabled_fraction.size else 0.0,
                "count": int(probe_fair_expression_enabled_fraction.size),
            },
            "probe_fair_expression_force_muted_fraction": {
                "median": float(np.median(probe_fair_expression_force_muted_fraction)) if probe_fair_expression_force_muted_fraction.size else 0.0,
                "mean": float(np.mean(probe_fair_expression_force_muted_fraction)) if probe_fair_expression_force_muted_fraction.size else 0.0,
                "count": int(probe_fair_expression_force_muted_fraction.size),
            },
            "probe_fair_ready_confidence_median": {
                "median": float(np.median(probe_fair_ready_confidence_median)) if probe_fair_ready_confidence_median.size else 0.0,
                "mean": float(np.mean(probe_fair_ready_confidence_median)) if probe_fair_ready_confidence_median.size else 0.0,
                "count": int(probe_fair_ready_confidence_median.size),
            },
            "probe_fair_muted_confidence_median": {
                "median": float(np.median(probe_fair_muted_confidence_median)) if probe_fair_muted_confidence_median.size else 0.0,
                "mean": float(np.mean(probe_fair_muted_confidence_median)) if probe_fair_muted_confidence_median.size else 0.0,
                "count": int(probe_fair_muted_confidence_median.size),
            },
            "probe_expression_ready_but_muted_fraction": {
                "median": float(np.median(probe_expression_ready_but_muted_fraction)) if probe_expression_ready_but_muted_fraction.size else 0.0,
                "mean": float(np.mean(probe_expression_ready_but_muted_fraction)) if probe_expression_ready_but_muted_fraction.size else 0.0,
                "count": int(probe_expression_ready_but_muted_fraction.size),
            },
            "probe_shadow_expression_enabled_fraction": {
                "median": float(np.median(probe_shadow_expression_enabled_fraction)) if probe_shadow_expression_enabled_fraction.size else 0.0,
                "mean": float(np.mean(probe_shadow_expression_enabled_fraction)) if probe_shadow_expression_enabled_fraction.size else 0.0,
                "count": int(probe_shadow_expression_enabled_fraction.size),
            },
            "probe_shadow_expression_scale_median": {
                "median": float(np.median(probe_shadow_expression_scale_median)) if probe_shadow_expression_scale_median.size else 0.0,
                "mean": float(np.mean(probe_shadow_expression_scale_median)) if probe_shadow_expression_scale_median.size else 0.0,
                "count": int(probe_shadow_expression_scale_median.size),
            },
            "probe_shadow_confidence_median": {
                "median": float(np.median(probe_shadow_confidence_median)) if probe_shadow_confidence_median.size else 0.0,
                "mean": float(np.mean(probe_shadow_confidence_median)) if probe_shadow_confidence_median.size else 0.0,
                "count": int(probe_shadow_confidence_median.size),
            },
            "probe_shadow_strict_miss_fraction": {
                "median": float(np.median(probe_shadow_strict_miss_fraction)) if probe_shadow_strict_miss_fraction.size else 0.0,
                "mean": float(np.mean(probe_shadow_strict_miss_fraction)) if probe_shadow_strict_miss_fraction.size else 0.0,
                "count": int(probe_shadow_strict_miss_fraction.size),
            },
            "probe_second_probe_raw_future_gain_mean": {
                "median": float(np.median(probe_second_probe_raw_future_gain_mean)) if probe_second_probe_raw_future_gain_mean.size else 0.0,
                "mean": float(np.mean(probe_second_probe_raw_future_gain_mean)) if probe_second_probe_raw_future_gain_mean.size else 0.0,
                "count": int(probe_second_probe_raw_future_gain_mean.size),
            },
            "probe_second_probe_future_estimate_mean": {
                "median": float(np.median(probe_second_probe_future_estimate_mean)) if probe_second_probe_future_estimate_mean.size else 0.0,
                "mean": float(np.mean(probe_second_probe_future_estimate_mean)) if probe_second_probe_future_estimate_mean.size else 0.0,
                "count": int(probe_second_probe_future_estimate_mean.size),
            },
            "probe_second_probe_choice_future_gain_mean": {
                "median": float(np.median(probe_second_probe_choice_future_gain_mean)) if probe_second_probe_choice_future_gain_mean.size else 0.0,
                "mean": float(np.mean(probe_second_probe_choice_future_gain_mean)) if probe_second_probe_choice_future_gain_mean.size else 0.0,
                "count": int(probe_second_probe_choice_future_gain_mean.size),
            },
            "probe_family_coverage_satisfied_fraction": {
                "median": float(np.median(probe_family_coverage_satisfied_fraction)) if probe_family_coverage_satisfied_fraction.size else 0.0,
                "mean": float(np.mean(probe_family_coverage_satisfied_fraction)) if probe_family_coverage_satisfied_fraction.size else 0.0,
                "count": int(probe_family_coverage_satisfied_fraction.size),
            },
            "probe_second_probe_value_driven_fraction": {
                "median": float(np.median(probe_second_probe_value_driven_fraction)) if probe_second_probe_value_driven_fraction.size else 0.0,
                "mean": float(np.mean(probe_second_probe_value_driven_fraction)) if probe_second_probe_value_driven_fraction.size else 0.0,
                "count": int(probe_second_probe_value_driven_fraction.size),
            },
            "probe_uniformity_pressure_active_fraction": {
                "median": float(np.median(probe_uniformity_pressure_active_fraction)) if probe_uniformity_pressure_active_fraction.size else 0.0,
                "mean": float(np.mean(probe_uniformity_pressure_active_fraction)) if probe_uniformity_pressure_active_fraction.size else 0.0,
                "count": int(probe_uniformity_pressure_active_fraction.size),
            },
            "probe_env_expression_delta": {
                "median": float(np.median(probe_env_expression_delta)) if probe_env_expression_delta.size else 0.0,
                "mean": float(np.mean(probe_env_expression_delta)) if probe_env_expression_delta.size else 0.0,
                "count": int(probe_env_expression_delta.size),
            },
            "probe_forced_env_expression_delta": {
                "median": float(np.median(probe_forced_env_expression_delta)) if probe_forced_env_expression_delta.size else 0.0,
                "mean": float(np.mean(probe_forced_env_expression_delta)) if probe_forced_env_expression_delta.size else 0.0,
                "count": int(probe_forced_env_expression_delta.size),
            },
            "probe_forced_env_expression_scale": {
                "median": float(np.median(probe_forced_env_expression_scale)) if probe_forced_env_expression_scale.size else 0.0,
                "mean": float(np.mean(probe_forced_env_expression_scale)) if probe_forced_env_expression_scale.size else 0.0,
                "count": int(probe_forced_env_expression_scale.size),
            },
            "probe_online_offline_gap_mean": {
                "median": float(np.median(probe_online_offline_gap_mean)) if probe_online_offline_gap_mean.size else 0.0,
                "mean": float(np.mean(probe_online_offline_gap_mean)) if probe_online_offline_gap_mean.size else 0.0,
                "count": int(probe_online_offline_gap_mean.size),
            },
            "probe_online_subset_stability_mean": {
                "median": float(np.median(probe_online_subset_stability_mean)) if probe_online_subset_stability_mean.size else 0.0,
                "mean": float(np.mean(probe_online_subset_stability_mean)) if probe_online_subset_stability_mean.size else 0.0,
                "count": int(probe_online_subset_stability_mean.size),
            },
            "probe_online_geometry_complete_fraction": {
                "median": float(np.median(probe_online_geometry_complete_fraction)) if probe_online_geometry_complete_fraction.size else 0.0,
                "mean": float(np.mean(probe_online_geometry_complete_fraction)) if probe_online_geometry_complete_fraction.size else 0.0,
                "count": int(probe_online_geometry_complete_fraction.size),
            },
            "probe_online_split_latent_disagreement_mean": {
                "median": float(np.median(probe_online_split_latent_disagreement_mean)) if probe_online_split_latent_disagreement_mean.size else 0.0,
                "mean": float(np.mean(probe_online_split_latent_disagreement_mean)) if probe_online_split_latent_disagreement_mean.size else 0.0,
                "count": int(probe_online_split_latent_disagreement_mean.size),
            },
            "probe_online_split_retrieval_margin_deficit_mean": {
                "median": float(np.median(probe_online_split_retrieval_margin_deficit_mean)) if probe_online_split_retrieval_margin_deficit_mean.size else 0.0,
                "mean": float(np.mean(probe_online_split_retrieval_margin_deficit_mean)) if probe_online_split_retrieval_margin_deficit_mean.size else 0.0,
                "count": int(probe_online_split_retrieval_margin_deficit_mean.size),
            },
            "probe_online_leaveout_shift_mean": {
                "median": float(np.median(probe_online_leaveout_shift_mean)) if probe_online_leaveout_shift_mean.size else 0.0,
                "mean": float(np.mean(probe_online_leaveout_shift_mean)) if probe_online_leaveout_shift_mean.size else 0.0,
                "count": int(probe_online_leaveout_shift_mean.size),
            },
            "probe_teacher_action_agreement": {
                "median": float(np.median(probe_teacher_action_agreement)) if probe_teacher_action_agreement.size else 0.0,
                "mean": float(np.mean(probe_teacher_action_agreement)) if probe_teacher_action_agreement.size else 0.0,
                "count": int(probe_teacher_action_agreement.size),
            },
            "full_system_zero_context_ablation_delta": {
                "median": float(np.median(full_system_zero_context_ablation_delta)) if full_system_zero_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_zero_context_ablation_delta)) if full_system_zero_context_ablation_delta.size else 0.0,
                "count": int(full_system_zero_context_ablation_delta.size),
            },
            "full_system_shuffled_context_ablation_delta": {
                "median": float(np.median(full_system_shuffled_context_ablation_delta)) if full_system_shuffled_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_shuffled_context_ablation_delta)) if full_system_shuffled_context_ablation_delta.size else 0.0,
                "count": int(full_system_shuffled_context_ablation_delta.size),
            },
            "full_system_stale_context_ablation_delta": {
                "median": float(np.median(full_system_stale_context_ablation_delta)) if full_system_stale_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_stale_context_ablation_delta)) if full_system_stale_context_ablation_delta.size else 0.0,
                "count": int(full_system_stale_context_ablation_delta.size),
            },
            "full_system_online_refinement_ablation_delta": {
                "median": float(np.median(full_system_online_refinement_ablation_delta)) if full_system_online_refinement_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_online_refinement_ablation_delta)) if full_system_online_refinement_ablation_delta.size else 0.0,
                "count": int(full_system_online_refinement_ablation_delta.size),
            },
            "full_system_frozen_context_ablation_delta": {
                "median": float(np.median(full_system_frozen_context_ablation_delta)) if full_system_frozen_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_frozen_context_ablation_delta)) if full_system_frozen_context_ablation_delta.size else 0.0,
                "count": int(full_system_frozen_context_ablation_delta.size),
            },
            "full_system_actor_only_ablation_delta": {
                "median": float(np.median(full_system_actor_only_ablation_delta)) if full_system_actor_only_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_actor_only_ablation_delta)) if full_system_actor_only_ablation_delta.size else 0.0,
                "count": int(full_system_actor_only_ablation_delta.size),
            },
            "full_system_learned_eval": summarize_matched_eval_rows(
                full_system_learned_eval_summary_rows
            ),
            "full_system_state_only_eval": summarize_matched_eval_rows(
                full_system_state_only_eval_summary_rows
            ),
            "full_system_zero_context_eval": summarize_matched_eval_rows(
                full_system_zero_context_eval_summary_rows
            ),
            "full_system_shuffled_context_eval": summarize_matched_eval_rows(
                full_system_shuffled_context_eval_summary_rows
            ),
            "full_system_stale_context_eval": summarize_matched_eval_rows(
                full_system_stale_context_eval_summary_rows
            ),
            "full_system_online_refinement_eval": summarize_matched_eval_rows(
                full_system_online_refinement_eval_summary_rows
            ),
            "full_system_frozen_context_eval": summarize_matched_eval_rows(
                full_system_frozen_context_eval_summary_rows
            ),
            "full_system_actor_only_eval": summarize_matched_eval_rows(
                full_system_actor_only_eval_summary_rows
            ),
            "full_system_state_only_ablation_delta": {
                "median": float(np.median(full_system_state_only_ablation_delta)) if full_system_state_only_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_state_only_ablation_delta)) if full_system_state_only_ablation_delta.size else 0.0,
                "count": int(full_system_state_only_ablation_delta.size),
            },
            "full_system_oracle_zero_context_ablation_delta": {
                "median": float(np.median(full_system_oracle_zero_context_ablation_delta)) if full_system_oracle_zero_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_zero_context_ablation_delta)) if full_system_oracle_zero_context_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_zero_context_ablation_delta.size),
            },
            "full_system_oracle_shuffled_context_ablation_delta": {
                "median": float(np.median(full_system_oracle_shuffled_context_ablation_delta)) if full_system_oracle_shuffled_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_shuffled_context_ablation_delta)) if full_system_oracle_shuffled_context_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_shuffled_context_ablation_delta.size),
            },
            "full_system_oracle_stale_context_ablation_delta": {
                "median": float(np.median(full_system_oracle_stale_context_ablation_delta)) if full_system_oracle_stale_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_stale_context_ablation_delta)) if full_system_oracle_stale_context_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_stale_context_ablation_delta.size),
            },
            "full_system_oracle_online_refinement_ablation_delta": {
                "median": float(np.median(full_system_oracle_online_refinement_ablation_delta)) if full_system_oracle_online_refinement_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_online_refinement_ablation_delta)) if full_system_oracle_online_refinement_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_online_refinement_ablation_delta.size),
            },
            "full_system_oracle_frozen_context_ablation_delta": {
                "median": float(np.median(full_system_oracle_frozen_context_ablation_delta)) if full_system_oracle_frozen_context_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_frozen_context_ablation_delta)) if full_system_oracle_frozen_context_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_frozen_context_ablation_delta.size),
            },
            "full_system_oracle_actor_only_ablation_delta": {
                "median": float(np.median(full_system_oracle_actor_only_ablation_delta)) if full_system_oracle_actor_only_ablation_delta.size else 0.0,
                "mean": float(np.mean(full_system_oracle_actor_only_ablation_delta)) if full_system_oracle_actor_only_ablation_delta.size else 0.0,
                "count": int(full_system_oracle_actor_only_ablation_delta.size),
            },
            "full_system_oracle_learned_eval": summarize_matched_eval_rows(
                full_system_oracle_learned_eval_summary_rows
            ),
            "full_system_oracle_zero_context_eval": summarize_matched_eval_rows(
                full_system_oracle_zero_context_eval_summary_rows
            ),
            "full_system_oracle_shuffled_context_eval": summarize_matched_eval_rows(
                full_system_oracle_shuffled_context_eval_summary_rows
            ),
            "full_system_oracle_stale_context_eval": summarize_matched_eval_rows(
                full_system_oracle_stale_context_eval_summary_rows
            ),
            "full_system_oracle_online_refinement_eval": summarize_matched_eval_rows(
                full_system_oracle_online_refinement_eval_summary_rows
            ),
            "full_system_oracle_frozen_context_eval": summarize_matched_eval_rows(
                full_system_oracle_frozen_context_eval_summary_rows
            ),
            "full_system_oracle_actor_only_eval": summarize_matched_eval_rows(
                full_system_oracle_actor_only_eval_summary_rows
            ),
            "probe_readiness_reason_counts": readiness_reason_totals,
            "probe_readiness_component_means": readiness_component_means,
            "probe_fair_stop_blocker_counts": fair_stop_blocker_totals,
            "probe_shadow_blocker_counts": shadow_blocker_totals,
            "probe_second_probe_selection_count": second_probe_selection_totals,
            "probe_fair_handoff_pair_count": fair_handoff_pair_totals,
            "probe_strict_usage_counts": strict_usage_counts,
        },
    }
