"""Top-level experiment driver.

This file wires the whole benchmark together:

1. collect probe data from the environment
2. train a latent encoder on those probe windows
3. train a plain PPO baseline
4. train a PPO agent conditioned on the learned probe belief
5. compare how quickly each variant solves the task across seeds
"""

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    save_dashboard_context,
    save_training_artifacts,
)
from .diagnostics import build_latent_support_diagnostics
from .reporting import finalize_benchmark_run
from .support import (
    benchmark_profile_flags,
    belief_source_from_mode,
    classify_probe_run,
    compute_belief_progress_index,
    compute_system_id_progress_index,
    default_seeds_for_profile,
    apply_system_id_representation_override,
    evaluate_representation_gate,
    matched_eval_summary_dict,
    print_solve_summary,
    print_return_summary,
    probe_strict_usage_status,
    resolve_benchmark_profile,
    solve_eval_episodes_for_profile,
)
from ..config import ExperimentConfig, build_experiment_config
from ...crawler import CrawlerModelBundle, train_crawler_library
from ...envs import (
    BIPEDAL_WALKER_NAME,
    CONTINUOUS_CARTPOLE_NAME,
)
from ...crawler.probes.data import ProbeCrawler
from ...cognition.representation import build_latent_snapshot, save_latent_snapshot
from ...rl.probe_policy import train_plain_ppo, train_probe_conditioned_ppo
from ...viz.diagnostics import (
    compute_linear_env_fit,
    compute_mode_leakage,
    compute_neighbor_env_alignment,
    compute_same_env_gap_ratio,
    compute_split_retrieval_stats,
    compute_uncertainty_error_alignment,
)
from ...viz.live import LiveTrainingTraceWriter


def set_seed(seed: int):
    """Keep Python, NumPy, and Torch aligned for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_array_shapes(title: str, arrays: dict[str, np.ndarray]):
    """Small helper for inspecting the collected dataset before training."""
    print(title)
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")


def snapshot_mean(snapshot: dict, key: str, default: float = 0.0) -> float:
    """Read one scalar mean from a snapshot field if it exists."""
    values = np.asarray(snapshot.get(key, np.asarray([], dtype=np.float32)), dtype=np.float32)
    return float(np.mean(values)) if values.size else float(default)


def _empty_matched_eval_fields() -> dict[str, object]:
    """Default matched-eval fields for representation-only runs."""
    return {
        "full_system_learned_eval_summary": None,
        "full_system_state_only_eval_summary": None,
        "full_system_zero_context_eval_summary": None,
        "full_system_shuffled_context_eval_summary": None,
        "full_system_stale_context_eval_summary": None,
        "full_system_online_refinement_eval_summary": None,
        "full_system_frozen_context_eval_summary": None,
        "full_system_actor_only_eval_summary": None,
        "full_system_state_only_eval_returns": None,
        "full_system_state_only_ablation_delta": None,
        "full_system_zero_context_ablation_delta": None,
        "full_system_shuffled_context_ablation_delta": None,
        "full_system_stale_context_ablation_delta": None,
        "full_system_online_refinement_ablation_delta": None,
        "full_system_frozen_context_ablation_delta": None,
        "full_system_actor_only_ablation_delta": None,
        "full_system_oracle_zero_context_ablation_delta": None,
        "full_system_oracle_shuffled_context_ablation_delta": None,
        "full_system_oracle_stale_context_ablation_delta": None,
        "full_system_oracle_online_refinement_ablation_delta": None,
        "full_system_oracle_frozen_context_ablation_delta": None,
        "full_system_oracle_actor_only_ablation_delta": None,
        "full_system_oracle_learned_eval_summary": None,
        "full_system_oracle_zero_context_eval_summary": None,
        "full_system_oracle_shuffled_context_eval_summary": None,
        "full_system_oracle_stale_context_eval_summary": None,
        "full_system_oracle_online_refinement_eval_summary": None,
        "full_system_oracle_frozen_context_eval_summary": None,
        "full_system_oracle_actor_only_eval_summary": None,
    }


def _representation_only_seed_result(
    *,
    seed: int,
    benchmark_profile: str,
    encoder_probe_steps: int,
    windows: dict[str, np.ndarray],
    latent_mechanics_fit: float,
    latent_split_stats: dict[str, float],
    latent_cross_split_stats: dict[str, float],
    latent_neighbor_alignment: float,
    latent_paired_gap_ratio: float,
    latent_cross_gap_ratio: float,
    latent_heldout_probe_error: float,
    latent_probe_leakage: float,
    latent_uncert_error_corr: float,
    latent_support_diagnostics: dict,
    belief_progress_index: float,
    belief_mode: str,
    belief_source: str,
    representation_gate: dict[str, object],
    config: ExperimentConfig,
    system_id_progress_index: float,
    sysid_trusted: bool,
    sysid_validation_top1: float,
    sysid_validation_margin: float,
    sysid_validation_nll: float,
    particle_entropy_mean: float,
    particle_entropy_norm_mean: float,
    particle_ess_ratio_mean: float,
    particle_leaveout_shift_mean: float,
    particle_subset_stability_mean: float,
) -> dict:
    """Return the normal benchmark schema without pretending PPO ran."""
    result = {
        "seed": seed,
        "benchmark_profile": benchmark_profile,
        "baseline_solve_episode": None,
        "probe_solve_episode": None,
        "probe_shadow_solve_episode": None,
        "probe_no_expression_solve_episode": None,
        "full_system_solve_episode": None,
        "full_system_state_only_solve_episode": None,
        "full_system_oracle_solve_episode": None,
        "sim_fanout_solve_episode": None,
        "baseline_solve_env_steps": None,
        "probe_solve_env_steps": None,
        "probe_shadow_solve_env_steps": None,
        "probe_no_expression_solve_env_steps": None,
        "full_system_solve_env_steps": None,
        "full_system_state_only_solve_env_steps": None,
        "full_system_oracle_solve_env_steps": None,
        "sim_fanout_solve_env_steps": None,
        "baseline_total_env_steps": 0,
        "probe_total_env_steps": int(encoder_probe_steps),
        "probe_shadow_total_env_steps": 0,
        "probe_no_expression_total_env_steps": -1,
        "full_system_total_env_steps": 0,
        "full_system_state_only_total_env_steps": 0,
        "full_system_oracle_total_env_steps": 0,
        "sim_fanout_total_env_steps": 0,
        "baseline_control_env_steps": 0,
        "probe_probe_env_steps": int(encoder_probe_steps),
        "probe_control_env_steps": 0,
        "probe_shadow_probe_env_steps": 0,
        "probe_shadow_control_env_steps": 0,
        "probe_no_expression_probe_env_steps": -1,
        "probe_no_expression_control_env_steps": -1,
        "full_system_probe_env_steps": 0,
        "full_system_control_env_steps": 0,
        "full_system_oracle_probe_env_steps": 0,
        "full_system_oracle_control_env_steps": 0,
        "sim_fanout_probe_env_steps": 0,
        "sim_fanout_control_env_steps": 0,
        "probe_post_expression_env_steps": -1,
        "probe_post_expression_episodes": -1,
        "probe_shadow_post_expression_env_steps": -1,
        "probe_shadow_post_expression_episodes": -1,
        "probe_no_expression_post_expression_env_steps": -1,
        "probe_no_expression_post_expression_episodes": -1,
        "full_system_post_context_env_steps": -1,
        "full_system_post_context_episodes": -1,
        "full_system_oracle_post_context_env_steps": -1,
        "full_system_oracle_post_context_episodes": -1,
        "sim_fanout_post_context_env_steps": -1,
        "sim_fanout_post_context_episodes": -1,
        "baseline_completed_episodes": 0,
        "probe_completed_episodes": 0,
        "probe_shadow_completed_episodes": 0,
        "probe_no_expression_completed_episodes": 0,
        "full_system_completed_episodes": 0,
        "full_system_state_only_completed_episodes": 0,
        "full_system_oracle_completed_episodes": 0,
        "sim_fanout_completed_episodes": 0,
        "full_system_controller_style": None,
        "full_system_oracle_controller_style": None,
        "sim_fanout_controller_style": None,
        "probe_encoder_steps": int(encoder_probe_steps),
        "probe_windows_total": int(len(windows.get("probe_mode", []))),
        "probe_stop_reasons": {"representation_gate_failed": 1},
        "probe_final_stop_reason": "representation_gate_failed",
        "probe_family_expected_gain": {},
        "probe_family_realized_gain": {},
        "probe_family_future_error": {},
        "probe_family_selection_count": {},
        "probe_env_expression_delta": None,
        "probe_forced_env_expression_delta": None,
        "probe_forced_env_expression_scale": None,
        "probe_strict_usage_status": "unused",
        "probe_expression_scale_median": 0.0,
        "probe_expression_scale_active_fraction": 0.0,
        "probe_fair_ready_handoff_fraction": 0.0,
        "probe_fair_expression_enabled_fraction": 0.0,
        "probe_fair_expression_force_muted_fraction": 0.0,
        "probe_fair_ready_confidence_median": 0.0,
        "probe_fair_muted_confidence_median": 0.0,
        "probe_expression_ready_but_muted_fraction": 0.0,
        "probe_shadow_expression_enabled_fraction": None,
        "probe_shadow_expression_scale_median": None,
        "probe_shadow_confidence_median": None,
        "probe_shadow_strict_miss_fraction": None,
        "probe_readiness_reason_counts": {"representation_gate_failed": 1},
        "probe_readiness_component_means": {},
        "probe_fair_stop_blocker_counts": {"representation_gate_failed": 1},
        "probe_shadow_blocker_counts": None,
        "probe_second_probe_selection_count": {},
        "probe_second_probe_raw_future_gain_mean": 0.0,
        "probe_second_probe_future_estimate_mean": 0.0,
        "probe_second_probe_choice_future_gain_mean": 0.0,
        "probe_family_coverage_satisfied_fraction": 0.0,
        "probe_second_probe_value_driven_fraction": 0.0,
        "probe_uniformity_pressure_active_fraction": 0.0,
        "probe_fair_handoff_probe_families": [],
        "probe_readiness_component_timeline": [],
        "probe_online_future_quality_trace": [],
        "probe_online_subset_stability_trace": [],
        "probe_online_offline_gap_trace": [],
        "probe_online_subset_stability_mean": 0.0,
        "probe_online_offline_gap_mean": 0.0,
        "probe_online_geometry_complete_fraction": 0.0,
        "probe_online_split_latent_disagreement_mean": 0.0,
        "probe_online_split_retrieval_margin_deficit_mean": 0.0,
        "probe_online_leaveout_shift_mean": 0.0,
        "probe_message_input_delta_mean": 0.0,
        "probe_message_input_delta_max": 0.0,
        "probe_muted_message_input_delta_mean": 0.0,
        "probe_muted_message_input_delta_max": 0.0,
        "probe_actor_message_norm_mean": 0.0,
        "probe_actor_message_nonzero_fraction": 0.0,
        "probe_muted_actor_message_nonzero_fraction": 0.0,
        "probe_matched_mute_parity_fraction": 0.0,
        "probe_message_off_fraction": 1.0,
        "probe_message_diag_fraction": 0.0,
        "probe_message_on_fraction": 0.0,
        "probe_message_ablation_config_diff": {},
        "probe_teacher_action_agreement": 0.0,
        "latent_mechanics_fit": float(latent_mechanics_fit),
        "latent_split_mrr": float(latent_cross_split_stats["mrr"]),
        "latent_cross_split_mrr": float(latent_cross_split_stats["mrr"]),
        "latent_paired_split_mrr": float(latent_split_stats["mrr"]),
        "latent_split_top1": float(latent_cross_split_stats["top1"]),
        "latent_cross_split_top1": float(latent_cross_split_stats["top1"]),
        "latent_paired_split_top1": float(latent_split_stats["top1"]),
        "latent_neighbor_alignment": float(latent_neighbor_alignment),
        "latent_gap_ratio": float(latent_cross_gap_ratio),
        "latent_paired_gap_ratio": float(latent_paired_gap_ratio),
        "latent_heldout_probe_error": float(latent_heldout_probe_error),
        "latent_probe_leakage": float(latent_probe_leakage),
        "latent_uncert_error_corr": float(latent_uncert_error_corr),
        "latent_support_diagnostics": latent_support_diagnostics,
        "belief_progress_index": float(belief_progress_index),
        "belief_mode": belief_mode,
        "belief_source": belief_source,
        "representation_repair_mode": bool(config.representation_repair_mode),
        "representation_gate": representation_gate,
        "system_id_progress_index": float(system_id_progress_index),
        "sysid_trusted": bool(sysid_trusted),
        "sysid_validation_top1": float(sysid_validation_top1),
        "sysid_validation_margin": float(sysid_validation_margin),
        "sysid_validation_nll": float(sysid_validation_nll),
        "particle_entropy_mean": float(particle_entropy_mean),
        "particle_entropy_norm_mean": float(particle_entropy_norm_mean),
        "particle_ess_ratio_mean": float(particle_ess_ratio_mean),
        "particle_leaveout_shift_mean": float(particle_leaveout_shift_mean),
        "particle_subset_stability_mean": float(particle_subset_stability_mean),
        "probe_run_classification": "representation_blocked",
    }
    result.update(_empty_matched_eval_fields())
    return result


def run_single_seed(
    seed: int,
    config: ExperimentConfig,
    run_index: int = 1,
    total_runs: int = 1,
    live_trace: LiveTrainingTraceWriter | None = None,
):
    """Run the full benchmark pipeline for one seed."""
    set_seed(seed)
    benchmark_profile = resolve_benchmark_profile(config)
    profile_flags = benchmark_profile_flags(benchmark_profile)
    solve_eval_episodes = solve_eval_episodes_for_profile(config)

    artifact_tag = f"{config.benchmark_tag}_seed_{seed}"

    print(f"\n=== Seed {seed} | env={config.env_name} ===")
    print(f"Collecting probe data for {config.env_name}...")
    if live_trace is not None:
        live_trace.begin_seed(run_index=run_index, total_runs=total_runs, seed=seed)
    # First collect short probe rollouts that the encoder will learn from.
    crawler = ProbeCrawler(
        env_name=config.env_name,
        window_size=config.window_size,
        seed=seed,
        randomize_physics=config.randomize_physics,
        action_bins=config.action_bins,
        trace_writer=live_trace,
    )
    crawler.collect(episodes_per_mode=config.probe_episodes_per_mode, max_steps=config.probe_max_steps)

    transitions = crawler.get_transition_arrays()
    windows = crawler.get_window_arrays()
    encoder_probe_steps = int(transitions["state"].shape[0])

    print_array_shapes("Transitions:", transitions)
    print()
    print_array_shapes("Windows:", windows)
    print(f"\nProbe encoder data collection steps: {encoder_probe_steps}")

    print("\nTraining encoder + delta predictor...")
    if live_trace is not None:
        live_trace.set_stage(
            "encoder_training",
            "Belief Formation",
            "Compressing the support windows into an environment belief and message channel.",
            run_index=run_index,
            total_runs=total_runs,
            seed=seed,
        )
    # Train the latent encoder before either PPO variant so both runs see the same probe model.
    crawler_bundle = train_crawler_library(
        windows=windows,
        z_dim=config.z_dim,
        window_size=config.window_size,
        action_vocab_size=crawler.action_dim,
        belief_mode=config.belief_mode,
        sysid_epochs=config.sysid_epochs,
        sysid_batch_size=config.sysid_batch_size,
        sysid_lr=config.sysid_lr,
        sysid_negative_count=config.sysid_negative_count,
        sysid_particle_count=config.sysid_particle_count,
        sysid_likelihood_scale=config.sysid_likelihood_scale,
        progress_callback=None if live_trace is None else live_trace.record_encoder_epoch,
        epochs=config.encoder_epochs,
        batch_size=config.encoder_batch_size,
        lr=config.encoder_lr,
        physics_loss_weight=config.physics_loss_weight,
        affordance_loss_weight=config.affordance_loss_weight,
        decision_loss_weight=1.0,
        return_loss_weight=0.5,
        risk_loss_weight=0.25,
        kl_loss_weight=config.encoder_kl_loss_weight,
        contrastive_loss_weight=config.encoder_contrastive_loss_weight,
        env_consistency_loss_weight=config.encoder_env_consistency_loss_weight,
        env_geometry_loss_weight=config.encoder_env_geometry_loss_weight,
        mode_adversary_loss_weight=config.encoder_mode_adversary_loss_weight,
        latent_rollout_loss_weight=config.encoder_latent_rollout_loss_weight,
        env_within_between_loss_weight=config.encoder_env_within_between_loss_weight,
        belief_subset_count=config.encoder_belief_subset_count,
        belief_subset_size=config.encoder_belief_subset_size,
        contrastive_dim=config.encoder_contrastive_dim,
        ensemble_size=4,
        intervention_horizon=config.intervention_horizon,
        analytic_affordances=False,
        env_name=config.env_name,
        representation_repair_mode=config.representation_repair_mode,
    )

    latent_snapshot = build_latent_snapshot(
        encoder=crawler_bundle.encoder,
        belief_aggregator=crawler_bundle.belief_aggregator,
        env_param_predictor=crawler_bundle.env_param_predictor,
        env_future_predictor=crawler_bundle.env_future_predictor,
        env_metric_projector=crawler_bundle.env_metric_projector,
        belief_message_projector=crawler_bundle.env_expression_projector,
        device=crawler_bundle.device,
        windows=windows,
        env_name=config.env_name,
        benchmark_tag=config.benchmark_tag,
        support_size=config.encoder_belief_subset_size,
        subset_count=config.encoder_belief_subset_count,
        crawler_bundle=crawler_bundle,
    )
    latent_snapshot_path = Path("artifacts") / f"{artifact_tag}_latent_snapshot.npz"
    latent_predictive_mean = latent_snapshot["env_belief_mean"].astype(np.float32)
    latent_metric_mean = latent_snapshot.get(
        "env_metric_mean",
        latent_predictive_mean,
    ).astype(np.float32)
    latent_split_mean_a = latent_snapshot.get(
        "env_metric_split_mean_a",
        latent_snapshot["env_split_mean_a"],
    ).astype(np.float32)
    latent_split_mean_b = latent_snapshot.get(
        "env_metric_split_mean_b",
        latent_snapshot["env_split_mean_b"],
    ).astype(np.float32)
    latent_cross_split_mean_a = latent_snapshot.get(
        "env_cross_family_metric_split_mean_a",
        latent_split_mean_a,
    ).astype(np.float32)
    latent_cross_split_mean_b = latent_snapshot.get(
        "env_cross_family_metric_split_mean_b",
        latent_split_mean_b,
    ).astype(np.float32)
    latent_env_params = latent_snapshot["env_params"].astype(np.float32)
    latent_env_uncertainty = latent_snapshot.get(
        "env_uncertainty",
        np.zeros((latent_metric_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    latent_env_param_error = latent_snapshot.get(
        "env_param_error_mean",
        np.zeros((latent_metric_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    latent_future_probe_error = latent_snapshot.get(
        "env_future_prediction_error",
        np.zeros((latent_metric_mean.shape[0],), dtype=np.float32),
    ).astype(np.float32)
    latent_dominant_probe_mode = latent_snapshot.get(
        "env_dominant_probe_mode",
        np.asarray([], dtype="U"),
    ).astype("U")
    latent_split_stats = compute_split_retrieval_stats(
        latent_split_mean_a,
        latent_split_mean_b,
    )
    latent_cross_split_stats = compute_split_retrieval_stats(
        latent_cross_split_mean_a,
        latent_cross_split_mean_b,
    )
    latent_neighbor_alignment = compute_neighbor_env_alignment(
        latent_metric_mean,
        latent_env_params,
    )
    latent_mechanics_fit = compute_linear_env_fit(
        latent_predictive_mean,
        latent_env_params,
    )
    latent_gap_ratio = compute_same_env_gap_ratio(
        latent_metric_mean,
        latent_split_mean_a,
        latent_split_mean_b,
    )
    latent_cross_gap_ratio = compute_same_env_gap_ratio(
        latent_metric_mean,
        latent_cross_split_mean_a,
        latent_cross_split_mean_b,
    )
    latent_uncertainty_alignment = compute_uncertainty_error_alignment(
        latent_env_uncertainty,
        latent_env_param_error,
    )
    latent_probe_leakage = (
        0.0
        if latent_dominant_probe_mode.size == 0
        else compute_mode_leakage(latent_predictive_mean, latent_dominant_probe_mode)
    )
    latent_heldout_probe_error = (
        float(np.mean(latent_future_probe_error))
        if latent_future_probe_error.size
        else 0.0
    )
    latent_belief_progress_index = compute_belief_progress_index(
        mechanics_fit=latent_mechanics_fit,
        neighbor_alignment=latent_neighbor_alignment,
        split_retrieval=float(latent_cross_split_stats["top1"]),
        heldout_probe_error=latent_heldout_probe_error,
        uncert_error_corr=float(latent_uncertainty_alignment["correlation"]),
        probe_leakage=latent_probe_leakage,
    )
    sysid_metrics = dict(getattr(crawler_bundle, "sysid_validation_metrics", {}) or {})
    belief_mode = str(getattr(crawler_bundle, "belief_mode", config.belief_mode))
    belief_source = belief_source_from_mode(belief_mode)
    sysid_trusted = bool(getattr(crawler_bundle, "sysid_trusted", False))
    sysid_validation_top1 = float(sysid_metrics.get("validation_top1", 0.0))
    sysid_validation_margin = float(sysid_metrics.get("validation_margin", 0.0))
    sysid_validation_nll = float(sysid_metrics.get("validation_nll", 0.0))
    particle_entropy_mean = snapshot_mean(latent_snapshot, "particle_entropy")
    particle_entropy_norm_mean = snapshot_mean(latent_snapshot, "particle_entropy_norm")
    particle_ess_ratio_mean = snapshot_mean(latent_snapshot, "particle_ess_ratio")
    particle_leaveout_shift_mean = snapshot_mean(latent_snapshot, "particle_leaveout_shift")
    particle_subset_stability_mean = snapshot_mean(
        latent_snapshot,
        "particle_subset_stability",
        default=float("nan"),
    )
    if not np.isfinite(particle_subset_stability_mean):
        particle_subset_stability_mean = float(
            np.clip(1.0 - particle_leaveout_shift_mean / 0.35, 0.0, 1.0)
        )
    system_id_progress_index = compute_system_id_progress_index(
        trusted=sysid_trusted,
        validation_top1=sysid_validation_top1,
        validation_margin=sysid_validation_margin,
        particle_entropy_norm=particle_entropy_norm_mean,
        particle_ess_ratio=particle_ess_ratio_mean,
        particle_leaveout_shift=particle_leaveout_shift_mean,
    )
    latent_support_diagnostics = build_latent_support_diagnostics(latent_snapshot)

    representation_gate = evaluate_representation_gate(
        paired_split_top1=float(latent_split_stats["top1"]),
        cross_split_top1=float(latent_cross_split_stats["top1"]),
        neighbor_alignment=float(latent_neighbor_alignment),
        paired_gap_ratio=float(latent_gap_ratio["mean"]),
        belief_norm_std=float(latent_support_diagnostics.get("belief_norm_std", 0.0)),
        nearest_between_median=float(latent_support_diagnostics.get("nearest_between_median", 0.0)),
        min_paired_top1=config.representation_gate_min_paired_top1,
        min_cross_top1=config.representation_gate_min_cross_top1,
        min_neighbor_alignment=config.representation_gate_min_neighbor_alignment,
        max_paired_gap_ratio=config.representation_gate_max_paired_gap_ratio,
        min_belief_norm_std=config.representation_gate_min_belief_norm_std,
        min_nearest_between=config.representation_gate_min_nearest_between,
    )
    representation_gate = apply_system_id_representation_override(
        representation_gate,
        belief_mode=belief_mode,
        sysid_trusted=sysid_trusted,
        sysid_validation_top1=sysid_validation_top1,
        sysid_validation_margin=sysid_validation_margin,
    )
    representation_gate["enabled"] = bool(config.representation_gate_enabled)
    latent_snapshot["representation_gate_json"] = np.asarray(
        json.dumps(representation_gate, sort_keys=True),
        dtype="U",
    )
    latent_snapshot["belief_source"] = np.asarray(belief_source, dtype="U")
    save_latent_snapshot(latent_snapshot_path, latent_snapshot)
    print(f"Saved latent snapshot to {latent_snapshot_path}")
    if config.representation_gate_enabled:
        status = "PASS" if bool(representation_gate["pass"]) else "BLOCKED"
        print(
            "Representation gate: "
            f"{status} | "
            f"reasons={representation_gate['failure_reasons'] if representation_gate['failure_reasons'] else ['none']} | "
            f"override={representation_gate.get('override_reason') or 'none'}"
        )
    if (
        config.representation_gate_enabled
        and config.representation_only_until_gate_pass
        and not bool(representation_gate["pass"])
    ):
        crawler.close()
        return _representation_only_seed_result(
            seed=seed,
            benchmark_profile=benchmark_profile,
            encoder_probe_steps=encoder_probe_steps,
            windows=windows,
            latent_mechanics_fit=float(latent_mechanics_fit),
            latent_split_stats=latent_split_stats,
            latent_cross_split_stats=latent_cross_split_stats,
            latent_neighbor_alignment=float(latent_neighbor_alignment),
            latent_paired_gap_ratio=float(latent_gap_ratio["mean"]),
            latent_cross_gap_ratio=float(latent_cross_gap_ratio["mean"]),
            latent_heldout_probe_error=float(latent_heldout_probe_error),
            latent_probe_leakage=float(latent_probe_leakage),
            latent_uncert_error_corr=float(latent_uncertainty_alignment["correlation"]),
            latent_support_diagnostics=latent_support_diagnostics,
            belief_progress_index=float(latent_belief_progress_index),
            belief_mode=belief_mode,
            belief_source=belief_source,
            representation_gate=representation_gate,
            config=config,
            system_id_progress_index=float(system_id_progress_index),
            sysid_trusted=bool(sysid_trusted),
            sysid_validation_top1=float(sysid_validation_top1),
            sysid_validation_margin=float(sysid_validation_margin),
            sysid_validation_nll=float(sysid_validation_nll),
            particle_entropy_mean=float(particle_entropy_mean),
            particle_entropy_norm_mean=float(particle_entropy_norm_mean),
            particle_ess_ratio_mean=float(particle_ess_ratio_mean),
            particle_leaveout_shift_mean=float(particle_leaveout_shift_mean),
            particle_subset_stability_mean=float(particle_subset_stability_mean),
        )

    crawler.close()

    print("\nTraining baseline PPO...")
    # Baseline PPO gets only environment state.
    baseline_result = train_plain_ppo(
        env_name=config.env_name,
        num_episodes=config.num_episodes,
        belief_dim=crawler_bundle.env_expression_dim + 2,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        lr=config.lr,
        clip_ratio=config.clip_ratio,
        value_clip_ratio=config.value_clip_ratio,
        ppo_epochs=config.ppo_epochs,
        minibatch_size=config.minibatch_size,
        value_loss_weight=config.value_loss_weight,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        min_rollout_steps=config.min_rollout_steps,
        lr_anneal=config.lr_anneal,
        hidden_dim=config.hidden_dim,
        initial_log_std=config.initial_log_std,
        normalize_rewards=config.normalize_rewards,
        seed=seed,
        randomize_physics=config.randomize_physics,
        solved_return=config.solved_return,
        solve_eval_episodes=solve_eval_episodes,
        run_index=run_index,
        total_runs=total_runs,
        variant_label="baseline",
        peer_variant_label="probe",
        peer_solved_episode=None,
        trace_writer=live_trace,
    )

    baseline_returns = baseline_result.returns
    baseline_solve = baseline_result.solved_episode

    print("\nTraining probe-conditioned PPO...")
    # Probe-conditioned PPO gets both state and the probe-derived env expression.
    probe_result = train_probe_conditioned_ppo(
        env_name=config.env_name,
        crawler_bundle=crawler_bundle,
        num_episodes=config.num_episodes,
        window_size=config.window_size,
        action_bins=config.action_bins,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        lr=config.lr,
        clip_ratio=config.clip_ratio,
        value_clip_ratio=config.value_clip_ratio,
        ppo_epochs=config.ppo_epochs,
        minibatch_size=config.minibatch_size,
        value_loss_weight=config.value_loss_weight,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        min_rollout_steps=config.min_rollout_steps,
        lr_anneal=config.lr_anneal,
        hidden_dim=config.hidden_dim,
        initial_log_std=config.initial_log_std,
        normalize_rewards=config.normalize_rewards,
        seed=seed,
        randomize_physics=config.randomize_physics,
        latent_memory_capacity=config.latent_memory_capacity,
        base_probe_episodes=config.base_probe_episodes,
        max_probe_episodes=config.max_probe_episodes,
        benchmark_mode=config.benchmark_mode,
        probe_budget_mode=config.probe_budget_mode,
        probe_adaptive_budget=config.probe_adaptive_budget,
        probe_adaptive_policy_schedule=config.probe_adaptive_policy_schedule,
        belief_bits_per_dim=config.belief_bits_per_dim,
        belief_use_residual_sketch=config.belief_use_residual_sketch,
        novelty_probe_threshold=config.novelty_probe_threshold,
        low_return_probe_threshold=config.low_return_probe_threshold,
        exploit_return_threshold=config.exploit_return_threshold,
        uncertainty_probe_threshold=config.uncertainty_probe_threshold,
        uncertainty_focus_threshold=config.uncertainty_focus_threshold,
        surprise_probe_threshold=config.surprise_probe_threshold,
        online_z_update_alpha=config.online_z_update_alpha,
        online_z_update_freq=config.online_z_update_freq,
        sil_batch_size=config.sil_batch_size,
        sil_policy_weight=config.sil_policy_weight,
        sil_value_weight=config.sil_value_weight,
        min_elite_return=config.min_elite_return,
        elite_warmup_episodes=config.elite_warmup_episodes,
        elite_threshold_std_scale=config.elite_threshold_std_scale,
        solved_return=config.solved_return,
        solve_eval_episodes=solve_eval_episodes,
        run_index=run_index,
        total_runs=total_runs,
        variant_label="probe",
        peer_variant_label="baseline",
        peer_solved_episode=baseline_solve,
        trace_writer=live_trace,
    )

    probe_returns = probe_result.returns

    probe_shadow_result = None
    probe_shadow_returns: list[float] = []
    if profile_flags["run_probe_shadow"]:
        print("\nTraining probe-conditioned PPO with shadow env expression...")
        probe_shadow_result = train_probe_conditioned_ppo(
            env_name=config.env_name,
            crawler_bundle=crawler_bundle,
            num_episodes=config.num_episodes,
            window_size=config.window_size,
            action_bins=config.action_bins,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            lr=config.lr,
            clip_ratio=config.clip_ratio,
            value_clip_ratio=config.value_clip_ratio,
            ppo_epochs=config.ppo_epochs,
            minibatch_size=config.minibatch_size,
            value_loss_weight=config.value_loss_weight,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            min_rollout_steps=config.min_rollout_steps,
            lr_anneal=config.lr_anneal,
            hidden_dim=config.hidden_dim,
            initial_log_std=config.initial_log_std,
            normalize_rewards=config.normalize_rewards,
            seed=seed,
            randomize_physics=config.randomize_physics,
            latent_memory_capacity=config.latent_memory_capacity,
            base_probe_episodes=config.base_probe_episodes,
            max_probe_episodes=config.max_probe_episodes,
            benchmark_mode=config.benchmark_mode,
            probe_budget_mode=config.probe_budget_mode,
            probe_adaptive_budget=config.probe_adaptive_budget,
            probe_adaptive_policy_schedule=config.probe_adaptive_policy_schedule,
            belief_bits_per_dim=config.belief_bits_per_dim,
            belief_use_residual_sketch=config.belief_use_residual_sketch,
            novelty_probe_threshold=config.novelty_probe_threshold,
            low_return_probe_threshold=config.low_return_probe_threshold,
            exploit_return_threshold=config.exploit_return_threshold,
            uncertainty_probe_threshold=config.uncertainty_probe_threshold,
            uncertainty_focus_threshold=config.uncertainty_focus_threshold,
            surprise_probe_threshold=config.surprise_probe_threshold,
            online_z_update_alpha=config.online_z_update_alpha,
            online_z_update_freq=config.online_z_update_freq,
            sil_batch_size=config.sil_batch_size,
            sil_policy_weight=config.sil_policy_weight,
            sil_value_weight=config.sil_value_weight,
            min_elite_return=config.min_elite_return,
            elite_warmup_episodes=config.elite_warmup_episodes,
            elite_threshold_std_scale=config.elite_threshold_std_scale,
            solved_return=config.solved_return,
            solve_eval_episodes=solve_eval_episodes,
            run_index=run_index,
            total_runs=total_runs,
            variant_label="probe-shadowexpr",
            peer_variant_label="probe",
            peer_solved_episode=probe_result.solved_episode,
            shadow_env_expression=True,
            trace_writer=live_trace,
        )
        probe_shadow_returns = probe_shadow_result.returns

    probe_no_expression_result = None
    probe_no_expression_returns: list[float] = []
    if profile_flags["run_probe_no_expression_training"]:
        print("\nTraining probe-conditioned PPO without env expression...")
        probe_no_expression_result = train_probe_conditioned_ppo(
            env_name=config.env_name,
            crawler_bundle=crawler_bundle,
            num_episodes=config.num_episodes,
            window_size=config.window_size,
            action_bins=config.action_bins,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            lr=config.lr,
            clip_ratio=config.clip_ratio,
            value_clip_ratio=config.value_clip_ratio,
            ppo_epochs=config.ppo_epochs,
            minibatch_size=config.minibatch_size,
            value_loss_weight=config.value_loss_weight,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            min_rollout_steps=config.min_rollout_steps,
            lr_anneal=config.lr_anneal,
            hidden_dim=config.hidden_dim,
            initial_log_std=config.initial_log_std,
            normalize_rewards=config.normalize_rewards,
            seed=seed,
            randomize_physics=config.randomize_physics,
            latent_memory_capacity=config.latent_memory_capacity,
            base_probe_episodes=config.base_probe_episodes,
            max_probe_episodes=config.max_probe_episodes,
            benchmark_mode=config.benchmark_mode,
            probe_budget_mode=config.probe_budget_mode,
            probe_adaptive_budget=config.probe_adaptive_budget,
            probe_adaptive_policy_schedule=config.probe_adaptive_policy_schedule,
            belief_bits_per_dim=config.belief_bits_per_dim,
            belief_use_residual_sketch=config.belief_use_residual_sketch,
            novelty_probe_threshold=config.novelty_probe_threshold,
            low_return_probe_threshold=config.low_return_probe_threshold,
            exploit_return_threshold=config.exploit_return_threshold,
            uncertainty_probe_threshold=config.uncertainty_probe_threshold,
            uncertainty_focus_threshold=config.uncertainty_focus_threshold,
            surprise_probe_threshold=config.surprise_probe_threshold,
            online_z_update_alpha=config.online_z_update_alpha,
            online_z_update_freq=config.online_z_update_freq,
            sil_batch_size=config.sil_batch_size,
            sil_policy_weight=config.sil_policy_weight,
            sil_value_weight=config.sil_value_weight,
            min_elite_return=config.min_elite_return,
            elite_warmup_episodes=config.elite_warmup_episodes,
            elite_threshold_std_scale=config.elite_threshold_std_scale,
            solved_return=config.solved_return,
            solve_eval_episodes=solve_eval_episodes,
            run_index=run_index,
            total_runs=total_runs,
            variant_label="probe-noexpr",
            peer_variant_label="probe",
            peer_solved_episode=probe_result.solved_episode,
            disable_env_expression=True,
            trace_writer=live_trace,
        )
        probe_no_expression_returns = probe_no_expression_result.returns
    else:
        print(
            "\nSkipping separate probe-noexpr training in this profile; "
            "using matched env-expression eval-off instead."
        )

    full_system_result = None
    full_system_returns: list[float] = []
    full_system_oracle_result = None
    full_system_oracle_returns: list[float] = []
    sim_fanout_result = None
    sim_fanout_returns: list[float] = []

    save_training_artifacts(
        crawler_bundle=crawler_bundle,
        baseline_result=baseline_result,
        probe_result=probe_result,
        artifact_tag=artifact_tag,
        env_name=config.env_name,
        benchmark_profile=benchmark_profile,
        benchmark_mode=config.benchmark_mode,
        probe_budget_mode=config.probe_budget_mode,
        window_size=config.window_size,
        z_dim=config.z_dim,
        action_bins=config.action_bins,
        hidden_dim=config.hidden_dim,
        initial_log_std=config.initial_log_std,
        belief_bits_per_dim=config.belief_bits_per_dim,
        belief_use_residual_sketch=config.belief_use_residual_sketch,
        online_z_update_alpha=config.online_z_update_alpha,
        online_z_update_freq=config.online_z_update_freq,
        base_probe_episodes=config.base_probe_episodes,
        max_probe_episodes=config.max_probe_episodes,
        probe_adaptive_budget=config.probe_adaptive_budget,
        probe_adaptive_policy_schedule=config.probe_adaptive_policy_schedule,
        randomize_physics=config.randomize_physics,
        solve_eval_episodes=solve_eval_episodes,
        solved_return=config.solved_return,
        value_clip_ratio=config.value_clip_ratio,
        lr_anneal=config.lr_anneal,
        full_system_result=full_system_result,
        full_system_oracle_result=full_system_oracle_result,
        sim_fanout_result=sim_fanout_result,
    )

    probe_solve = probe_result.solved_episode
    probe_shadow_solve = None if probe_shadow_result is None else probe_shadow_result.solved_episode
    probe_no_expression_solve = (
        None
        if probe_no_expression_result is None
        else probe_no_expression_result.solved_episode
    )
    full_system_solve = None if full_system_result is None else full_system_result.solved_episode
    full_system_state_only_solve = (
        None
        if full_system_result is None
        else full_system_result.state_only_solved_episode
    )
    full_system_oracle_solve = (
        None
        if full_system_oracle_result is None
        else full_system_oracle_result.solved_episode
    )
    sim_fanout_solve = None if sim_fanout_result is None else sim_fanout_result.solved_episode
    probe_solve_env_steps = (
        None
        if probe_result.solved_env_steps is None
        else probe_result.solved_env_steps + encoder_probe_steps
    )
    probe_shadow_solve_env_steps = (
        None
        if probe_shadow_result is None or probe_shadow_result.solved_env_steps is None
        else probe_shadow_result.solved_env_steps + encoder_probe_steps
    )
    probe_no_expression_solve_env_steps = (
        None
        if probe_no_expression_result is None or probe_no_expression_result.solved_env_steps is None
        else probe_no_expression_result.solved_env_steps + encoder_probe_steps
    )
    full_system_solve_env_steps = (
        None
        if full_system_result is None or full_system_result.solved_env_steps is None
        else full_system_result.solved_env_steps + encoder_probe_steps
    )
    full_system_state_only_solve_env_steps = (
        None
        if full_system_result is None or full_system_result.state_only_solved_env_steps is None
        else full_system_result.state_only_solved_env_steps + encoder_probe_steps
    )
    full_system_oracle_solve_env_steps = (
        None
        if full_system_oracle_result is None or full_system_oracle_result.solved_env_steps is None
        else full_system_oracle_result.solved_env_steps + encoder_probe_steps
    )
    sim_fanout_solve_env_steps = (
        None
        if sim_fanout_result is None or sim_fanout_result.solved_env_steps is None
        else sim_fanout_result.solved_env_steps + encoder_probe_steps
    )
    probe_total_env_steps = probe_result.total_env_steps + encoder_probe_steps
    probe_shadow_total_env_steps = (
        0
        if probe_shadow_result is None
        else probe_shadow_result.total_env_steps + encoder_probe_steps
    )
    probe_no_expression_total_env_steps = (
        -1
        if probe_no_expression_result is None
        else probe_no_expression_result.total_env_steps + encoder_probe_steps
    )
    full_system_total_env_steps = (
        0
        if full_system_result is None
        else full_system_result.total_env_steps + encoder_probe_steps
    )
    full_system_state_only_total_env_steps = (
        0
        if full_system_result is None or full_system_result.state_only_total_env_steps is None
        else full_system_result.state_only_total_env_steps + encoder_probe_steps
    )
    full_system_oracle_total_env_steps = (
        0
        if full_system_oracle_result is None
        else full_system_oracle_result.total_env_steps + encoder_probe_steps
    )
    sim_fanout_total_env_steps = (
        0
        if sim_fanout_result is None
        else sim_fanout_result.total_env_steps + encoder_probe_steps
    )
    full_system_learned_eval_summary = (
        None
        if full_system_result is None
        else matched_eval_summary_dict(full_system_result.learned_eval_summary)
    )
    full_system_state_only_eval_summary = (
        None
        if full_system_result is None
        else matched_eval_summary_dict(full_system_result.state_only_eval_summary)
    )
    full_system_zero_context_eval_summary = (
        None
        if full_system_result is None
        else matched_eval_summary_dict(full_system_result.zero_context_eval_summary)
    )
    full_system_shuffled_context_eval_summary = (
        None
        if full_system_result is None
        else matched_eval_summary_dict(full_system_result.shuffled_context_eval_summary)
    )
    full_system_stale_context_eval_summary = (
        None
        if full_system_result is None
        else matched_eval_summary_dict(full_system_result.stale_context_eval_summary)
    )
    full_system_online_refinement_eval_summary = (
        None
        if full_system_result is None
        else matched_eval_summary_dict(full_system_result.no_online_refinement_eval_summary)
    )
    full_system_frozen_context_eval_summary = (
        None
        if full_system_result is None
        else matched_eval_summary_dict(full_system_result.frozen_context_eval_summary)
    )
    full_system_actor_only_eval_summary = (
        None
        if full_system_result is None
        else matched_eval_summary_dict(full_system_result.actor_only_eval_summary)
    )
    full_system_oracle_learned_eval_summary = (
        None
        if full_system_oracle_result is None
        else matched_eval_summary_dict(full_system_oracle_result.learned_eval_summary)
    )
    full_system_oracle_zero_context_eval_summary = (
        None
        if full_system_oracle_result is None
        else matched_eval_summary_dict(full_system_oracle_result.zero_context_eval_summary)
    )
    full_system_oracle_shuffled_context_eval_summary = (
        None
        if full_system_oracle_result is None
        else matched_eval_summary_dict(full_system_oracle_result.shuffled_context_eval_summary)
    )
    full_system_oracle_stale_context_eval_summary = (
        None
        if full_system_oracle_result is None
        else matched_eval_summary_dict(full_system_oracle_result.stale_context_eval_summary)
    )
    full_system_oracle_online_refinement_eval_summary = (
        None
        if full_system_oracle_result is None
        else matched_eval_summary_dict(full_system_oracle_result.no_online_refinement_eval_summary)
    )
    full_system_oracle_frozen_context_eval_summary = (
        None
        if full_system_oracle_result is None
        else matched_eval_summary_dict(full_system_oracle_result.frozen_context_eval_summary)
    )
    full_system_oracle_actor_only_eval_summary = (
        None
        if full_system_oracle_result is None
        else matched_eval_summary_dict(full_system_oracle_result.actor_only_eval_summary)
    )
    probe_run_classification = classify_probe_run(
        baseline_episode=baseline_solve,
        baseline_steps=baseline_result.solved_env_steps,
        probe_episode=probe_solve,
        probe_steps=probe_solve_env_steps,
        probe_no_expression_episode=probe_no_expression_solve,
        probe_no_expression_steps=probe_no_expression_solve_env_steps,
        probe_env_expression_delta=probe_result.env_expression_ablation_delta,
        probe_fair_ready_handoff_fraction=probe_result.fair_ready_handoff_fraction,
        probe_fair_expression_enabled_fraction=probe_result.fair_expression_enabled_fraction,
        full_system_zero_context_ablation_delta=(
            None if full_system_result is None else full_system_result.zero_context_ablation_delta
        ),
        full_system_shuffled_context_ablation_delta=(
            None if full_system_result is None else full_system_result.shuffled_context_ablation_delta
        ),
        full_system_stale_context_ablation_delta=(
            None if full_system_result is None else full_system_result.stale_context_ablation_delta
        ),
        benchmark_profile=benchmark_profile,
        seed_count=total_runs,
    )
    probe_usage_status = probe_strict_usage_status(
        probe_result.fair_expression_enabled_fraction
    )
    dominant_readiness_blocker = max(
        (probe_result.readiness_reason_counts or {}).items(),
        key=lambda item: (int(item[1]), str(item[0])),
        default=("none", 0),
    )[0]
    mean_leaveout_stability = float(
        (probe_result.readiness_component_means or {}).get("leaveout_stability", 0.0)
    )

    print_return_summary("Baseline PPO", baseline_returns)
    print_return_summary("Probe-conditioned PPO", probe_returns)
    if probe_shadow_result is not None:
        print_return_summary("Probe + shadow env expression PPO", probe_shadow_returns)
    if probe_no_expression_result is not None:
        print_return_summary("Probe + no env expression PPO", probe_no_expression_returns)
    elif probe_result.no_env_expression_eval_returns is not None:
        print(
            "Probe + no env expression training: not run | "
            f"matched_eval_off={np.mean(probe_result.no_env_expression_eval_returns):.2f}"
        )
    print(
        "Solve episodes: "
        f"baseline={baseline_solve} | "
        f"probe-conditioned={probe_solve} | "
        f"probe-shadowexpr={probe_shadow_solve} | "
        f"probe-noexpr={probe_no_expression_solve}"
    )
    print(
        "Solve env steps (end-to-end): "
        f"baseline={baseline_result.solved_env_steps} | "
        f"probe-conditioned={probe_solve_env_steps} | "
        f"probe-shadowexpr={probe_shadow_solve_env_steps} | "
        f"probe-noexpr={probe_no_expression_solve_env_steps}"
    )
    if probe_result.env_expression_ablation_delta is not None:
        print(
            "Env-expression eval delta: "
            f"probe-conditioned={probe_result.env_expression_ablation_delta:.2f} | "
            f"forced={float(probe_result.forced_env_expression_ablation_delta or 0.0):.2f}"
        )
        print(
            "Strict env-expression usage: "
            f"status={probe_usage_status} | "
            f"blocker={dominant_readiness_blocker} | "
            f"leaveout={mean_leaveout_stability:.2f} | "
            f"forced_scale={float(probe_result.forced_env_expression_scale or 0.0):.2f}"
        )
    print(
        "Belief progress: "
        f"bpi={latent_belief_progress_index:.3f} | "
        f"fit={latent_mechanics_fit:.3f} | "
        f"neighbor={latent_neighbor_alignment:.3f} | "
        f"paired-top1={float(latent_split_stats['top1']):.3f} | "
        f"cross-top1={float(latent_cross_split_stats['top1']):.3f} | "
        f"gap={float(latent_cross_gap_ratio['mean']):.3f} | "
        f"leak={latent_probe_leakage:.3f} | "
        f"uncert-corr={float(latent_uncertainty_alignment['correlation']):.3f}"
    )
    if belief_mode == "particle_sysid":
        print(
            "System-ID progress: "
            f"score={system_id_progress_index:.3f} | "
            f"trusted={sysid_trusted} | "
            f"top1={sysid_validation_top1:.3f} | "
            f"margin={sysid_validation_margin:.3f} | "
            f"entropy={particle_entropy_mean:.3f} | "
            f"ess={particle_ess_ratio_mean:.3f} | "
            f"leaveout={particle_leaveout_shift_mean:.3f}"
        )
    print(
        "Belief support: "
        f"center={float(latent_support_diagnostics['center_window_share']):.3f} | "
        f"directional={float(latent_support_diagnostics['directional_window_share']):.3f} | "
        f"mechanics={float(latent_support_diagnostics.get('mechanics_window_share', 0.0)):.3f} | "
        f"passive={float(latent_support_diagnostics.get('passive_window_share', 0.0)):.3f} | "
        f"stress={float(latent_support_diagnostics.get('stress_window_share', 0.0)):.3f} | "
        f"eff-family={float(latent_support_diagnostics['effective_window_families']):.2f} | "
        f"support/env={float(latent_support_diagnostics['support_count_mean']):.1f} | "
        f"split-overlap={float(latent_support_diagnostics['split_group_overlap_mean']):.3f} | "
        f"cross-overlap={float(latent_support_diagnostics.get('cross_family_split_group_overlap_mean', 0.0)):.3f} | "
        f"window-leak={float(latent_support_diagnostics['window_mode_leakage']):.3f} | "
        f"nearest={float(latent_support_diagnostics['nearest_between_median']):.4f}"
    )

    return {
        "seed": seed,
        "benchmark_profile": benchmark_profile,
        "baseline_solve_episode": baseline_solve,
        "probe_solve_episode": probe_solve,
        "probe_shadow_solve_episode": probe_shadow_solve,
        "probe_no_expression_solve_episode": probe_no_expression_solve,
        "full_system_solve_episode": full_system_solve,
        "full_system_state_only_solve_episode": full_system_state_only_solve,
        "full_system_oracle_solve_episode": full_system_oracle_solve,
        "sim_fanout_solve_episode": sim_fanout_solve,
        "baseline_solve_env_steps": baseline_result.solved_env_steps,
        "probe_solve_env_steps": probe_solve_env_steps,
        "probe_shadow_solve_env_steps": probe_shadow_solve_env_steps,
        "probe_no_expression_solve_env_steps": probe_no_expression_solve_env_steps,
        "full_system_solve_env_steps": full_system_solve_env_steps,
        "full_system_state_only_solve_env_steps": full_system_state_only_solve_env_steps,
        "full_system_oracle_solve_env_steps": full_system_oracle_solve_env_steps,
        "sim_fanout_solve_env_steps": sim_fanout_solve_env_steps,
        "baseline_best_return": baseline_result.best_return,
        "probe_best_return": probe_result.best_return,
        "baseline_peak_env_steps": baseline_result.best_env_steps,
        "probe_peak_env_steps_with_encoder": (
            None
            if probe_result.best_env_steps is None
            else int(probe_result.best_env_steps + encoder_probe_steps)
        ),
        "baseline_total_env_steps": baseline_result.total_env_steps,
        "probe_total_env_steps": probe_total_env_steps,
        "probe_shadow_total_env_steps": probe_shadow_total_env_steps,
        "probe_no_expression_total_env_steps": probe_no_expression_total_env_steps,
        "full_system_total_env_steps": full_system_total_env_steps,
        "full_system_state_only_total_env_steps": full_system_state_only_total_env_steps,
        "full_system_oracle_total_env_steps": full_system_oracle_total_env_steps,
        "sim_fanout_total_env_steps": sim_fanout_total_env_steps,
        "baseline_control_env_steps": baseline_result.control_env_steps_total,
        "probe_probe_env_steps": probe_result.probe_env_steps_total + encoder_probe_steps,
        "probe_control_env_steps": probe_result.control_env_steps_total,
        "probe_shadow_probe_env_steps": 0
        if probe_shadow_result is None
        else probe_shadow_result.probe_env_steps_total + encoder_probe_steps,
        "probe_shadow_control_env_steps": 0
        if probe_shadow_result is None
        else probe_shadow_result.control_env_steps_total,
        "probe_no_expression_probe_env_steps": (
            -1
            if probe_no_expression_result is None
            else probe_no_expression_result.probe_env_steps_total + encoder_probe_steps
        ),
        "probe_no_expression_control_env_steps": (
            -1
            if probe_no_expression_result is None
            else probe_no_expression_result.control_env_steps_total
        ),
        "full_system_probe_env_steps": 0
        if full_system_result is None
        else full_system_result.probe_env_steps_total + encoder_probe_steps,
        "full_system_control_env_steps": 0
        if full_system_result is None
        else full_system_result.control_env_steps_total,
        "full_system_oracle_probe_env_steps": 0
        if full_system_oracle_result is None
        else full_system_oracle_result.probe_env_steps_total + encoder_probe_steps,
        "full_system_oracle_control_env_steps": 0
        if full_system_oracle_result is None
        else full_system_oracle_result.control_env_steps_total,
        "sim_fanout_probe_env_steps": 0
        if sim_fanout_result is None
        else sim_fanout_result.probe_env_steps_total + encoder_probe_steps,
        "sim_fanout_control_env_steps": 0
        if sim_fanout_result is None
        else sim_fanout_result.control_env_steps_total,
        "probe_post_expression_env_steps": probe_result.post_expression_env_steps_total,
        "probe_post_expression_episodes": probe_result.post_expression_episode_count,
        "probe_shadow_post_expression_env_steps": (
            -1
            if probe_shadow_result is None or probe_shadow_result.post_expression_env_steps_total is None
            else int(probe_shadow_result.post_expression_env_steps_total)
        ),
        "probe_shadow_post_expression_episodes": (
            -1
            if probe_shadow_result is None or probe_shadow_result.post_expression_episode_count is None
            else int(probe_shadow_result.post_expression_episode_count)
        ),
        "probe_no_expression_post_expression_env_steps": (
            -1
            if probe_no_expression_result is None
            or probe_no_expression_result.post_expression_env_steps_total is None
            else int(probe_no_expression_result.post_expression_env_steps_total)
        ),
        "probe_no_expression_post_expression_episodes": (
            -1
            if probe_no_expression_result is None
            or probe_no_expression_result.post_expression_episode_count is None
            else int(probe_no_expression_result.post_expression_episode_count)
        ),
        "full_system_post_context_env_steps": (
            -1
            if full_system_result is None or full_system_result.post_expression_env_steps_total is None
            else int(full_system_result.post_expression_env_steps_total)
        ),
        "full_system_post_context_episodes": (
            -1
            if full_system_result is None or full_system_result.post_expression_episode_count is None
            else int(full_system_result.post_expression_episode_count)
        ),
        "full_system_oracle_post_context_env_steps": (
            -1
            if full_system_oracle_result is None or full_system_oracle_result.post_expression_env_steps_total is None
            else int(full_system_oracle_result.post_expression_env_steps_total)
        ),
        "full_system_oracle_post_context_episodes": (
            -1
            if full_system_oracle_result is None or full_system_oracle_result.post_expression_episode_count is None
            else int(full_system_oracle_result.post_expression_episode_count)
        ),
        "sim_fanout_post_context_env_steps": (
            -1
            if sim_fanout_result is None or sim_fanout_result.post_expression_env_steps_total is None
            else int(sim_fanout_result.post_expression_env_steps_total)
        ),
        "sim_fanout_post_context_episodes": (
            -1
            if sim_fanout_result is None or sim_fanout_result.post_expression_episode_count is None
            else int(sim_fanout_result.post_expression_episode_count)
        ),
        "baseline_completed_episodes": len(baseline_returns),
        "probe_completed_episodes": len(probe_returns),
        "probe_shadow_completed_episodes": len(probe_shadow_returns),
        "probe_no_expression_completed_episodes": len(probe_no_expression_returns),
        "full_system_completed_episodes": len(full_system_returns),
        "full_system_state_only_completed_episodes": (
            0
            if full_system_result is None or full_system_result.state_only_completed_episodes is None
            else int(full_system_result.state_only_completed_episodes)
        ),
        "full_system_oracle_completed_episodes": len(full_system_oracle_returns),
        "sim_fanout_completed_episodes": len(sim_fanout_returns),
        "full_system_controller_style": (
            None if full_system_result is None else full_system_result.controller_style
        ),
        "full_system_oracle_controller_style": (
            None if full_system_oracle_result is None else full_system_oracle_result.controller_style
        ),
        "sim_fanout_controller_style": (
            None if sim_fanout_result is None else sim_fanout_result.controller_style
        ),
        "probe_encoder_steps": encoder_probe_steps,
        "probe_windows_total": probe_result.probe_windows_total,
        "probe_stop_reasons": probe_result.probe_stop_reasons,
        "probe_final_stop_reason": (
            probe_result.solve_probe_stop_reason
            if probe_result.solve_probe_stop_reason is not None
            else probe_result.last_probe_stop_reason
        ),
        "probe_family_expected_gain": probe_result.probe_family_expected_gain,
        "probe_family_realized_gain": probe_result.probe_family_realized_gain,
        "probe_family_future_error": probe_result.probe_family_future_error,
        "probe_family_selection_count": probe_result.probe_family_selection_count,
        "probe_env_expression_delta": probe_result.env_expression_ablation_delta,
        "probe_forced_env_expression_delta": probe_result.forced_env_expression_ablation_delta,
        "probe_forced_env_expression_scale": probe_result.forced_env_expression_scale,
        "probe_strict_usage_status": probe_usage_status,
        "probe_expression_scale_median": probe_result.expression_scale_median,
        "probe_expression_scale_active_fraction": probe_result.expression_scale_active_fraction,
        "probe_fair_ready_handoff_fraction": probe_result.fair_ready_handoff_fraction,
        "probe_fair_expression_enabled_fraction": probe_result.fair_expression_enabled_fraction,
        "probe_fair_expression_force_muted_fraction": probe_result.fair_expression_force_muted_fraction,
        "probe_fair_ready_confidence_median": probe_result.fair_ready_confidence_median,
        "probe_fair_muted_confidence_median": probe_result.fair_muted_confidence_median,
        "probe_expression_ready_but_muted_fraction": probe_result.expression_ready_but_muted_fraction,
        "probe_shadow_expression_enabled_fraction": (
            None if probe_shadow_result is None else probe_shadow_result.shadow_expression_enabled_fraction
        ),
        "probe_shadow_expression_scale_median": (
            None if probe_shadow_result is None else probe_shadow_result.shadow_expression_scale_median
        ),
        "probe_shadow_confidence_median": (
            None if probe_shadow_result is None else probe_shadow_result.shadow_confidence_median
        ),
        "probe_shadow_strict_miss_fraction": (
            None if probe_shadow_result is None else probe_shadow_result.shadow_strict_miss_fraction
        ),
        "probe_readiness_reason_counts": probe_result.readiness_reason_counts,
        "probe_readiness_component_means": probe_result.readiness_component_means,
        "probe_fair_stop_blocker_counts": probe_result.fair_stop_blocker_counts,
        "probe_shadow_blocker_counts": (
            None if probe_shadow_result is None else probe_shadow_result.shadow_blocker_counts
        ),
        "probe_second_probe_selection_count": probe_result.second_probe_family_selection_count,
        "probe_second_probe_raw_future_gain_mean": probe_result.second_probe_raw_future_gain_mean,
        "probe_second_probe_future_estimate_mean": probe_result.second_probe_future_estimate_mean,
        "probe_second_probe_choice_future_gain_mean": probe_result.second_probe_choice_future_gain_mean,
        "probe_family_coverage_satisfied_fraction": probe_result.family_coverage_satisfied_fraction,
        "probe_second_probe_value_driven_fraction": probe_result.second_probe_value_driven_fraction,
        "probe_uniformity_pressure_active_fraction": probe_result.uniformity_pressure_active_fraction,
        "probe_fair_handoff_probe_families": probe_result.fair_handoff_probe_families,
        "probe_readiness_component_timeline": probe_result.readiness_component_timeline,
        "probe_online_future_quality_trace": probe_result.online_future_quality_trace,
        "probe_online_subset_stability_trace": probe_result.online_subset_stability_trace,
        "probe_online_offline_gap_trace": probe_result.online_offline_gap_trace,
        "probe_online_subset_stability_mean": (
            0.0
            if not probe_result.online_subset_stability_trace
            else float(np.mean(np.asarray(probe_result.online_subset_stability_trace, dtype=np.float32)))
        ),
        "probe_online_offline_gap_mean": probe_result.online_offline_gap_mean,
        "probe_online_geometry_complete_fraction": probe_result.online_geometry_complete_fraction,
        "probe_online_split_latent_disagreement_mean": probe_result.online_split_latent_disagreement_mean,
        "probe_online_split_retrieval_margin_deficit_mean": probe_result.online_split_retrieval_margin_deficit_mean,
        "probe_online_leaveout_shift_mean": probe_result.online_leaveout_shift_mean,
        "probe_message_input_delta_mean": probe_result.message_input_delta_mean,
        "probe_message_input_delta_max": probe_result.message_input_delta_max,
        "probe_muted_message_input_delta_mean": probe_result.muted_message_input_delta_mean,
        "probe_muted_message_input_delta_max": probe_result.muted_message_input_delta_max,
        "probe_actor_message_norm_mean": probe_result.actor_message_norm_mean,
        "probe_actor_message_nonzero_fraction": probe_result.actor_message_nonzero_fraction,
        "probe_muted_actor_message_nonzero_fraction": probe_result.muted_actor_message_nonzero_fraction,
        "probe_matched_mute_parity_fraction": probe_result.matched_mute_parity_fraction,
        "probe_message_off_fraction": probe_result.message_off_fraction,
        "probe_message_diag_fraction": probe_result.message_diag_fraction,
        "probe_message_on_fraction": probe_result.message_on_fraction,
        "probe_message_ablation_config_diff": probe_result.message_ablation_config_diff,
        "probe_teacher_action_agreement": probe_result.teacher_action_agreement,
        "latent_mechanics_fit": float(latent_mechanics_fit),
        "latent_split_mrr": float(latent_cross_split_stats["mrr"]),
        "latent_cross_split_mrr": float(latent_cross_split_stats["mrr"]),
        "latent_paired_split_mrr": float(latent_split_stats["mrr"]),
        "latent_split_top1": float(latent_cross_split_stats["top1"]),
        "latent_cross_split_top1": float(latent_cross_split_stats["top1"]),
        "latent_paired_split_top1": float(latent_split_stats["top1"]),
        "latent_neighbor_alignment": float(latent_neighbor_alignment),
        "latent_gap_ratio": float(latent_cross_gap_ratio["mean"]),
        "latent_paired_gap_ratio": float(latent_gap_ratio["mean"]),
        "latent_heldout_probe_error": float(latent_heldout_probe_error),
        "latent_probe_leakage": float(latent_probe_leakage),
        "latent_uncert_error_corr": float(latent_uncertainty_alignment["correlation"]),
        "latent_support_diagnostics": latent_support_diagnostics,
        "belief_progress_index": float(latent_belief_progress_index),
        "belief_mode": belief_mode,
        "belief_source": belief_source,
        "representation_repair_mode": bool(config.representation_repair_mode),
        "representation_gate": representation_gate,
        "system_id_progress_index": float(system_id_progress_index),
        "sysid_trusted": bool(sysid_trusted),
        "sysid_validation_top1": float(sysid_validation_top1),
        "sysid_validation_margin": float(sysid_validation_margin),
        "sysid_validation_nll": float(sysid_validation_nll),
        "particle_entropy_mean": float(particle_entropy_mean),
        "particle_entropy_norm_mean": float(particle_entropy_norm_mean),
        "particle_ess_ratio_mean": float(particle_ess_ratio_mean),
        "particle_leaveout_shift_mean": float(particle_leaveout_shift_mean),
        "particle_subset_stability_mean": float(particle_subset_stability_mean),
        "probe_run_classification": probe_run_classification,
        "full_system_learned_eval_summary": full_system_learned_eval_summary,
        "full_system_state_only_eval_summary": full_system_state_only_eval_summary,
        "full_system_zero_context_eval_summary": full_system_zero_context_eval_summary,
        "full_system_shuffled_context_eval_summary": full_system_shuffled_context_eval_summary,
        "full_system_stale_context_eval_summary": full_system_stale_context_eval_summary,
        "full_system_online_refinement_eval_summary": full_system_online_refinement_eval_summary,
        "full_system_frozen_context_eval_summary": full_system_frozen_context_eval_summary,
        "full_system_actor_only_eval_summary": full_system_actor_only_eval_summary,
        "full_system_state_only_eval_returns": (
            None
            if full_system_result is None or full_system_result.state_only_eval_returns is None
            else [float(value) for value in full_system_result.state_only_eval_returns]
        ),
        "full_system_state_only_ablation_delta": (
            None if full_system_result is None else full_system_result.state_only_ablation_delta
        ),
        "full_system_zero_context_ablation_delta": (
            None if full_system_result is None else full_system_result.zero_context_ablation_delta
        ),
        "full_system_shuffled_context_ablation_delta": (
            None if full_system_result is None else full_system_result.shuffled_context_ablation_delta
        ),
        "full_system_stale_context_ablation_delta": (
            None if full_system_result is None else full_system_result.stale_context_ablation_delta
        ),
        "full_system_online_refinement_ablation_delta": (
            None if full_system_result is None else full_system_result.online_refinement_ablation_delta
        ),
        "full_system_frozen_context_ablation_delta": (
            None if full_system_result is None else full_system_result.frozen_context_ablation_delta
        ),
        "full_system_actor_only_ablation_delta": (
            None if full_system_result is None else full_system_result.actor_only_ablation_delta
        ),
        "full_system_oracle_zero_context_ablation_delta": (
            None
            if full_system_oracle_result is None
            else full_system_oracle_result.zero_context_ablation_delta
        ),
        "full_system_oracle_shuffled_context_ablation_delta": (
            None
            if full_system_oracle_result is None
            else full_system_oracle_result.shuffled_context_ablation_delta
        ),
        "full_system_oracle_stale_context_ablation_delta": (
            None
            if full_system_oracle_result is None
            else full_system_oracle_result.stale_context_ablation_delta
        ),
        "full_system_oracle_online_refinement_ablation_delta": (
            None
            if full_system_oracle_result is None
            else full_system_oracle_result.online_refinement_ablation_delta
        ),
        "full_system_oracle_frozen_context_ablation_delta": (
            None
            if full_system_oracle_result is None
            else full_system_oracle_result.frozen_context_ablation_delta
        ),
        "full_system_oracle_actor_only_ablation_delta": (
            None
            if full_system_oracle_result is None
            else full_system_oracle_result.actor_only_ablation_delta
        ),
        "full_system_oracle_learned_eval_summary": full_system_oracle_learned_eval_summary,
        "full_system_oracle_zero_context_eval_summary": full_system_oracle_zero_context_eval_summary,
        "full_system_oracle_shuffled_context_eval_summary": full_system_oracle_shuffled_context_eval_summary,
        "full_system_oracle_stale_context_eval_summary": full_system_oracle_stale_context_eval_summary,
        "full_system_oracle_online_refinement_eval_summary": full_system_oracle_online_refinement_eval_summary,
        "full_system_oracle_frozen_context_eval_summary": full_system_oracle_frozen_context_eval_summary,
        "full_system_oracle_actor_only_eval_summary": full_system_oracle_actor_only_eval_summary,
    }


def run_training_pipeline(
    env_name: str = BIPEDAL_WALKER_NAME,
    seeds: list[int] | None = None,
    config_override: dict | None = None,
):
    """Benchmark the current setup across a small fixed seed set."""
    config = build_experiment_config(env_name)
    if config_override:
        config = replace(config, **config_override)
    benchmark_profile = resolve_benchmark_profile(config)
    if seeds is None:
        seeds = default_seeds_for_profile(benchmark_profile)
    artifact_dir = Path("artifacts")
    save_dashboard_context(
        env_name=config.env_name,
        benchmark_tag=config.benchmark_tag,
        seeds=seeds,
        artifact_dir=artifact_dir,
        benchmark_profile=benchmark_profile,
    )
    live_trace = LiveTrainingTraceWriter(
        artifact_dir=artifact_dir,
        enabled=True,
    )
    live_trace.reset_session(
        env_name=config.env_name,
        benchmark_tag=config.benchmark_tag,
        seeds=seeds,
        total_runs=len(seeds),
    )
    results = []

    total_runs = len(seeds)
    # Treat each seed as one benchmark run so the logs can report progress cleanly.
    for run_index, seed in enumerate(seeds, start=1):
        results.append(
            run_single_seed(
                seed,
                config=config,
                run_index=run_index,
                total_runs=total_runs,
                live_trace=live_trace,
            )
        )
    return finalize_benchmark_run(
        config=config,
        benchmark_profile=benchmark_profile,
        seeds=seeds,
        results=results,
        live_trace=live_trace,
    )


if __name__ == "__main__":
    run_training_pipeline()
