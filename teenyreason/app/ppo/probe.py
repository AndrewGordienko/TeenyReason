"""Probe-conditioned PPO training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from ..benchmark import print_array_shapes, set_seed
from ..benchmark.artifacts import (
    save_dashboard_context,
    serialize_normalizer,
    serialize_normalizer_state,
)
from ..benchmark.support import (
    default_seeds_for_profile,
    resolve_benchmark_profile,
    solve_eval_episodes_for_profile,
)
from ..config import build_experiment_config
from ...crawler import train_crawler_library
from ...envs import CONTINUOUS_CARTPOLE_NAME
from ...crawler.probes.data import ProbeCrawler
from ...cognition.representation import build_latent_snapshot, save_latent_snapshot
from ...rl.probe_policy import train_probe_conditioned_ppo
from ...viz.live import LiveTrainingTraceWriter


@dataclass(frozen=True)
class ProbePPOSeedResult:
    """One seed from a probe-conditioned PPO training run."""

    seed: int
    returns: list[float]
    solved_episode: int | None
    solved_env_steps: int | None
    checkpoint_path: Path


@dataclass(frozen=True)
class ProbePPOTrainingResult:
    """Small result object for the probe-only training entrypoint."""

    env_name: str
    profile: str
    seeds: tuple[int, ...]
    seed_results: tuple[ProbePPOSeedResult, ...]

    @property
    def solved_episodes(self) -> tuple[int | None, ...]:
        return tuple(result.solved_episode for result in self.seed_results)


def run_probe_conditioned_pipeline(
    env_name: str = CONTINUOUS_CARTPOLE_NAME,
    *,
    seeds: list[int] | None = None,
    config_override: dict | None = None,
) -> ProbePPOTrainingResult:
    """Train only the probe-conditioned PPO track."""
    config = build_experiment_config(env_name)
    if config_override:
        config = replace(config, **config_override)
    profile = resolve_benchmark_profile(config)
    run_seeds = default_seeds_for_profile(profile) if seeds is None else [int(seed) for seed in seeds]
    artifact_dir = Path("artifacts")
    save_dashboard_context(
        env_name=config.env_name,
        benchmark_tag=config.benchmark_tag,
        seeds=run_seeds,
        artifact_dir=artifact_dir,
        benchmark_profile=profile,
    )
    live_trace = LiveTrainingTraceWriter(artifact_dir=artifact_dir, enabled=True)
    live_trace.reset_session(
        env_name=config.env_name,
        benchmark_tag=config.benchmark_tag,
        seeds=run_seeds,
        total_runs=len(run_seeds),
    )

    results: list[ProbePPOSeedResult] = []
    for run_index, seed in enumerate(run_seeds, start=1):
        results.append(
            _run_probe_conditioned_seed(
                seed=seed,
                config=config,
                profile=profile,
                run_index=run_index,
                total_runs=len(run_seeds),
                live_trace=live_trace,
            )
        )
    live_trace.finish(
        summary={
            "probe_episode_solves": [
                -1 if result.solved_episode is None else int(result.solved_episode)
                for result in results
            ],
            "probe_env_step_solves": [
                -1 if result.solved_env_steps is None else int(result.solved_env_steps)
                for result in results
            ],
            "benchmark_tag": config.benchmark_tag,
            "benchmark_profile": profile,
            "pipeline": "probe_conditioned_ppo",
        }
    )
    return ProbePPOTrainingResult(
        env_name=config.env_name,
        profile=profile,
        seeds=tuple(run_seeds),
        seed_results=tuple(results),
    )


def _run_probe_conditioned_seed(
    *,
    seed: int,
    config,
    profile: str,
    run_index: int,
    total_runs: int,
    live_trace,
) -> ProbePPOSeedResult:
    set_seed(seed)
    artifact_tag = f"{config.benchmark_tag}_seed_{seed}"
    print(f"\n=== Probe-conditioned PPO | seed {seed} | env={config.env_name} ===")
    print(f"Collecting probe data for {config.env_name}...")
    live_trace.begin_seed(run_index=run_index, total_runs=total_runs, seed=seed)

    crawler = ProbeCrawler(
        env_name=config.env_name,
        window_size=config.window_size,
        seed=seed,
        randomize_physics=config.randomize_physics,
        action_bins=config.action_bins,
        trace_writer=live_trace,
    )
    crawler.collect(
        episodes_per_mode=config.probe_episodes_per_mode,
        max_steps=config.probe_max_steps,
    )
    transitions = crawler.get_transition_arrays()
    windows = crawler.get_window_arrays()
    encoder_probe_steps = int(transitions["state"].shape[0])

    print_array_shapes("Transitions:", transitions)
    print()
    print_array_shapes("Windows:", windows)
    print(f"\nProbe encoder data collection steps: {encoder_probe_steps}")
    print("\nTraining encoder + delta predictor...")
    live_trace.set_stage(
        "encoder_training",
        "Belief Formation",
        "Compressing support windows into an environment belief.",
        run_index=run_index,
        total_runs=total_runs,
        seed=seed,
    )
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
        progress_callback=live_trace.record_encoder_epoch,
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
    )
    snapshot = build_latent_snapshot(
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
    snapshot_path = Path("artifacts") / f"{artifact_tag}_latent_snapshot.npz"
    save_latent_snapshot(snapshot_path, snapshot)
    crawler.close()

    print(f"Saved latent snapshot to {snapshot_path}")
    print("\nTraining probe-conditioned PPO...")
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
        solve_eval_episodes=solve_eval_episodes_for_profile(config),
        run_index=run_index,
        total_runs=total_runs,
        variant_label="probe",
        peer_variant_label="probe",
        peer_solved_episode=None,
        trace_writer=live_trace,
    )
    checkpoint_path = _save_probe_artifacts(
        artifact_tag=artifact_tag,
        env_name=config.env_name,
        profile=profile,
        config=config,
        crawler_bundle=crawler_bundle,
        probe_result=probe_result,
    )
    print(
        "Probe-conditioned PPO done | "
        f"solve_ep={probe_result.solved_episode} | "
        f"solve_steps={probe_result.solved_env_steps} | "
        f"checkpoint={checkpoint_path}"
    )
    return ProbePPOSeedResult(
        seed=int(seed),
        returns=list(probe_result.returns),
        solved_episode=probe_result.solved_episode,
        solved_env_steps=probe_result.solved_env_steps,
        checkpoint_path=checkpoint_path,
    )


def _save_probe_artifacts(
    *,
    artifact_tag: str,
    env_name: str,
    profile: str,
    config,
    crawler_bundle,
    probe_result,
) -> Path:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    returns_path = output_dir / f"{artifact_tag}_probe_ppo_returns.npy"
    checkpoint_path = output_dir / f"{artifact_tag}_probe_ppo_checkpoint.pt"
    encoder_path = output_dir / f"{artifact_tag}_encoder_state_dict.pt"
    np.save(returns_path, np.asarray(probe_result.returns, dtype=np.float32))
    torch.save(crawler_bundle.encoder.state_dict(), encoder_path)
    torch.save(
        {
            "env_name": env_name,
            "window_size": config.window_size,
            "z_dim": config.z_dim,
            "belief_dim": int(crawler_bundle.env_expression_dim) + 2,
            "belief_style": "env_expression_plus_confidence_and_uncertainty",
            "action_bins": config.action_bins,
            "hidden_dim": config.hidden_dim,
            "initial_log_std": config.initial_log_std,
            "benchmark_mode": config.benchmark_mode,
            "benchmark_profile": str(profile),
            "probe_budget_mode": config.probe_budget_mode,
            "belief_bits_per_dim": int(config.belief_bits_per_dim),
            "belief_use_residual_sketch": bool(config.belief_use_residual_sketch),
            "value_clip_ratio": config.value_clip_ratio,
            "lr_anneal": bool(config.lr_anneal),
            "online_z_update_alpha": config.online_z_update_alpha,
            "online_z_update_freq": config.online_z_update_freq,
            "base_probe_episodes": config.base_probe_episodes,
            "max_probe_episodes": config.max_probe_episodes,
            "probe_adaptive_budget": config.probe_adaptive_budget,
            "probe_adaptive_policy_schedule": config.probe_adaptive_policy_schedule,
            "encoder_state_dict": crawler_bundle.encoder.state_dict(),
            "predictor_state_dict": crawler_bundle.predictor.state_dict(),
            "predictor_ensemble_size": int(crawler_bundle.predictor.ensemble_size),
            "belief_aggregator_state_dict": crawler_bundle.belief_aggregator.state_dict(),
            "env_param_predictor_state_dict": crawler_bundle.env_param_predictor.state_dict(),
            "env_param_predictor_ensemble_size": int(crawler_bundle.env_param_predictor.ensemble_size),
            "env_param_dim": int(crawler_bundle.env_param_predictor.output_dim),
            "env_future_predictor_state_dict": None
            if crawler_bundle.env_future_predictor is None
            else crawler_bundle.env_future_predictor.state_dict(),
            "env_future_summary_dim": 0
            if crawler_bundle.env_future_predictor is None
            else int(crawler_bundle.env_future_predictor.output_dim),
            "env_family_future_predictor_state_dict": None
            if crawler_bundle.env_family_future_predictor is None
            else crawler_bundle.env_family_future_predictor.state_dict(),
            "env_family_count": 0
            if crawler_bundle.env_family_future_predictor is None
            else int(crawler_bundle.env_family_future_predictor.num_families),
            "family_value_predictor_state_dict": None
            if crawler_bundle.family_value_predictor is None
            else crawler_bundle.family_value_predictor.state_dict(),
            "family_value_input_dim": 0
            if crawler_bundle.family_value_predictor is None
            else int(crawler_bundle.family_value_predictor.input_dim),
            "family_value_output_dim": 0
            if crawler_bundle.family_value_predictor is None
            else int(crawler_bundle.family_value_predictor.output_dim),
            "env_metric_projector_state_dict": None
            if crawler_bundle.env_metric_projector is None
            else crawler_bundle.env_metric_projector.state_dict(),
            "env_metric_dim": 0
            if crawler_bundle.env_metric_projector is None
            else int(crawler_bundle.env_metric_projector.net[-1].out_features),
            "belief_message_projector_state_dict": None
            if crawler_bundle.env_expression_projector is None
            else crawler_bundle.env_expression_projector.state_dict(),
            "belief_message_dim": int(crawler_bundle.env_expression_dim),
            "env_expression_dim": int(crawler_bundle.env_expression_dim),
            "controller_context_projector_state_dict": None
            if crawler_bundle.controller_context_projector is None
            else crawler_bundle.controller_context_projector.state_dict(),
            "oracle_context_projector_state_dict": None
            if crawler_bundle.oracle_context_projector is None
            else crawler_bundle.oracle_context_projector.state_dict(),
            "controller_trust_predictor_state_dict": None
            if crawler_bundle.controller_trust_predictor is None
            else crawler_bundle.controller_trust_predictor.state_dict(),
            "controller_context_dim": int(crawler_bundle.full_system_controller_dim),
            "controller_mechanics_dim": int(crawler_bundle.z_dim),
            "controller_affordance_dim": int(crawler_bundle.z_dim),
            "env_param_normalizer_mean": None
            if crawler_bundle.env_param_normalizer_mean is None
            else torch.tensor(crawler_bundle.env_param_normalizer_mean, dtype=torch.float32),
            "env_param_normalizer_std": None
            if crawler_bundle.env_param_normalizer_std is None
            else torch.tensor(crawler_bundle.env_param_normalizer_std, dtype=torch.float32),
            "probe_family_names": np.asarray(crawler_bundle.family_names, dtype="U"),
            "belief_mode": str(getattr(crawler_bundle, "belief_mode", "latent_pool")),
            "sysid_model_state_dict": None
            if getattr(crawler_bundle, "sysid_model", None) is None
            else crawler_bundle.sysid_model.state_dict(),
            "sysid_feature_stats": None
            if getattr(crawler_bundle, "sysid_stats", None) is None
            else crawler_bundle.sysid_stats.to_dict(),
            "sysid_particles_raw": None
            if getattr(crawler_bundle, "sysid_particles_raw", None) is None
            else np.asarray(crawler_bundle.sysid_particles_raw, dtype=np.float32),
            "sysid_trusted": bool(getattr(crawler_bundle, "sysid_trusted", False)),
            "sysid_validation_metrics": dict(getattr(crawler_bundle, "sysid_validation_metrics", {}) or {}),
            "sysid_likelihood_scale": float(getattr(crawler_bundle, "sysid_likelihood_scale", 0.35)),
            "policy_state_dict": probe_result.policy.state_dict(),
            "state_normalizer": serialize_normalizer(probe_result.state_normalizer),
            "best_policy_state_dict": probe_result.best_policy_state_dict,
            "best_state_normalizer": serialize_normalizer_state(
                probe_result.best_state_normalizer_state
            ),
            "best_return": float(probe_result.best_return),
            "best_episode": probe_result.best_episode,
            "solve_policy_state_dict": probe_result.solve_policy_state_dict,
            "solve_state_normalizer": None
            if probe_result.solve_state_normalizer_state is None
            else serialize_normalizer_state(probe_result.solve_state_normalizer_state),
            "solve_eval_returns": None
            if probe_result.solve_eval_returns is None
            else torch.tensor(probe_result.solve_eval_returns, dtype=torch.float32),
            "solve_probe_count": probe_result.solve_probe_count,
            "solved_episode": probe_result.solved_episode,
            "solved_env_steps": probe_result.solved_env_steps,
            "total_env_steps": probe_result.total_env_steps,
            "probe_env_steps_total": probe_result.probe_env_steps_total,
            "control_env_steps_total": probe_result.control_env_steps_total,
            "post_expression_env_steps_total": probe_result.post_expression_env_steps_total,
            "post_expression_episode_count": probe_result.post_expression_episode_count,
            "expression_scale_median": probe_result.expression_scale_median,
            "expression_scale_active_fraction": probe_result.expression_scale_active_fraction,
            "fair_ready_handoff_fraction": probe_result.fair_ready_handoff_fraction,
            "fair_expression_enabled_fraction": probe_result.fair_expression_enabled_fraction,
            "fair_expression_force_muted_fraction": probe_result.fair_expression_force_muted_fraction,
            "fair_ready_confidence_median": probe_result.fair_ready_confidence_median,
            "fair_muted_confidence_median": probe_result.fair_muted_confidence_median,
            "probe_windows_total": probe_result.probe_windows_total,
            "probe_stop_reasons": probe_result.probe_stop_reasons,
            "last_probe_stop_reason": probe_result.last_probe_stop_reason,
            "solve_probe_stop_reason": probe_result.solve_probe_stop_reason,
            "probe_family_expected_gain": probe_result.probe_family_expected_gain,
            "probe_family_realized_gain": probe_result.probe_family_realized_gain,
            "probe_family_future_error": probe_result.probe_family_future_error,
            "probe_family_selection_count": probe_result.probe_family_selection_count,
            "randomize_physics": config.randomize_physics,
            "solve_eval_episodes": solve_eval_episodes_for_profile(config),
            "solved_return": config.solved_return,
        },
        checkpoint_path,
    )
    print(f"Saved encoder to {encoder_path}")
    print(f"Saved probe-conditioned returns to {returns_path}")
    print(f"Saved probe-conditioned PPO checkpoint to {checkpoint_path}")
    return checkpoint_path


__all__ = [
    "ProbePPOSeedResult",
    "ProbePPOTrainingResult",
    "run_probe_conditioned_pipeline",
]
