"""Small PPO comparison runner for live dashboard experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmark import print_array_shapes, set_seed
from ..benchmark.artifacts import save_dashboard_context
from ..benchmark.support import (
    default_seeds_for_profile,
    resolve_benchmark_profile,
    solve_eval_episodes_for_profile,
)
from ..config import ExperimentConfig, build_experiment_config
from ...crawler import train_crawler_library
from ...crawler.probes.data import ProbeCrawler
from ...cognition.representation import build_latent_snapshot, save_latent_snapshot
from ...rl.probe_policy import train_plain_ppo, train_probe_conditioned_ppo
from ...viz.live import LiveTrainingTraceWriter, clear_live_trace_history
from .types import (
    DEFAULT_COMPARISON_ENVS,
    GYM_SOLVE_AVG_WINDOW,
    GYM_SOLVE_GRACE_WINDOWS,
    GYM_SOLVED_RETURNS,
    PPOComparisonEnvResult,
    PPOComparisonSeedResult,
    PPOComparisonSuiteResult,
)


def run_ppo_comparison(
    env_names: tuple[str, ...] | list[str] = DEFAULT_COMPARISON_ENVS,
    *,
    seeds: int | list[int] | tuple[int, ...] | None = 1,
    profile: str | None = "fast",
    common_overrides: dict[str, Any] | None = None,
    env_overrides: dict[str, dict[str, Any]] | None = None,
    artifact_dir: str | Path = "artifacts",
    reset_live_history: bool = False,
) -> PPOComparisonSuiteResult:
    """Run standard PPO against probe-conditioned PPO with live dashboard curves."""
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(exist_ok=True)
    if reset_live_history:
        clear_live_trace_history(artifact_root)
    env_results: list[PPOComparisonEnvResult] = []
    comparison_suite_id = f"ppo-comparison-{int(time.time() * 1000)}"

    for env_name in env_names:
        config = _build_comparison_config(
            env_name,
            profile=profile,
            common_overrides=common_overrides,
            env_overrides=(env_overrides or {}).get(env_name),
        )
        active_profile = resolve_benchmark_profile(config)
        run_seeds = _resolve_seeds(seeds, active_profile)
        live_trace = LiveTrainingTraceWriter(
            artifact_dir=artifact_root,
            enabled=True,
            max_curve_points=max(256, int(config.num_episodes) + 8),
            max_archived_runs=30,
        )
        save_dashboard_context(
            env_name=config.env_name,
            benchmark_tag=config.benchmark_tag,
            seeds=run_seeds,
            artifact_dir=artifact_root,
            benchmark_profile=active_profile,
        )
        live_trace.reset_session(
            env_name=config.env_name,
            benchmark_tag=f"{config.benchmark_tag}_comparison",
            seeds=run_seeds,
            total_runs=len(run_seeds),
            comparison_suite_id=comparison_suite_id,
        )

        seed_results = []
        for run_index, seed in enumerate(run_seeds, start=1):
            seed_result = _run_comparison_seed(
                seed=seed,
                config=config,
                profile=active_profile,
                run_index=run_index,
                total_runs=len(run_seeds),
                artifact_dir=artifact_root,
                live_trace=live_trace,
            )
            seed_results.append(seed_result)
            live_trace.update_summary(
                _env_summary(
                    PPOComparisonEnvResult(
                        env_name=config.env_name,
                        benchmark_tag=config.benchmark_tag,
                        profile=active_profile,
                        solved_return=float(_gym_solved_return(config.env_name, config.solved_return)),
                        solve_avg_window=GYM_SOLVE_AVG_WINDOW,
                        seed_results=tuple(seed_results),
                    )
                )
            )
        env_result = PPOComparisonEnvResult(
            env_name=config.env_name,
            benchmark_tag=config.benchmark_tag,
            profile=active_profile,
            solved_return=float(_gym_solved_return(config.env_name, config.solved_return)),
            solve_avg_window=GYM_SOLVE_AVG_WINDOW,
            seed_results=tuple(seed_results),
        )
        env_results.append(env_result)
        live_trace.finish(summary=_env_summary(env_result))

    summary_path = artifact_root / "ppo_comparison_summary.json"
    _write_suite_summary(summary_path, env_results)
    return PPOComparisonSuiteResult(env_results=tuple(env_results), summary_path=summary_path)


def _build_comparison_config(
    env_name: str,
    *,
    profile: str | None,
    common_overrides: dict[str, Any] | None,
    env_overrides: dict[str, Any] | None,
) -> ExperimentConfig:
    config = build_experiment_config(env_name)
    updates: dict[str, Any] = {}
    if profile is not None:
        updates["benchmark_profile"] = str(profile)
    if common_overrides:
        updates.update(common_overrides)
    if env_overrides:
        updates.update(env_overrides)
    return replace(config, **updates) if updates else config


def _resolve_seeds(
    seeds: int | list[int] | tuple[int, ...] | None,
    profile: str,
) -> list[int]:
    if seeds is None:
        return list(default_seeds_for_profile(profile))
    if isinstance(seeds, int):
        return list(range(max(1, int(seeds))))
    return [int(seed) for seed in seeds]


def _gym_solved_return(env_name: str, fallback: float) -> float:
    """Use Gym-style comparison thresholds for the three headline tasks."""
    return float(GYM_SOLVED_RETURNS.get(env_name, float(fallback)))


def _run_comparison_seed(
    *,
    seed: int,
    config: ExperimentConfig,
    profile: str,
    run_index: int,
    total_runs: int,
    artifact_dir: Path,
    live_trace: LiveTrainingTraceWriter,
) -> PPOComparisonSeedResult:
    set_seed(seed)
    artifact_tag = f"{config.benchmark_tag}_seed_{seed}"
    solve_eval_episodes = solve_eval_episodes_for_profile(config)
    solved_return = _gym_solved_return(config.env_name, config.solved_return)

    print(f"\n=== PPO comparison | seed {seed} | env={config.env_name} ===")
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

    live_trace.set_stage(
        "encoder_training",
        "Belief Formation",
        "Training the shared probe encoder before the policy comparison.",
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
        representation_repair_mode=config.representation_repair_mode,
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
    save_latent_snapshot(artifact_dir / f"{artifact_tag}_latent_snapshot.npz", snapshot)
    crawler.close()

    print("\nTraining standard PPO baseline...")
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
        sil_policy_weight=0.0,
        sil_value_weight=0.0,
        min_elite_return=float("inf"),
        seed=seed,
        randomize_physics=config.randomize_physics,
        solved_return=solved_return,
        solve_avg_window=GYM_SOLVE_AVG_WINDOW,
        solve_grace_episodes=GYM_SOLVE_GRACE_WINDOWS * GYM_SOLVE_AVG_WINDOW,
        solve_eval_episodes=solve_eval_episodes,
        late_exploitation_enabled=False,
        run_index=run_index,
        total_runs=total_runs,
        variant_label="baseline",
        peer_variant_label="probe",
        peer_solved_episode=None,
        trace_writer=live_trace,
    )
    np.save(
        artifact_dir / f"{artifact_tag}_comparison_baseline_returns.npy",
        np.asarray(baseline_result.returns, dtype=np.float32),
    )
    live_trace.update_summary(
        _partial_env_summary(
            config=config,
            profile=profile,
            solved_return=solved_return,
            seed=seed,
            encoder_probe_steps=encoder_probe_steps,
            baseline_result=baseline_result,
        )
    )

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
        solved_return=solved_return,
        solve_avg_window=GYM_SOLVE_AVG_WINDOW,
        solve_grace_episodes=GYM_SOLVE_GRACE_WINDOWS * GYM_SOLVE_AVG_WINDOW,
        solve_eval_episodes=solve_eval_episodes,
        run_index=run_index,
        total_runs=total_runs,
        variant_label="probe",
        peer_variant_label="baseline",
        peer_solved_episode=baseline_result.solved_episode,
        trace_writer=live_trace,
    )
    np.save(
        artifact_dir / f"{artifact_tag}_comparison_probe_returns.npy",
        np.asarray(probe_result.returns, dtype=np.float32),
    )

    probe_solved_with_encoder = (
        None
        if probe_result.solved_env_steps is None
        else int(probe_result.solved_env_steps + encoder_probe_steps)
    )
    probe_best_with_encoder = (
        None
        if probe_result.best_env_steps is None
        else int(probe_result.best_env_steps + encoder_probe_steps)
    )
    return PPOComparisonSeedResult(
        seed=int(seed),
        encoder_probe_steps=encoder_probe_steps,
        baseline_best_return=float(baseline_result.best_return),
        baseline_best_episode=baseline_result.best_episode,
        baseline_best_env_steps=baseline_result.best_env_steps,
        baseline_solved_episode=baseline_result.solved_episode,
        baseline_solved_env_steps=baseline_result.solved_env_steps,
        baseline_total_env_steps=int(baseline_result.total_env_steps),
        probe_best_return=float(probe_result.best_return),
        probe_best_episode=probe_result.best_episode,
        probe_best_env_steps=probe_result.best_env_steps,
        probe_best_env_steps_with_encoder=probe_best_with_encoder,
        probe_solved_episode=probe_result.solved_episode,
        probe_solved_env_steps=probe_result.solved_env_steps,
        probe_solved_env_steps_with_encoder=probe_solved_with_encoder,
        probe_total_env_steps=int(probe_result.total_env_steps),
        probe_total_env_steps_with_encoder=int(probe_result.total_env_steps + encoder_probe_steps),
    )


def _partial_env_summary(
    *,
    config: ExperimentConfig,
    profile: str,
    solved_return: float,
    seed: int,
    encoder_probe_steps: int,
    baseline_result: Any,
) -> dict[str, Any]:
    """Publish baseline comparison data before the probe branch finishes."""
    seed_row = {
        "seed": int(seed),
        "encoder_probe_steps": int(encoder_probe_steps),
        "baseline_best_return": float(baseline_result.best_return),
        "baseline_best_episode": baseline_result.best_episode,
        "baseline_best_env_steps": baseline_result.best_env_steps,
        "baseline_solved_episode": baseline_result.solved_episode,
        "baseline_solved_env_steps": baseline_result.solved_env_steps,
        "baseline_total_env_steps": int(baseline_result.total_env_steps),
    }
    return {
        "pipeline": "ppo_comparison",
        "benchmark_tag": config.benchmark_tag,
        "benchmark_profile": profile,
        "env_name": config.env_name,
        "solved_return": float(solved_return),
        "solve_avg_window": GYM_SOLVE_AVG_WINDOW,
        "solve_rule": f"rolling_avg_{GYM_SOLVE_AVG_WINDOW}_episodes",
        "encoder_probe_steps": [int(encoder_probe_steps)],
        "baseline_episode_solves": [
            -1 if baseline_result.solved_episode is None else int(baseline_result.solved_episode)
        ],
        "baseline_best_returns": [float(baseline_result.best_return)],
        "baseline_env_step_solves": [
            -1 if baseline_result.solved_env_steps is None else int(baseline_result.solved_env_steps)
        ],
        "baseline_peak_env_steps": [
            -1 if baseline_result.best_env_steps is None else int(baseline_result.best_env_steps)
        ],
        "baseline_total_env_steps": [int(baseline_result.total_env_steps)],
        "seed_results": [seed_row],
    }


def _env_summary(result: PPOComparisonEnvResult) -> dict[str, Any]:
    seed_rows = [asdict(row) for row in result.seed_results]
    return {
        "pipeline": "ppo_comparison",
        "benchmark_tag": result.benchmark_tag,
        "benchmark_profile": result.profile,
        "env_name": result.env_name,
        "solved_return": result.solved_return,
        "solve_avg_window": result.solve_avg_window,
        "solve_rule": f"rolling_avg_{result.solve_avg_window}_episodes",
        "encoder_probe_steps": [
            int(row.encoder_probe_steps)
            for row in result.seed_results
        ],
        "baseline_episode_solves": [
            -1 if row.baseline_solved_episode is None else int(row.baseline_solved_episode)
            for row in result.seed_results
        ],
        "baseline_best_returns": [
            float(row.baseline_best_return)
            for row in result.seed_results
        ],
        "probe_episode_solves": [
            -1 if row.probe_solved_episode is None else int(row.probe_solved_episode)
            for row in result.seed_results
        ],
        "probe_best_returns": [
            float(row.probe_best_return)
            for row in result.seed_results
        ],
        "baseline_env_step_solves": [
            -1 if row.baseline_solved_env_steps is None else int(row.baseline_solved_env_steps)
            for row in result.seed_results
        ],
        "baseline_peak_env_steps": [
            -1 if row.baseline_best_env_steps is None else int(row.baseline_best_env_steps)
            for row in result.seed_results
        ],
        "baseline_total_env_steps": [
            int(row.baseline_total_env_steps)
            for row in result.seed_results
        ],
        "probe_env_step_solves_with_encoder": [
            -1
            if row.probe_solved_env_steps_with_encoder is None
            else int(row.probe_solved_env_steps_with_encoder)
            for row in result.seed_results
        ],
        "probe_peak_env_steps_with_encoder": [
            -1
            if row.probe_best_env_steps_with_encoder is None
            else int(row.probe_best_env_steps_with_encoder)
            for row in result.seed_results
        ],
        "probe_total_env_steps_with_encoder": [
            int(row.probe_total_env_steps_with_encoder)
            for row in result.seed_results
        ],
        "seed_results": seed_rows,
    }


def _write_suite_summary(path: Path, env_results: list[PPOComparisonEnvResult]) -> None:
    payload = {
        "pipeline": "ppo_comparison",
        "env_results": [_env_summary(result) for result in env_results],
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)
    print(f"Saved PPO comparison summary to {path}")


__all__ = [
    "DEFAULT_COMPARISON_ENVS",
    "PPOComparisonEnvResult",
    "PPOComparisonSeedResult",
    "PPOComparisonSuiteResult",
    "run_ppo_comparison",
]
