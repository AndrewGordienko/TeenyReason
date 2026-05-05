"""Deliberate-practice curriculum collector for generic continuous control."""

from __future__ import annotations

import numpy as np

from ....gym_mpc import TransitionBatch
from ...config import AdvancedGymMPCConfig
from ...options import ControlFactorModel, FailureFrontierMiner, MotorPriorModel, OptionSegmentMiner, select_self_demo_trajectories
from ...options.curriculum import (
    CurriculumState,
    FrontierRestartSearcher,
    GoalConditionedHindsightPolicy,
    SuccessiveHalvingRepairSearcher,
    curriculum_diagnostics,
    select_quality_training_trajectories,
)
from ..options import (
    accepted_option_fraction,
    collect_option_rollouts,
    collect_reflex_seed_trajectories,
    empty_option_stats,
    frontier_floor,
    select_option_training_trajectories,
)
from ..replay import max_return_to_go, mean_or_zero, trajectory_steps
from ..trajectory import ReplayTrajectory, collect_probe_trajectories, trajectories_to_batch


def collect_curriculum_repair_archive_transitions(
    config: AdvancedGymMPCConfig,
) -> tuple[TransitionBatch, np.ndarray, np.ndarray, dict[str, object]]:
    """Collect data by converting local failures into self-generated lessons."""
    bootstrap, action_low, action_high = collect_probe_trajectories(config)
    reflex = collect_reflex_seed_trajectories(config, bootstrap, action_low, action_high)
    archive = list(bootstrap) + list(reflex)
    repairs: list[ReplayTrajectory] = []
    frontier_repairs: list[ReplayTrajectory] = []
    repair_results = []
    frontier_results = []
    window_count = 0
    hindsight_stats: dict[str, object] = {}
    cycles = max(1, int(config.curriculum_cycles))
    for cycle in range(cycles):
        factor_model = ControlFactorModel.fit(archive, action_low, action_high)
        motor_prior = MotorPriorModel.fit(archive, factor_model, action_low, action_high)
        curriculum_state = CurriculumState.from_trajectories(
            archive,
            solve_return=float(config.solve_return),
            control_steps=int(config.control_steps),
        )
        hindsight = GoalConditionedHindsightPolicy.fit(
            archive,
            factor_model,
            action_low,
            action_high,
            horizon=int(config.curriculum_hindsight_horizon),
        )
        hindsight_stats = hindsight.diagnostics()
        windows = FailureFrontierMiner(
            horizon=int(config.curriculum_halving_long_steps),
            max_windows=int(config.curriculum_windows),
        ).mine(archive, factor_model)
        window_count += len(windows)
        result = SuccessiveHalvingRepairSearcher(
            env_name=config.env_name,
            discount=float(config.discount),
            action_low=action_low,
            action_high=action_high,
            initial_candidates=int(config.curriculum_halving_initial_candidates),
            mid_candidates=int(config.curriculum_halving_mid_candidates),
            final_candidates=int(config.curriculum_halving_final_candidates),
            short_steps=int(config.curriculum_halving_short_steps),
            mid_steps=int(config.curriculum_halving_mid_steps),
            long_steps=int(config.curriculum_halving_long_steps),
            accept_lift=float(config.curriculum_accept_lift),
            seed=int(config.seed + 760_000 + cycle * 10_000),
        ).search(
            archive,
            windows,
            curriculum_state,
            factor_model,
            hindsight,
            motor_prior=motor_prior,
            inverse_candidates=int(config.option_inverse_repair_candidates),
            intrinsic_fraction=float(config.curriculum_intrinsic_candidate_fraction),
        )
        repair_results.append(result)
        repairs.extend(result.accepted)
        archive.extend(result.accepted)
        restart_factor_model = ControlFactorModel.fit(archive, action_low, action_high)
        restart_state = CurriculumState.from_trajectories(
            archive,
            solve_return=float(config.solve_return),
            control_steps=int(config.control_steps),
        )
        restart = FrontierRestartSearcher(
            env_name=config.env_name,
            discount=float(config.discount),
            action_low=action_low,
            action_high=action_high,
            restart_count=int(config.curriculum_frontier_restart_count),
            candidate_count=int(config.curriculum_frontier_restart_candidates),
            branch_steps=int(config.curriculum_frontier_restart_steps),
            noise=float(config.curriculum_frontier_restart_noise),
            accept_lift=float(config.curriculum_frontier_accept_lift),
            seed=int(config.seed + 780_000 + cycle * 10_000),
        ).search(archive, restart_state, restart_factor_model)
        frontier_results.append(restart)
        frontier_repairs.extend(restart.accepted)
        archive.extend(restart.accepted)
        if not result.accepted and not restart.accepted and cycle > 0:
            break
    final_factor_model = ControlFactorModel.fit(archive, action_low, action_high)
    final_state = CurriculumState.from_trajectories(
        archive,
        solve_return=float(config.solve_return),
        control_steps=int(config.control_steps),
    )
    self_demos = select_self_demo_trajectories(archive, count=int(config.option_self_demo_count))
    all_repairs = repairs + frontier_repairs
    option_source = self_demos + all_repairs
    segments = OptionSegmentMiner(
        duration=int(config.option_segment_duration),
        max_segments=int(config.option_max_segments),
    ).mine(option_source, final_factor_model)
    option_rollouts, option_stats = collect_option_rollouts(config, segments, action_low, action_high)
    seed_selected = select_option_training_trajectories(config, archive, all_repairs, option_rollouts, self_demos)
    quality_selected, quality_stats = select_quality_training_trajectories(
        base=seed_selected,
        repairs=all_repairs,
        options=option_rollouts,
        self_demos=self_demos,
        state=final_state,
        factor_model=final_factor_model,
        keep_count=int(config.option_archive_keep_count),
        quality_gate=bool(config.curriculum_quality_gate),
    )
    coverage_selected = seed_selected
    batch = trajectories_to_batch(coverage_selected)
    value_actor_batch = trajectories_to_batch(quality_selected)
    stats = curriculum_archive_diagnostics(
        config,
        bootstrap=bootstrap,
        reflex=reflex,
        repairs=all_repairs,
        repair_stats=aggregate_repair_results(repair_results),
        frontier_stats=aggregate_frontier_results(frontier_results),
        quality_stats=quality_stats,
        self_demos=self_demos,
        segments=segments,
        option_rollouts=option_rollouts,
        option_stats=option_stats if option_stats else empty_option_stats(),
        failure_window_count=window_count,
        selected=coverage_selected,
        batch=batch,
        curriculum_state=final_state,
        hindsight_stats=hindsight_stats,
    )
    stats["_value_actor_batch"] = value_actor_batch
    stats["_replay_trajectories"] = list(quality_selected)
    return batch, action_low, action_high, stats


def curriculum_archive_diagnostics(
    config: AdvancedGymMPCConfig,
    *,
    bootstrap: list[ReplayTrajectory],
    reflex: list[ReplayTrajectory],
    repairs: list[ReplayTrajectory],
    repair_stats: dict[str, object],
    frontier_stats: dict[str, object],
    quality_stats: dict[str, object],
    self_demos: list[ReplayTrajectory],
    segments: list[object],
    option_rollouts: list[ReplayTrajectory],
    option_stats: dict[str, float],
    failure_window_count: int,
    selected: list[ReplayTrajectory],
    batch: TransitionBatch,
    curriculum_state: CurriculumState,
    hindsight_stats: dict[str, object],
) -> dict[str, object]:
    all_seen = list(bootstrap) + list(reflex) + list(repairs) + list(option_rollouts)
    returns = [item.episode_return for item in all_seen]
    best = max_or_zero(returns)
    interaction_steps = (
        trajectory_steps(bootstrap)
        + trajectory_steps(reflex)
        + int(repair_stats.get("curriculum_repair_interaction_steps", 0))
        + int(frontier_stats.get("frontier_restart_interaction_steps", 0))
        + trajectory_steps(option_rollouts)
    )
    selected_steps = trajectory_steps(selected)
    charged_steps = max(int(interaction_steps), int(selected_steps))
    floor = frontier_floor(config, list(bootstrap) + list(reflex) + list(repairs))
    validated_options = [item for item in option_rollouts if float(item.episode_return) >= floor]
    stats = {
        "collector": "curriculum_repair_archive",
        "collector_samples": int(batch.observations.shape[0]),
        "collector_interaction_steps": int(charged_steps),
        "collector_episode_count": int(len(all_seen)),
        "collector_best_return": float(best),
        "collector_return_mean": mean_or_zero(returns),
        "collector_solve_gap": float(float(config.solve_return) - best),
        "curriculum_cycle_count": int(config.curriculum_cycles),
        "failure_window_count": int(failure_window_count),
        "self_demo_count": int(len(self_demos)),
        "self_demo_best_return": max_or_zero([item.episode_return for item in self_demos]),
        "option_segment_count": int(len(segments)),
        "option_selected_steps": int(selected_steps),
        "option_selected_fraction": float(selected_steps / charged_steps) if charged_steps > 0 else 0.0,
        "option_rollout_count": int(len(option_rollouts)),
        "option_rollout_return_max": max_or_zero([item.episode_return for item in option_rollouts]),
        "option_value_target_max": max_return_to_go(batch, discount=float(config.discount)),
        "accepted_data_real_option_fraction": accepted_option_fraction(option_rollouts, selected),
        "curriculum_chain_validated_count": int(len(validated_options)),
        "curriculum_chain_validation_rate": float(len(validated_options) / len(option_rollouts)) if option_rollouts else 0.0,
        "curriculum_internal_demo_only": 1.0,
        **curriculum_diagnostics(all_seen, curriculum_state),
        **repair_stats,
        **frontier_stats,
        **quality_stats,
        **hindsight_stats,
        **option_stats,
    }
    return stats


def aggregate_repair_results(results: list[object]) -> dict[str, object]:
    if not results:
        return {
            "curriculum_repair_attempt_count": 0,
            "curriculum_repair_cheap_attempt_count": 0,
            "curriculum_repair_accept_count": 0,
            "curriculum_repair_accept_rate": 0.0,
            "curriculum_repair_interaction_steps": 0,
            "curriculum_repair_stage1_keep": 0,
            "curriculum_repair_stage2_keep": 0,
            "curriculum_slow_teacher_finalists": 0,
            "curriculum_repair_return_lift_mean": 0.0,
            "curriculum_repair_return_lift_max": 0.0,
            "curriculum_repair_score_lift_max": 0.0,
            "curriculum_repair_survival_lift_max": 0.0,
            "curriculum_repair_terminal_avoid_count": 0,
        }
    attempts = sum(int(item.attempted) for item in results)
    cheap_attempts = sum(int(item.cheap_attempted) for item in results)
    accepted = sum(int(item.accepted_count) for item in results)
    return_lifts = [float(value) for item in results for value in item.return_lifts]
    score_lifts = [float(value) for item in results for value in item.score_lifts]
    survival_lifts = [float(value) for item in results for value in item.survival_lifts]
    terminal_avoids = [float(value) for item in results for value in item.terminal_avoids]
    return {
        "curriculum_repair_attempt_count": int(attempts),
        "curriculum_repair_cheap_attempt_count": int(cheap_attempts),
        "curriculum_repair_accept_count": int(accepted),
        "curriculum_repair_accept_rate": float(accepted / attempts) if attempts else 0.0,
        "curriculum_repair_interaction_steps": int(sum(int(item.interaction_steps) for item in results)),
        "curriculum_repair_stage1_keep": int(sum(int(item.stage1_keep) for item in results)),
        "curriculum_repair_stage2_keep": int(sum(int(item.stage2_keep) for item in results)),
        "curriculum_slow_teacher_finalists": int(sum(int(item.final_count) for item in results)),
        "curriculum_repair_return_lift_mean": mean_or_zero(return_lifts),
        "curriculum_repair_return_lift_max": max_or_zero(return_lifts),
        "curriculum_repair_score_lift_max": max_or_zero(score_lifts),
        "curriculum_repair_survival_lift_max": max_or_zero(survival_lifts),
        "curriculum_repair_terminal_avoid_count": int(np.sum(np.asarray(terminal_avoids, dtype=np.float32))),
    }


def aggregate_frontier_results(results: list[object]) -> dict[str, object]:
    if not results:
        return {
            "frontier_restart_attempt_count": 0,
            "frontier_restart_accept_count": 0,
            "frontier_restart_accept_rate": 0.0,
            "frontier_restart_interaction_steps": 0,
            "frontier_restart_return_lift_mean": 0.0,
            "frontier_restart_return_lift_max": 0.0,
            "frontier_restart_score_lift_max": 0.0,
        }
    attempts = sum(int(item.attempted) for item in results)
    accepted = sum(int(item.accepted_count) for item in results)
    return_lifts = [float(value) for item in results for value in item.return_lifts]
    score_lifts = [float(value) for item in results for value in item.score_lifts]
    return {
        "frontier_restart_attempt_count": int(attempts),
        "frontier_restart_accept_count": int(accepted),
        "frontier_restart_accept_rate": float(accepted / attempts) if attempts else 0.0,
        "frontier_restart_interaction_steps": int(sum(int(item.interaction_steps) for item in results)),
        "frontier_restart_return_lift_mean": mean_or_zero(return_lifts),
        "frontier_restart_return_lift_max": max_or_zero(return_lifts),
        "frontier_restart_score_lift_max": max_or_zero(score_lifts),
    }


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


__all__ = ["collect_curriculum_repair_archive_transitions"]
