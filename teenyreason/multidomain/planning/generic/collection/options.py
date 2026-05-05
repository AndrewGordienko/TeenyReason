"""Option-repair archive collector for generic continuous Gym environments."""

from __future__ import annotations

import numpy as np

from ...gym_mpc import TransitionBatch
from ..config import AdvancedGymMPCConfig
from ..options import (
    ControlFactorModel,
    CounterfactualRepairSearcher,
    FailureFrontierMiner,
    MotorPriorModel,
    OptionPlanner,
    OptionSegmentMiner,
    collect_option_planner_episode,
    select_self_demo_trajectories,
)
from ..options.models import OptionActor, OptionOutcomeModel
from .reflex import collect_reflex_trajectory, make_search_state, observation_stats, sample_generation, update_search_state
from .replay import max_return_to_go, mean_or_zero, trajectory_steps
from .trajectory import ReplayTrajectory, collect_probe_trajectories, trajectories_to_batch


def collect_option_archive_transitions(
    config: AdvancedGymMPCConfig,
) -> tuple[TransitionBatch, np.ndarray, np.ndarray, dict[str, object]]:
    """Collect data by repairing failures and composing learned options."""
    bootstrap, action_low, action_high = collect_probe_trajectories(config)
    reflex = collect_reflex_seed_trajectories(config, bootstrap, action_low, action_high)
    base_trajectories = list(bootstrap) + list(reflex)
    factor_model = ControlFactorModel.fit(base_trajectories, action_low, action_high)
    motor_prior = MotorPriorModel.fit(base_trajectories, factor_model, action_low, action_high)
    windows = FailureFrontierMiner(
        horizon=int(config.option_repair_branch_steps),
        max_windows=int(config.option_failure_windows),
    ).mine(base_trajectories, factor_model)
    repair = CounterfactualRepairSearcher(
        env_name=config.env_name,
        discount=float(config.discount),
        action_low=action_low,
        action_high=action_high,
        branch_steps=int(config.option_repair_branch_steps),
        attempts_per_window=int(config.option_repair_attempts_per_window),
        accept_delta=float(config.option_repair_accept_delta),
        survival_lift=int(config.option_repair_survival_lift),
        seed=int(config.seed + 610_000),
        motor_prior=motor_prior,
        inverse_candidates=int(config.option_inverse_repair_candidates),
    ).search(base_trajectories, windows)
    repair_trajectories = list(repair.accepted)
    self_demos = select_self_demo_trajectories(
        base_trajectories + repair_trajectories,
        count=int(config.option_self_demo_count),
    )
    option_source = self_demos + repair_trajectories
    segments = OptionSegmentMiner(
        duration=int(config.option_segment_duration),
        max_segments=int(config.option_max_segments),
    ).mine(option_source, factor_model)
    option_rollouts, option_stats = collect_option_rollouts(config, segments, action_low, action_high)
    selected = select_option_training_trajectories(config, base_trajectories, repair_trajectories, option_rollouts, self_demos)
    batch = trajectories_to_batch(selected)
    stats = option_archive_diagnostics(
        config,
        bootstrap=bootstrap,
        reflex=reflex,
        repair=repair,
        self_demos=self_demos,
        segments=segments,
        option_rollouts=option_rollouts,
        option_stats=option_stats,
        failure_window_count=len(windows),
        selected=selected,
        batch=batch,
    )
    stats["_replay_trajectories"] = list(selected)
    return batch, action_low, action_high, stats


def collect_reflex_seed_trajectories(
    config: AdvancedGymMPCConfig,
    bootstrap: list[ReplayTrajectory],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> list[ReplayTrajectory]:
    obs_mean, obs_std = observation_stats(bootstrap)
    state = make_search_state(config, obs_mean, obs_std, action_low, action_high)
    rng = np.random.default_rng(int(config.seed) + 510_000)
    out: list[ReplayTrajectory] = []
    generations = max(1, int(config.reflex_generations))
    for generation in range(generations):
        policies = sample_generation(config, state, rng, generation=generation)
        evaluated: list[ReplayTrajectory] = []
        for index, policy in enumerate(policies):
            seed = int(config.seed + 520_000 + generation * 4099 + index)
            trajectory = collect_reflex_trajectory(
                config,
                policy,
                seed=seed,
                max_steps=max(int(config.probe_steps), int(config.control_steps)),
            )
            evaluated.append(trajectory)
            out.append(trajectory)
        state = update_search_state(state, policies, evaluated, config)
    return out


def collect_option_rollouts(
    config: AdvancedGymMPCConfig,
    segments: list[object],
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[list[ReplayTrajectory], dict[str, float]]:
    if not segments:
        return [], empty_option_stats()
    actor = OptionActor.fit(
        segments,
        action_low=action_low,
        action_high=action_high,
        ridge=float(config.option_actor_ridge),
    )
    outcomes = OptionOutcomeModel.fit(segments)
    planner = OptionPlanner.from_segments(
        actor,
        outcomes,
        segments,
        action_low=action_low,
        action_high=action_high,
        done_penalty=float(config.done_penalty),
        uncertainty_penalty=float(config.uncertainty_penalty),
        initiation_slack=float(config.option_initiation_slack),
        min_real_roi=float(config.option_min_real_roi),
    )
    rollouts: list[ReplayTrajectory] = []
    stats_rows: list[dict[str, float]] = []
    for index in range(max(0, int(config.option_planner_episodes))):
        trajectory, stats = collect_option_planner_episode(
            config.env_name,
            planner,
            seed=int(config.seed + 650_000 + index),
            max_steps=int(config.control_steps),
            discount=float(config.discount),
        )
        rollouts.append(trajectory)
        stats_rows.append(stats)
    stats = aggregate_option_stats(stats_rows)
    stats["option_count"] = float(len(actor.option_ids))
    stats["option_prediction_error"] = float(outcomes.prediction_error(segments))
    stats["option_real_roi_mean"] = mean_or_zero([segment.real_roi for segment in segments])
    return rollouts, stats


def select_option_training_trajectories(
    config: AdvancedGymMPCConfig,
    base: list[ReplayTrajectory],
    repairs: list[ReplayTrajectory],
    option_rollouts: list[ReplayTrajectory],
    self_demos: list[ReplayTrajectory],
) -> list[ReplayTrajectory]:
    floor = frontier_floor(config, base + repairs)
    accepted_options = [item for item in option_rollouts if float(item.episode_return) >= floor]
    bootstrap_count = max(1, min(int(config.probe_episodes), len(base)))
    bootstrap = list(base[:bootstrap_count])
    frontier = sorted(base[bootstrap_count:], key=lambda item: item.episode_return, reverse=True)
    frontier = [item for item in frontier if float(item.episode_return) >= floor]
    if not frontier and len(base) > bootstrap_count:
        frontier = sorted(base[bootstrap_count:], key=lambda item: item.episode_return, reverse=True)[: max(1, int(config.reflex_elite_count))]
    ranked_options = sorted(accepted_options, key=lambda item: item.episode_return, reverse=True)
    keep_frontier = frontier[: max(1, int(config.option_archive_keep_count))]
    demo_rows = [item for item in self_demos if not any(item is other for other in bootstrap + keep_frontier)]
    return bootstrap + keep_frontier + demo_rows + list(repairs) + ranked_options


def option_archive_diagnostics(
    config: AdvancedGymMPCConfig,
    *,
    bootstrap: list[ReplayTrajectory],
    reflex: list[ReplayTrajectory],
    repair,
    self_demos: list[ReplayTrajectory],
    segments: list[object],
    option_rollouts: list[ReplayTrajectory],
    option_stats: dict[str, float],
    failure_window_count: int,
    selected: list[ReplayTrajectory],
    batch: TransitionBatch,
) -> dict[str, object]:
    all_seen = list(bootstrap) + list(reflex) + list(repair.accepted) + list(option_rollouts)
    returns = [item.episode_return for item in all_seen]
    best = max_or_zero(returns)
    interaction_steps = (
        trajectory_steps(bootstrap)
        + trajectory_steps(reflex)
        + int(repair.interaction_steps)
        + trajectory_steps(option_rollouts)
    )
    selected_steps = trajectory_steps(selected)
    charged_steps = max(int(interaction_steps), int(selected_steps))
    stats = {
        "collector": "option_archive",
        "collector_samples": int(batch.observations.shape[0]),
        "collector_interaction_steps": int(charged_steps),
        "collector_episode_count": int(len(all_seen)),
        "collector_best_return": float(best),
        "collector_return_mean": mean_or_zero(returns),
        "collector_solve_gap": float(float(config.solve_return) - best),
        "failure_window_count": int(failure_window_count),
        "self_demo_count": int(len(self_demos)),
        "self_demo_best_return": max_or_zero([item.episode_return for item in self_demos]),
        "option_segment_count": int(len(segments)),
        "option_selected_steps": int(selected_steps),
        "option_selected_fraction": float(selected_steps / interaction_steps) if interaction_steps > 0 else 0.0,
        "option_rollout_count": int(len(option_rollouts)),
        "option_rollout_return_max": max_or_zero([item.episode_return for item in option_rollouts]),
        "option_value_target_max": max_return_to_go(batch, discount=float(config.discount)),
        "accepted_data_real_option_fraction": accepted_option_fraction(option_rollouts, selected),
        **repair.diagnostics(),
        **option_stats,
    }
    return stats


def aggregate_option_stats(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return empty_option_stats()
    keys = sorted({key for row in rows for key in row})
    return {key: mean_or_zero([row.get(key, 0.0) for row in rows]) for key in keys}


def empty_option_stats() -> dict[str, float]:
    return {
        "option_count": 0.0,
        "option_reuse_rate": 0.0,
        "option_real_roi_mean": 0.0,
        "option_predicted_roi_mean": 0.0,
        "option_prediction_error": 0.0,
        "option_planner_return": 0.0,
        "raw_fallback_rate": 1.0,
    }


def frontier_floor(config: AdvancedGymMPCConfig, trajectories: list[ReplayTrajectory]) -> float:
    best = max_or_zero([item.episode_return for item in trajectories])
    gap = abs(float(config.solve_return) - best)
    window = max(float(config.success_archive_frontier_floor), float(config.success_archive_frontier_gap_fraction) * gap)
    return float(best - window)


def accepted_option_fraction(option_rollouts: list[ReplayTrajectory], selected: list[ReplayTrajectory]) -> float:
    if not selected:
        return 0.0
    option_ids = {id(item) for item in option_rollouts}
    option_steps = sum(item.length for item in selected if id(item) in option_ids)
    total_steps = sum(item.length for item in selected)
    return float(option_steps / total_steps) if total_steps else 0.0


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


__all__ = ["collect_option_archive_transitions"]
