"""Generic intrinsic-skill practice with model proposals and real validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teenyreason.multidomain.planning.generic.collection.trajectory import ReplayTrajectory, rows_to_trajectory
from teenyreason.multidomain.planning.generic.config import AdvancedGymMPCConfig
from teenyreason.multidomain.planning.generic.options.curriculum.hindsight import GoalConditionedHindsightPolicy
from teenyreason.multidomain.planning.generic.options.factors import ControlFactorModel
from teenyreason.multidomain.planning.generic.options.failures import FailureFrontierMiner, FailureWindow
from teenyreason.multidomain.planning.generic.options.priors import MotorPriorModel
from teenyreason.multidomain.planning.generic.options.repair import CounterfactualRepairSearcher, repair_candidates

from .discovery import DiscoveryContext, build_discovery_context
from .reflex import action_sequence, feedback_repair_candidates
from .schema import IntrinsicGoal, SkillRecord


@dataclass
class SkillPracticeResult:
    """Output of one intrinsic skill-discovery round."""

    skills: list[SkillRecord]
    accepted_trajectories: list[ReplayTrajectory]
    goal_actor: GoalConditionedHindsightPolicy | None
    diagnostics: dict[str, float]


def run_skill_discovery_round(
    config: AdvancedGymMPCConfig,
    trajectories: list[ReplayTrajectory],
    model,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    round_idx: int,
) -> SkillPracticeResult:
    if not bool(config.skill_discovery_enabled) or not trajectories:
        return empty_result()
    windows = FailureFrontierMiner(
        horizon=max(1, int(config.skill_branch_steps)),
        max_windows=max(1, int(config.skill_search_windows)),
    ).mine(trajectories, ControlFactorModel.fit(trajectories, action_low, action_high))
    if not windows:
        windows = synthetic_frontier_windows(trajectories, action_low, action_high, max_windows=int(config.skill_search_windows))
    context = build_discovery_context(
        trajectories,
        action_low,
        action_high,
        max_islands=int(config.skill_stable_island_count),
        max_goals=int(config.skill_intrinsic_goal_count),
        model=model,
        windows=windows,
    )
    motor_prior = MotorPriorModel.fit(trajectories, context.factor_model, action_low, action_high)
    goal_actor = GoalConditionedHindsightPolicy.fit(
        trajectories,
        context.factor_model,
        action_low,
        action_high,
        horizon=max(1, int(config.skill_goal_actor_horizon)),
        ridge=float(config.option_actor_ridge),
    )
    searcher = CounterfactualRepairSearcher(
        env_name=config.env_name,
        discount=float(config.discount),
        action_low=action_low,
        action_high=action_high,
        branch_steps=max(1, int(config.skill_branch_steps)),
        attempts_per_window=1,
        accept_delta=float(config.skill_accept_return_lift),
        survival_lift=int(config.skill_accept_survival_lift),
        seed=int(config.seed + 260_000 + int(round_idx) * 997),
    )
    search_result = search_skills(
        config,
        trajectories,
        context,
        model,
        action_low,
        action_high,
        windows,
        motor_prior,
        goal_actor,
        searcher,
        round_idx=round_idx,
    )
    diagnostics = {
        **discovery_diagnostics(context),
        **goal_actor_diagnostics(goal_actor),
        **search_result["diagnostics"],
    }
    return SkillPracticeResult(
        skills=search_result["skills"],
        accepted_trajectories=search_result["accepted_trajectories"],
        goal_actor=goal_actor,
        diagnostics=diagnostics,
    )


def search_skills(
    config: AdvancedGymMPCConfig,
    trajectories: list[ReplayTrajectory],
    context: DiscoveryContext,
    model,
    action_low: np.ndarray,
    action_high: np.ndarray,
    windows: list[FailureWindow],
    motor_prior: MotorPriorModel,
    goal_actor: GoalConditionedHindsightPolicy,
    searcher: CounterfactualRepairSearcher,
    *,
    round_idx: int,
) -> dict[str, object]:
    selected_windows = windows[: max(1, int(config.skill_search_windows))]
    if context.intrinsic_goals:
        goals = context.intrinsic_goals
    elif windows:
        goals = [frontier_goal_from_window(windows[0], trajectories)]
    else:
        goals = []
    skill_id = int(round_idx) * 10_000
    skills: list[SkillRecord] = []
    accepted_trajectories: list[ReplayTrajectory] = []
    attempted = 0
    proposed_count = 0
    feedback_count = 0
    survivor_count = 0
    validation_steps = 0
    return_lifts: list[float] = []
    survival_lifts: list[float] = []
    terminal_avoids: list[float] = []
    for window_index, window in enumerate(selected_windows):
        if not (0 <= int(window.trajectory_index) < len(trajectories)):
            continue
        goal = goals[window_index % len(goals)] if goals else frontier_goal_from_window(window, trajectories)
        candidate_goals = [goal] + goals[: min(3, len(goals))]
        candidates = propose_candidate_sequences(
            config,
            trajectories[int(window.trajectory_index)],
            window,
            candidate_goals,
            action_low,
            action_high,
            motor_prior,
            goal_actor,
            context.factor_model,
            seed=int(config.seed + 261_000 + int(round_idx) * 997 + window_index),
        )
        proposed_count += len(candidates)
        feedback_count += sum(1 for sequence, _goal in candidates if hasattr(sequence, "action"))
        top_candidates = successive_halving_candidates(
            config,
            trajectories[int(window.trajectory_index)],
            window,
            context,
            model,
            candidates,
        )
        survivor_count += len(top_candidates)
        for sequence, candidate_goal, _score in top_candidates[: max(1, int(config.skill_real_validate_top))]:
            branch = searcher.branch(
                trajectories[int(window.trajectory_index)],
                window,
                sequence,
                seed_offset=window_index * 997 + attempted,
            )
            attempted += 1
            validation_steps += int(branch["prefix_steps"]) + int(branch["branch_steps"])
            return_lifts.append(float(branch["return_lift"]))
            survival_lifts.append(float(branch["survival_lift"]))
            terminal_avoids.append(float(branch["terminal_avoid"]))
            if not bool(branch["accepted"]):
                continue
            trajectory = rows_to_trajectory(branch["rows"], seed=int(trajectories[int(window.trajectory_index)].seed), discount=float(config.discount))
            record = skill_record_from_branch(
                skill_id=skill_id + len(skills),
                goal=candidate_goal,
                rows=branch["rows"],
                prefix_steps=int(branch["prefix_steps"]),
                factor_model=context.factor_model,
                return_lift=float(branch["return_lift"]),
                survival_lift=float(branch["survival_lift"]),
                terminal_avoid=float(branch["terminal_avoid"]),
            )
            if trajectory is not None and record is not None:
                accepted_trajectories.append(trajectory)
                skills.append(record)
    diagnostics = {
        "skill_search_window_count": float(len(selected_windows)),
        "skill_candidate_count": float(proposed_count),
        "skill_feedback_candidate_count": float(feedback_count),
        "skill_halving_survivor_count": float(survivor_count),
        "skill_real_validation_count": float(attempted),
        "skill_real_validation_steps": float(validation_steps),
        "skill_accept_count": float(len(skills)),
        "skill_accept_rate": float(len(skills) / attempted) if attempted else 0.0,
        "skill_return_lift_mean": mean(return_lifts),
        "skill_return_lift_max": max_or_zero(return_lifts),
        "skill_survival_lift_mean": mean(survival_lifts),
        "skill_survival_lift_max": max_or_zero(survival_lifts),
        "skill_terminal_avoid_count": float(np.sum(np.asarray(terminal_avoids, dtype=np.float32))),
        "skill_external_reward_last_weight": 0.20,
    }
    return {"skills": skills, "accepted_trajectories": accepted_trajectories, "diagnostics": diagnostics}


def propose_candidate_sequences(
    config: AdvancedGymMPCConfig,
    trajectory: ReplayTrajectory,
    window: FailureWindow,
    goals: list[IntrinsicGoal],
    action_low: np.ndarray,
    action_high: np.ndarray,
    motor_prior: MotorPriorModel,
    goal_actor: GoalConditionedHindsightPolicy,
    factor_model: ControlFactorModel,
    *,
    seed: int,
) -> list[tuple[object, IntrinsicGoal]]:
    count = max(4, int(config.skill_candidate_count))
    base = repair_candidates(
        trajectory,
        window,
        action_low,
        action_high,
        count=count,
        seed=seed,
        motor_prior=motor_prior,
        inverse_candidates=max(1, int(config.option_inverse_repair_candidates)),
    )
    primary_goal = goals[0] if goals else frontier_goal_from_window(window, [trajectory])
    candidates = [(sequence, primary_goal) for sequence in base]
    start_obs = trajectory.observations[int(window.start)]
    duration = max(1, int(config.skill_branch_steps))
    for goal in goals:
        candidates.append((goal_actor.action_sequence(start_obs, goal.target_delta, duration=duration), goal))
    candidates.extend(
        feedback_repair_candidates(
            config,
            trajectory,
            window,
            goals,
            factor_model,
            action_low,
            action_high,
            seed=seed + 13_000,
        )
    )
    return candidates


def successive_halving_candidates(
    config: AdvancedGymMPCConfig,
    trajectory: ReplayTrajectory,
    window: FailureWindow,
    context: DiscoveryContext,
    model,
    candidates: list[tuple[object, IntrinsicGoal]],
) -> list[tuple[object, IntrinsicGoal, float]]:
    if not candidates:
        return []
    current = [(sequence, goal, 0.0) for sequence, goal in candidates]
    budgets = unique_budgets([5, 15, int(config.skill_branch_steps)])
    keep_counts = [max(4, int(config.skill_halving_keep_count) * 2), max(2, int(config.skill_halving_keep_count)), max(1, int(config.skill_real_validate_top))]
    for budget, keep in zip(budgets, keep_counts):
        scored = []
        for sequence, goal, _old_score in current:
            truncated = action_sequence(
                sequence,
                model,
                trajectory.observations[int(window.start)],
                duration=max(1, int(budget)),
            )
            score = score_sequence_candidate(config, trajectory, window, context, model, truncated, goal)
            scored.append((sequence, goal, score))
        scored.sort(key=lambda item: item[2], reverse=True)
        current = scored[: max(1, min(int(keep), len(scored)))]
    return current


def score_sequence_candidate(
    config: AdvancedGymMPCConfig,
    trajectory: ReplayTrajectory,
    window: FailureWindow,
    context: DiscoveryContext,
    model,
    sequence: np.ndarray,
    goal: IntrinsicGoal,
) -> float:
    start_obs = trajectory.observations[int(window.start)]
    predicted_obs, model_score, uncertainty = rollout_model_score(config, model, start_obs, sequence)
    target_delta = np.asarray(goal.target_delta, dtype=np.float32).reshape(-1)
    actual_delta = context.factor_model.delta_z(start_obs.reshape(1, -1), predicted_obs.reshape(1, -1))[0]
    target_norm = float(np.linalg.norm(target_delta) + 1e-4)
    progress = float(actual_delta @ (target_delta / target_norm)) if target_norm > 1e-4 else -float(np.linalg.norm(actual_delta))
    stable_bonus = stable_island_bonus(context, predicted_obs)
    smoothness = action_smoothness(sequence)
    return float(0.45 * model_score + 0.25 * progress + 0.20 * stable_bonus + 0.10 * smoothness - 0.15 * uncertainty)


def rollout_model_score(config: AdvancedGymMPCConfig, model, observation: np.ndarray, sequence: np.ndarray) -> tuple[np.ndarray, float, float]:
    sequence = np.asarray(sequence, dtype=np.float32)
    if hasattr(model, "score_sequence_components"):
        components = model.score_sequence_components(
            observation,
            sequence.reshape(1, sequence.shape[0], -1),
            discount=float(config.discount),
            done_penalty=float(config.done_penalty),
            uncertainty_penalty=float(config.uncertainty_penalty),
        )
        model_score = float(np.asarray(components["score"], dtype=np.float32).reshape(-1)[0])
        uncertainty = float(np.asarray(components["uncertainty_total"], dtype=np.float32).reshape(-1)[0])
    else:
        model_score = 0.0
        uncertainty = 0.0
    obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    for action in sequence:
        pred = model.predict_batch(obs, np.asarray(action, dtype=np.float32).reshape(1, -1))
        obs = np.asarray(pred["next_observation"], dtype=np.float32).reshape(1, -1)
    return obs.reshape(-1).astype(np.float32), float(model_score), float(uncertainty)


def skill_record_from_branch(
    *,
    skill_id: int,
    goal: IntrinsicGoal,
    rows: list[dict[str, np.ndarray | float]],
    prefix_steps: int,
    factor_model,
    return_lift: float,
    survival_lift: float,
    terminal_avoid: float,
) -> SkillRecord | None:
    branch_rows = rows[max(0, int(prefix_steps)) :]
    if not branch_rows:
        return None
    start = np.asarray(branch_rows[0]["observation"], dtype=np.float32).reshape(-1)
    end = np.asarray(branch_rows[-1]["next_observation"], dtype=np.float32).reshape(-1)
    actions = np.asarray([row["action"] for row in branch_rows], dtype=np.float32)
    outcome_delta = factor_model.delta_z(start.reshape(1, -1), end.reshape(1, -1))[0]
    goal_alignment = float(outcome_delta @ goal.target_delta / (np.linalg.norm(outcome_delta) * np.linalg.norm(goal.target_delta) + 1e-4))
    reliability = 0.35 + 0.20 * np.tanh(max(0.0, return_lift) / 10.0)
    reliability += 0.20 * np.tanh(max(0.0, survival_lift) / 12.0)
    reliability += 0.15 * max(0.0, goal_alignment)
    reliability += 0.10 * float(terminal_avoid > 0.5)
    return SkillRecord(
        skill_id=int(skill_id),
        goal=goal,
        initiation_observation=start,
        termination_observation=end,
        actions=actions.astype(np.float32),
        outcome_delta=outcome_delta.astype(np.float32),
        real_return_lift=float(return_lift),
        survival_lift=float(survival_lift),
        terminal_avoid=float(terminal_avoid),
        reliability=float(np.clip(reliability, 0.05, 1.0)),
    )


def synthetic_frontier_windows(
    trajectories: list[ReplayTrajectory],
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    max_windows: int,
) -> list[FailureWindow]:
    factor_model = ControlFactorModel.fit(trajectories, action_low, action_high)
    ranked: list[tuple[float, FailureWindow]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        if trajectory.length <= 1:
            continue
        scores = float(np.max(trajectory.returns_to_go)) - trajectory.returns_to_go + 2.0 * trajectory.dones
        for step in np.argsort(scores)[::-1][:2]:
            start = max(0, int(step) - 1)
            end = min(trajectory.length, start + 8)
            if end <= start:
                continue
            delta = factor_model.delta_z(trajectory.observations[start : start + 1], trajectory.next_observations[end - 1 : end])[0]
            target = factor_model.repair_target_delta(delta)
            window = FailureWindow(
                trajectory_index=int(trajectory_index),
                start=int(start),
                end=int(end),
                priority=float(scores[int(step)]),
                target_delta=target,
                reference_return=float(np.sum(trajectory.rewards[start:end])),
                reference_survival=int(end - start),
            )
            ranked.append((float(scores[int(step)]), window))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[: max(1, int(max_windows))]]


def frontier_goal_from_window(window: FailureWindow, trajectories: list[ReplayTrajectory]) -> IntrinsicGoal:
    trajectory = trajectories[int(window.trajectory_index)]
    return IntrinsicGoal(
        goal_id=0,
        goal_kind="extend_survival_from_frontier",
        target_delta=np.asarray(window.target_delta, dtype=np.float32).reshape(-1),
        anchor_observation=trajectory.observations[int(window.start)].astype(np.float32).copy(),
        priority=float(window.priority),
        source="failure_frontier",
    )


def discovery_diagnostics(context: DiscoveryContext) -> dict[str, float]:
    return {
        "skill_factor_count": float(context.controllable_scores.size),
        "skill_controllable_factor_count": float(np.sum(context.controllable_scores > 0.15)),
        "skill_danger_factor_count": float(np.sum(context.danger_scores > 0.15)),
        "skill_stable_island_count": float(len(context.stable_islands)),
        "skill_stable_island_score_mean": mean([item.score for item in context.stable_islands]),
        "skill_intrinsic_goal_count": float(len(context.intrinsic_goals)),
    }


def goal_actor_diagnostics(goal_actor: GoalConditionedHindsightPolicy | None) -> dict[str, float]:
    if goal_actor is None:
        return {"skill_goal_actor_train_rows": 0.0, "skill_goal_actor_train_loss": 0.0}
    return {
        "skill_goal_actor_train_rows": float(goal_actor.train_rows),
        "skill_goal_actor_train_loss": float(goal_actor.train_loss),
    }


def stable_island_bonus(context: DiscoveryContext, observation: np.ndarray) -> float:
    if not context.stable_islands:
        return 0.0
    obs_z = ((np.asarray(observation, dtype=np.float32).reshape(1, -1) - context.factor_model.obs_mean.reshape(1, -1)) / context.factor_model.obs_std.reshape(1, -1))[0]
    distances = [float(np.mean(np.square(obs_z - island.factor_center))) for island in context.stable_islands]
    return float(1.0 / (1.0 + min(distances)))


def action_smoothness(sequence: np.ndarray) -> float:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape[0] <= 1:
        return 1.0
    jerk = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
    return float(1.0 / (1.0 + float(np.mean(jerk))))


def unique_budgets(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if int(value) > 0 and int(value) not in out:
            out.append(int(value))
    return out


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


def max_or_zero(values: list[float]) -> float:
    return float(np.max(np.asarray(values, dtype=np.float32))) if values else 0.0


def empty_result() -> SkillPracticeResult:
    return SkillPracticeResult(skills=[], accepted_trajectories=[], goal_actor=None, diagnostics={})


__all__ = ["SkillPracticeResult", "run_skill_discovery_round"]
