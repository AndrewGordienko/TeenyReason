"""Successive-halving repair search for internal curriculum practice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ......envs import make_env
from ....gym_mpc import assert_box_spaces
from ...collection.trajectory import ReplayTrajectory, rows_to_trajectory
from ..factors import ControlFactorModel
from ..failures import FailureWindow
from ..repair import row_return, step_with_action
from .hindsight import GoalConditionedHindsightPolicy
from .levels import CurriculumState, action_jerk


@dataclass(frozen=True)
class HalvingBranch:
    actions: np.ndarray
    source: str


@dataclass
class SuccessiveHalvingRepairResult:
    accepted: list[ReplayTrajectory]
    attempted: int
    cheap_attempted: int
    accepted_count: int
    interaction_steps: int
    return_lifts: list[float]
    survival_lifts: list[float]
    terminal_avoids: list[float]
    score_lifts: list[float]
    stage1_keep: int
    stage2_keep: int
    final_count: int

    def diagnostics(self) -> dict[str, object]:
        return {
            "curriculum_repair_attempt_count": int(self.attempted),
            "curriculum_repair_cheap_attempt_count": int(self.cheap_attempted),
            "curriculum_repair_accept_count": int(self.accepted_count),
            "curriculum_repair_accept_rate": float(self.accepted_count / self.attempted) if self.attempted else 0.0,
            "curriculum_repair_interaction_steps": int(self.interaction_steps),
            "curriculum_repair_stage1_keep": int(self.stage1_keep),
            "curriculum_repair_stage2_keep": int(self.stage2_keep),
            "curriculum_slow_teacher_finalists": int(self.final_count),
            "curriculum_repair_return_lift_mean": mean_or_zero(self.return_lifts),
            "curriculum_repair_return_lift_max": max_or_zero(self.return_lifts),
            "curriculum_repair_score_lift_max": max_or_zero(self.score_lifts),
            "curriculum_repair_survival_lift_max": max_or_zero(self.survival_lifts),
            "curriculum_repair_terminal_avoid_count": int(np.sum(np.asarray(self.terminal_avoids, dtype=np.float32))),
        }


class SuccessiveHalvingRepairSearcher:
    """Try many tiny repairs, extend only winners, keep real improvements."""

    def __init__(
        self,
        *,
        env_name: str,
        discount: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        initial_candidates: int,
        mid_candidates: int,
        final_candidates: int,
        short_steps: int,
        mid_steps: int,
        long_steps: int,
        accept_lift: float,
        seed: int,
    ):
        self.env_name = str(env_name)
        self.discount = float(discount)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.initial_candidates = max(1, int(initial_candidates))
        self.mid_candidates = max(1, int(mid_candidates))
        self.final_candidates = max(1, int(final_candidates))
        self.short_steps = max(1, int(short_steps))
        self.mid_steps = max(self.short_steps, int(mid_steps))
        self.long_steps = max(self.mid_steps, int(long_steps))
        self.accept_lift = float(accept_lift)
        self.seed = int(seed)

    def search(
        self,
        trajectories: list[ReplayTrajectory],
        windows: list[FailureWindow],
        curriculum_state: CurriculumState,
        factor_model: ControlFactorModel,
        hindsight_policy: GoalConditionedHindsightPolicy,
        *,
        motor_prior=None,
        inverse_candidates: int = 0,
        intrinsic_fraction: float = 0.25,
    ) -> SuccessiveHalvingRepairResult:
        accepted: list[ReplayTrajectory] = []
        stats = empty_stats()
        cheap_weights = fit_cheap_action_delta_weights(motor_prior, self.action_low, self.action_high)
        for window_index, window in enumerate(windows):
            if not (0 <= int(window.trajectory_index) < len(trajectories)):
                continue
            trajectory = trajectories[int(window.trajectory_index)]
            candidates = curriculum_candidates(
                trajectory,
                window,
                self.action_low,
                self.action_high,
                self.initial_candidates,
                self.long_steps,
                self.seed + window_index * 4099,
                factor_model,
                hindsight_policy,
                motor_prior=motor_prior,
                inverse_candidates=int(inverse_candidates),
                intrinsic_fraction=float(intrinsic_fraction),
            )
            stage1 = self.evaluate_cheap_stage(candidates, window, self.short_steps, factor_model, cheap_weights)
            stats["cheap_attempted"] = int(stats["cheap_attempted"]) + len(stage1)
            stage1 = keep_best(stage1, self.mid_candidates)
            stats["stage1_keep"] += len(stage1)
            stage2 = self.evaluate_cheap_stage([item["candidate"] for item in stage1], window, self.mid_steps, factor_model, cheap_weights)
            stats["cheap_attempted"] = int(stats["cheap_attempted"]) + len(stage2)
            stage2 = keep_best(stage2, self.final_candidates)
            stats["stage2_keep"] += len(stage2)
            final = self.evaluate_stage(trajectory, window, [item["candidate"] for item in stage2], self.long_steps, curriculum_state, window_index)
            stats["final_count"] += len(final)
            for result in final:
                record_result(stats, result)
                if bool(result["accepted"]):
                    candidate = rows_to_trajectory(result["rows"], seed=int(trajectory.seed), discount=self.discount)
                    if candidate is not None:
                        accepted.append(candidate)
        return SuccessiveHalvingRepairResult(
            accepted=accepted,
            attempted=int(stats["attempted"]),
            cheap_attempted=int(stats["cheap_attempted"]),
            accepted_count=len(accepted),
            interaction_steps=int(stats["interaction_steps"]),
            return_lifts=list(stats["return_lifts"]),
            survival_lifts=list(stats["survival_lifts"]),
            terminal_avoids=list(stats["terminal_avoids"]),
            score_lifts=list(stats["score_lifts"]),
            stage1_keep=int(stats["stage1_keep"]),
            stage2_keep=int(stats["stage2_keep"]),
            final_count=int(stats["final_count"]),
        )

    def evaluate_cheap_stage(
        self,
        candidates: list[HalvingBranch],
        window: FailureWindow,
        steps: int,
        factor_model: ControlFactorModel,
        cheap_weights: np.ndarray | None,
    ) -> list[dict[str, object]]:
        return [
            {
                "candidate": candidate,
                "score_lift": cheap_candidate_score(candidate, window, steps, factor_model, cheap_weights, self.action_low, self.action_high),
            }
            for candidate in candidates
        ]

    def evaluate_stage(
        self,
        trajectory: ReplayTrajectory,
        window: FailureWindow,
        candidates: list[HalvingBranch],
        steps: int,
        curriculum_state: CurriculumState,
        window_index: int,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, candidate in enumerate(candidates):
            result = self.branch(
                trajectory,
                window,
                candidate,
                branch_steps=int(steps),
                seed_offset=int(window_index * 997 + index),
                curriculum_state=curriculum_state,
            )
            rows.append({**result, "candidate": candidate})
        return rows

    def branch(
        self,
        trajectory: ReplayTrajectory,
        window: FailureWindow,
        candidate: HalvingBranch,
        *,
        branch_steps: int,
        seed_offset: int,
        curriculum_state: CurriculumState,
    ) -> dict[str, object]:
        max_steps = max(1, int(window.start) + int(branch_steps) + 1)
        env = make_env(self.env_name, max_episode_steps=max_steps)
        try:
            assert_box_spaces(env)
            observation, _info = env.reset(seed=int(trajectory.seed))
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            prefix_rows: list[dict[str, np.ndarray | float]] = []
            for prefix_step in range(max(0, int(window.start))):
                row, observation, done = step_with_action(env, observation, trajectory.actions[prefix_step])
                prefix_rows.append(row)
                if done:
                    return empty_branch(prefix_rows, candidate)
            branch_rows: list[dict[str, np.ndarray | float]] = []
            for local_step in range(max(1, int(branch_steps))):
                action = candidate.actions[min(local_step, candidate.actions.shape[0] - 1)]
                row, observation, done = step_with_action(env, observation, action)
                branch_rows.append(row)
                if done:
                    break
            return score_branch(trajectory, window, prefix_rows, branch_rows, candidate, curriculum_state, self.accept_lift)
        finally:
            env.close()


def score_branch(
    trajectory: ReplayTrajectory,
    window: FailureWindow,
    prefix_rows: list[dict[str, np.ndarray | float]],
    branch_rows: list[dict[str, np.ndarray | float]],
    candidate: HalvingBranch,
    curriculum_state: CurriculumState,
    accept_lift: float,
) -> dict[str, object]:
    start = int(window.start)
    reference_end = min(trajectory.length, start + len(branch_rows))
    reference_return = float(np.sum(trajectory.rewards[start:reference_end])) if reference_end > start else 0.0
    branch_return = row_return(branch_rows)
    return_lift = float(branch_return - reference_return)
    survival_lift = float(len(branch_rows) - max(0, reference_end - start))
    reference_terminal = bool(np.any(trajectory.dones[start:reference_end] > 0.5))
    branch_terminal = bool(branch_rows and float(branch_rows[-1]["done"]) > 0.5)
    terminal_avoid = float(reference_terminal and not branch_terminal)
    smoothness = action_jerk(candidate.actions[: max(1, len(branch_rows))])
    progress_bonus = 0.05 * max(0.0, curriculum_state.frontier_return - abs(reference_return))
    score_lift = float(return_lift + 0.35 * survival_lift + 5.0 * terminal_avoid - 3.0 * branch_terminal - 0.03 * smoothness + progress_bonus)
    accepted = bool(return_lift >= accept_lift or score_lift >= accept_lift or survival_lift >= 2.0 or terminal_avoid > 0.5)
    return {
        "rows": prefix_rows + branch_rows,
        "prefix_steps": len(prefix_rows),
        "branch_steps": len(branch_rows),
        "interaction_steps": len(prefix_rows) + len(branch_rows),
        "branch_return": branch_return,
        "reference_return": reference_return,
        "return_lift": return_lift,
        "survival_lift": survival_lift,
        "terminal_avoid": terminal_avoid,
        "score_lift": score_lift,
        "accepted": accepted,
        "candidate_source": candidate.source,
    }


def curriculum_candidates(
    trajectory: ReplayTrajectory,
    window: FailureWindow,
    action_low: np.ndarray,
    action_high: np.ndarray,
    count: int,
    duration: int,
    seed: int,
    factor_model: ControlFactorModel,
    hindsight_policy: GoalConditionedHindsightPolicy,
    *,
    motor_prior=None,
    inverse_candidates: int = 0,
    intrinsic_fraction: float = 0.25,
) -> list[HalvingBranch]:
    rng = np.random.default_rng(int(seed) + int(window.start) * 37 + int(trajectory.seed) * 13)
    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    duration = max(1, int(duration))
    center = np.clip(np.zeros_like(action_low), action_low, action_high)
    base = padded_base_actions(trajectory, window, duration, center)
    rows = [HalvingBranch(base, "reference_suffix"), HalvingBranch(np.repeat(center.reshape(1, -1), duration, axis=0), "center_hold")]
    if motor_prior is not None and int(inverse_candidates) > 0:
        for sequence in motor_prior.repair_sequences(window.target_delta, base, count=int(inverse_candidates), seed=int(seed)):
            rows.append(HalvingBranch(fit_duration(sequence, duration, center), "inverse_prior"))
    start_obs = trajectory.observations[min(max(0, int(window.start)), trajectory.length - 1)]
    for target in target_deltas(window, factor_model):
        rows.append(HalvingBranch(hindsight_policy.action_sequence(start_obs, target, duration=duration), "hindsight_goal"))
    rows.extend(intrinsic_action_sequences(action_low, action_high, duration, count=max(1, int(count * intrinsic_fraction))))
    scale = np.maximum(action_high - action_low, 1e-6)
    while len(rows) < max(1, int(count)):
        noise = rng.normal(0.0, 0.30, size=(duration, action_low.size)).astype(np.float32)
        smooth = np.cumsum(noise, axis=0) / np.sqrt(np.arange(duration, dtype=np.float32).reshape(-1, 1) + 1.0)
        rows.append(HalvingBranch(np.clip(base + smooth * scale, action_low, action_high).astype(np.float32), "smoothed_random"))
    return rows[: max(1, int(count))]


def target_deltas(window: FailureWindow, factor_model: ControlFactorModel) -> list[np.ndarray]:
    base = np.asarray(window.target_delta, dtype=np.float32).reshape(-1)
    scale = max(float(np.linalg.norm(base)), 1.0)
    rows = [base, -base]
    weights = np.asarray(factor_model.delta_reward_weights - np.maximum(factor_model.delta_terminal_weights, 0.0), dtype=np.float32)
    for axis in np.argsort(np.abs(weights))[::-1][: min(4, weights.size)]:
        target = np.zeros_like(base)
        target[int(axis)] = scale * (1.0 if weights[int(axis)] >= 0.0 else -1.0)
        rows.append(target)
        rows.append(-target)
    return rows


def fit_cheap_action_delta_weights(motor_prior, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray | None:
    if motor_prior is None or getattr(motor_prior, "actions", np.asarray([])).size == 0:
        return None
    actions = normalize_actions_local(np.asarray(motor_prior.actions, dtype=np.float32), action_low, action_high)
    deltas = np.asarray(motor_prior.delta_z, dtype=np.float32)
    if actions.shape[0] < max(2, actions.shape[1] + 1):
        return None
    x = np.concatenate([np.ones((actions.shape[0], 1), dtype=np.float32), actions], axis=1).astype(np.float64)
    y = deltas.astype(np.float64)
    reg = 1e-3 * np.eye(x.shape[1], dtype=np.float64)
    return np.linalg.solve(x.T @ x + reg, x.T @ y).astype(np.float32)


def cheap_candidate_score(
    candidate: HalvingBranch,
    window: FailureWindow,
    steps: int,
    factor_model: ControlFactorModel,
    cheap_weights: np.ndarray | None,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> float:
    duration = max(1, min(int(steps), candidate.actions.shape[0]))
    sequence = np.asarray(candidate.actions[:duration], dtype=np.float32)
    target = np.asarray(window.target_delta, dtype=np.float32).reshape(-1)
    if cheap_weights is None:
        predicted_delta = np.zeros_like(target)
    else:
        actions_z = normalize_actions_local(sequence, action_low, action_high)
        x = np.concatenate([np.ones((actions_z.shape[0], 1), dtype=np.float32), actions_z], axis=1)
        predicted_delta = np.mean(x @ cheap_weights, axis=0).astype(np.float32)
    target_norm = float(np.linalg.norm(target) + 1e-4)
    align = float(predicted_delta @ target / target_norm) if target.size else 0.0
    reward_gain = float(predicted_delta @ factor_model.delta_reward_weights) if predicted_delta.size else 0.0
    terminal_risk = float(predicted_delta @ np.maximum(factor_model.delta_terminal_weights, 0.0)) if predicted_delta.size else 0.0
    source_bonus = {
        "hindsight_goal": 0.35,
        "inverse_prior": 0.30,
        "rhythm": 0.10,
        "axis_increase": 0.05,
        "axis_decrease": 0.05,
        "reference_suffix": -0.10,
    }.get(candidate.source, 0.0)
    return float(align + 0.50 * reward_gain - 0.60 * terminal_risk - 0.03 * action_jerk(sequence) + source_bonus)


def normalize_actions_local(actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    return np.clip(2.0 * (actions - low) / np.maximum(high - low, 1e-6) - 1.0, -1.0, 1.0).astype(np.float32)


def intrinsic_action_sequences(action_low: np.ndarray, action_high: np.ndarray, duration: int, *, count: int) -> list[HalvingBranch]:
    center = np.clip(np.zeros_like(action_low), action_low, action_high)
    rows: list[HalvingBranch] = []
    phase = np.linspace(0.0, 2.0 * np.pi, max(1, int(duration)), dtype=np.float32).reshape(-1, 1)
    for axis in range(action_low.size):
        low = center.copy()
        high = center.copy()
        low[axis] = action_low[axis]
        high[axis] = action_high[axis]
        rows.append(HalvingBranch(np.repeat(low.reshape(1, -1), duration, axis=0), "axis_decrease"))
        rows.append(HalvingBranch(np.repeat(high.reshape(1, -1), duration, axis=0), "axis_increase"))
        wave = np.repeat(center.reshape(1, -1), duration, axis=0)
        wave[:, axis] = center[axis] + 0.5 * (action_high[axis] - action_low[axis]) * np.sin(phase[:, 0])
        rows.append(HalvingBranch(np.clip(wave, action_low, action_high).astype(np.float32), "rhythm"))
        if len(rows) >= max(1, int(count)):
            break
    return rows[: max(1, int(count))]


def padded_base_actions(trajectory: ReplayTrajectory, window: FailureWindow, duration: int, center: np.ndarray) -> np.ndarray:
    start = max(0, int(window.start))
    base = np.asarray(trajectory.actions[start : start + duration], dtype=np.float32)
    if base.shape[0] < duration:
        pad = np.repeat(center.reshape(1, -1), duration - base.shape[0], axis=0)
        base = np.concatenate([base, pad], axis=0)
    return base.astype(np.float32)


def fit_duration(sequence: np.ndarray, duration: int, center: np.ndarray) -> np.ndarray:
    row = np.asarray(sequence, dtype=np.float32)
    if row.shape[0] < duration:
        pad = np.repeat(center.reshape(1, -1), duration - row.shape[0], axis=0)
        row = np.concatenate([row, pad], axis=0)
    return row[:duration].astype(np.float32)


def keep_best(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    return sorted(rows, key=lambda item: float(item.get("score_lift", 0.0)), reverse=True)[: max(1, int(count))]


def empty_branch(prefix_rows: list[dict[str, np.ndarray | float]], candidate: HalvingBranch) -> dict[str, object]:
    return {
        "rows": prefix_rows,
        "prefix_steps": len(prefix_rows),
        "branch_steps": 0,
        "interaction_steps": len(prefix_rows),
        "return_lift": 0.0,
        "survival_lift": 0.0,
        "terminal_avoid": 0.0,
        "score_lift": 0.0,
        "accepted": False,
        "candidate_source": candidate.source,
    }


def empty_stats() -> dict[str, object]:
    return {
        "attempted": 0,
        "cheap_attempted": 0,
        "interaction_steps": 0,
        "return_lifts": [],
        "survival_lifts": [],
        "terminal_avoids": [],
        "score_lifts": [],
        "stage1_keep": 0,
        "stage2_keep": 0,
        "final_count": 0,
    }


def record_result(stats: dict[str, object], result: dict[str, object]) -> None:
    stats["attempted"] = int(stats["attempted"]) + 1
    stats["interaction_steps"] = int(stats["interaction_steps"]) + int(result.get("interaction_steps", 0))
    stats["return_lifts"].append(float(result.get("return_lift", 0.0)))
    stats["survival_lifts"].append(float(result.get("survival_lift", 0.0)))
    stats["terminal_avoids"].append(float(result.get("terminal_avoid", 0.0)))
    stats["score_lifts"].append(float(result.get("score_lift", 0.0)))


def mean_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.mean(np.asarray(rows, dtype=np.float32))) if rows else 0.0


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


__all__ = ["SuccessiveHalvingRepairResult", "SuccessiveHalvingRepairSearcher"]
