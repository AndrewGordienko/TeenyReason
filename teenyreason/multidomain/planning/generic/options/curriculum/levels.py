"""Generic curriculum levels for deliberate-practice repair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...collection.trajectory import ReplayTrajectory


@dataclass(frozen=True)
class CurriculumState:
    best_return: float
    best_survival: int
    frontier_return: float
    return_thresholds: tuple[float, ...]
    survival_thresholds: tuple[int, ...]

    @classmethod
    def from_trajectories(cls, trajectories: list[ReplayTrajectory], *, solve_return: float, control_steps: int) -> "CurriculumState":
        returns = [float(item.episode_return) for item in trajectories]
        survival = [int(item.length) for item in trajectories]
        best_return = max(returns) if returns else 0.0
        best_survival = max(survival) if survival else 0
        return_thresholds = generic_return_thresholds(best_return, float(solve_return))
        survival_thresholds = generic_survival_thresholds(best_survival, int(control_steps))
        return cls(
            best_return=float(best_return),
            best_survival=int(best_survival),
            frontier_return=float(best_return),
            return_thresholds=return_thresholds,
            survival_thresholds=survival_thresholds,
        )


def trajectory_curriculum_score(trajectory: ReplayTrajectory, state: CurriculumState) -> float:
    terminal = float(trajectory.dones.size > 0 and trajectory.dones[-1] > 0.5)
    smoothness = action_jerk(trajectory.actions)
    return_lift = float(trajectory.episode_return - state.frontier_return)
    survival_lift = float(trajectory.length - state.best_survival)
    return_level = passed_count(float(trajectory.episode_return), state.return_thresholds)
    survival_level = passed_count(float(trajectory.length), tuple(float(item) for item in state.survival_thresholds))
    return float(
        trajectory.episode_return
        + 0.10 * trajectory.length
        + 4.0 * return_level
        + 2.0 * survival_level
        + 0.5 * max(return_lift, 0.0)
        + 0.05 * max(survival_lift, 0.0)
        - 8.0 * terminal
        - 0.25 * smoothness
    )


def curriculum_diagnostics(trajectories: list[ReplayTrajectory], state: CurriculumState) -> dict[str, object]:
    if not trajectories:
        return {
            "curriculum_best_return": 0.0,
            "curriculum_best_survival": 0,
            "curriculum_return_level": 0,
            "curriculum_survival_level": 0,
            "curriculum_state_change_stability": 0.0,
            "curriculum_action_smoothness": 0.0,
        }
    best = max(trajectories, key=lambda item: trajectory_curriculum_score(item, state))
    return {
        "curriculum_best_return": float(max(item.episode_return for item in trajectories)),
        "curriculum_best_survival": int(max(item.length for item in trajectories)),
        "curriculum_return_level": int(passed_count(float(best.episode_return), state.return_thresholds)),
        "curriculum_survival_level": int(passed_count(float(best.length), tuple(float(item) for item in state.survival_thresholds))),
        "curriculum_state_change_stability": state_change_stability(best),
        "curriculum_action_smoothness": float(1.0 / (1.0 + action_jerk(best.actions))),
    }


def generic_return_thresholds(best_return: float, solve_return: float) -> tuple[float, ...]:
    low = min(best_return, -100.0)
    if solve_return <= low:
        return (low,)
    raw = [low, -100.0, 0.0, 50.0, solve_return]
    raw.extend([best_return + 0.25 * (solve_return - best_return), best_return + 0.50 * (solve_return - best_return)])
    ordered = sorted({float(value) for value in raw if value > best_return - 200.0 and value <= solve_return})
    return tuple(ordered)


def generic_survival_thresholds(best_survival: int, control_steps: int) -> tuple[int, ...]:
    raw = [25, 50, 100, 200, int(control_steps)]
    raw.extend([best_survival + 25, best_survival + 75])
    return tuple(sorted({max(1, min(int(value), int(control_steps))) for value in raw}))


def passed_count(value: float, thresholds: tuple[float, ...]) -> int:
    return int(sum(1 for threshold in thresholds if float(value) >= float(threshold)))


def action_jerk(actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[0] <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(actions, axis=0))))


def state_change_stability(trajectory: ReplayTrajectory) -> float:
    deltas = np.asarray(trajectory.next_observations - trajectory.observations, dtype=np.float32)
    if deltas.shape[0] <= 1:
        return 0.0
    jerk = np.mean(np.abs(np.diff(deltas, axis=0)))
    return float(1.0 / (1.0 + jerk))


__all__ = ["CurriculumState", "curriculum_diagnostics", "trajectory_curriculum_score"]
