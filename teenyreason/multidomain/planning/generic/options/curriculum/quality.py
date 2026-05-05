"""Generic partial-win and archive-quality scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...collection.trajectory import ReplayTrajectory
from ..factors import ControlFactorModel
from .levels import CurriculumState, action_jerk, passed_count, state_change_stability


@dataclass(frozen=True)
class QualityRow:
    trajectory: ReplayTrajectory
    score: float
    trusted: bool
    reason: str
    metrics: dict[str, float]


def trajectory_quality_metrics(
    trajectory: ReplayTrajectory,
    state: CurriculumState,
    factor_model: ControlFactorModel,
) -> dict[str, float]:
    delta_z = factor_model.delta_z(trajectory.observations, trajectory.next_observations)
    reward_progress = reward_aligned_progress(delta_z, factor_model)
    terminal = float(trajectory.dones.size > 0 and trajectory.dones[-1] > 0.5)
    terminal_shock = terminal_shock_score(trajectory, factor_model)
    smoothness = float(1.0 / (1.0 + action_jerk(trajectory.actions)))
    stability = state_change_stability(trajectory)
    rhythmicity = action_rhythm_score(trajectory.actions)
    value = float(np.max(trajectory.returns_to_go)) if trajectory.returns_to_go.size else float(trajectory.episode_return)
    return {
        "return": float(trajectory.episode_return),
        "survival": float(trajectory.length),
        "value": float(value),
        "terminal": float(terminal),
        "return_lift": float(trajectory.episode_return - state.best_return),
        "survival_lift": float(trajectory.length - state.best_survival),
        "return_level": float(passed_count(float(trajectory.episode_return), state.return_thresholds)),
        "survival_level": float(passed_count(float(trajectory.length), tuple(float(item) for item in state.survival_thresholds))),
        "reward_aligned_progress": float(reward_progress),
        "terminal_shock": float(terminal_shock),
        "state_change_stability": float(stability),
        "action_smoothness": float(smoothness),
        "action_rhythm": float(rhythmicity),
    }


def trajectory_quality_score(
    trajectory: ReplayTrajectory,
    state: CurriculumState,
    factor_model: ControlFactorModel,
) -> float:
    metrics = trajectory_quality_metrics(trajectory, state, factor_model)
    return quality_score_from_metrics(metrics)


def quality_score_from_metrics(metrics: dict[str, float]) -> float:
    return float(
        metrics["return"]
        + 0.08 * metrics["survival"]
        + 4.0 * metrics["return_level"]
        + 2.5 * metrics["survival_level"]
        + 3.0 * metrics["reward_aligned_progress"]
        + 2.0 * metrics["state_change_stability"]
        + 1.0 * metrics["action_smoothness"]
        + 0.8 * metrics["action_rhythm"]
        - 8.0 * metrics["terminal"]
        - 2.5 * metrics["terminal_shock"]
    )


def score_archive(
    trajectories: list[ReplayTrajectory],
    state: CurriculumState,
    factor_model: ControlFactorModel,
) -> list[QualityRow]:
    rows = []
    for trajectory in trajectories:
        metrics = trajectory_quality_metrics(trajectory, state, factor_model)
        rows.append(
            QualityRow(
                trajectory=trajectory,
                score=quality_score_from_metrics(metrics),
                trusted=False,
                reason="untrusted",
                metrics=metrics,
            )
        )
    return rows


def select_quality_training_trajectories(
    *,
    base: list[ReplayTrajectory],
    repairs: list[ReplayTrajectory],
    options: list[ReplayTrajectory],
    self_demos: list[ReplayTrajectory],
    state: CurriculumState,
    factor_model: ControlFactorModel,
    keep_count: int,
    quality_gate: bool,
) -> tuple[list[ReplayTrajectory], dict[str, object]]:
    if not quality_gate:
        selected = unique_trajectories(base + self_demos + repairs + options)
        return selected, quality_selection_diagnostics(selected, [], state, factor_model)
    base_rows = mark_trusted_frontier(score_archive(base, state, factor_model), keep_count=max(1, int(keep_count)))
    repair_rows = mark_trusted_improvements(score_archive(repairs, state, factor_model), state)
    option_rows = mark_trusted_improvements(score_archive(options, state, factor_model), state)
    demo_rows = mark_all(score_archive(self_demos, state, factor_model), reason="self_demo")
    trusted = [row.trajectory for row in base_rows + repair_rows + option_rows + demo_rows if row.trusted]
    if not trusted:
        trusted = [row.trajectory for row in sorted(base_rows, key=lambda item: item.score, reverse=True)[: max(1, int(keep_count))]]
    rejected = [row.trajectory for row in repair_rows + option_rows if not row.trusted]
    selected = unique_trajectories(trusted)
    return selected, quality_selection_diagnostics(selected, rejected, state, factor_model)


def mark_trusted_frontier(rows: list[QualityRow], *, keep_count: int) -> list[QualityRow]:
    ranked = sorted(rows, key=lambda item: item.score, reverse=True)
    trusted_ids = {id(row.trajectory) for row in ranked[: max(1, int(keep_count))]}
    return [
        QualityRow(row.trajectory, row.score, id(row.trajectory) in trusted_ids, "frontier" if id(row.trajectory) in trusted_ids else "below_frontier", row.metrics)
        for row in rows
    ]


def mark_trusted_improvements(rows: list[QualityRow], state: CurriculumState) -> list[QualityRow]:
    if not rows:
        return []
    scores = np.asarray([row.score for row in rows], dtype=np.float32)
    score_floor = float(np.percentile(scores, 65.0))
    out = []
    for row in rows:
        reason = improvement_reason(row, state, score_floor)
        out.append(QualityRow(row.trajectory, row.score, reason != "rejected", reason, row.metrics))
    return out


def mark_all(rows: list[QualityRow], *, reason: str) -> list[QualityRow]:
    return [QualityRow(row.trajectory, row.score, True, reason, row.metrics) for row in rows]


def improvement_reason(row: QualityRow, state: CurriculumState, score_floor: float) -> str:
    metrics = row.metrics
    if metrics["return_lift"] > 0.0:
        return "return_lift"
    if metrics["survival_lift"] > 0.0 and metrics["return"] >= state.frontier_return - 25.0:
        return "survival_lift"
    if metrics["terminal"] < 0.5 and metrics["survival"] >= 0.75 * max(1.0, state.best_survival):
        return "terminal_risk_reduction"
    if row.score >= score_floor and metrics["reward_aligned_progress"] > 0.0:
        return "partial_win_quality"
    return "rejected"


def quality_selection_diagnostics(
    selected: list[ReplayTrajectory],
    rejected: list[ReplayTrajectory],
    state: CurriculumState,
    factor_model: ControlFactorModel,
) -> dict[str, object]:
    selected_rows = score_archive(selected, state, factor_model) if selected else []
    best_metrics = selected_rows[0].metrics if selected_rows else {}
    if selected_rows:
        best_metrics = max(selected_rows, key=lambda item: item.score).metrics
    return {
        "quality_gate_enabled": 1.0,
        "quality_selected_trajectory_count": int(len(selected)),
        "quality_rejected_trajectory_count": int(len(rejected)),
        "quality_selected_steps": int(sum(item.length for item in selected)),
        "quality_best_score": max_or_zero([row.score for row in selected_rows]),
        "quality_best_return_lift": float(best_metrics.get("return_lift", 0.0)),
        "quality_best_survival_lift": float(best_metrics.get("survival_lift", 0.0)),
        "quality_reward_aligned_progress": float(best_metrics.get("reward_aligned_progress", 0.0)),
        "quality_terminal_shock": float(best_metrics.get("terminal_shock", 0.0)),
        "quality_action_rhythm": float(best_metrics.get("action_rhythm", 0.0)),
    }


def reward_aligned_progress(delta_z: np.ndarray, factor_model: ControlFactorModel) -> float:
    if delta_z.size == 0:
        return 0.0
    weights = np.asarray(factor_model.delta_reward_weights, dtype=np.float32).reshape(-1)
    progress = np.asarray(delta_z, dtype=np.float32) @ weights
    return float(np.mean(np.maximum(progress, 0.0)))


def terminal_shock_score(trajectory: ReplayTrajectory, factor_model: ControlFactorModel) -> float:
    if trajectory.length <= 0 or not (trajectory.dones.size and trajectory.dones[-1] > 0.5):
        return 0.0
    delta_z = factor_model.delta_z(trajectory.observations[-1:], trajectory.next_observations[-1:])
    reward_penalty = max(0.0, -float(trajectory.rewards[-1])) / 100.0
    return float(np.linalg.norm(delta_z.reshape(-1)) + reward_penalty)


def action_rhythm_score(actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[0] < 4:
        return 0.0
    centered = actions - np.mean(actions, axis=0, keepdims=True)
    left = centered[:-2]
    right = centered[2:]
    denom = float(np.linalg.norm(left) * np.linalg.norm(right) + 1e-4)
    return float(max(0.0, float(np.sum(left * right)) / denom))


def unique_trajectories(trajectories: list[ReplayTrajectory]) -> list[ReplayTrajectory]:
    out: list[ReplayTrajectory] = []
    seen: set[int] = set()
    for trajectory in trajectories:
        key = id(trajectory)
        if key not in seen:
            seen.add(key)
            out.append(trajectory)
    return out


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


__all__ = [
    "QualityRow",
    "select_quality_training_trajectories",
    "trajectory_quality_metrics",
    "trajectory_quality_score",
]
