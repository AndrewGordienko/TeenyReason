"""Generic controllable-factor, danger, island, and intrinsic-goal mining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teenyreason.multidomain.planning.generic.collection.trajectory import ReplayTrajectory, trajectories_to_batch
from teenyreason.multidomain.planning.generic.options.factors import ControlFactorModel
from teenyreason.multidomain.planning.generic.options.failures import FailureWindow

from .schema import IntrinsicGoal, StableIsland


@dataclass(frozen=True)
class DiscoveryContext:
    """All generic structure mined from current real experience."""

    factor_model: ControlFactorModel
    controllable_scores: np.ndarray
    danger_scores: np.ndarray
    stable_islands: list[StableIsland]
    intrinsic_goals: list[IntrinsicGoal]


def build_discovery_context(
    trajectories: list[ReplayTrajectory],
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    max_islands: int,
    max_goals: int,
    model=None,
    windows: list[FailureWindow] | None = None,
) -> DiscoveryContext:
    factor_model = ControlFactorModel.fit(trajectories, action_low, action_high)
    batch = trajectories_to_batch(trajectories)
    controllable_scores = mine_controllable_factors(batch, factor_model, action_low, action_high)
    danger_scores = mine_danger_factors(factor_model)
    stable_islands = mine_stable_islands(
        trajectories,
        factor_model,
        controllable_scores,
        count=max_islands,
        model=model,
    )
    intrinsic_goals = generate_intrinsic_goals(
        trajectories,
        factor_model,
        controllable_scores,
        danger_scores,
        stable_islands,
        count=max_goals,
        windows=list(windows or []),
    )
    return DiscoveryContext(
        factor_model=factor_model,
        controllable_scores=controllable_scores,
        danger_scores=danger_scores,
        stable_islands=stable_islands,
        intrinsic_goals=intrinsic_goals,
    )


def mine_controllable_factors(
    batch,
    factor_model: ControlFactorModel,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Score observation dimensions by reliable action-conditioned movement."""
    delta = factor_model.delta_z(batch.observations, batch.next_observations)
    actions = normalize_actions(np.asarray(batch.actions, dtype=np.float32), action_low, action_high)
    scores = np.zeros((delta.shape[1],), dtype=np.float32)
    for obs_dim in range(delta.shape[1]):
        dim_scores: list[float] = []
        for action_dim in range(actions.shape[1]):
            dim_scores.append(abs(safe_corr(actions[:, action_dim], delta[:, obs_dim])))
        scores[obs_dim] = max(dim_scores) * float(np.std(delta[:, obs_dim]) + 1e-4)
    if float(np.max(scores)) > 1e-8:
        scores = scores / float(np.max(scores))
    return scores.astype(np.float32)


def mine_danger_factors(factor_model: ControlFactorModel) -> np.ndarray:
    """Score observation dimensions that predict terminal or bad-return movement."""
    terminal = np.maximum(np.asarray(factor_model.delta_terminal_weights, dtype=np.float32), 0.0)
    bad_reward = np.maximum(-np.asarray(factor_model.delta_reward_weights, dtype=np.float32), 0.0)
    danger = terminal + 0.75 * bad_reward
    if float(np.max(danger)) > 1e-8:
        danger = danger / float(np.max(danger))
    return danger.astype(np.float32)


def mine_stable_islands(
    trajectories: list[ReplayTrajectory],
    factor_model: ControlFactorModel,
    controllable_scores: np.ndarray,
    *,
    count: int,
    model=None,
) -> list[StableIsland]:
    candidates: list[tuple[float, StableIsland]] = []
    island_id = 0
    for trajectory in trajectories:
        surprises = transition_surprises(trajectory, factor_model, model)
        action_jerk = action_jerks(trajectory)
        state_jerk = state_jerks(trajectory, factor_model)
        reward_scale = max(1.0, float(np.std(trajectory.returns_to_go) + abs(np.mean(trajectory.returns_to_go))))
        for step in range(trajectory.length):
            survival = float((trajectory.length - step) / max(1, trajectory.length))
            terminal_risk = terminal_risk_near(trajectory, step)
            smoothness = 1.0 / (1.0 + float(action_jerk[step]) + 0.5 * float(state_jerk[step]))
            surprise = float(surprises[step])
            surprise_score = 1.0 / (1.0 + surprise)
            return_score = float(np.clip(trajectory.returns_to_go[step] / reward_scale, -1.0, 1.0))
            score = 0.35 * survival + 0.25 * smoothness + 0.20 * surprise_score + 0.20 * max(0.0, return_score)
            score -= 0.45 * terminal_risk
            obs = trajectory.observations[step].astype(np.float32).copy()
            factor_center = ((obs - factor_model.obs_mean) / factor_model.obs_std).astype(np.float32)
            support = local_support(trajectory, step, factor_model, controllable_scores)
            candidates.append(
                (
                    float(score),
                    StableIsland(
                        island_id=island_id,
                        center=obs,
                        factor_center=factor_center,
                        score=float(score),
                        survival=float(survival),
                        smoothness=float(smoothness),
                        terminal_risk=float(terminal_risk),
                        surprise=float(surprise),
                        support=int(support),
                    ),
                )
            )
            island_id += 1
    selected: list[StableIsland] = []
    for _score, island in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(island_distance(island, old) > 0.35 for old in selected):
            selected.append(island)
        if len(selected) >= max(1, int(count)):
            break
    return [replace_island_id(item, idx) for idx, item in enumerate(selected)]


def generate_intrinsic_goals(
    trajectories: list[ReplayTrajectory],
    factor_model: ControlFactorModel,
    controllable_scores: np.ndarray,
    danger_scores: np.ndarray,
    stable_islands: list[StableIsland],
    *,
    count: int,
    windows: list[FailureWindow],
) -> list[IntrinsicGoal]:
    goals: list[IntrinsicGoal] = []
    anchors = frontier_anchor_observations(trajectories, count=max(4, int(count)))
    for window in windows:
        anchor = trajectories[int(window.trajectory_index)].observations[int(window.start)]
        goals.append(make_goal("extend_survival_from_frontier", window.target_delta, anchor, window.priority, "failure_frontier"))
    for island in stable_islands:
        anchor = nearest_anchor(island.center, anchors, factor_model)
        target_delta = factor_model.delta_z(anchor.reshape(1, -1), island.center.reshape(1, -1))[0]
        goals.append(make_goal("return_to_stable_island", target_delta, anchor, island.score, "stable_island"))
        goals.append(make_goal("hold_factor_stable", np.zeros_like(target_delta), island.center, 0.5 * island.score, "stable_island"))
    repair_direction = factor_model.delta_reward_weights - np.maximum(factor_model.delta_terminal_weights, 0.0)
    if float(np.linalg.norm(repair_direction)) > 1e-6 and anchors:
        goals.append(make_goal("reduce_instability", repair_direction, anchors[0], float(np.max(danger_scores)), "danger_factor"))
    for factor_index in top_indices(controllable_scores, count=min(4, int(controllable_scores.size))):
        direction = np.zeros_like(controllable_scores, dtype=np.float32)
        direction[int(factor_index)] = 1.0
        anchor = anchors[int(factor_index) % len(anchors)] if anchors else factor_model.obs_mean
        goals.append(make_goal("increase_controllable_factor", direction, anchor, float(controllable_scores[int(factor_index)]), "controllable_factor"))
        goals.append(make_goal("decrease_controllable_factor", -direction, anchor, float(controllable_scores[int(factor_index)]), "controllable_factor"))
    phase_goal = repeated_phase_goal(trajectories, factor_model)
    if phase_goal is not None and anchors:
        goals.append(make_goal("make_repeated_phase_pattern", phase_goal, anchors[0], 0.5, "phase_pattern"))
    goals = [goal for goal in goals if np.all(np.isfinite(goal.target_delta))]
    goals.sort(key=lambda item: item.priority, reverse=True)
    return [replace_goal_id(goal, idx) for idx, goal in enumerate(goals[: max(1, int(count))])]


def make_goal(kind: str, target_delta: np.ndarray, anchor: np.ndarray, priority: float, source: str) -> IntrinsicGoal:
    return IntrinsicGoal(
        goal_id=0,
        goal_kind=str(kind),
        target_delta=np.asarray(target_delta, dtype=np.float32).reshape(-1),
        anchor_observation=np.asarray(anchor, dtype=np.float32).reshape(-1),
        priority=float(priority),
        source=str(source),
    )


def transition_surprises(trajectory: ReplayTrajectory, factor_model: ControlFactorModel, model) -> np.ndarray:
    if model is None:
        return np.zeros((trajectory.length,), dtype=np.float32)
    pred = model.predict_batch(trajectory.observations, trajectory.actions)
    errors = np.square((pred["next_observation"] - trajectory.next_observations) / factor_model.obs_std.reshape(1, -1))
    return np.mean(errors, axis=1).astype(np.float32)


def action_jerks(trajectory: ReplayTrajectory) -> np.ndarray:
    if trajectory.length <= 1:
        return np.zeros((trajectory.length,), dtype=np.float32)
    diffs = np.diff(trajectory.actions, axis=0, prepend=trajectory.actions[:1])
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


def state_jerks(trajectory: ReplayTrajectory, factor_model: ControlFactorModel) -> np.ndarray:
    delta = factor_model.delta_z(trajectory.observations, trajectory.next_observations)
    if trajectory.length <= 1:
        return np.zeros((trajectory.length,), dtype=np.float32)
    return np.linalg.norm(np.diff(delta, axis=0, prepend=delta[:1]), axis=1).astype(np.float32)


def terminal_risk_near(trajectory: ReplayTrajectory, step: int) -> float:
    if trajectory.length <= 0:
        return 0.0
    terminal_indices = np.flatnonzero(trajectory.dones > 0.5)
    if terminal_indices.size == 0:
        return 0.0
    distance = int(np.min(np.abs(terminal_indices - int(step))))
    return float(np.exp(-distance / 12.0))


def local_support(trajectory: ReplayTrajectory, step: int, factor_model: ControlFactorModel, controllable_scores: np.ndarray) -> int:
    obs = trajectory.observations
    center = obs[int(step)].reshape(1, -1)
    z = (obs - factor_model.obs_mean.reshape(1, -1)) / factor_model.obs_std.reshape(1, -1)
    center_z = (center - factor_model.obs_mean.reshape(1, -1)) / factor_model.obs_std.reshape(1, -1)
    weights = 0.25 + np.asarray(controllable_scores, dtype=np.float32).reshape(1, -1)
    distances = np.mean(np.square(z - center_z) * weights, axis=1)
    return int(np.sum(distances < 0.75))


def frontier_anchor_observations(trajectories: list[ReplayTrajectory], *, count: int) -> list[np.ndarray]:
    anchors: list[tuple[float, np.ndarray]] = []
    for trajectory in trajectories:
        for step in range(trajectory.length):
            score = float(trajectory.dones[step]) * 3.0 - float(trajectory.returns_to_go[step])
            anchors.append((score, trajectory.observations[step].astype(np.float32).copy()))
    anchors.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in anchors[: max(1, int(count))]]


def repeated_phase_goal(trajectories: list[ReplayTrajectory], factor_model: ControlFactorModel) -> np.ndarray | None:
    deltas: list[np.ndarray] = []
    for trajectory in trajectories:
        if trajectory.length < 6:
            continue
        delta = factor_model.delta_z(trajectory.observations, trajectory.next_observations)
        smoothness = 1.0 / (1.0 + action_jerks(trajectory))
        keep = np.argsort(smoothness)[-min(8, trajectory.length) :]
        deltas.append(np.mean(delta[keep], axis=0))
    if not deltas:
        return None
    return np.mean(np.asarray(deltas, dtype=np.float32), axis=0).astype(np.float32)


def nearest_anchor(center: np.ndarray, anchors: list[np.ndarray], factor_model: ControlFactorModel) -> np.ndarray:
    if not anchors:
        return np.asarray(center, dtype=np.float32).reshape(-1)
    obs = np.asarray(anchors, dtype=np.float32)
    center_z = (np.asarray(center, dtype=np.float32).reshape(1, -1) - factor_model.obs_mean.reshape(1, -1)) / factor_model.obs_std.reshape(1, -1)
    obs_z = (obs - factor_model.obs_mean.reshape(1, -1)) / factor_model.obs_std.reshape(1, -1)
    return obs[int(np.argmin(np.mean(np.square(obs_z - center_z), axis=1)))].astype(np.float32)


def normalize_actions(actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    action_low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    center = 0.5 * (action_high + action_low)
    scale = np.maximum(0.5 * (action_high - action_low), 1e-6)
    return np.clip((np.asarray(actions, dtype=np.float32) - center) / scale, -1.0, 1.0).astype(np.float32)


def island_distance(left: StableIsland, right: StableIsland) -> float:
    return float(np.mean(np.square(left.factor_center - right.factor_center)))


def replace_island_id(island: StableIsland, island_id: int) -> StableIsland:
    return StableIsland(
        island_id=int(island_id),
        center=island.center,
        factor_center=island.factor_center,
        score=island.score,
        survival=island.survival,
        smoothness=island.smoothness,
        terminal_risk=island.terminal_risk,
        surprise=island.surprise,
        support=island.support,
    )


def replace_goal_id(goal: IntrinsicGoal, goal_id: int) -> IntrinsicGoal:
    return IntrinsicGoal(
        goal_id=int(goal_id),
        goal_kind=goal.goal_kind,
        target_delta=goal.target_delta,
        anchor_observation=goal.anchor_observation,
        priority=goal.priority,
        source=goal.source,
    )


def top_indices(values: np.ndarray, *, count: int) -> list[int]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return [int(idx) for idx in np.argsort(values)[::-1][: max(0, int(count))]]


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size < 2 or right.size < 2 or float(np.std(left)) < 1e-8 or float(np.std(right)) < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


__all__ = [
    "DiscoveryContext",
    "build_discovery_context",
    "generate_intrinsic_goals",
    "mine_controllable_factors",
    "mine_danger_factors",
    "mine_stable_islands",
]
