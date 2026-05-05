"""Model-free frontier restart search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ......envs import make_env
from ....gym_mpc import assert_box_spaces
from ...collection.trajectory import ReplayTrajectory, rows_to_trajectory
from ..factors import ControlFactorModel
from ..repair import row_return, step_with_action
from .levels import CurriculumState
from .quality import trajectory_quality_score


@dataclass(frozen=True)
class FrontierSpec:
    trajectory: ReplayTrajectory
    index: int


@dataclass
class FrontierRestartResult:
    accepted: list[ReplayTrajectory]
    attempted: int
    accepted_count: int
    interaction_steps: int
    return_lifts: list[float]
    score_lifts: list[float]

    def diagnostics(self) -> dict[str, object]:
        return {
            "frontier_restart_attempt_count": int(self.attempted),
            "frontier_restart_accept_count": int(self.accepted_count),
            "frontier_restart_accept_rate": float(self.accepted_count / self.attempted) if self.attempted else 0.0,
            "frontier_restart_interaction_steps": int(self.interaction_steps),
            "frontier_restart_return_lift_mean": mean_or_zero(self.return_lifts),
            "frontier_restart_return_lift_max": max_or_zero(self.return_lifts),
            "frontier_restart_score_lift_max": max_or_zero(self.score_lifts),
        }


class FrontierRestartSearcher:
    """Replay from strong prefixes and spend real samples on local repairs."""

    def __init__(
        self,
        *,
        env_name: str,
        discount: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        restart_count: int,
        candidate_count: int,
        branch_steps: int,
        noise: float,
        accept_lift: float,
        seed: int,
    ):
        self.env_name = str(env_name)
        self.discount = float(discount)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.restart_count = max(0, int(restart_count))
        self.candidate_count = max(1, int(candidate_count))
        self.branch_steps = max(1, int(branch_steps))
        self.noise = max(0.0, float(noise))
        self.accept_lift = float(accept_lift)
        self.seed = int(seed)

    def search(
        self,
        trajectories: list[ReplayTrajectory],
        state: CurriculumState,
        factor_model: ControlFactorModel,
    ) -> FrontierRestartResult:
        accepted: list[ReplayTrajectory] = []
        attempted = 0
        interaction_steps = 0
        return_lifts: list[float] = []
        score_lifts: list[float] = []
        for spec_index, spec in enumerate(frontier_specs(trajectories, state, factor_model, count=self.restart_count)):
            candidates = frontier_candidate_sequences(
                spec.trajectory,
                spec.index,
                self.action_low,
                self.action_high,
                count=self.candidate_count,
                duration=self.branch_steps,
                noise=self.noise,
                seed=int(self.seed + spec_index * 4099),
            )
            for candidate_index, sequence in enumerate(candidates):
                result = self.branch(spec, sequence, state, factor_model, seed_offset=spec_index * 997 + candidate_index)
                attempted += 1
                interaction_steps += int(result["interaction_steps"])
                return_lifts.append(float(result["return_lift"]))
                score_lifts.append(float(result["score_lift"]))
                if bool(result["accepted"]):
                    trajectory = rows_to_trajectory(result["rows"], seed=int(spec.trajectory.seed), discount=self.discount)
                    if trajectory is not None:
                        accepted.append(trajectory)
        return FrontierRestartResult(
            accepted=accepted,
            attempted=int(attempted),
            accepted_count=len(accepted),
            interaction_steps=int(interaction_steps),
            return_lifts=return_lifts,
            score_lifts=score_lifts,
        )

    def branch(
        self,
        spec: FrontierSpec,
        sequence: np.ndarray,
        state: CurriculumState,
        factor_model: ControlFactorModel,
        *,
        seed_offset: int,
    ) -> dict[str, object]:
        max_steps = max(1, int(spec.index) + int(self.branch_steps) + 1)
        env = make_env(self.env_name, max_episode_steps=max_steps)
        try:
            assert_box_spaces(env)
            observation, _info = env.reset(seed=int(spec.trajectory.seed))
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            prefix_rows: list[dict[str, np.ndarray | float]] = []
            for prefix_step in range(max(0, int(spec.index))):
                row, observation, done = step_with_action(env, observation, spec.trajectory.actions[prefix_step])
                prefix_rows.append(row)
                if done:
                    return empty_result(prefix_rows)
            branch_rows: list[dict[str, np.ndarray | float]] = []
            for local_step in range(self.branch_steps):
                action = sequence[min(local_step, sequence.shape[0] - 1)]
                row, observation, done = step_with_action(env, observation, action)
                branch_rows.append(row)
                if done:
                    break
            return self.score(spec, prefix_rows, branch_rows, state, factor_model)
        finally:
            env.close()

    def score(
        self,
        spec: FrontierSpec,
        prefix_rows: list[dict[str, np.ndarray | float]],
        branch_rows: list[dict[str, np.ndarray | float]],
        state: CurriculumState,
        factor_model: ControlFactorModel,
    ) -> dict[str, object]:
        rows = prefix_rows + branch_rows
        candidate = rows_to_trajectory(rows, seed=int(spec.trajectory.seed), discount=self.discount)
        if candidate is None:
            return empty_result(rows)
        reference_score = trajectory_quality_score(spec.trajectory, state, factor_model)
        candidate_score = trajectory_quality_score(candidate, state, factor_model)
        return_lift = float(candidate.episode_return - spec.trajectory.episode_return)
        score_lift = float(candidate_score - reference_score)
        survival_lift = float(candidate.length - spec.trajectory.length)
        accepted = bool(return_lift > 0.0 or score_lift >= self.accept_lift or (survival_lift > 0.0 and candidate.episode_return >= state.frontier_return - 25.0))
        return {
            "rows": rows,
            "interaction_steps": len(rows),
            "return_lift": return_lift,
            "score_lift": score_lift,
            "accepted": accepted,
        }


def frontier_specs(
    trajectories: list[ReplayTrajectory],
    state: CurriculumState,
    factor_model: ControlFactorModel,
    *,
    count: int,
) -> list[FrontierSpec]:
    ranked = sorted(trajectories, key=lambda item: trajectory_quality_score(item, state, factor_model), reverse=True)
    specs: list[FrontierSpec] = []
    for trajectory in ranked[: max(1, int(count))]:
        for index in frontier_indices(trajectory):
            if not any(item.trajectory is trajectory and item.index == index for item in specs):
                specs.append(FrontierSpec(trajectory, index))
        if len(specs) >= max(1, int(count)):
            break
    return specs[: max(1, int(count))]


def frontier_indices(trajectory: ReplayTrajectory) -> list[int]:
    if trajectory.length <= 1:
        return [0]
    best_value = int(np.argmax(trajectory.returns_to_go)) if trajectory.returns_to_go.size else 0
    late = int(round(0.70 * float(trajectory.length - 1)))
    pre_terminal = max(0, int(trajectory.length - 2))
    return unique_indices([best_value, late, pre_terminal], trajectory.length)


def frontier_candidate_sequences(
    trajectory: ReplayTrajectory,
    start: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    count: int,
    duration: int,
    noise: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed) + int(start) * 31)
    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    center = np.clip(np.zeros_like(action_low), action_low, action_high)
    base = padded_suffix(trajectory, start, duration, center)
    rows = [base, np.repeat(center.reshape(1, -1), duration, axis=0)]
    for axis in range(action_low.size):
        high = base.copy()
        low = base.copy()
        high[:, axis] = action_high[axis]
        low[:, axis] = action_low[axis]
        rows.extend([high, low])
    scale = np.maximum(action_high - action_low, 1e-6)
    while len(rows) < max(1, int(count)):
        eps = rng.normal(0.0, float(noise), size=base.shape).astype(np.float32)
        smooth = np.cumsum(eps, axis=0) / np.sqrt(np.arange(duration, dtype=np.float32).reshape(-1, 1) + 1.0)
        rows.append(np.clip(base + smooth * scale, action_low, action_high).astype(np.float32))
    return [np.asarray(row, dtype=np.float32) for row in rows[: max(1, int(count))]]


def padded_suffix(trajectory: ReplayTrajectory, start: int, duration: int, center: np.ndarray) -> np.ndarray:
    base = np.asarray(trajectory.actions[int(start) : int(start) + int(duration)], dtype=np.float32)
    if base.shape[0] < int(duration):
        pad = np.repeat(center.reshape(1, -1), int(duration) - base.shape[0], axis=0)
        base = np.concatenate([base, pad], axis=0)
    return base[: int(duration)].astype(np.float32)


def unique_indices(values: list[int], length: int) -> list[int]:
    out: list[int] = []
    for value in values:
        index = max(0, min(int(value), int(length) - 1))
        if index not in out:
            out.append(index)
    return out


def empty_result(rows: list[dict[str, np.ndarray | float]]) -> dict[str, object]:
    return {"rows": rows, "interaction_steps": len(rows), "return_lift": 0.0, "score_lift": 0.0, "accepted": False}


def mean_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.mean(np.asarray(rows, dtype=np.float32))) if rows else 0.0


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


__all__ = ["FrontierRestartResult", "FrontierRestartSearcher"]
