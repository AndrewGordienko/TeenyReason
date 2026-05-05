"""Counterfactual real-env repair search from failure prefixes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .....envs import make_env
from ...gym_mpc import assert_box_spaces
from ..collection.trajectory import ReplayTrajectory, make_trajectory, rows_to_trajectory
from .failures import FailureWindow


@dataclass
class RepairResult:
    accepted: list[ReplayTrajectory]
    attempted: int
    accepted_count: int
    interaction_steps: int
    return_lifts: list[float]
    survival_lifts: list[float]
    terminal_avoids: list[float]

    def diagnostics(self) -> dict[str, object]:
        return {
            "repair_attempt_count": int(self.attempted),
            "repair_accept_count": int(self.accepted_count),
            "repair_accept_rate": float(self.accepted_count / self.attempted) if self.attempted else 0.0,
            "repair_interaction_steps": int(self.interaction_steps),
            "repair_return_lift_mean": mean_or_zero(self.return_lifts),
            "repair_return_lift_max": max_or_zero(self.return_lifts),
            "repair_survival_lift_max": max_or_zero(self.survival_lifts),
            "repair_terminal_avoid_count": int(np.sum(np.asarray(self.terminal_avoids, dtype=np.float32))),
        }


class CounterfactualRepairSearcher:
    """Replay stored prefixes, branch with real actions, keep only repairs."""

    def __init__(
        self,
        *,
        env_name: str,
        discount: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        branch_steps: int,
        attempts_per_window: int,
        accept_delta: float,
        survival_lift: int,
        seed: int,
        motor_prior=None,
        inverse_candidates: int = 0,
    ):
        self.env_name = str(env_name)
        self.discount = float(discount)
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.branch_steps = max(1, int(branch_steps))
        self.attempts_per_window = max(1, int(attempts_per_window))
        self.accept_delta = float(accept_delta)
        self.survival_lift = max(1, int(survival_lift))
        self.seed = int(seed)
        self.motor_prior = motor_prior
        self.inverse_candidates = max(0, int(inverse_candidates))

    def search(
        self,
        trajectories: list[ReplayTrajectory],
        windows: list[FailureWindow],
    ) -> RepairResult:
        accepted: list[ReplayTrajectory] = []
        attempted = 0
        interaction_steps = 0
        return_lifts: list[float] = []
        survival_lifts: list[float] = []
        terminal_avoids: list[float] = []
        for window_index, window in enumerate(windows):
            if not (0 <= int(window.trajectory_index) < len(trajectories)):
                continue
            trajectory = trajectories[int(window.trajectory_index)]
            candidates = repair_candidates(
                trajectory,
                window,
                self.action_low,
                self.action_high,
                self.attempts_per_window,
                self.seed,
                motor_prior=self.motor_prior,
                inverse_candidates=self.inverse_candidates,
            )
            for attempt_index, sequence in enumerate(candidates):
                result = self.branch(trajectory, window, sequence, seed_offset=window_index * 997 + attempt_index)
                attempted += 1
                interaction_steps += int(result["prefix_steps"]) + int(result["branch_steps"])
                return_lifts.append(float(result["return_lift"]))
                survival_lifts.append(float(result["survival_lift"]))
                terminal_avoids.append(float(result["terminal_avoid"]))
                if bool(result["accepted"]):
                    candidate = rows_to_trajectory(
                        result["rows"],
                        seed=int(trajectory.seed),
                        discount=self.discount,
                    )
                    if candidate is not None:
                        accepted.append(candidate)
        return RepairResult(
            accepted=accepted,
            attempted=attempted,
            accepted_count=len(accepted),
            interaction_steps=interaction_steps,
            return_lifts=return_lifts,
            survival_lifts=survival_lifts,
            terminal_avoids=terminal_avoids,
        )

    def branch(
        self,
        trajectory: ReplayTrajectory,
        window: FailureWindow,
        sequence: object,
        *,
        seed_offset: int,
    ) -> dict[str, object]:
        del seed_offset
        max_steps = max(1, int(window.start) + int(self.branch_steps) + 1)
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
                    return empty_branch(prefix_rows)
            branch_rows: list[dict[str, np.ndarray | float]] = []
            fixed_sequence = None if hasattr(sequence, "action") else np.asarray(sequence, dtype=np.float32)
            previous_action = np.clip(np.zeros_like(self.action_low, dtype=np.float32), self.action_low, self.action_high)
            for local_step in range(max(1, int(self.branch_steps))):
                if fixed_sequence is None:
                    action = sequence.action(observation, step=local_step, previous_action=previous_action)
                else:
                    action = fixed_sequence[min(local_step, fixed_sequence.shape[0] - 1)]
                row, observation, done = step_with_action(env, observation, action)
                branch_rows.append(row)
                previous_action = np.asarray(action, dtype=np.float32).reshape(-1)
                if done:
                    break
            return self.score_branch(trajectory, window, prefix_rows, branch_rows)
        finally:
            env.close()

    def score_branch(
        self,
        trajectory: ReplayTrajectory,
        window: FailureWindow,
        prefix_rows: list[dict[str, np.ndarray | float]],
        branch_rows: list[dict[str, np.ndarray | float]],
    ) -> dict[str, object]:
        reference_end = min(trajectory.length, int(window.start) + len(branch_rows))
        reference_rewards = trajectory.rewards[int(window.start) : reference_end]
        branch_return = row_return(branch_rows)
        reference_return = float(np.sum(reference_rewards)) if reference_rewards.size else 0.0
        return_lift = float(branch_return - reference_return)
        survival_lift = float(len(branch_rows) - max(0, reference_end - int(window.start)))
        reference_terminal = bool(np.any(trajectory.dones[int(window.start) : reference_end] > 0.5))
        branch_terminal = bool(branch_rows and float(branch_rows[-1]["done"]) > 0.5)
        terminal_avoid = float(reference_terminal and not branch_terminal)
        accepted = bool(
            return_lift >= self.accept_delta
            or survival_lift >= self.survival_lift
            or terminal_avoid > 0.5
        )
        return {
            "rows": prefix_rows + branch_rows,
            "prefix_steps": len(prefix_rows),
            "branch_steps": len(branch_rows),
            "branch_return": branch_return,
            "reference_return": reference_return,
            "return_lift": return_lift,
            "survival_lift": survival_lift,
            "terminal_avoid": terminal_avoid,
            "accepted": accepted,
        }


def repair_candidates(
    trajectory: ReplayTrajectory,
    window: FailureWindow,
    action_low: np.ndarray,
    action_high: np.ndarray,
    count: int,
    seed: int,
    motor_prior=None,
    inverse_candidates: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed) + int(window.start) * 37 + int(trajectory.seed) * 13)
    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    duration = max(1, int(window.end) - int(window.start))
    action_dim = int(action_low.shape[0])
    center = np.clip(np.zeros((action_dim,), dtype=np.float32), action_low, action_high)
    base = trajectory.actions[int(window.start) : int(window.start) + duration]
    if base.shape[0] < duration:
        pad = np.repeat(center.reshape(1, -1), duration - base.shape[0], axis=0)
        base = np.concatenate([base, pad], axis=0)
    rows = [np.asarray(base, dtype=np.float32), np.repeat(center.reshape(1, -1), duration, axis=0)]
    if motor_prior is not None and int(inverse_candidates) > 0:
        rows.extend(
            motor_prior.repair_sequences(
                window.target_delta,
                base,
                count=int(inverse_candidates),
                seed=int(seed + window.start),
            )
        )
    for axis in range(action_dim):
        low = center.copy()
        high = center.copy()
        low[axis] = action_low[axis]
        high[axis] = action_high[axis]
        rows.append(np.repeat(low.reshape(1, -1), duration, axis=0))
        rows.append(np.repeat(high.reshape(1, -1), duration, axis=0))
    scale = np.maximum(action_high - action_low, 1e-6)
    while len(rows) < max(1, int(count)):
        noise = rng.normal(0.0, 0.35, size=(duration, action_dim)).astype(np.float32)
        smooth = np.cumsum(noise, axis=0) / np.sqrt(np.arange(duration, dtype=np.float32).reshape(-1, 1) + 1.0)
        rows.append(np.clip(base + smooth * scale, action_low, action_high).astype(np.float32))
    return [np.asarray(row, dtype=np.float32) for row in rows[: max(1, int(count))]]


def step_with_action(env, observation: np.ndarray, action: np.ndarray) -> tuple[dict[str, np.ndarray | float], np.ndarray, bool]:
    next_obs, reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=np.float32))
    next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
    done = bool(terminated or truncated)
    return (
        {
            "observation": observation.copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "reward": float(reward),
            "next_observation": next_obs.copy(),
            "done": float(done),
        },
        next_obs,
        done,
    )


def empty_branch(prefix_rows: list[dict[str, np.ndarray | float]]) -> dict[str, object]:
    return {
        "rows": prefix_rows,
        "prefix_steps": len(prefix_rows),
        "branch_steps": 0,
        "return_lift": 0.0,
        "survival_lift": 0.0,
        "terminal_avoid": 0.0,
        "accepted": False,
    }


def row_return(rows: list[dict[str, np.ndarray | float]]) -> float:
    return float(np.sum(np.asarray([float(row["reward"]) for row in rows], dtype=np.float32))) if rows else 0.0


def mean_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.mean(np.asarray(rows, dtype=np.float32))) if rows else 0.0


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


__all__ = ["CounterfactualRepairSearcher", "RepairResult"]
