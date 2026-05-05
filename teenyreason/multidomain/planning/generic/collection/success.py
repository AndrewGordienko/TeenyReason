"""Success-seeking archive collector for generic continuous Gym environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...gym_mpc import TransitionBatch
from ..config import AdvancedGymMPCConfig
from .frontier import fit_frontier_models
from .replay import (
    collect_replay_branch,
    max_return_to_go,
    mean_or_zero,
    reference_suffix_return,
    row_return,
    sum_or_zero,
    trajectory_steps,
)
from .trajectory import ReplayTrajectory, collect_probe_trajectories, rows_to_trajectory, trajectories_to_batch


@dataclass(frozen=True)
class SuccessBranchSpec:
    trajectory: ReplayTrajectory
    index: int


class SuccessArchive:
    """Small real-trajectory archive organized around frontier progress."""

    def __init__(self, trajectories: list[ReplayTrajectory]):
        self.trajectories = list(trajectories)

    @property
    def best_return(self) -> float:
        return max_or_zero([item.episode_return for item in self.trajectories])

    @property
    def best_survival(self) -> int:
        return max((item.length for item in self.trajectories), default=0)

    @property
    def best_value(self) -> float:
        values = [float(np.max(item.returns_to_go)) for item in self.trajectories if item.returns_to_go.size]
        return max_or_zero(values)

    def add(self, trajectory: ReplayTrajectory) -> None:
        self.trajectories.append(trajectory)


def collect_success_archive_transitions(
    config: AdvancedGymMPCConfig,
) -> tuple[TransitionBatch, np.ndarray, np.ndarray, dict[str, object]]:
    """Practice from the edge of competence and keep real improvements."""
    trajectories, action_low, action_high = collect_probe_trajectories(config)
    archive = SuccessArchive(trajectories)
    interaction_steps = trajectory_steps(trajectories)
    max_steps = max(interaction_steps, int(config.probe_episodes) * int(config.probe_steps))
    stats = make_success_stats(archive)
    for cycle in range(max(0, int(config.frontier_cycles))):
        remaining = max_steps - int(interaction_steps)
        if remaining <= 0:
            break
        batch = trajectories_to_batch(archive.trajectories)
        model, value_model, action_value_model, actor_model = fit_frontier_models(
            config,
            batch,
            action_low,
            action_high,
            cycle=cycle,
        )
        for spec in success_branch_specs(archive, config, cycle=cycle):
            remaining = max_steps - int(interaction_steps)
            if remaining <= 0:
                break
            result = collect_replay_branch(
                config,
                model,
                value_model,
                action_value_model,
                actor_model,
                spec.trajectory,
                branch_index=int(spec.index),
                action_low=action_low,
                action_high=action_high,
                cycle=cycle,
                branch_limit=min(max(1, int(config.replay_branch_steps)), remaining),
            )
            interaction_steps += int(result["prefix_steps"]) + int(result["branch_steps"])
            accepted_rows, accepted_steps, reason, metrics = accepted_success_rows(config, archive, spec, result)
            accepted = rows_to_trajectory(
                accepted_rows,
                seed=int(spec.trajectory.seed),
                discount=float(config.discount),
            )
            if accepted is not None:
                archive.add(accepted)
            update_success_stats(stats, result, reason, metrics, accepted_branch_steps=accepted_steps)
    batch = trajectories_to_batch(archive.trajectories)
    diagnostics = success_diagnostics(stats, archive, batch, interaction_steps, config)
    diagnostics["_replay_trajectories"] = list(archive.trajectories)
    return batch, action_low, action_high, diagnostics


def success_branch_specs(
    archive: SuccessArchive,
    config: AdvancedGymMPCConfig,
    *,
    cycle: int,
) -> list[SuccessBranchSpec]:
    elite_count = max(1, int(config.success_archive_elite_count))
    candidate_trajectories = archive_elites(archive.trajectories, elite_count)
    specs: list[SuccessBranchSpec] = []
    for trajectory in candidate_trajectories:
        for index in success_branch_indices(trajectory):
            spec = SuccessBranchSpec(trajectory, index)
            if not has_spec(specs, spec):
                specs.append(spec)
    if not specs:
        return []
    specs.sort(key=success_spec_score, reverse=True)
    start = int(cycle * max(1, int(config.success_archive_branches_per_cycle))) % len(specs)
    rotated = specs[start:] + specs[:start]
    return rotated[: max(1, int(config.success_archive_branches_per_cycle))]


def archive_elites(trajectories: list[ReplayTrajectory], elite_count: int) -> list[ReplayTrajectory]:
    ranked: list[ReplayTrajectory] = []
    for key in (
        lambda item: item.episode_return,
        lambda item: item.length,
        lambda item: float(np.max(item.returns_to_go)) if item.returns_to_go.size else -float("inf"),
    ):
        for trajectory in sorted(trajectories, key=key, reverse=True)[:elite_count]:
            if not any(item is trajectory for item in ranked):
                ranked.append(trajectory)
    return ranked


def success_branch_indices(trajectory: ReplayTrajectory) -> list[int]:
    if trajectory.length <= 1:
        return [0]
    best_rtg = int(np.argmax(trajectory.returns_to_go))
    worst_reward = max(0, int(np.argmin(trajectory.rewards)) - 1)
    middle = int(round(0.50 * float(trajectory.length - 1)))
    late = int(round(0.75 * float(trajectory.length - 1)))
    pre_terminal = max(0, int(trajectory.length - 2))
    return unique_indices([best_rtg, worst_reward, middle, late, pre_terminal], trajectory.length)


def accepted_success_rows(
    config: AdvancedGymMPCConfig,
    archive: SuccessArchive,
    spec: SuccessBranchSpec,
    result: dict[str, object],
) -> tuple[list[dict[str, np.ndarray | float]], int, str, dict[str, float]]:
    prefix_rows = list(result["prefix_rows"])
    branch_rows = list(result["branch_rows"])
    all_rows = prefix_rows + branch_rows
    metrics = success_metrics(config, archive, spec, result, all_rows)
    reason = acceptance_reason(config, metrics)
    if reason != "rejected":
        return all_rows, len(branch_rows), reason, metrics
    if bool(config.success_archive_keep_rejected_terminal):
        terminal_rows = [row for row in branch_rows if float(row["done"]) > 0.5]
        return terminal_rows[-1:], len(terminal_rows[-1:]), reason, metrics
    return [], 0, reason, metrics


def success_metrics(
    config: AdvancedGymMPCConfig,
    archive: SuccessArchive,
    spec: SuccessBranchSpec,
    result: dict[str, object],
    rows: list[dict[str, np.ndarray | float]],
) -> dict[str, float]:
    branch_steps = int(result["branch_steps"])
    reference_steps = max(0, min(branch_steps, spec.trajectory.length - int(spec.index)))
    reference_return = reference_suffix_return(spec.trajectory, int(spec.index), branch_steps)
    total_return = row_return(rows)
    value = max_row_return_to_go(rows, discount=float(config.discount))
    return {
        "branch_return": float(result["branch_return"]),
        "branch_delta": float(result["branch_return"]) - float(reference_return),
        "total_return": float(total_return),
        "total_lift": float(total_return - spec.trajectory.episode_return),
        "survival_lift": float(branch_steps - reference_steps),
        "value_lift": float(value - archive.best_value),
        "new_best_lift": float(total_return - archive.best_return),
        "frontier_floor": frontier_return_floor(config, archive),
    }


def acceptance_reason(config: AdvancedGymMPCConfig, metrics: dict[str, float]) -> str:
    if metrics["new_best_lift"] > 0.0:
        return "new_best"
    if metrics["total_return"] < metrics["frontier_floor"]:
        return "rejected"
    if metrics["total_lift"] >= float(config.replay_accept_delta):
        return "total_lift"
    if metrics["branch_delta"] >= float(config.replay_accept_delta):
        return "suffix_lift"
    if metrics["survival_lift"] >= float(config.success_archive_survival_lift):
        return "survival_lift"
    if metrics["value_lift"] >= float(config.success_archive_value_lift):
        return "value_lift"
    return "rejected"


def frontier_return_floor(config: AdvancedGymMPCConfig, archive: SuccessArchive) -> float:
    gap = abs(float(config.solve_return) - float(archive.best_return))
    window = max(float(config.success_archive_frontier_floor), float(config.success_archive_frontier_gap_fraction) * gap)
    return float(archive.best_return - window)


def make_success_stats(archive: SuccessArchive) -> dict[str, object]:
    return {
        "bootstrap_samples": trajectory_steps(archive.trajectories),
        "bootstrap_returns": [item.episode_return for item in archive.trajectories],
        "branch_returns": [],
        "branch_deltas": [],
        "total_returns": [],
        "survival_lifts": [],
        "value_lifts": [],
        "accepted_branch_steps": [],
        "branch_steps": [],
        "accepted_reasons": {},
    }


def update_success_stats(
    stats: dict[str, object],
    result: dict[str, object],
    reason: str,
    metrics: dict[str, float],
    *,
    accepted_branch_steps: int,
) -> None:
    list_stat(stats, "branch_returns").append(float(metrics["branch_return"]))
    list_stat(stats, "branch_deltas").append(float(metrics["branch_delta"]))
    list_stat(stats, "total_returns").append(float(metrics["total_return"]))
    list_stat(stats, "survival_lifts").append(float(metrics["survival_lift"]))
    list_stat(stats, "value_lifts").append(float(metrics["value_lift"]))
    list_stat(stats, "branch_steps").append(float(result["branch_steps"]))
    list_stat(stats, "accepted_branch_steps").append(float(accepted_branch_steps))
    reasons = stats["accepted_reasons"]
    if not isinstance(reasons, dict):
        raise TypeError("accepted_reasons must be a dict")
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def success_diagnostics(
    stats: dict[str, object],
    archive: SuccessArchive,
    batch: TransitionBatch,
    interaction_steps: int,
    config: AdvancedGymMPCConfig,
) -> dict[str, object]:
    bootstrap_returns = list_stat(stats, "bootstrap_returns")
    branch_steps = sum_or_zero(list_stat(stats, "branch_steps"))
    accepted_steps = sum_or_zero(list_stat(stats, "accepted_branch_steps"))
    reasons = stats["accepted_reasons"] if isinstance(stats["accepted_reasons"], dict) else {}
    return {
        "collector": "success_archive",
        "collector_samples": int(batch.observations.shape[0]),
        "collector_interaction_steps": int(interaction_steps),
        "collector_episode_count": int(len(bootstrap_returns)),
        "collector_best_return": archive.best_return,
        "collector_return_mean": mean_or_zero(bootstrap_returns),
        "collector_solve_gap": float(float(config.solve_return) - archive.best_return),
        "success_archive_best_return": archive.best_return,
        "success_archive_frontier_floor": frontier_return_floor(config, archive),
        "success_archive_best_survival": int(archive.best_survival),
        "success_archive_best_value": float(archive.best_value),
        "success_archive_trajectory_count": int(len(archive.trajectories)),
        "success_archive_branch_steps": int(branch_steps),
        "success_archive_accepted_branch_steps": int(accepted_steps),
        "success_archive_acceptance_rate": float(accepted_steps / branch_steps) if branch_steps > 0.0 else 0.0,
        "success_archive_branch_return_mean": mean_or_zero(list_stat(stats, "branch_returns")),
        "success_archive_branch_return_max": max_or_zero(list_stat(stats, "branch_returns")),
        "success_archive_branch_delta_mean": mean_or_zero(list_stat(stats, "branch_deltas")),
        "success_archive_branch_delta_max": max_or_zero(list_stat(stats, "branch_deltas")),
        "success_archive_total_return_max": max_or_zero(list_stat(stats, "total_returns")),
        "success_archive_survival_lift_max": max_or_zero(list_stat(stats, "survival_lifts")),
        "success_archive_value_lift_max": max_or_zero(list_stat(stats, "value_lifts")),
        "success_archive_new_best_count": int(reasons.get("new_best", 0)),
        "success_archive_total_lift_count": int(reasons.get("total_lift", 0)),
        "success_archive_suffix_lift_count": int(reasons.get("suffix_lift", 0)),
        "success_archive_survival_lift_count": int(reasons.get("survival_lift", 0)),
        "success_archive_value_lift_count": int(reasons.get("value_lift", 0)),
        "success_archive_rejected_count": int(reasons.get("rejected", 0)),
        "success_archive_value_target_max": max_return_to_go(batch, discount=float(config.discount)),
    }


def success_spec_score(spec: SuccessBranchSpec) -> float:
    trajectory = spec.trajectory
    value = float(trajectory.returns_to_go[int(spec.index)]) if trajectory.returns_to_go.size else 0.0
    prefix = float(np.sum(trajectory.rewards[: int(spec.index)])) if trajectory.rewards.size else 0.0
    return float(prefix + 0.5 * value + 0.02 * trajectory.length)


def max_row_return_to_go(rows: list[dict[str, np.ndarray | float]], *, discount: float) -> float:
    if not rows:
        return 0.0
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float32)
    dones = np.asarray([float(row["done"]) for row in rows], dtype=np.float32)
    running = 0.0
    best = -float("inf")
    for idx in range(rewards.size - 1, -1, -1):
        if idx < rewards.size - 1 and float(dones[idx]) > 0.5:
            running = 0.0
        running = float(rewards[idx]) + float(discount) * running
        best = max(best, running)
    return float(best)


def unique_indices(values: list[int], length: int) -> list[int]:
    out: list[int] = []
    for value in values:
        index = max(0, min(int(value), int(length) - 1))
        if index not in out:
            out.append(index)
    return out


def has_spec(specs: list[SuccessBranchSpec], candidate: SuccessBranchSpec) -> bool:
    return any(item.trajectory is candidate.trajectory and item.index == candidate.index for item in specs)


def list_stat(stats: dict[str, object], key: str) -> list[float]:
    values = stats[key]
    if not isinstance(values, list):
        raise TypeError(f"expected success stat list for {key}")
    return values


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


__all__ = ["collect_success_archive_transitions"]
