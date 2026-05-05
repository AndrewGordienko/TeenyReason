"""Replay-fork collector for generic continuous Gym environments."""

from __future__ import annotations

import numpy as np

from .....envs import make_env
from ...gym_mpc import TransitionBatch, assert_box_spaces
from ..config import AdvancedGymMPCConfig
from ..control import ActorPolicyModel
from ..model import ActionValueModel, EnsembleMLPWorldModel, ValueBootstrapModel
from ..planner import CEMPlanner
from .frontier import actor_collection_enabled, fit_frontier_models, make_frontier_planner, safe_frontier_action
from .trajectory import (
    ReplayTrajectory,
    collect_probe_trajectories,
    rows_to_trajectory,
    trajectories_to_batch,
)


def collect_replay_frontier_transitions(
    config: AdvancedGymMPCConfig,
) -> tuple[TransitionBatch, np.ndarray, np.ndarray, dict[str, object]]:
    """Collect trajectories, replay elite prefixes, and fork near high-value failures."""
    trajectories, action_low, action_high = collect_probe_trajectories(config)
    training_trajectories = list(trajectories)
    interaction_steps = trajectory_steps(trajectories)
    stats = make_replay_stats(trajectories)
    max_steps = max(interaction_steps, int(config.probe_episodes) * int(config.probe_steps))
    for cycle in range(max(0, int(config.frontier_cycles))):
        remaining = max_steps - int(interaction_steps)
        if remaining <= 0:
            break
        batch = trajectories_to_batch(training_trajectories)
        model, value_model, action_value_model, actor_model = fit_frontier_models(
            config,
            batch,
            action_low,
            action_high,
            cycle=cycle,
        )
        specs = branch_specs(training_trajectories, config, cycle=cycle)
        for spec in specs:
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
            accepted_rows, accepted_branch_steps = accepted_replay_rows(config, result)
            accepted = rows_to_trajectory(
                accepted_rows,
                seed=int(spec.trajectory.seed),
                discount=float(config.discount),
            )
            if accepted is not None:
                training_trajectories.append(accepted)
            update_replay_stats(stats, result, accepted_branch_steps=accepted_branch_steps)
    batch = trajectories_to_batch(training_trajectories)
    diagnostics = replay_diagnostics(stats, batch, interaction_steps, config)
    diagnostics["_replay_trajectories"] = list(training_trajectories)
    return batch, action_low, action_high, diagnostics


class BranchSpec:
    def __init__(self, trajectory: ReplayTrajectory, index: int):
        self.trajectory = trajectory
        self.index = int(index)


def branch_specs(
    trajectories: list[ReplayTrajectory],
    config: AdvancedGymMPCConfig,
    *,
    cycle: int,
) -> list[BranchSpec]:
    elites = sorted(trajectories, key=lambda item: item.episode_return, reverse=True)
    elites = elites[: max(1, int(config.replay_elite_count))]
    specs: list[BranchSpec] = []
    for trajectory in elites:
        for index in branch_indices(trajectory):
            specs.append(BranchSpec(trajectory, index))
    if not specs:
        return []
    start = int(cycle * max(1, int(config.replay_branches_per_cycle))) % len(specs)
    rotated = specs[start:] + specs[:start]
    return rotated[: max(1, int(config.replay_branches_per_cycle))]


def branch_indices(trajectory: ReplayTrajectory) -> list[int]:
    if trajectory.length <= 1:
        return [0]
    reward_min = max(0, int(np.argmin(trajectory.rewards)) - 1)
    best_rtg = int(np.argmax(trajectory.returns_to_go))
    late = int(round(0.65 * float(trajectory.length - 1)))
    middle = int(round(0.50 * float(trajectory.length - 1)))
    raw = [best_rtg, reward_min, late, middle]
    out: list[int] = []
    for value in raw:
        index = max(0, min(int(value), trajectory.length - 1))
        if index not in out:
            out.append(index)
    return out


def collect_replay_branch(
    config: AdvancedGymMPCConfig,
    model: EnsembleMLPWorldModel,
    value_model: ValueBootstrapModel | None,
    action_value_model: ActionValueModel | None,
    actor_model: ActorPolicyModel | None,
    trajectory: ReplayTrajectory,
    *,
    branch_index: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    cycle: int,
    branch_limit: int,
) -> dict[str, object]:
    env = make_env(config.env_name, max_episode_steps=max(1, int(branch_index + branch_limit + 1)))
    rng = np.random.default_rng(int(config.seed + 131_000 + cycle * 4099 + branch_index * 17))
    try:
        assert_box_spaces(env)
        planner = make_frontier_planner(config, action_low, action_high)
        observation, _info = env.reset(seed=int(trajectory.seed))
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        prefix_rows: list[dict[str, np.ndarray | float]] = []
        for prefix_step in range(max(0, min(int(branch_index), trajectory.length))):
            row, observation, done = step_with_action(env, observation, trajectory.actions[prefix_step])
            prefix_rows.append(row)
            if done:
                return empty_branch_result(prefix_rows, branch_index)
        branch_rows: list[dict[str, np.ndarray | float]] = []
        plan_scores: list[float] = []
        plan_uncertainties: list[float] = []
        for local_step in range(max(1, int(branch_limit))):
            action, plan = choose_replay_action(
                config,
                model,
                value_model,
                action_value_model,
                actor_model,
                planner,
                trajectory,
                branch_index + local_step,
                observation,
                rng,
            )
            row, observation, done = step_with_action(env, observation, action)
            branch_rows.append(row)
            plan_scores.append(float(plan.get("predicted_score", 0.0)))
            plan_uncertainties.append(float(plan.get("predicted_uncertainty", 0.0)))
            if done:
                break
        branch_return = row_return(branch_rows)
        reference_return = reference_suffix_return(trajectory, branch_index, len(branch_rows))
        return {
            "prefix_rows": prefix_rows,
            "branch_rows": branch_rows,
            "prefix_steps": len(prefix_rows),
            "branch_steps": len(branch_rows),
            "branch_return": branch_return,
            "reference_return": reference_return,
            "branch_delta": float(branch_return - reference_return),
            "total_return": float(row_return(prefix_rows) + branch_return),
            "plan_score_mean": mean_or_zero(plan_scores),
            "plan_uncertainty_mean": mean_or_zero(plan_uncertainties),
        }
    finally:
        env.close()


def choose_replay_action(
    config: AdvancedGymMPCConfig,
    model: EnsembleMLPWorldModel,
    value_model: ValueBootstrapModel | None,
    action_value_model: ActionValueModel | None,
    actor_model: ActorPolicyModel | None,
    planner: CEMPlanner,
    trajectory: ReplayTrajectory,
    suffix_index: int,
    observation: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    cem_plan = planner.choose_plan(
        model,
        observation,
        seed=int(config.seed + 141_000),
        step=int(suffix_index),
        discount=float(config.discount),
        done_penalty=float(config.done_penalty),
        uncertainty_penalty=float(config.uncertainty_penalty),
        uncertainty_gate_quantile=float(config.uncertainty_gate_quantile),
        value_model=value_model if bool(config.value_bootstrap) else None,
        action_value_model=(
            action_value_model if bool(config.action_value_bootstrap) else None
        ),
        action_value_weight=float(config.action_value_score_weight),
        actor_model=actor_model if actor_collection_enabled(config) else None,
        actor_center_prior=bool(config.actor_center_prior),
        actor_prior_candidates=int(config.actor_prior_candidates),
        actor_noise=float(config.actor_noise),
    )
    candidates = local_candidate_sequences(
        config,
        planner,
        model,
        actor_model,
        trajectory,
        suffix_index,
        cem_plan["sequence"],
        observation,
        rng,
    )
    scores, uncertainties = model.score_sequences(
        observation,
        candidates,
        discount=float(config.discount),
        done_penalty=float(config.done_penalty),
        uncertainty_penalty=float(config.uncertainty_penalty),
        value_model=value_model if bool(config.value_bootstrap) else None,
        action_value_model=(
            action_value_model if bool(config.action_value_bootstrap) else None
        ),
        action_value_weight=float(config.action_value_score_weight),
    )
    scores = penalize_uncertainty(scores, uncertainties, float(config.uncertainty_penalty))
    best = int(np.argmax(scores))
    plan = {
        "action": candidates[best, 0].astype(np.float32),
        "predicted_score": float(scores[best]),
        "predicted_uncertainty": float(uncertainties[best]),
        "uncertainty": float(uncertainties[best]),
        "candidate_uncertainty_gate": percentile_or_zero(uncertainties, float(config.uncertainty_gate_quantile) * 100.0),
    }
    return safe_frontier_action(config, planner, plan), plan


def local_candidate_sequences(
    config: AdvancedGymMPCConfig,
    planner: CEMPlanner,
    model: EnsembleMLPWorldModel,
    actor_model: ActorPolicyModel | None,
    trajectory: ReplayTrajectory,
    suffix_index: int,
    cem_sequence: object,
    observation: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    horizon = int(planner.horizon)
    action_dim = int(planner.action_low.shape[0])
    base = suffix_from_trajectory(trajectory, suffix_index, horizon, planner.action_low, planner.action_high)
    rows = [np.asarray(cem_sequence, dtype=np.float32).reshape(horizon, action_dim), base]
    actor_sequence = replay_actor_sequence(config, model, actor_model, observation, horizon)
    if actor_sequence is not None:
        rows.append(actor_sequence)
    scale = np.maximum(planner.action_high - planner.action_low, 1e-6)
    for _idx in range(max(0, int(config.replay_local_candidates))):
        noise = rng.normal(0.0, float(config.replay_local_noise), size=(horizon, action_dim)).astype(np.float32)
        rows.append(np.clip(base + noise * scale, planner.action_low, planner.action_high).astype(np.float32))
    center = np.clip(np.zeros((action_dim,), dtype=np.float32), planner.action_low, planner.action_high)
    rows.append(np.tile(center, (horizon, 1)).astype(np.float32))
    return np.stack(rows, axis=0).astype(np.float32)


def replay_actor_sequence(
    config: AdvancedGymMPCConfig,
    model: EnsembleMLPWorldModel,
    actor_model: ActorPolicyModel | None,
    observation: np.ndarray,
    horizon: int,
) -> np.ndarray | None:
    if actor_model is None or not actor_collection_enabled(config):
        return None
    try:
        sequence = actor_model.plan_sequence(model, observation, horizon=horizon)
    except (RuntimeError, ValueError):
        return None
    return np.asarray(sequence, dtype=np.float32)


def suffix_from_trajectory(
    trajectory: ReplayTrajectory,
    start: int,
    horizon: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    center = np.clip(np.zeros_like(action_low, dtype=np.float32), action_low, action_high)
    rows = []
    for offset in range(max(1, int(horizon))):
        index = int(start + offset)
        action = trajectory.actions[index] if 0 <= index < trajectory.length else center
        rows.append(np.asarray(action, dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0).astype(np.float32)


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


def accepted_replay_rows(
    config: AdvancedGymMPCConfig,
    result: dict[str, object],
) -> tuple[list[dict[str, np.ndarray | float]], int]:
    prefix_rows = list(result["prefix_rows"])
    branch_rows = list(result["branch_rows"])
    branch_delta = float(result["branch_delta"])
    if branch_delta >= float(config.replay_accept_delta):
        return prefix_rows + branch_rows, len(branch_rows)
    terminal_rows = [row for row in branch_rows if float(row["done"]) > 0.5]
    return terminal_rows[-1:], len(terminal_rows[-1:])


def empty_branch_result(prefix_rows: list[dict[str, np.ndarray | float]], branch_index: int) -> dict[str, object]:
    return {
        "prefix_rows": prefix_rows,
        "branch_rows": [],
        "prefix_steps": len(prefix_rows),
        "branch_steps": 0,
        "branch_return": 0.0,
        "reference_return": 0.0,
        "branch_delta": 0.0,
        "total_return": row_return(prefix_rows),
        "plan_score_mean": 0.0,
        "plan_uncertainty_mean": 0.0,
        "branch_index": int(branch_index),
    }


def make_replay_stats(trajectories: list[ReplayTrajectory]) -> dict[str, object]:
    returns = [item.episode_return for item in trajectories]
    return {
        "bootstrap_samples": trajectory_steps(trajectories),
        "bootstrap_returns": returns,
        "branch_returns": [],
        "branch_deltas": [],
        "branch_steps": [],
        "accepted_branch_steps": [],
        "prefix_steps": [],
        "plan_scores": [],
        "plan_uncertainties": [],
    }


def update_replay_stats(
    stats: dict[str, object],
    result: dict[str, object],
    *,
    accepted_branch_steps: int,
) -> None:
    list_stat(stats, "branch_returns").append(float(result["branch_return"]))
    list_stat(stats, "branch_deltas").append(float(result["branch_delta"]))
    list_stat(stats, "branch_steps").append(float(result["branch_steps"]))
    list_stat(stats, "accepted_branch_steps").append(float(accepted_branch_steps))
    list_stat(stats, "prefix_steps").append(float(result["prefix_steps"]))
    list_stat(stats, "plan_scores").append(float(result["plan_score_mean"]))
    list_stat(stats, "plan_uncertainties").append(float(result["plan_uncertainty_mean"]))


def replay_diagnostics(
    stats: dict[str, object],
    batch: TransitionBatch,
    interaction_steps: int,
    config: AdvancedGymMPCConfig,
) -> dict[str, object]:
    bootstrap_returns = list_stat(stats, "bootstrap_returns")
    best = max_or_zero(bootstrap_returns)
    branch_steps = sum_or_zero(list_stat(stats, "branch_steps"))
    accepted_steps = sum_or_zero(list_stat(stats, "accepted_branch_steps"))
    value_target_max = max_return_to_go(batch, discount=float(config.discount))
    return {
        "collector": "replay_frontier",
        "collector_samples": int(batch.observations.shape[0]),
        "collector_interaction_steps": int(interaction_steps),
        "collector_episode_count": int(len(bootstrap_returns)),
        "collector_best_return": best,
        "collector_return_mean": mean_or_zero(bootstrap_returns),
        "collector_solve_gap": float(float(config.solve_return) - best),
        "replay_bootstrap_samples": int(stats["bootstrap_samples"]),
        "replay_prefix_steps": int(sum_or_zero(list_stat(stats, "prefix_steps"))),
        "replay_branch_steps": int(branch_steps),
        "replay_accepted_branch_steps": int(accepted_steps),
        "replay_branch_acceptance_rate": float(accepted_steps / branch_steps) if branch_steps > 0.0 else 0.0,
        "replay_branch_return_mean": mean_or_zero(list_stat(stats, "branch_returns")),
        "replay_branch_return_max": max_or_zero(list_stat(stats, "branch_returns")),
        "replay_branch_delta_mean": mean_or_zero(list_stat(stats, "branch_deltas")),
        "replay_branch_delta_max": max_or_zero(list_stat(stats, "branch_deltas")),
        "replay_plan_score_mean": mean_or_zero(list_stat(stats, "plan_scores")),
        "replay_plan_uncertainty_mean": mean_or_zero(list_stat(stats, "plan_uncertainties")),
        "replay_value_target_max": value_target_max,
    }


def reference_suffix_return(trajectory: ReplayTrajectory, start: int, count: int) -> float:
    if count <= 0 or start >= trajectory.length:
        return 0.0
    end = max(0, min(int(start + count), trajectory.length))
    return float(np.sum(trajectory.rewards[int(start) : end]))


def row_return(rows: list[dict[str, np.ndarray | float]]) -> float:
    return float(np.sum(np.asarray([float(row["reward"]) for row in rows], dtype=np.float32))) if rows else 0.0


def trajectory_steps(trajectories: list[ReplayTrajectory]) -> int:
    return int(sum(item.length for item in trajectories))


def max_return_to_go(batch: TransitionBatch, *, discount: float) -> float:
    rewards = np.asarray(batch.rewards, dtype=np.float32)
    if rewards.size == 0:
        return 0.0
    dones = np.asarray(batch.dones, dtype=np.float32).reshape(-1)
    running = 0.0
    best = -float("inf")
    for idx in range(rewards.size - 1, -1, -1):
        if idx < rewards.size - 1 and float(dones[idx]) > 0.5:
            running = 0.0
        running = float(rewards[idx]) + float(discount) * running
        best = max(best, running)
    return float(best)


def penalize_uncertainty(scores: np.ndarray, uncertainties: np.ndarray, weight: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    uncertainties = np.asarray(uncertainties, dtype=np.float32)
    if scores.size < 2 or float(weight) <= 0.0:
        return scores
    excess = np.maximum(uncertainties - float(np.percentile(uncertainties, 75.0)), 0.0)
    return (scores - float(weight) * float(np.std(scores) + 1e-4) * excess).astype(np.float32)


def list_stat(stats: dict[str, object], key: str) -> list[float]:
    values = stats[key]
    if not isinstance(values, list):
        raise TypeError(f"expected replay stat list for {key}")
    return values


def mean_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.mean(np.asarray(rows, dtype=np.float32))) if rows else 0.0


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


def sum_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.sum(np.asarray(rows, dtype=np.float32))) if rows else 0.0


def percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return float(np.percentile(values, percentile)) if values.size else 0.0


__all__ = ["collect_replay_frontier_transitions"]
