"""Active frontier collector for generic continuous Gym environments."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .....envs import make_env
from ...gym_mpc import (
    TransitionBatch,
    assert_box_spaces,
    collect_probe_transitions,
    probe_action,
)
from ..config import AdvancedGymMPCConfig
from ..control import ActorPolicyModel
from ..diagnostics import batch_from_rows, merge_transition_batches
from ..events import transition_event_weights
from ..model import ActionValueModel, EnsembleMLPWorldModel, ValueBootstrapModel
from ..planner import CEMPlanner


def collect_frontier_transitions(
    config: AdvancedGymMPCConfig,
) -> tuple[TransitionBatch, np.ndarray, np.ndarray, dict[str, object]]:
    """Collect broad bootstrap data, then spend remaining probe budget near the frontier."""
    bootstrap_episodes = max(1, min(int(config.probe_episodes), int(config.frontier_bootstrap_episodes)))
    bootstrap = replace(config, probe_episodes=bootstrap_episodes)
    batch, action_low, action_high = collect_probe_transitions(bootstrap)
    max_samples = max(int(batch.observations.shape[0]), int(config.probe_episodes) * int(config.probe_steps))
    stats = make_frontier_stats(batch, config)
    for cycle in range(max(0, int(config.frontier_cycles))):
        remaining = max_samples - int(batch.observations.shape[0])
        if remaining <= 0:
            break
        model, value_model, action_value_model, actor_model = fit_frontier_models(
            config,
            batch,
            action_low,
            action_high,
            cycle=cycle,
        )
        rollout = collect_frontier_rollout(
            config,
            model,
            value_model,
            action_value_model,
            actor_model,
            action_low=action_low,
            action_high=action_high,
            cycle=cycle,
            step_limit=min(max(1, int(config.frontier_chunk_steps)), remaining),
        )
        rows = accepted_rollout_rows(batch, rollout)
        extra = batch_from_rows(rows)
        if extra is not None:
            batch = merge_transition_batches(batch, extra)
        update_frontier_stats(stats, rollout, accepted_steps=len(rows))
    return batch, action_low, action_high, summarize_frontier_stats(stats, batch, config)


def fit_frontier_models(
    config: AdvancedGymMPCConfig,
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    cycle: int,
) -> tuple[
    EnsembleMLPWorldModel,
    ValueBootstrapModel | None,
    ActionValueModel | None,
    ActorPolicyModel | None,
]:
    weights = frontier_weights(config, batch, action_low, action_high)
    model, _loss = EnsembleMLPWorldModel.fit(
        batch,
        action_low=action_low,
        action_high=action_high,
        ensemble_size=int(config.ensemble_size),
        hidden_dim=int(config.hidden_dim),
        epochs=max(1, int(config.frontier_refit_epochs)),
        batch_size=int(config.batch_size),
        lr=float(config.lr),
        seed=int(config.seed + 81_000 + int(cycle) * 997),
        sample_weights=weights,
    )
    value_model = None
    if bool(config.value_bootstrap):
        value_model, _value_loss, _stats = ValueBootstrapModel.fit(
            batch,
            discount=float(config.discount),
            hidden_dim=int(config.hidden_dim),
            epochs=max(1, int(config.frontier_refit_epochs)),
            batch_size=int(config.batch_size),
            lr=float(config.lr),
            seed=int(config.seed + 83_000 + int(cycle) * 997),
            sample_weights=weights,
        )
    action_value_model = None
    if bool(config.action_value_bootstrap):
        action_value_model, _q_loss, _q_stats = ActionValueModel.fit(
            batch,
            action_low=action_low,
            action_high=action_high,
            discount=float(config.discount),
            hidden_dim=int(config.hidden_dim),
            epochs=max(1, int(config.frontier_refit_epochs)),
            batch_size=int(config.batch_size),
            lr=float(config.lr),
            seed=int(config.seed + 85_000 + int(cycle) * 997),
            sample_weights=weights,
        )
    actor_model = None
    if actor_collection_enabled(config):
        actor_model, _actor_loss, _actor_stats = ActorPolicyModel.fit(
            batch,
            action_low=action_low,
            action_high=action_high,
            discount=float(config.discount),
            hidden_dim=int(config.hidden_dim),
            epochs=max(1, int(config.frontier_refit_epochs)),
            batch_size=int(config.batch_size),
            lr=float(config.lr),
            seed=int(config.seed + 87_000 + int(cycle) * 997),
            sample_weights=weights,
        )
    return model, value_model, action_value_model, actor_model


def frontier_weights(
    config: AdvancedGymMPCConfig,
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray | None:
    if not bool(config.event_weighting):
        return None
    weights, _stats = transition_event_weights(
        batch,
        action_low=action_low,
        action_high=action_high,
        strength=float(config.event_weight_strength),
        terminal_weight=float(config.event_terminal_weight),
        quantile=float(config.event_quantile),
        floor=float(config.event_floor),
        action_saturation_floor=float(config.event_action_saturation_floor),
    )
    return weights


def collect_frontier_rollout(
    config: AdvancedGymMPCConfig,
    model: EnsembleMLPWorldModel,
    value_model: ValueBootstrapModel | None,
    action_value_model: ActionValueModel | None,
    actor_model: ActorPolicyModel | None,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    cycle: int,
    step_limit: int,
) -> dict[str, object]:
    env = make_env(config.env_name, max_episode_steps=max(1, int(step_limit)))
    rng = np.random.default_rng(int(config.seed + 91_000 + int(cycle) * 4099))
    try:
        assert_box_spaces(env)
        planner = make_frontier_planner(config, action_low, action_high)
        observation, _info = env.reset(seed=int(config.seed + 20_000 + int(cycle)))
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        rows: list[dict[str, np.ndarray | float]] = []
        stats = {"surprise": [], "reward_error": [], "plan_score": [], "plan_uncertainty": []}
        total_return = 0.0
        for step in range(max(1, int(step_limit))):
            action, plan = choose_frontier_action(
                config,
                planner,
                model,
                value_model,
                action_value_model,
                actor_model,
                observation,
                rng,
                cycle,
                step,
            )
            pred = model.predict_batch(observation.reshape(1, -1), action.reshape(1, -1))
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            done = bool(terminated or truncated)
            update_rollout_stats(stats, model, pred, next_obs, float(reward), plan)
            rows.append(
                {
                    "observation": observation.copy(),
                    "action": action.copy(),
                    "reward": float(reward),
                    "next_observation": next_obs.copy(),
                    "done": float(done),
                }
            )
            total_return += float(reward)
            observation = next_obs
            if done:
                break
        return {
            "return": float(total_return),
            "steps": int(len(rows)),
            "rows": rows,
            "surprise_mean": mean_or_zero(stats["surprise"]),
            "surprise_max": max_or_zero(stats["surprise"]),
            "reward_error_mean": mean_or_zero(stats["reward_error"]),
            "plan_score_mean": mean_or_zero(stats["plan_score"]),
            "plan_uncertainty_mean": mean_or_zero(stats["plan_uncertainty"]),
        }
    finally:
        env.close()


def make_frontier_planner(
    config: AdvancedGymMPCConfig,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> CEMPlanner:
    return CEMPlanner(
        action_low=action_low,
        action_high=action_high,
        horizon=int(config.horizon),
        candidate_count=int(config.candidate_count),
        iterations=int(config.cem_iterations),
        elite_fraction=float(config.elite_fraction),
        noise_floor=float(config.action_noise_floor),
        temporal_chunk_size=int(config.temporal_chunk_size),
        temporal_chunk_candidates=int(config.temporal_chunk_candidates),
        temporal_smoothness_penalty=float(config.temporal_smoothness_penalty),
    )


def choose_frontier_action(
    config: AdvancedGymMPCConfig,
    planner: CEMPlanner,
    model: EnsembleMLPWorldModel,
    value_model: ValueBootstrapModel | None,
    action_value_model: ActionValueModel | None,
    actor_model: ActorPolicyModel | None,
    observation: np.ndarray,
    rng: np.random.Generator,
    cycle: int,
    step: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if rng.random() < float(np.clip(config.frontier_random_action_rate, 0.0, 1.0)):
        action = probe_action(rng, planner.action_low, planner.action_high, episode=cycle, step=step)
        return action.astype(np.float32), {}
    plan = planner.choose_plan(
        model,
        observation,
        seed=int(config.seed + int(cycle) * 701),
        step=step,
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
        uncertainty_bonus=float(config.frontier_uncertainty_bonus),
    )
    return safe_frontier_action(config, planner, plan), plan


def safe_frontier_action(
    config: AdvancedGymMPCConfig,
    planner: CEMPlanner,
    plan: dict[str, object],
) -> np.ndarray:
    action = np.asarray(plan["action"], dtype=np.float32).reshape(-1)
    if not bool(config.uncertainty_execution_gate):
        return action
    uncertainty = float(plan.get("uncertainty", 0.0))
    threshold = float(plan.get("candidate_uncertainty_gate", plan.get("candidate_uncertainty_p75", 0.0)))
    if uncertainty <= threshold or threshold <= 0.0:
        return action
    center = np.clip(np.zeros_like(action, dtype=np.float32), planner.action_low, planner.action_high)
    blend = float(np.clip(config.uncertainty_safe_action_blend, 0.0, 1.0))
    return np.clip((1.0 - blend) * action + blend * center, planner.action_low, planner.action_high).astype(np.float32)


def actor_collection_enabled(config: AdvancedGymMPCConfig) -> bool:
    return bool(config.actor_policy) and bool(config.actor_collection_prior)


def update_rollout_stats(
    stats: dict[str, list[float]],
    model: EnsembleMLPWorldModel,
    pred: dict[str, np.ndarray],
    next_obs: np.ndarray,
    reward: float,
    plan: dict[str, object],
) -> None:
    stats["surprise"].append(model.normalized_transition_error(pred["next_observation"][0], next_obs))
    stats["reward_error"].append(abs(float(pred["reward"][0]) - float(reward)))
    if plan:
        stats["plan_score"].append(float(plan.get("predicted_score", plan.get("score", 0.0))))
        stats["plan_uncertainty"].append(float(plan.get("predicted_uncertainty", 0.0)))


def make_frontier_stats(batch: TransitionBatch, config: AdvancedGymMPCConfig) -> dict[str, object]:
    returns = np.asarray(batch.episode_returns, dtype=np.float32).reshape(-1)
    return {
        "bootstrap_samples": int(batch.observations.shape[0]),
        "bootstrap_best_return": float(np.max(returns)) if returns.size else 0.0,
        "rollout_returns": [],
        "rollout_steps": [],
        "accepted_steps": [],
        "surprises": [],
        "reward_errors": [],
        "plan_scores": [],
        "plan_uncertainties": [],
        "solve_return": float(config.solve_return),
    }


def accepted_rollout_rows(
    batch: TransitionBatch,
    rollout: dict[str, object],
) -> list[dict[str, np.ndarray | float]]:
    rows = rollout["rows"]
    if not isinstance(rows, list):
        return []
    returns = np.asarray(batch.episode_returns, dtype=np.float32).reshape(-1)
    current_mean = float(np.mean(returns)) if returns.size else -float("inf")
    current_best = float(np.max(returns)) if returns.size else -float("inf")
    rollout_return = float(rollout["return"])
    if rollout_return >= current_best or rollout_return >= current_mean:
        return rows
    terminal_rows = [row for row in rows if float(row["done"]) > 0.5]
    return terminal_rows[-1:]


def update_frontier_stats(
    stats: dict[str, object],
    rollout: dict[str, object],
    *,
    accepted_steps: int,
) -> None:
    list_stat(stats, "rollout_returns").append(float(rollout["return"]))
    list_stat(stats, "rollout_steps").append(float(rollout["steps"]))
    list_stat(stats, "accepted_steps").append(float(accepted_steps))
    list_stat(stats, "surprises").append(float(rollout["surprise_mean"]))
    list_stat(stats, "reward_errors").append(float(rollout["reward_error_mean"]))
    list_stat(stats, "plan_scores").append(float(rollout["plan_score_mean"]))
    list_stat(stats, "plan_uncertainties").append(float(rollout["plan_uncertainty_mean"]))


def summarize_frontier_stats(
    stats: dict[str, object],
    batch: TransitionBatch,
    config: AdvancedGymMPCConfig,
) -> dict[str, object]:
    returns = np.asarray(batch.episode_returns, dtype=np.float32).reshape(-1)
    best = float(np.max(returns)) if returns.size else 0.0
    bootstrap_best = float(stats["bootstrap_best_return"])
    return {
        "collector": "frontier",
        "collector_samples": int(batch.observations.shape[0]),
        "collector_interaction_steps": int(int(stats["bootstrap_samples"]) + sum_or_zero(list_stat(stats, "rollout_steps"))),
        "collector_episode_count": int(returns.size),
        "collector_best_return": best,
        "collector_return_mean": float(np.mean(returns)) if returns.size else 0.0,
        "collector_solve_gap": float(float(config.solve_return) - best),
        "frontier_bootstrap_samples": int(stats["bootstrap_samples"]),
        "frontier_bootstrap_best_return": bootstrap_best,
        "frontier_best_lift": float(best - bootstrap_best),
        "frontier_rollout_count": len(list_stat(stats, "rollout_returns")),
        "frontier_rollout_return_mean": mean_or_zero(list_stat(stats, "rollout_returns")),
        "frontier_rollout_return_max": max_or_zero(list_stat(stats, "rollout_returns")),
        "frontier_rollout_steps_mean": mean_or_zero(list_stat(stats, "rollout_steps")),
        "frontier_accepted_steps": int(np.sum(np.asarray(list_stat(stats, "accepted_steps"), dtype=np.float32))),
        "frontier_acceptance_rate": acceptance_rate(stats),
        "frontier_surprise_mean": mean_or_zero(list_stat(stats, "surprises")),
        "frontier_reward_error_mean": mean_or_zero(list_stat(stats, "reward_errors")),
        "frontier_plan_score_mean": mean_or_zero(list_stat(stats, "plan_scores")),
        "frontier_plan_uncertainty_mean": mean_or_zero(list_stat(stats, "plan_uncertainties")),
    }


def list_stat(stats: dict[str, object], key: str) -> list[float]:
    values = stats[key]
    if not isinstance(values, list):
        raise TypeError(f"expected list frontier stat for {key}")
    return values


def acceptance_rate(stats: dict[str, object]) -> float:
    steps = float(np.sum(np.asarray(list_stat(stats, "rollout_steps"), dtype=np.float32)))
    if steps <= 0.0:
        return 0.0
    accepted = float(np.sum(np.asarray(list_stat(stats, "accepted_steps"), dtype=np.float32)))
    return float(accepted / steps)


def mean_or_zero(values: object) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return float(np.mean(np.asarray(rows, dtype=np.float32)))


def max_or_zero(values: object) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return float(np.max(np.asarray(rows, dtype=np.float32)))


def sum_or_zero(values: object) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return float(np.sum(np.asarray(rows, dtype=np.float32)))


__all__ = ["collect_frontier_transitions"]
