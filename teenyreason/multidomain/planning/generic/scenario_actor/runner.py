"""Scenario-memory actor loop without planner-as-teacher training."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from teenyreason.crawler.experience import build_control_crawler_experience
from teenyreason.cognition.scenario import ScenarioMemory, retrieve_windows
from teenyreason.cognition.skills import SkillMemory, run_skill_discovery_round

from .....envs import make_env
from ...gym_mpc import TransitionBatch, assert_box_spaces
from ..collection import collect_training_transitions
from ..collection.trajectory import trajectories_to_batch
from ..config import AdvancedGymMPCConfig
from ..diagnostics import batch_from_rows, merge_transition_batches, model_validation_diagnostics
from ..fitting import fit_action_value_model, fit_model, fit_value_model, make_planner
from .actor_critic import ActorCriticAgent
from .policy import append_imagined_variants, append_real_batch, create_policy_state, restore_agent, snapshot_agent, train_scenario_policy


@dataclass(frozen=True)
class ScenarioActorConfig:
    base: AdvancedGymMPCConfig
    rounds: int = 3
    eval_steps: int | None = None


@dataclass
class ScenarioActorResult:
    config: ScenarioActorConfig
    diagnostics: dict[str, object]
    action_low: np.ndarray
    action_high: np.ndarray
    rows: list[dict[str, object]]

    def summary(self) -> dict[str, object]:
        rows = list(self.rows)
        final = rows[-1] if rows else {}
        collector_best = float(self.diagnostics.get("collector_best_return", 0.0))
        best_row = max(rows, key=lambda row: float(row.get("best_return", -float("inf")))) if rows else {}
        solved = [row for row in rows if bool(row.get("solved"))]
        probe_samples = int(self.diagnostics.get("collector_interaction_steps", 0))
        total_samples = int(final.get("real_samples", probe_samples))
        best_return = max(collector_best, float(best_row.get("best_return", collector_best)))
        peak_samples = probe_samples if collector_best >= best_return else int(best_row.get("samples_to_peak", total_samples))
        final_eval_return = float(final.get("scenario_actor_real_return", final.get("eval_return", 0.0)))
        final_eval_steps = int(final.get("scenario_actor_real_steps", 0))
        retained_actor_return = float(final.get("retained_actor_return", collector_best))
        retained_actor_steps = int(final.get("retained_actor_steps", 0))
        return {
            "model_family": "ScenarioActor",
            "env_name": self.config.base.env_name,
            "seed": int(self.config.base.seed),
            "collector": str(self.config.base.collector),
            "method": "scenario_actor",
            "probe_samples": probe_samples,
            "probe_return_mean": float(self.diagnostics.get("collector_return_mean", 0.0)),
            "control_return": final_eval_return,
            "control_steps": final_eval_steps,
            "latest_eval_return": final_eval_return,
            "latest_eval_steps": final_eval_steps,
            "retained_actor_return": retained_actor_return,
            "retained_actor_steps": retained_actor_steps,
            "collector_best_return": collector_best,
            "best_return": float(best_return),
            "samples_to_peak": int(peak_samples),
            "samples_to_solve": int(solved[0]["real_samples"]) if solved else None,
            "total_samples": total_samples,
            "solved": bool(solved),
            "solve_return": float(self.config.base.solve_return),
            "horizon": int(self.config.base.scenario_variant_horizon),
            "candidate_count": int(self.config.base.scenario_variants_per_window),
            "train_loss": float(final.get("scenario_actor_refine_loss", 0.0)),
            "diagnostics": {**self.diagnostics, **final},
            "rows": rows,
        }


def run_scenario_actor(config: ScenarioActorConfig, *, render_mode: str | None = None) -> ScenarioActorResult:
    if render_mode == "rgb_array":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    batch, action_low, action_high, collector_stats = collect_training_transitions(config.base)
    replay_trajectories = list(collector_stats.pop("_replay_trajectories", []))
    collector_stats.pop("_value_actor_batch", None)
    experience = build_control_crawler_experience(
        config.base,
        batch,
        action_low,
        action_high,
        collector_stats,
        replay_trajectories=replay_trajectories,
    )
    replay_trajectories = experience.replay_trajectories
    memory = experience.memory
    skill_memory = experience.skill_memory
    diagnostics = initial_diagnostics(config.base, batch, action_low, action_high, collector_stats)
    diagnostics.update(experience.summary())
    real_samples = int(diagnostics.get("collector_interaction_steps", batch.observations.shape[0]))
    best_return = float(diagnostics.get("collector_best_return", max_or_zero(batch.episode_returns)))
    policy = create_policy_state(config.base, batch, action_low, action_high)
    focus_observations = experience.focus_observations(
        initial_focus_observations(config.base, memory),
        count=int(config.base.scenario_focus_window_count),
    )
    best_snapshot = snapshot_agent(policy.agent)
    best_actor_return = float(best_return)
    best_actor_steps = 0
    rows: list[dict[str, object]] = []
    for round_idx in range(max(1, int(config.rounds))):
        seed_offset = 231_000 + round_idx * 997
        model, model_loss = fit_model(config.base, batch, action_low, action_high, seed_offset=seed_offset)
        value_model, value_loss, value_stats = fit_value_model(config.base, batch, action_low, action_high, seed_offset=seed_offset)
        action_value_model, action_value_loss, action_value_stats = fit_action_value_model(config.base, batch, action_low, action_high, seed_offset=seed_offset)
        skill_round = run_skill_discovery_round(
            config.base,
            replay_trajectories,
            model,
            action_low,
            action_high,
            round_idx=round_idx,
        )
        accepted_skill_trajectories = experience.absorb_skill_round(skill_round)
        real_samples += int(skill_round.diagnostics.get("skill_real_validation_steps", 0.0))
        if accepted_skill_trajectories:
            skill_batch = trajectories_to_batch(accepted_skill_trajectories)
            batch = merge_transition_batches(batch, skill_batch)
            experience.refresh_batch(batch)
            append_real_batch(policy, skill_batch, config.base)
        windows = retrieve_windows(
            memory,
            count=int(config.base.scenario_window_count),
            window_size=int(config.base.scenario_window_size),
            focus_observations=focus_observations,
            focus_weight=float(config.base.scenario_focus_distance_weight),
        )
        variants = experience.generate_variants(
            model,
            value_model,
            action_value_model,
            windows,
            round_idx=round_idx,
        )
        trainable_variants = experience.trainable_variants(variants)
        imagined_stats = append_imagined_variants(policy, trainable_variants, config.base)
        actor_stats = train_scenario_policy(config.base, policy, round_idx=round_idx)
        planner = make_planner(config.base, action_low, action_high)
        rollout = rollout_actor(
            config,
            policy.agent,
            model,
            planner,
            memory,
            skill_memory,
            action_low,
            action_high,
            render_mode=render_mode if round_idx == max(1, int(config.rounds)) - 1 else None,
            seed_offset=round_idx * 10_000,
        )
        real_samples += int(rollout["steps"])
        rollout_return = float(rollout["return"])
        previous_best = float(best_return)
        best_return = max(previous_best, rollout_return)
        actor_accepted = rollout_return >= best_actor_return
        if actor_accepted:
            best_snapshot = snapshot_agent(policy.agent)
            best_actor_return = rollout_return
            best_actor_steps = int(rollout["steps"])
        else:
            restore_agent(policy.agent, best_snapshot)
        surprise = scenario_surprise(variants, rollout_return)
        experience.record_real_rollout(
            rollout["rows"],
            seed=int(config.base.seed + 233_000 + round_idx),
            discount=float(config.base.discount),
            surprise=surprise,
        )
        experience.record_imagined_variants(variants, round_idx=round_idx, surprise=surprise)
        experience.validate_imagination_against_rollout(rollout["rows"])
        rollout_batch = batch_from_rows(rollout["rows"])
        batch = merge_transition_batches(batch, rollout_batch)
        experience.refresh_batch(batch)
        append_real_batch(policy, rollout_batch, config.base)
        focus_observations = experience.focus_observations(
            focus_from_rollout(config.base, rollout["rows"], model),
            count=int(config.base.scenario_focus_window_count),
        )
        row = {
            "round": int(round_idx + 1),
            "real_samples": int(real_samples),
            "samples_to_peak": int(real_samples if rollout_return > previous_best else rows[-1]["samples_to_peak"] if rows else real_samples),
            "eval_return": float(rollout_return),
            "best_return": float(best_return),
            "solved": bool(best_return >= float(config.base.solve_return)),
            "world_model_train_loss": float(model_loss),
            "value_train_loss": float(value_loss),
            "action_value_train_loss": float(action_value_loss),
            "scenario_actor_refine_loss": float(actor_stats.get("scenario_critic_td_loss", 0.0)),
            "scenario_actor_real_return": float(rollout_return),
            "scenario_actor_real_steps": float(rollout["steps"]),
            "scenario_actor_accepted": float(actor_accepted),
            "retained_actor_return": float(best_actor_return),
            "retained_actor_steps": float(best_actor_steps),
            **scenario_diagnostics(memory, windows, variants, surprise, rollout_return, focus_observations),
            **rollout["diagnostics"],
            **imagined_stats,
            **skill_round.diagnostics,
            **skill_memory.summary(prefix="skill_memory"),
            **experience.summary(),
            **value_stats,
            **action_value_stats,
            **actor_stats,
        }
        rows.append(row)
    return ScenarioActorResult(config=config, diagnostics=diagnostics, action_low=action_low, action_high=action_high, rows=rows)


def rollout_actor(
    config: ScenarioActorConfig,
    actor: ActorCriticAgent,
    model,
    planner,
    memory: ScenarioMemory,
    skill_memory: SkillMemory,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    render_mode: str | None,
    seed_offset: int,
) -> dict[str, object]:
    max_steps = max(1, int(config.eval_steps or config.base.control_steps))
    env = make_env(config.base.env_name, max_episode_steps=max_steps, render_mode=render_mode)
    try:
        assert_box_spaces(env)
        observation, _info = env.reset(seed=int(config.base.seed + 234_000 + int(seed_offset)))
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        rows: list[dict[str, np.ndarray | float]] = []
        total_return = 0.0
        memory_blends: list[float] = []
        plan_blends: list[float] = []
        familiarities: list[float] = []
        skill_blends: list[float] = []
        skill_familiarities: list[float] = []
        skill_reliabilities: list[float] = []
        uncertainties: list[float] = []
        done_risks: list[float] = []
        for step in range(max_steps):
            if render_mode is not None:
                env.render()
            actor_action = actor.act(observation)
            action, prior_stats = runtime_memory_action(
                config.base,
                model,
                planner,
                memory,
                skill_memory,
                observation,
                actor_action,
                action_low,
                action_high,
                step=step,
            )
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            done = bool(terminated or truncated)
            rows.append({"observation": observation.copy(), "action": action.copy(), "reward": float(reward), "next_observation": next_obs.copy(), "done": float(done)})
            total_return += float(reward)
            memory_blends.append(float(prior_stats["memory_blend"]))
            plan_blends.append(float(prior_stats["plan_blend"]))
            familiarities.append(float(prior_stats["familiarity"]))
            skill_blends.append(float(prior_stats["skill_blend"]))
            skill_familiarities.append(float(prior_stats["skill_familiarity"]))
            skill_reliabilities.append(float(prior_stats["skill_reliability"]))
            uncertainties.append(float(prior_stats["uncertainty"]))
            done_risks.append(float(prior_stats["done_risk"]))
            observation = next_obs
            if done:
                break
        diagnostics = {
            "scenario_runtime_memory_blend_mean": mean(memory_blends),
            "scenario_runtime_plan_blend_mean": mean(plan_blends),
            "scenario_runtime_memory_familiarity_mean": mean(familiarities),
            "skill_runtime_blend_mean": mean(skill_blends),
            "skill_runtime_familiarity_mean": mean(skill_familiarities),
            "skill_runtime_reliability_mean": mean(skill_reliabilities),
            "skill_chain_action_fraction": float(np.mean(np.asarray(skill_blends, dtype=np.float32) > 1e-5)) if skill_blends else 0.0,
            "scenario_runtime_model_uncertainty_mean": mean(uncertainties),
            "scenario_runtime_done_risk_mean": mean(done_risks),
            "scenario_runtime_memory_prior_used_fraction": float(np.mean(np.asarray(memory_blends, dtype=np.float32) > 1e-5)) if memory_blends else 0.0,
            "scenario_runtime_plan_used_fraction": float(np.mean(np.asarray(plan_blends, dtype=np.float32) > 1e-5)) if plan_blends else 0.0,
        }
        return {"return": float(total_return), "steps": int(len(rows)), "rows": rows, "diagnostics": diagnostics}
    finally:
        env.close()


def runtime_memory_action(
    config: AdvancedGymMPCConfig,
    model,
    planner,
    memory: ScenarioMemory,
    skill_memory: SkillMemory,
    observation: np.ndarray,
    actor_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    step: int,
) -> tuple[np.ndarray, dict[str, float]]:
    prior_action, familiarity = nearest_memory_action(memory, model, observation, action_low, action_high)
    skill_action, skill_stats = skill_memory.action_prior(
        observation,
        model,
        action_low,
        action_high,
        step=step,
        count=max(1, int(config.skill_runtime_retrieval_count)),
    )
    plan = planner.choose_plan(
        model,
        observation,
        seed=int(config.seed + 242_000),
        step=int(step),
        discount=float(config.discount),
        done_penalty=float(config.done_penalty),
        uncertainty_penalty=float(config.uncertainty_penalty),
    )
    plan_action = np.asarray(plan["action"], dtype=np.float32).reshape(-1)
    pred = model.predict_batch(np.asarray(observation, dtype=np.float32).reshape(1, -1), np.asarray(actor_action, dtype=np.float32).reshape(1, -1))
    uncertainty = float(pred["uncertainty"][0])
    done_risk = float(pred["done_risk"][0])
    uncertainty_term = float(uncertainty / (1.0 + max(0.0, uncertainty)))
    pressure = float(np.clip((1.0 - familiarity + uncertainty_term + done_risk) / 3.0, 0.0, 1.0))
    skill_pressure = float(np.clip(skill_stats["skill_familiarity"] * skill_stats["skill_reliability"], 0.0, 1.0))
    skill_blend = float(config.skill_runtime_blend) * skill_pressure
    memory_blend = float(config.scenario_runtime_memory_blend) * pressure * (1.0 - 0.50 * skill_pressure)
    plan_blend = float(config.scenario_runtime_plan_blend) * (0.35 + 0.65 * pressure) * (1.0 - 0.25 * skill_pressure)
    memory_blend, plan_blend, skill_blend = cap_blends(memory_blend, plan_blend, skill_blend, cap=0.95)
    total_blend = memory_blend + plan_blend + skill_blend
    actor_weight = 1.0 - total_blend
    action = (
        actor_weight * np.asarray(actor_action, dtype=np.float32).reshape(-1)
        + memory_blend * prior_action
        + plan_blend * plan_action
        + skill_blend * skill_action
    )
    return np.clip(action, action_low, action_high).astype(np.float32), {
        "memory_blend": memory_blend,
        "plan_blend": plan_blend,
        "skill_blend": skill_blend,
        "familiarity": familiarity,
        "skill_familiarity": float(skill_stats["skill_familiarity"]),
        "skill_reliability": float(skill_stats["skill_reliability"]),
        "uncertainty": uncertainty,
        "done_risk": done_risk,
    }


def cap_blends(memory_blend: float, plan_blend: float, skill_blend: float, *, cap: float) -> tuple[float, float, float]:
    total = float(memory_blend + plan_blend + skill_blend)
    if total <= float(cap) or total <= 1e-8:
        return float(memory_blend), float(plan_blend), float(skill_blend)
    scale = float(cap) / total
    return float(memory_blend * scale), float(plan_blend * scale), float(skill_blend * scale)


def nearest_memory_action(
    memory: ScenarioMemory,
    model,
    observation: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, float]:
    rows = memory.real_tracelets()
    if not rows:
        center = np.clip(np.zeros_like(action_low, dtype=np.float32), action_low, action_high)
        return center, 0.0
    obs = np.asarray([item.observation for item in rows], dtype=np.float32)
    actions = np.asarray([item.action for item in rows], dtype=np.float32)
    current = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    obs_z = (obs - model.obs_mean.reshape(1, -1)) / np.maximum(model.obs_std.reshape(1, -1), 1e-4)
    current_z = (current - model.obs_mean.reshape(1, -1)) / np.maximum(model.obs_std.reshape(1, -1), 1e-4)
    distances = np.mean(np.square(obs_z - current_z), axis=1)
    k = max(1, min(8, int(distances.shape[0])))
    order = np.argsort(distances)[:k]
    scale = float(np.median(distances[order]) + 1e-4)
    weights = np.exp(-distances[order] / scale).astype(np.float32)
    weights = weights / max(float(np.sum(weights)), 1e-6)
    action = np.sum(actions[order] * weights.reshape(-1, 1), axis=0)
    familiarity = float(1.0 / (1.0 + float(np.min(distances))))
    return np.clip(action, action_low, action_high).astype(np.float32), familiarity


def scenario_diagnostics(
    memory: ScenarioMemory,
    windows: list[object],
    variants: list[object],
    surprise: float,
    rollout_return: float,
    focus_observations: list[np.ndarray],
) -> dict[str, float]:
    predicted = [float(item.predicted_return + item.predicted_value) for item in variants]
    weights = [float(item.weights.combined) for item in variants]
    real_count = max(1, len(memory.real_tracelets()))
    imagined_count = len(memory.imagined_tracelets())
    return {
        **memory.summary(),
        "scenario_window_count": float(len(windows)),
        "scenario_variant_count": float(len(variants)),
        "scenario_focus_count": float(len(focus_observations)),
        "scenario_failure_conditioned_retrieval": 1.0,
        "scenario_retrieval_familiarity_mean": mean([float(item.familiarity) for item in windows]),
        "scenario_weight_mean": mean(weights),
        "scenario_weight_max": max_or_zero(weights),
        "scenario_surprise_mean": float(surprise),
        "scenario_prediction_gap": float(mean(predicted) - float(rollout_return)) if predicted else 0.0,
        "scenario_imagined_to_real_weight_ratio": float(imagined_count / real_count),
    }


def scenario_surprise(variants: list[object], rollout_return: float) -> float:
    predicted = [float(item.predicted_return + item.predicted_value) for item in variants]
    if not predicted:
        return 0.0
    return float(abs(mean(predicted) - float(rollout_return)) / max(1.0, abs(float(rollout_return))))


def initial_diagnostics(config: AdvancedGymMPCConfig, batch: TransitionBatch, action_low: np.ndarray, action_high: np.ndarray, collector_stats: dict[str, object]) -> dict[str, object]:
    model, model_loss = fit_model(config, batch, action_low, action_high, seed_offset=230_000)
    return {**collector_stats, "world_model_train_loss": float(model_loss), **model_validation_diagnostics(model, batch, rollout_horizons=(1, 5, 10))}


def initial_focus_observations(config: AdvancedGymMPCConfig, memory: ScenarioMemory) -> list[np.ndarray]:
    rows = memory.real_tracelets()
    if not rows:
        return []
    score = np.asarray([float(item.done) * 2.0 - float(item.return_to_go) for item in rows], dtype=np.float32)
    order = np.argsort(score)[::-1][: max(1, int(config.scenario_focus_window_count))]
    return [rows[int(idx)].observation.astype(np.float32).copy() for idx in order]


def focus_from_rollout(config: AdvancedGymMPCConfig, rows: list[dict[str, np.ndarray | float]], model) -> list[np.ndarray]:
    if not rows:
        return []
    scores: list[float] = []
    for row in rows:
        obs = np.asarray(row["observation"], dtype=np.float32).reshape(1, -1)
        action = np.asarray(row["action"], dtype=np.float32).reshape(1, -1)
        next_obs = np.asarray(row["next_observation"], dtype=np.float32).reshape(1, -1)
        pred = model.predict_batch(obs, action)
        transition_error = model.normalized_transition_error(pred["next_observation"][0], next_obs[0])
        score = transition_error + float(row["done"]) * 3.0 - float(row["reward"])
        scores.append(float(score))
    order = np.argsort(np.asarray(scores, dtype=np.float32))[::-1][: max(1, int(config.scenario_focus_window_count))]
    return [np.asarray(rows[int(idx)]["observation"], dtype=np.float32).copy() for idx in order]


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


def max_or_zero(values: object) -> float:
    rows = list(np.asarray(values, dtype=np.float32).reshape(-1))
    return float(np.max(rows)) if rows else 0.0


__all__ = ["ScenarioActorConfig", "ScenarioActorResult", "run_scenario_actor"]
