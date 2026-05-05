"""Planner that composes learned options before falling back to raw actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .....envs import make_env
from ...gym_mpc import assert_box_spaces
from ..collection.trajectory import make_trajectory
from .models import OptionActor, OptionOutcomeModel
from .segments import OptionSegment


@dataclass(frozen=True)
class OptionPlanner:
    actor: OptionActor
    outcomes: OptionOutcomeModel
    initiation_centers: dict[int, np.ndarray]
    initiation_radii: dict[int, float]
    action_low: np.ndarray
    action_high: np.ndarray
    done_penalty: float
    uncertainty_penalty: float
    uncertainty_gate: float
    initiation_slack: float
    min_real_roi: float

    @classmethod
    def from_segments(
        cls,
        actor: OptionActor,
        outcomes: OptionOutcomeModel,
        segments: list[OptionSegment],
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        done_penalty: float,
        uncertainty_penalty: float,
        initiation_slack: float = 1.25,
        min_real_roi: float = -0.05,
    ) -> "OptionPlanner":
        centers: dict[int, np.ndarray] = {}
        radii: dict[int, float] = {}
        for option_id in actor.option_ids:
            starts = [segment.start_obs for segment in segments if int(segment.option_id) == int(option_id)]
            center = np.mean(np.stack(starts, axis=0), axis=0).astype(np.float32) if starts else actor.obs_mean
            centers[int(option_id)] = center
            radii[int(option_id)] = initiation_radius(starts, center, actor.obs_std)
        uncertainties = list(outcomes.uncertainty_by_option.values())
        gate = float(np.percentile(np.asarray(uncertainties, dtype=np.float32), 75.0)) if uncertainties else 0.0
        return cls(
            actor=actor,
            outcomes=outcomes,
            initiation_centers=centers,
            initiation_radii=radii,
            action_low=np.asarray(action_low, dtype=np.float32).reshape(-1),
            action_high=np.asarray(action_high, dtype=np.float32).reshape(-1),
            done_penalty=float(done_penalty),
            uncertainty_penalty=float(uncertainty_penalty),
            uncertainty_gate=float(gate),
            initiation_slack=float(initiation_slack),
            min_real_roi=float(min_real_roi),
        )

    def choose(self, observation: np.ndarray) -> dict[str, object]:
        if not self.actor.option_ids:
            return self.raw_fallback("no_options")
        scores: list[tuple[float, int, dict[str, object]]] = []
        for option_id in self.actor.option_ids:
            pred = self.outcomes.predict(observation, option_id)
            initiation = self.initiation_distance(observation, option_id)
            radius = max(float(self.initiation_radii.get(option_id, 1.0)), 1e-4)
            if initiation > radius * max(0.1, self.initiation_slack):
                continue
            if float(self.actor.option_roi.get(option_id, 0.0)) < self.min_real_roi:
                continue
            score = (
                float(pred["total_reward"])
                + 0.5 * float(self.actor.option_roi.get(option_id, 0.0))
                - self.done_penalty * float(pred["done_risk"])
                - self.uncertainty_penalty * float(pred["uncertainty"])
                - 0.05 * initiation
            )
            scores.append((float(score), int(option_id), pred))
        if not scores:
            return self.raw_fallback("no_trusted_initiation")
        score, option_id, pred = max(scores, key=lambda item: item[0])
        trusted = bool(float(pred["uncertainty"]) <= max(self.uncertainty_gate, 1e-6) * 1.5)
        if not trusted:
            return self.raw_fallback("uncertain_option")
        return {
            "kind": "option",
            "option_id": int(option_id),
            "duration": int(pred["duration"]),
            "score": float(score),
            "predicted_roi": float(float(pred["total_reward"]) / max(1, int(pred["duration"]))),
            "uncertainty": float(pred["uncertainty"]),
        }

    def initiation_distance(self, observation: np.ndarray, option_id: int) -> float:
        center = self.initiation_centers.get(int(option_id), self.actor.obs_mean)
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        z = (obs - center.reshape(-1)) / np.maximum(self.actor.obs_std, 1e-4)
        return float(np.mean(np.abs(z)))

    def raw_fallback(self, reason: str) -> dict[str, object]:
        return {
            "kind": "raw",
            "option_id": -1,
            "duration": 1,
            "score": 0.0,
            "predicted_roi": 0.0,
            "uncertainty": 0.0,
            "reason": reason,
        }


def collect_option_planner_episode(
    env_name: str,
    planner: OptionPlanner,
    *,
    seed: int,
    max_steps: int,
    discount: float,
) -> tuple[object, dict[str, float]]:
    env = make_env(env_name, max_episode_steps=max(1, int(max_steps)))
    try:
        assert_box_spaces(env)
        obs, _info = env.reset(seed=int(seed))
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        previous = np.clip(np.zeros_like(planner.action_low), planner.action_low, planner.action_high)
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        next_observations: list[np.ndarray] = []
        dones: list[float] = []
        option_steps = 0
        fallback_steps = 0
        predicted_rois: list[float] = []
        option_ids: list[int] = []
        step = 0
        while step < max(1, int(max_steps)):
            plan = planner.choose(obs)
            duration = max(1, int(plan["duration"]))
            if str(plan["kind"]) == "option":
                option_ids.append(int(plan["option_id"]))
                predicted_rois.append(float(plan["predicted_roi"]))
            for phase in range(duration):
                if step >= max(1, int(max_steps)):
                    break
                if str(plan["kind"]) == "option":
                    action = planner.actor.act(int(plan["option_id"]), obs, phase=phase)
                    option_steps += 1
                else:
                    action = previous.copy()
                    fallback_steps += 1
                next_obs, reward, terminated, truncated, _info = env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                done = bool(terminated or truncated)
                observations.append(obs.copy())
                actions.append(np.asarray(action, dtype=np.float32).copy())
                rewards.append(float(reward))
                next_observations.append(next_obs.copy())
                dones.append(float(done))
                previous = np.asarray(action, dtype=np.float32)
                obs = next_obs
                step += 1
                if done:
                    break
            if dones and dones[-1] > 0.5:
                break
        trajectory = make_trajectory(
            seed=seed,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            discount=float(discount),
        )
        total_steps = max(1, option_steps + fallback_steps)
        stats = {
            "option_planner_return": float(trajectory.episode_return),
            "option_reuse_rate": float(option_steps / total_steps),
            "raw_fallback_rate": float(fallback_steps / total_steps),
            "option_predicted_roi_mean": mean_or_zero(predicted_rois),
            "option_unique_count_used": float(len(set(option_ids))),
        }
        return trajectory, stats
    finally:
        env.close()


def mean_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.mean(np.asarray(rows, dtype=np.float32))) if rows else 0.0


def initiation_radius(starts: list[np.ndarray], center: np.ndarray, obs_std: np.ndarray) -> float:
    if not starts:
        return 1.0
    rows = np.stack(starts, axis=0).astype(np.float32)
    z = (rows - center.reshape(1, -1)) / np.maximum(obs_std.reshape(1, -1), 1e-4)
    distances = np.mean(np.abs(z), axis=1)
    return float(max(np.percentile(distances, 80.0), 0.25))


__all__ = ["OptionPlanner", "collect_option_planner_episode"]
