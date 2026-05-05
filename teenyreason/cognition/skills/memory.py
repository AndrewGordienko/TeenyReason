"""Validated skill memory and runtime skill priors."""

from __future__ import annotations

import numpy as np

from .schema import SkillRecord


class SkillMemory:
    """Append-only memory of real-validated skills."""

    def __init__(self, records: list[SkillRecord] | None = None):
        self._records = list(records or [])
        self.goal_actor = None

    def add(self, record: SkillRecord) -> None:
        self._records.append(record)

    def extend(self, records: list[SkillRecord]) -> None:
        self._records.extend(records)

    def set_goal_actor(self, actor: object | None) -> None:
        self.goal_actor = actor

    def records(self) -> list[SkillRecord]:
        return list(self._records)

    def retrieve(self, observation: np.ndarray, model, *, count: int = 6) -> list[tuple[SkillRecord, float]]:
        if not self._records:
            return []
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        centers = np.asarray([item.initiation_observation for item in self._records], dtype=np.float32)
        obs_mean = np.asarray(getattr(model, "obs_mean", np.mean(centers, axis=0)), dtype=np.float32).reshape(1, -1)
        obs_std = np.maximum(np.asarray(getattr(model, "obs_std", np.std(centers, axis=0) + 1e-4), dtype=np.float32).reshape(1, -1), 1e-4)
        center_z = (centers - obs_mean) / obs_std
        obs_z = (obs - obs_mean) / obs_std
        distances = np.mean(np.square(center_z - obs_z), axis=1)
        order = np.argsort(distances)[: max(1, min(int(count), len(self._records)))]
        out: list[tuple[SkillRecord, float]] = []
        for index in order:
            record = self._records[int(index)]
            familiarity = float(1.0 / (1.0 + float(distances[int(index)])))
            weight = familiarity * max(0.05, float(record.reliability))
            out.append((record, float(weight)))
        return out

    def action_prior(
        self,
        observation: np.ndarray,
        model,
        action_low: np.ndarray,
        action_high: np.ndarray,
        *,
        step: int,
        count: int = 6,
    ) -> tuple[np.ndarray, dict[str, float]]:
        retrieved = self.retrieve(observation, model, count=count)
        action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if not retrieved:
            return np.clip(np.zeros_like(action_low), action_low, action_high).astype(np.float32), empty_stats()
        actions: list[np.ndarray] = []
        weights: list[float] = []
        reliabilities: list[float] = []
        for record, weight in retrieved:
            local_step = min(max(0, int(step)), max(0, int(record.duration) - 1))
            stored_action = np.asarray(record.actions[local_step], dtype=np.float32).reshape(-1)
            if self.goal_actor is not None:
                goal_action = self.goal_actor.act(
                    observation,
                    record.goal.target_delta,
                    phase=local_step,
                )
                stored_action = 0.65 * stored_action + 0.35 * np.asarray(goal_action, dtype=np.float32).reshape(-1)
            actions.append(stored_action)
            weights.append(float(weight))
            reliabilities.append(float(record.reliability))
        weight_arr = np.maximum(np.asarray(weights, dtype=np.float32), 1e-6)
        weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1e-6)
        action = np.sum(np.asarray(actions, dtype=np.float32) * weight_arr.reshape(-1, 1), axis=0)
        familiarity = float(np.clip(np.max(weight_arr) * len(weight_arr), 0.0, 1.0))
        reliability = float(np.mean(np.asarray(reliabilities, dtype=np.float32))) if reliabilities else 0.0
        return np.clip(action, action_low, action_high).astype(np.float32), {
            "skill_familiarity": familiarity,
            "skill_reliability": reliability,
            "skill_retrieved_count": float(len(retrieved)),
            "skill_goal_actor_used": float(self.goal_actor is not None),
        }

    def summary(self, *, prefix: str = "skill_memory") -> dict[str, float]:
        records = self._records
        return {
            f"{prefix}_count": float(len(records)),
            f"{prefix}_reliability_mean": mean([item.reliability for item in records]),
            f"{prefix}_return_lift_mean": mean([item.real_return_lift for item in records]),
            f"{prefix}_survival_lift_mean": mean([item.survival_lift for item in records]),
            f"{prefix}_terminal_avoid_count": float(sum(1 for item in records if item.terminal_avoid > 0.5)),
        }


def empty_stats() -> dict[str, float]:
    return {
        "skill_familiarity": 0.0,
        "skill_reliability": 0.0,
        "skill_retrieved_count": 0.0,
        "skill_goal_actor_used": 0.0,
    }


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


__all__ = ["SkillMemory"]
