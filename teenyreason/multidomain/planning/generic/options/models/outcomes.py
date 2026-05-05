"""Option-level outcome model for planning over learned skills."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..segments import OptionSegment


@dataclass(frozen=True)
class OptionOutcomeModel:
    """Predict k-step option effect from real segment statistics."""

    delta_by_option: dict[int, np.ndarray]
    reward_by_option: dict[int, float]
    done_risk_by_option: dict[int, float]
    uncertainty_by_option: dict[int, float]
    duration_by_option: dict[int, int]
    obs_mean: np.ndarray
    obs_std: np.ndarray

    @classmethod
    def fit(cls, segments: list[OptionSegment]) -> "OptionOutcomeModel":
        if not segments:
            raise ValueError("cannot fit OptionOutcomeModel without option segments")
        starts = np.stack([segment.start_obs for segment in segments], axis=0)
        obs_mean = np.mean(starts, axis=0).astype(np.float32)
        obs_std = (np.std(starts, axis=0) + 1e-4).astype(np.float32)
        deltas: dict[int, np.ndarray] = {}
        rewards: dict[int, float] = {}
        dones: dict[int, float] = {}
        uncertainties: dict[int, float] = {}
        durations: dict[int, int] = {}
        for option_id in sorted({int(segment.option_id) for segment in segments}):
            group = [segment for segment in segments if int(segment.option_id) == option_id]
            delta_rows = np.stack([segment.delta_z for segment in group], axis=0).astype(np.float32)
            reward_rows = np.asarray([float(np.sum(segment.rewards)) for segment in group], dtype=np.float32)
            done_rows = np.asarray([float(segment.rewards.size == 0) for segment in group], dtype=np.float32)
            deltas[option_id] = np.mean(delta_rows, axis=0).astype(np.float32)
            rewards[option_id] = float(np.mean(reward_rows))
            dones[option_id] = float(np.mean(done_rows))
            uncertainties[option_id] = float(np.mean(np.linalg.norm(delta_rows - deltas[option_id].reshape(1, -1), axis=1)))
            durations[option_id] = int(round(np.mean([segment.duration for segment in group])))
        return cls(deltas, rewards, dones, uncertainties, durations, obs_mean, obs_std)

    def predict(self, observation: np.ndarray, option_id: int, *, duration: int | None = None) -> dict[str, object]:
        option = int(option_id)
        if option not in self.delta_by_option:
            option = self.best_option_id()
        base_duration = max(1, int(self.duration_by_option.get(option, 1)))
        requested = max(1, int(base_duration if duration is None else duration))
        scale = float(requested / base_duration)
        delta_z = self.delta_by_option[option] * scale
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        next_obs = obs + delta_z * self.obs_std
        return {
            "next_observation": next_obs.astype(np.float32),
            "delta_z": delta_z.astype(np.float32),
            "total_reward": float(self.reward_by_option[option] * scale),
            "done_risk": float(np.clip(self.done_risk_by_option[option] * scale, 0.0, 1.0)),
            "uncertainty": float(self.uncertainty_by_option[option] * np.sqrt(scale)),
            "duration": int(requested),
        }

    def best_option_id(self) -> int:
        return max(self.reward_by_option, key=lambda option_id: self.reward_by_option[option_id] - self.uncertainty_by_option[option_id])

    def prediction_error(self, segments: list[OptionSegment]) -> float:
        errors: list[float] = []
        for segment in segments:
            pred = self.predict(segment.start_obs, int(segment.option_id), duration=int(segment.duration))
            errors.append(float(np.linalg.norm(np.asarray(pred["delta_z"], dtype=np.float32) - segment.delta_z)))
        return float(np.mean(np.asarray(errors, dtype=np.float32))) if errors else 0.0


__all__ = ["OptionOutcomeModel"]
