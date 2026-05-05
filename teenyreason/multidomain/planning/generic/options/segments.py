"""Mine reusable option segments from real trajectories and repairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..collection.trajectory import ReplayTrajectory, trajectories_to_batch
from .factors import ControlFactorModel


@dataclass(frozen=True)
class OptionSegment:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    start_obs: np.ndarray
    end_obs: np.ndarray
    delta_z: np.ndarray
    duration: int
    score: float
    real_roi: float
    option_id: int = -1


class OptionSegmentMiner:
    """Extract short real windows with reward, survival, or controllability lift."""

    def __init__(self, *, duration: int, max_segments: int):
        self.duration = max(1, int(duration))
        self.max_segments = max(1, int(max_segments))

    def mine(
        self,
        trajectories: list[ReplayTrajectory],
        factor_model: ControlFactorModel,
    ) -> list[OptionSegment]:
        if not trajectories:
            return []
        batch = trajectories_to_batch(trajectories)
        scores = factor_model.transition_scores(batch)
        out: list[OptionSegment] = []
        offset = 0
        for trajectory in trajectories:
            length = int(trajectory.length)
            priorities = scores["factor_priority"][offset : offset + length]
            deltas = scores["delta_z"][offset : offset + length]
            out.extend(mine_trajectory_segments(trajectory, priorities, deltas, self.duration))
            offset += length
        out.sort(key=lambda item: item.score, reverse=True)
        return assign_option_ids(out[: self.max_segments])


def mine_trajectory_segments(
    trajectory: ReplayTrajectory,
    priorities: np.ndarray,
    deltas: np.ndarray,
    duration: int,
) -> list[OptionSegment]:
    rows: list[OptionSegment] = []
    length = int(trajectory.length)
    span = max(1, int(duration))
    if length <= 0:
        return rows
    for start in range(0, max(1, length - span + 1), max(1, span // 2)):
        end = min(length, start + span)
        if end <= start:
            continue
        rewards = trajectory.rewards[start:end]
        local_return = float(np.sum(rewards))
        survival = float(end - start)
        terminal = float(np.any(trajectory.dones[start:end] > 0.5))
        factor = float(np.mean(priorities[start:end])) if priorities.size else 0.0
        smoothness = action_smoothness(trajectory.actions[start:end])
        score = local_return + 0.10 * survival + 0.50 * factor - 0.25 * terminal - 0.05 * smoothness
        delta_z = np.mean(deltas[start:end], axis=0) if deltas.size else np.zeros_like(trajectory.observations[0])
        rows.append(
            OptionSegment(
                observations=np.asarray(trajectory.observations[start:end], dtype=np.float32),
                actions=np.asarray(trajectory.actions[start:end], dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                start_obs=np.asarray(trajectory.observations[start], dtype=np.float32),
                end_obs=np.asarray(trajectory.next_observations[end - 1], dtype=np.float32),
                delta_z=np.asarray(delta_z, dtype=np.float32),
                duration=int(end - start),
                score=float(score),
                real_roi=float(local_return / max(1, end - start)),
            )
        )
    return rows


def assign_option_ids(segments: list[OptionSegment]) -> list[OptionSegment]:
    if not segments:
        return []
    centers: list[np.ndarray] = []
    out: list[OptionSegment] = []
    for segment in segments:
        key = normalized_key(segment)
        option_id = nearest_or_new(centers, key)
        out.append(
            OptionSegment(
                observations=segment.observations,
                actions=segment.actions,
                rewards=segment.rewards,
                start_obs=segment.start_obs,
                end_obs=segment.end_obs,
                delta_z=segment.delta_z,
                duration=segment.duration,
                score=segment.score,
                real_roi=segment.real_roi,
                option_id=option_id,
            )
        )
    return out


def nearest_or_new(centers: list[np.ndarray], key: np.ndarray) -> int:
    if not centers:
        centers.append(key.copy())
        return 0
    distances = [float(np.linalg.norm(key - center)) for center in centers]
    best = int(np.argmin(np.asarray(distances, dtype=np.float32)))
    if distances[best] > 1.5 and len(centers) < 12:
        centers.append(key.copy())
        return len(centers) - 1
    centers[best] = (0.8 * centers[best] + 0.2 * key).astype(np.float32)
    return best


def normalized_key(segment: OptionSegment) -> np.ndarray:
    action_mean = np.mean(segment.actions, axis=0)
    action_std = np.std(segment.actions, axis=0)
    delta = np.asarray(segment.delta_z, dtype=np.float32).reshape(-1)
    delta = delta / float(np.linalg.norm(delta) + 1e-4)
    return np.concatenate([delta, action_mean, action_std]).astype(np.float32)


def action_smoothness(actions: np.ndarray) -> float:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[0] <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(actions, axis=0))))


__all__ = ["OptionSegment", "OptionSegmentMiner"]
