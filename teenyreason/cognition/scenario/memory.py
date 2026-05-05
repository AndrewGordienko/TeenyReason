"""Tracelet memory used by scenario-based imagination."""

from __future__ import annotations

import numpy as np

from teenyreason.multidomain.planning.generic.collection.trajectory import ReplayTrajectory
from teenyreason.multidomain.planning.gym_mpc import TransitionBatch

from .schema import ScenarioTracelet


class ScenarioMemory:
    """Append-only memory of real and imagined tracelets."""

    def __init__(self, tracelets: list[ScenarioTracelet] | None = None):
        self._tracelets = list(tracelets or [])

    @classmethod
    def from_trajectories(cls, trajectories: list[ReplayTrajectory], *, source: str = "real") -> "ScenarioMemory":
        memory = cls()
        for trajectory_id, trajectory in enumerate(trajectories):
            memory.add_trajectory(trajectory, trajectory_id=trajectory_id, source=source)
        return memory

    @classmethod
    def from_batch(cls, batch: TransitionBatch, *, discount: float, source: str = "real") -> "ScenarioMemory":
        memory = cls()
        rewards = np.asarray(batch.rewards, dtype=np.float32)
        dones = np.asarray(batch.dones, dtype=np.float32)
        returns = returns_to_go(rewards, dones, discount=float(discount))
        trajectory_id = 0
        step = 0
        for idx in range(int(batch.actions.shape[0])):
            memory._tracelets.append(
                ScenarioTracelet(
                    observation=batch.observations[idx].astype(np.float32).copy(),
                    action=batch.actions[idx].astype(np.float32).copy(),
                    next_observation=batch.next_observations[idx].astype(np.float32).copy(),
                    reward=float(batch.rewards[idx]),
                    done=float(batch.dones[idx]),
                    return_to_go=float(returns[idx]),
                    trajectory_id=int(trajectory_id),
                    step=int(step),
                    source=str(source),
                )
            )
            step += 1
            if float(batch.dones[idx]) > 0.5:
                trajectory_id += 1
                step = 0
        return memory

    def add_trajectory(
        self,
        trajectory: ReplayTrajectory,
        *,
        trajectory_id: int | None = None,
        source: str = "real",
        surprise: float = 0.0,
    ) -> None:
        tid = int(len(self.trajectory_ids()) if trajectory_id is None else trajectory_id)
        for step in range(trajectory.length):
            rtg = float(trajectory.returns_to_go[step]) if trajectory.returns_to_go.size else float(trajectory.episode_return)
            self._tracelets.append(
                ScenarioTracelet(
                    observation=trajectory.observations[step].astype(np.float32).copy(),
                    action=trajectory.actions[step].astype(np.float32).copy(),
                    next_observation=trajectory.next_observations[step].astype(np.float32).copy(),
                    reward=float(trajectory.rewards[step]),
                    done=float(trajectory.dones[step]),
                    return_to_go=rtg,
                    trajectory_id=tid,
                    step=int(step),
                    source=str(source),
                    surprise=float(surprise),
                )
            )

    def add_rows(
        self,
        rows: list[dict[str, np.ndarray | float]] | tuple[dict[str, np.ndarray | float], ...],
        *,
        source: str,
        surprise: float = 0.0,
    ) -> None:
        if not rows:
            return
        tid = self.next_trajectory_id()
        rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float32)
        dones = np.asarray([float(row["done"]) for row in rows], dtype=np.float32)
        returns = returns_to_go(rewards, dones, discount=0.99)
        for step, row in enumerate(rows):
            self._tracelets.append(
                ScenarioTracelet(
                    observation=np.asarray(row["observation"], dtype=np.float32).copy(),
                    action=np.asarray(row["action"], dtype=np.float32).copy(),
                    next_observation=np.asarray(row["next_observation"], dtype=np.float32).copy(),
                    reward=float(row["reward"]),
                    done=float(row["done"]),
                    return_to_go=float(returns[step]),
                    trajectory_id=tid,
                    step=int(step),
                    source=str(source),
                    surprise=float(surprise),
                )
            )

    def tracelets(self) -> list[ScenarioTracelet]:
        return list(self._tracelets)

    def real_tracelets(self) -> list[ScenarioTracelet]:
        return [item for item in self._tracelets if item.source == "real"]

    def imagined_tracelets(self) -> list[ScenarioTracelet]:
        return [item for item in self._tracelets if item.source != "real"]

    def trajectory_ids(self) -> set[int]:
        return {int(item.trajectory_id) for item in self._tracelets}

    def next_trajectory_id(self) -> int:
        ids = self.trajectory_ids()
        return max(ids) + 1 if ids else 0

    def by_trajectory(self) -> dict[int, list[ScenarioTracelet]]:
        out: dict[int, list[ScenarioTracelet]] = {}
        for item in sorted(self._tracelets, key=lambda row: (row.trajectory_id, row.step)):
            out.setdefault(int(item.trajectory_id), []).append(item)
        return out

    def observation_stats(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.real_tracelets() or self._tracelets
        if not rows:
            return np.zeros((1,), dtype=np.float32), np.ones((1,), dtype=np.float32)
        obs = np.asarray([item.observation for item in rows], dtype=np.float32)
        return np.mean(obs, axis=0).astype(np.float32), np.maximum(np.std(obs, axis=0), 1e-4).astype(np.float32)

    def summary(self, *, prefix: str = "scenario_memory") -> dict[str, float]:
        real = self.real_tracelets()
        imagined = self.imagined_tracelets()
        surprise = [item.surprise for item in self._tracelets]
        return {
            f"{prefix}_real_count": float(len(real)),
            f"{prefix}_imagined_count": float(len(imagined)),
            f"{prefix}_trajectory_count": float(len(self.trajectory_ids())),
            f"{prefix}_surprise_mean": mean(surprise),
        }


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


def returns_to_go(rewards: np.ndarray, dones: np.ndarray, *, discount: float) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(rewards.size - 1, -1, -1):
        if idx < rewards.size - 1 and float(dones[idx]) > 0.5:
            running = 0.0
        running = float(rewards[idx]) + float(discount) * running
        out[idx] = running
    return out


__all__ = ["ScenarioMemory"]
