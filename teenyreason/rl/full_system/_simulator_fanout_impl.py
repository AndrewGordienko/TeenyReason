"""Cheap simulator-side fan-out and cached teacher labels for CartPole."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import numpy as np

from ...envs import CONTINUOUS_CARTPOLE_NAME


@dataclass(frozen=True)
class ContinuousCartPoleSnapshot:
    """Explicit mutable simulator state needed for short branching rollouts."""

    state: np.ndarray
    steps_beyond_done: int | None
    elapsed_steps: int
    gravity: float
    masscart: float
    masspole: float
    length: float
    force_mag: float


@dataclass(frozen=True)
class FanoutLabel:
    """One cached short-horizon fan-out supervision target."""

    candidate_actions: np.ndarray
    returns: np.ndarray
    risks: np.ndarray
    recoverability: np.ndarray
    scores: np.ndarray
    best_idx: int
    best_vs_actor_margin: float

    @classmethod
    def from_result(
        cls,
        *,
        candidate_actions: np.ndarray,
        result: dict[str, np.ndarray | int],
    ) -> FanoutLabel:
        candidate_actions = np.asarray(candidate_actions, dtype=np.float32)
        scores = np.asarray(result["scores"], dtype=np.float32)
        best_idx = int(result["best_idx"])
        actor_score = float(scores[0]) if scores.size else 0.0
        best_score = float(scores[best_idx]) if scores.size else 0.0
        return cls(
            candidate_actions=candidate_actions.astype(np.float32),
            returns=np.asarray(result["returns"], dtype=np.float32),
            risks=np.asarray(result["risks"], dtype=np.float32),
            recoverability=np.asarray(result["recoverability"], dtype=np.float32),
            scores=scores.astype(np.float32),
            best_idx=best_idx,
            best_vs_actor_margin=float(best_score - actor_score),
        )

    def to_payload(self) -> dict[str, np.ndarray | int | float]:
        return {
            "candidate_actions": self.candidate_actions.astype(np.float32),
            "returns": self.returns.astype(np.float32),
            "risks": self.risks.astype(np.float32),
            "recoverability": self.recoverability.astype(np.float32),
            "scores": self.scores.astype(np.float32),
            "best_idx": int(self.best_idx),
            "best_vs_actor_margin": float(self.best_vs_actor_margin),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, np.ndarray | int | float]) -> FanoutLabel:
        return cls(
            candidate_actions=np.asarray(payload["candidate_actions"], dtype=np.float32),
            returns=np.asarray(payload["returns"], dtype=np.float32),
            risks=np.asarray(payload["risks"], dtype=np.float32),
            recoverability=np.asarray(payload["recoverability"], dtype=np.float32),
            scores=np.asarray(payload["scores"], dtype=np.float32),
            best_idx=int(payload["best_idx"]),
            best_vs_actor_margin=float(payload["best_vs_actor_margin"]),
        )


def cartpole_recoverability_from_state(state: np.ndarray) -> float:
    """Cheap recoverability heuristic used across the cheap-control pivot."""
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.shape[0] < 4:
        return 1.0
    x = abs(float(values[0]))
    dx = abs(float(values[1]))
    theta = abs(float(values[2]))
    dtheta = abs(float(values[3]))
    centered = 1.0 - np.clip(x / 2.4, 0.0, 1.0)
    upright = 1.0 - np.clip(theta / 0.35, 0.0, 1.0)
    calm = 1.0 / (1.0 + dx + dtheta)
    return float(np.clip(0.45 * upright + 0.35 * centered + 0.20 * calm, 0.0, 1.0))


def candidate_score(
    returns: np.ndarray,
    risks: np.ndarray,
    recoverability: np.ndarray,
) -> np.ndarray:
    """Shared candidate score used by the cheap belief-conditioned controller."""
    returns = np.asarray(returns, dtype=np.float32)
    risks = np.asarray(risks, dtype=np.float32)
    recoverability = np.asarray(recoverability, dtype=np.float32)
    return (
        returns
        + 0.5 * recoverability
        - 2.0 * risks
    ).astype(np.float32)


class PersistentFanoutLabelCache:
    """Small persistent cache for repeated short-horizon fan-out supervision."""

    def __init__(
        self,
        *,
        env_name: str,
        cache_dir: str | Path = "artifacts/fanout_label_cache",
        rounding_decimals: int = 4,
    ):
        self.env_name = str(env_name)
        self.cache_dir = Path(cache_dir) / self.env_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rounding_decimals = int(rounding_decimals)
        self._memory: dict[str, FanoutLabel] = {}

    def _key_payload(
        self,
        *,
        snapshot: ContinuousCartPoleSnapshot,
        candidate_actions: np.ndarray,
        horizon: int,
        gamma: float,
    ) -> bytes:
        rounded_snapshot = {
            "state": np.round(np.asarray(snapshot.state, dtype=np.float32), self.rounding_decimals).tolist(),
            "steps_beyond_done": -1 if snapshot.steps_beyond_done is None else int(snapshot.steps_beyond_done),
            "elapsed_steps": int(snapshot.elapsed_steps),
            "gravity": round(float(snapshot.gravity), self.rounding_decimals),
            "masscart": round(float(snapshot.masscart), self.rounding_decimals),
            "masspole": round(float(snapshot.masspole), self.rounding_decimals),
            "length": round(float(snapshot.length), self.rounding_decimals),
            "force_mag": round(float(snapshot.force_mag), self.rounding_decimals),
        }
        rounded_actions = np.round(
            np.asarray(candidate_actions, dtype=np.float32),
            self.rounding_decimals,
        ).tolist()
        payload = {
            "env_name": self.env_name,
            "snapshot": rounded_snapshot,
            "candidate_actions": rounded_actions,
            "horizon": int(horizon),
            "gamma": round(float(gamma), 6),
        }
        return repr(payload).encode("utf-8")

    def _key(
        self,
        *,
        snapshot: ContinuousCartPoleSnapshot,
        candidate_actions: np.ndarray,
        horizon: int,
        gamma: float,
    ) -> str:
        return sha1(
            self._key_payload(
                snapshot=snapshot,
                candidate_actions=candidate_actions,
                horizon=horizon,
                gamma=gamma,
            )
        ).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def lookup(
        self,
        *,
        snapshot: ContinuousCartPoleSnapshot,
        candidate_actions: np.ndarray,
        horizon: int,
        gamma: float,
    ) -> FanoutLabel | None:
        key = self._key(
            snapshot=snapshot,
            candidate_actions=candidate_actions,
            horizon=horizon,
            gamma=gamma,
        )
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        path = self._path_for_key(key)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as payload:
            label = FanoutLabel.from_payload({name: payload[name] for name in payload.files})
        self._memory[key] = label
        return label

    def store(
        self,
        *,
        snapshot: ContinuousCartPoleSnapshot,
        candidate_actions: np.ndarray,
        horizon: int,
        gamma: float,
        label: FanoutLabel,
    ) -> FanoutLabel:
        key = self._key(
            snapshot=snapshot,
            candidate_actions=candidate_actions,
            horizon=horizon,
            gamma=gamma,
        )
        self._memory[key] = label
        np.savez_compressed(self._path_for_key(key), **label.to_payload())
        return label

    def get_or_compute(
        self,
        *,
        env,
        adapter: SimulatorFanoutAdapter,
        candidate_actions: np.ndarray,
        horizon: int = 4,
        gamma: float = 0.99,
        snapshot: ContinuousCartPoleSnapshot | None = None,
    ) -> FanoutLabel:
        baseline = snapshot if snapshot is not None else adapter.snapshot(env)
        cached = self.lookup(
            snapshot=baseline,
            candidate_actions=candidate_actions,
            horizon=horizon,
            gamma=gamma,
        )
        if cached is not None:
            return cached
        result = adapter.evaluate_constant_action_candidates(
            env,
            candidate_actions=candidate_actions,
            horizon=horizon,
            gamma=gamma,
            baseline_snapshot=baseline,
        )
        return self.store(
            snapshot=baseline,
            candidate_actions=candidate_actions,
            horizon=horizon,
            gamma=gamma,
            label=FanoutLabel.from_result(candidate_actions=candidate_actions, result=result),
        )


class SimulatorFanoutAdapter:
    """Snapshot/restore helper for cheap simulator-side branching."""

    def __init__(self, env_name: str):
        self.env_name = str(env_name)
        if self.env_name != CONTINUOUS_CARTPOLE_NAME:
            raise ValueError(
                "SimulatorFanoutAdapter currently supports only Continuous CartPole"
            )

    def snapshot(self, env) -> ContinuousCartPoleSnapshot:
        """Capture the current physics and state without mutating the live env."""
        base_env = env.unwrapped
        state = np.asarray(base_env.state, dtype=np.float32).copy()
        return ContinuousCartPoleSnapshot(
            state=state,
            steps_beyond_done=base_env.steps_beyond_done,
            elapsed_steps=int(getattr(env, "_elapsed_steps", 0)),
            gravity=float(base_env.gravity),
            masscart=float(base_env.masscart),
            masspole=float(base_env.masspole),
            length=float(base_env.length),
            force_mag=float(base_env.force_mag),
        )

    def restore(self, env, snapshot: ContinuousCartPoleSnapshot) -> None:
        """Restore a previously captured live-environment snapshot."""
        base_env = env.unwrapped
        base_env.gravity = float(snapshot.gravity)
        base_env.masscart = float(snapshot.masscart)
        base_env.masspole = float(snapshot.masspole)
        base_env.length = float(snapshot.length)
        base_env.force_mag = float(snapshot.force_mag)
        base_env.total_mass = base_env.masspole + base_env.masscart
        base_env.polemass_length = base_env.masspole * base_env.length
        base_env.state = tuple(np.asarray(snapshot.state, dtype=np.float32).tolist())
        base_env.steps_beyond_done = snapshot.steps_beyond_done
        if hasattr(env, "_elapsed_steps"):
            env._elapsed_steps = int(snapshot.elapsed_steps)

    def evaluate_constant_action_candidates(
        self,
        env,
        *,
        candidate_actions: np.ndarray,
        horizon: int = 4,
        gamma: float = 0.99,
        baseline_snapshot: ContinuousCartPoleSnapshot | None = None,
    ) -> dict[str, np.ndarray | int]:
        """Roll out short constant-action branches from the current live snapshot."""
        baseline = baseline_snapshot if baseline_snapshot is not None else self.snapshot(env)
        candidate_actions = np.asarray(candidate_actions, dtype=np.float32)
        returns = np.zeros((candidate_actions.shape[0],), dtype=np.float32)
        risks = np.zeros((candidate_actions.shape[0],), dtype=np.float32)
        recoverability = np.zeros((candidate_actions.shape[0],), dtype=np.float32)

        for candidate_idx, action in enumerate(candidate_actions):
            self.restore(env, baseline)
            episode_return = 0.0
            final_state = np.asarray(baseline.state, dtype=np.float32)
            terminated_early = False
            for step_idx in range(max(1, int(horizon))):
                next_state, reward, terminated, truncated, _info = env.step(action)
                final_state = np.asarray(next_state, dtype=np.float32)
                episode_return += float(gamma**step_idx) * float(reward)
                if terminated or truncated:
                    terminated_early = True
                    break
            returns[candidate_idx] = float(episode_return)
            risks[candidate_idx] = 1.0 if terminated_early else 0.0
            recoverability[candidate_idx] = cartpole_recoverability_from_state(final_state)

        self.restore(env, baseline)
        scores = candidate_score(returns, risks, recoverability)
        best_idx = int(np.argmax(scores))
        return {
            "returns": returns.astype(np.float32),
            "risks": risks.astype(np.float32),
            "recoverability": recoverability.astype(np.float32),
            "scores": scores.astype(np.float32),
            "best_idx": best_idx,
        }


__all__ = [
    "ContinuousCartPoleSnapshot",
    "FanoutLabel",
    "PersistentFanoutLabelCache",
    "SimulatorFanoutAdapter",
    "candidate_score",
    "cartpole_recoverability_from_state",
]
