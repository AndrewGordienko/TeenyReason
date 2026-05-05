"""Generic action-conditioned world-model planning contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class ActionConditionedWorldModel(Protocol):
    """Predict future latent/state consequences under a belief."""

    def rollout(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        *,
        belief: object,
    ) -> np.ndarray:
        """Return the predicted trajectory for one candidate action sequence."""

    def score_rollout(self, states: np.ndarray, *, belief: object) -> float:
        """Return a higher-is-better score for a predicted trajectory."""


@dataclass(frozen=True)
class PlanningResult:
    """One MPC action-selection result."""

    action: float
    score: float
    actions: np.ndarray
    predicted_states: np.ndarray
    candidate_count: int


@dataclass(frozen=True)
class RandomShootingPlanner:
    """Small deterministic MPC planner over a discrete action grid."""

    horizon: int = 8
    candidate_count: int = 48
    action_grid: tuple[float, ...] = (-1.0, 0.0, 1.0)
    seed_offset: int = 17_000

    def choose_action(
        self,
        model: ActionConditionedWorldModel,
        state: np.ndarray,
        *,
        belief: object,
        seed: int,
        step: int = 0,
    ) -> PlanningResult:
        """Choose the first action from the best predicted action sequence."""
        candidates = self._candidate_sequences(seed=seed, step=step)
        best_score = -float("inf")
        best_actions = candidates[0]
        best_states = model.rollout(state, best_actions, belief=belief)
        for actions in candidates:
            states = model.rollout(state, actions, belief=belief)
            score = float(model.score_rollout(states, belief=belief))
            if score > best_score:
                best_score = score
                best_actions = actions
                best_states = states
        return PlanningResult(
            action=float(best_actions[0]),
            score=float(best_score),
            actions=best_actions.astype(np.float32),
            predicted_states=best_states.astype(np.float32),
            candidate_count=int(candidates.shape[0]),
        )

    def _candidate_sequences(self, *, seed: int, step: int) -> np.ndarray:
        rng = np.random.default_rng(int(self.seed_offset + seed * 997 + step * 37))
        action_values = np.asarray(self.action_grid, dtype=np.float32)
        random_count = max(0, int(self.candidate_count) - len(action_values) - 2)
        random_actions = rng.choice(
            action_values,
            size=(random_count, int(self.horizon)),
            replace=True,
        ).astype(np.float32)
        scripted = [np.full((int(self.horizon),), value, dtype=np.float32) for value in action_values]
        scripted.append(np.zeros((int(self.horizon),), dtype=np.float32))
        scripted.append(
            np.asarray(
                [action_values[idx % len(action_values)] for idx in range(int(self.horizon))],
                dtype=np.float32,
            )
        )
        return np.concatenate([np.stack(scripted, axis=0), random_actions], axis=0)


def planner_action_match_fraction(
    reference_actions: list[float],
    candidate_actions: list[float],
) -> float:
    """Return how often two planners chose the same first action."""
    count = min(len(reference_actions), len(candidate_actions))
    if count <= 0:
        return 0.0
    matches = [
        float(reference_actions[idx] == candidate_actions[idx])
        for idx in range(count)
    ]
    return float(np.mean(matches))


def dict_mean(rows: list[dict[str, Any]], key: str) -> float:
    """Mean helper for small benchmark rows."""
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))
