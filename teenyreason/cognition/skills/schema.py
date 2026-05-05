"""Records for generic intrinsic skills and stable state memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StableIsland:
    """A state region that was empirically survivable and easy to control."""

    island_id: int
    center: np.ndarray
    factor_center: np.ndarray
    score: float
    survival: float
    smoothness: float
    terminal_risk: float
    surprise: float
    support: int


@dataclass(frozen=True)
class IntrinsicGoal:
    """A generic practice target expressed as a latent state-delta request."""

    goal_id: int
    goal_kind: str
    target_delta: np.ndarray
    anchor_observation: np.ndarray
    priority: float
    source: str


@dataclass(frozen=True)
class SkillRecord:
    """A real-validated option-like action chunk."""

    skill_id: int
    goal: IntrinsicGoal
    initiation_observation: np.ndarray
    termination_observation: np.ndarray
    actions: np.ndarray
    outcome_delta: np.ndarray
    real_return_lift: float
    survival_lift: float
    terminal_avoid: float
    reliability: float
    source: str = "real_validated"

    @property
    def duration(self) -> int:
        return int(self.actions.shape[0])


__all__ = ["IntrinsicGoal", "SkillRecord", "StableIsland"]
