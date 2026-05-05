"""Data records for scenario-based memory and imagination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenarioTracelet:
    """One remembered transition from real or imagined experience."""

    observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    reward: float
    done: float
    return_to_go: float
    trajectory_id: int
    step: int
    source: str = "real"
    surprise: float = 0.0


@dataclass(frozen=True)
class ScenarioWindow:
    """A short contiguous remembered scenario."""

    tracelets: tuple[ScenarioTracelet, ...]
    familiarity: float
    terminal_distance: float
    mean_surprise: float

    @property
    def start_observation(self) -> np.ndarray:
        return self.tracelets[0].observation

    @property
    def actions(self) -> np.ndarray:
        return np.asarray([item.action for item in self.tracelets], dtype=np.float32)

    @property
    def observed_return(self) -> float:
        return float(sum(float(item.reward) for item in self.tracelets))

    @property
    def best_return_to_go(self) -> float:
        return float(max(float(item.return_to_go) for item in self.tracelets))


@dataclass(frozen=True)
class ScenarioWeights:
    """Continuous influence weights for an imagined or real scenario."""

    familiarity: float
    plausibility: float
    usefulness: float
    inverse_surprise: float

    @property
    def combined(self) -> float:
        return float(self.familiarity * self.plausibility * self.usefulness * self.inverse_surprise)


@dataclass(frozen=True)
class ScenarioVariant:
    """A local imagined variation of a remembered window."""

    window: ScenarioWindow
    actions: np.ndarray
    rows: tuple[dict[str, np.ndarray | float], ...]
    predicted_return: float
    predicted_value: float
    predicted_lift: float
    uncertainty: float
    done_risk: float
    weights: ScenarioWeights
    variant_kind: str


__all__ = ["ScenarioTracelet", "ScenarioVariant", "ScenarioWeights", "ScenarioWindow"]
