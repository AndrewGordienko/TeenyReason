"""Public crawler-library data contracts.

The canonical crawler API is intentionally world-agnostic:

- `EvidenceSlice`
- `BeliefState`
- `CrawlerMessage`
- `CrawlerStep`
- `CrawlerRunResult`

The older RL-facing objects are kept below as compatibility adapters so the
current benchmark, dashboard, and PPO stack can keep working during the
library refactor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvidenceSlice:
    """One generic evidence slice gathered by the crawler.

    The payload is intentionally opaque to the crawler core. Adapters and
    backends decide how to interpret it.
    """

    query_name: str
    source_id: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeliefState:
    """Generic belief emitted by a crawler belief backend."""

    latent: np.ndarray
    uncertainty: float
    support_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrawlerMessage:
    """Generic downstream-facing message emitted by the crawler."""

    vector: np.ndarray
    confidence: float
    ready: bool
    uncertainty: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrawlerStep:
    """One generic crawler update."""

    query_name: str | None
    evidence: EvidenceSlice | None
    belief_state: BeliefState
    message: CrawlerMessage
    stop_reason: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrawlerRunResult:
    """Final generic crawler output before a downstream consumer takes over."""

    steps: tuple[CrawlerStep, ...]
    final_belief_state: BeliefState
    final_message: CrawlerMessage
    stop_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceWindow:
    """Compatibility window object for the RL research stack."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool
    truncated: bool
    probe_family: str
    env_instance_id: int


@dataclass(frozen=True)
class EvidenceBatch:
    """Compatibility batch object for window-based RL evidence collection."""

    windows: tuple[EvidenceWindow, ...]
    env_name: str
    window_size: int
    action_vocab_size: int


@dataclass(frozen=True)
class PredictiveBelief:
    """Compatibility predictive belief used by the current RL stack."""

    mean_raw: np.ndarray
    mean_unit: np.ndarray
    logvar: np.ndarray
    view_spread: np.ndarray
    env_param_mean: np.ndarray
    env_param_std: np.ndarray
    future_probe_error: float
    support_count: int
    support_diversity_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricBelief:
    """Compatibility metric belief used by the current RL stack."""

    mean_raw: np.ndarray
    mean_unit: np.ndarray
    split_mean_a: np.ndarray
    split_mean_b: np.ndarray
    nearest_between_distance: float
    gap_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Compatibility uncertainty object used by the current RL stack."""

    vector: np.ndarray
    scalar: float
    feature_names: tuple[str, ...]
    feature_weights: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvExpression:
    """Compatibility controller-facing world expression plus trust metadata."""

    vector: np.ndarray
    confidence: float
    ready: bool
    uncertainty_scalar: float
    compressed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bits_per_dim(self) -> int:
        """Compatibility shim for the old belief-message contract."""
        return int(self.metadata.get("bits_per_dim", 0))

    @property
    def residual_vector(self) -> np.ndarray | None:
        """Compatibility shim for legacy dashboard/artifact consumers."""
        residual = self.metadata.get("residual_vector")
        if residual is None:
            return None
        return np.asarray(residual, dtype=np.float32)


BeliefMessage = EnvExpression


@dataclass(frozen=True)
class ControllerBeliefContext:
    """Structured controller-facing belief for the full-system control path."""

    mechanics_code: np.ndarray
    affordance_code: np.ndarray
    confidence: float
    uncertainty_scalar: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def vector(self) -> np.ndarray:
        """Return the canonical flat controller input view of the context."""
        return np.concatenate(
            [
                np.asarray(self.mechanics_code, dtype=np.float32).reshape(-1),
                np.asarray(self.affordance_code, dtype=np.float32).reshape(-1),
                np.asarray(
                    [float(self.confidence), float(self.uncertainty_scalar)],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ).astype(np.float32)


@dataclass(frozen=True)
class LegacyCrawlerStepResult:
    """Compatibility crawler update used by the current RL benchmark stack."""

    predictive_belief: PredictiveBelief
    metric_belief: MetricBelief
    uncertainty: UncertaintyEstimate
    env_expression: EnvExpression
    controller_context: ControllerBeliefContext | None
    expected_family_gain: dict[str, dict[str, float]]
    realized_family_gain: dict[str, float]
    stop_reason: str | None

    @property
    def belief_message(self) -> EnvExpression:
        """Compatibility alias for one transition cycle."""
        return self.env_expression


@dataclass(frozen=True)
class LegacyCrawlerRunResult:
    """Compatibility crawler run object used by the current RL benchmark stack."""

    step_results: tuple[LegacyCrawlerStepResult, ...]
    final_step: LegacyCrawlerStepResult
    total_probe_windows: int
    total_probe_steps: int
    stop_reason: str


# Compatibility aliases kept during the migration to the generic crawler API.
CrawlerStepResult = LegacyCrawlerStepResult
