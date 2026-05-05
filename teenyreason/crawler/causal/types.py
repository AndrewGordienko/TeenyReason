"""Generic causal-world contracts for the crawler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CausalWorldSpec:
    """Static contract for one crawlable world family."""

    domain: str
    modality: str
    hidden_target: str
    outcome_name: str


@dataclass(frozen=True)
class Intervention:
    """One action/query the crawler can use to learn the world."""

    name: str
    family: str
    cost: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservedOutcome:
    """Observed result of applying an intervention."""

    intervention: Intervention
    value: Any
    cost: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldBelief:
    """Crawler belief inferred from observed interventions."""

    label: str
    message: Any
    confidence: float
    uncertainty: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CounterfactualPrediction:
    """Predicted result for an intervention under a belief."""

    intervention: Intervention
    value: Any
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


class CausalWorldAdapter(Protocol):
    """Domain boundary for modality-independent crawler understanding."""

    spec: CausalWorldSpec

    def world_for_seed(self, seed: int) -> Any:
        ...

    def intervention_space(self, world: Any, *, seed: int) -> tuple[Intervention, ...]:
        ...

    def observe(self, world: Any, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        ...

    def infer_belief(
        self,
        world: Any,
        observations: tuple[ObservedOutcome, ...],
        *,
        seed: int,
    ) -> WorldBelief:
        ...

    def predict_outcome(
        self,
        world: Any,
        belief: WorldBelief,
        intervention: Intervention,
        *,
        seed: int,
    ) -> CounterfactualPrediction:
        ...

    def true_outcome(
        self,
        world: Any,
        intervention: Intervention,
        *,
        seed: int,
    ) -> ObservedOutcome:
        ...

    def score_prediction(
        self,
        prediction: CounterfactualPrediction,
        truth: ObservedOutcome,
    ) -> float:
        ...

    def world_label(self, world: Any) -> str:
        ...
