"""Shared imagination data contract across domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Target:
    """A compact desired latent outcome for generated practice."""

    target_id: str
    kind: str
    latent: tuple[float, ...] = ()
    utility: float = 0.0
    stability: float = 0.0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    """A generated counterfactual that may or may not be worth validation."""

    proposal_id: str
    domain: str
    context_id: str
    context_latent: tuple[float, ...]
    target: Target | None
    intervention: Any
    predicted_latent: tuple[float, ...] = ()
    predicted_utility: float = 0.0
    predicted_lift: float = 0.0
    uncertainty: float = 0.0
    support_confidence: float = 0.0
    reachability: float = 0.0
    consistency: float = 0.0
    graph_path: tuple[str, ...] = ()
    source_edges: tuple[str, ...] = ()
    trust_score: float = 0.0
    expected_solver_utility: float = 0.0
    horizon: int = 1
    generation_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationResult:
    """Truth signal from a real env, held-out check, parser, rule engine, or oracle."""

    proposal_id: str
    accepted: bool
    real_utility: float = 0.0
    real_lift: float = 0.0
    validation_cost: float = 0.0
    validator: str = "unknown"
    rejected_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImaginationSample:
    """A proposal paired with optional validation and derived training weight."""

    proposal: Proposal
    validation: ValidationResult | None = None
    accepted_weight: float = 0.0

    @property
    def validated(self) -> bool:
        return self.validation is not None

    @property
    def accepted(self) -> bool:
        return bool(self.validation and self.validation.accepted)


def as_latent(values: Any, *, limit: int = 32) -> tuple[float, ...]:
    """Convert a numeric vector-like value into a compact immutable latent tuple."""
    if values is None:
        return ()
    try:
        rows = list(values.reshape(-1))  # numpy arrays
    except AttributeError:
        try:
            rows = list(values)
        except TypeError:
            rows = [values]
    return tuple(float(value) for value in rows[: max(0, int(limit))])


__all__ = ["ImaginationSample", "Proposal", "Target", "ValidationResult", "as_latent"]
