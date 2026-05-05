"""Mindmap records and small numeric helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MindNode:
    """A compact concept/factor/event/episode node."""

    node_id: str
    kind: str
    vector: tuple[float, ...] = ()
    label: str = ""
    support: float = 1.0
    confidence: float = 0.5
    utility: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MindEdge:
    """A typed relation between remembered nodes."""

    edge_id: str
    source: str
    relation: str
    target: str
    strength: float = 0.0
    confidence: float = 0.5
    utility: float = 0.0
    cost: float = 1.0
    evidence_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResidualCorrection:
    """Correction retrieved from similar imagined-vs-real mismatches."""

    penalty: float
    corrected_predicted_lift: float
    support: float
    acceptance_rate: float
    nearest_distance: float


@dataclass(frozen=True)
class ResidualRecord:
    """A remembered prediction error."""

    residual_id: str
    proposal_id: str
    domain: str
    context_vector: tuple[float, ...]
    intervention_vector: tuple[float, ...]
    predicted_lift: float
    corrected_predicted_lift: float
    real_lift: float
    accepted: bool
    validation_cost: float
    nearest_distance: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_error(self) -> float:
        return float(self.predicted_lift - self.real_lift)

    @property
    def corrected_error(self) -> float:
        return float(self.corrected_predicted_lift - self.real_lift)


def merge_node(left: MindNode, right: MindNode) -> MindNode:
    support = float(left.support) + float(right.support)
    return MindNode(
        node_id=left.node_id,
        kind=left.kind,
        vector=left.vector or right.vector,
        label=left.label or right.label,
        support=support,
        confidence=weighted(left.confidence, right.confidence, left.support, right.support),
        utility=weighted(left.utility, right.utility, left.support, right.support),
        metadata={**right.metadata, **left.metadata},
    )


def merge_edge(left: MindEdge, right: MindEdge) -> MindEdge:
    left_count = max(1, int(left.evidence_count))
    right_count = max(1, int(right.evidence_count))
    return MindEdge(
        edge_id=left.edge_id,
        source=left.source,
        relation=left.relation,
        target=left.target,
        strength=weighted(left.strength, right.strength, left_count, right_count),
        confidence=weighted(left.confidence, right.confidence, left_count, right_count),
        utility=weighted(left.utility, right.utility, left_count, right_count),
        cost=weighted(left.cost, right.cost, left_count, right_count),
        evidence_count=left_count + right_count,
        metadata={**right.metadata, **left.metadata},
    )


def context_node_id(domain: str, vector: tuple[float, ...]) -> str:
    return f"context:{domain}:{stable_vector_key(vector)}"


def action_node_id(domain: str, intervention: Any) -> str:
    return f"intervention:{domain}:{stable_vector_key(intervention_latent(intervention))}"


def skill_node_id(skill: Any) -> str:
    return f"skill:{int(getattr(skill, 'skill_id', 0))}"


def intervention_latent(values: Any, *, limit: int = 48) -> tuple[float, ...]:
    return vector_tuple(values, limit=limit)


def vector_tuple(values: Any, *, limit: int = 64) -> tuple[float, ...]:
    if values is None:
        return ()
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    return tuple(float(item) for item in array[: max(0, int(limit))])


def stable_vector_key(vector: tuple[float, ...], *, limit: int = 8) -> str:
    if not vector:
        return "empty"
    rounded = [f"{float(item):.2f}" for item in vector[: max(1, int(limit))]]
    return "_".join(rounded)


def residual_distance(
    context_a: tuple[float, ...],
    action_a: tuple[float, ...],
    context_b: tuple[float, ...],
    action_b: tuple[float, ...],
) -> float:
    return 0.70 * vector_distance(context_a, context_b) + 0.30 * vector_distance(action_a, action_b)


def vector_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 1.0
    count = min(len(left), len(right))
    if count <= 0:
        return 1.0
    a = np.asarray(left[:count], dtype=np.float32)
    b = np.asarray(right[:count], dtype=np.float32)
    return float(np.mean(np.square(a - b)) / (1.0 + float(np.mean(np.square(a)))))


def weighted(left: float, right: float, left_weight: float, right_weight: float) -> float:
    total = max(1e-6, float(left_weight) + float(right_weight))
    return float((float(left) * float(left_weight) + float(right) * float(right_weight)) / total)


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


__all__ = [
    "MindEdge",
    "MindNode",
    "ResidualCorrection",
    "ResidualRecord",
    "action_node_id",
    "context_node_id",
    "intervention_latent",
    "mean",
    "merge_edge",
    "merge_node",
    "residual_distance",
    "skill_node_id",
    "vector_distance",
    "vector_tuple",
]
