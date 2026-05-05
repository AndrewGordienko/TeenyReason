"""Data types for explicit world-map memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceRef:
    """Pointer to evidence supporting or rejecting a relation."""

    source: str
    ref_id: str
    accepted: bool = True
    utility: float = 0.0
    cost: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldNode:
    """A reusable factor, event, skill, goal, or failure concept."""

    node_id: str
    kind: str
    label: str = ""
    vector: tuple[float, ...] = ()
    support: float = 0.0
    confidence: float = 0.0
    utility: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorldEdge:
    """A typed relation between two world-map nodes."""

    edge_id: str
    source: str
    relation: str
    target: str
    effect_size: float = 0.0
    confidence: float = 0.0
    utility: float = 0.0
    sample_cost: float = 1.0
    evidence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def edge_id(source: str, relation: str, target: str) -> str:
    return f"{source}|{relation}|{target}"


__all__ = ["EvidenceRef", "WorldEdge", "WorldNode", "edge_id"]
