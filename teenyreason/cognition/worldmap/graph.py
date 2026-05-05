"""Small explicit graph memory for crawler world knowledge."""

from __future__ import annotations

from .schema import EvidenceRef, WorldEdge, WorldNode, edge_id


class WorldMap:
    """Mutable graph of reusable factors and relations."""

    def __init__(self):
        self._nodes: dict[str, WorldNode] = {}
        self._edges: dict[str, WorldEdge] = {}

    def add_node(self, node: WorldNode) -> WorldNode:
        current = self._nodes.get(node.node_id)
        if current is None:
            self._nodes[node.node_id] = node
            return node
        merged = merge_node(current, node)
        self._nodes[node.node_id] = merged
        return merged

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        effect_size: float,
        confidence: float,
        utility: float = 0.0,
        sample_cost: float = 1.0,
        evidence: EvidenceRef | None = None,
        metadata: dict[str, object] | None = None,
    ) -> WorldEdge:
        key = edge_id(source, relation, target)
        incoming = WorldEdge(
            edge_id=key,
            source=source,
            relation=relation,
            target=target,
            effect_size=float(effect_size),
            confidence=float(confidence),
            utility=float(utility),
            sample_cost=max(1e-6, float(sample_cost)),
            evidence_count=1 if evidence is not None else 0,
            success_count=1 if evidence is not None and evidence.accepted else 0,
            failure_count=1 if evidence is not None and not evidence.accepted else 0,
            metadata=dict(metadata or {}),
        )
        current = self._edges.get(key)
        edge = incoming if current is None else merge_edge(current, incoming)
        self._edges[key] = edge
        return edge

    def validate_edge(self, edge_key: str, evidence: EvidenceRef) -> WorldEdge | None:
        edge = self._edges.get(edge_key)
        if edge is None:
            return None
        count = max(1, int(edge.evidence_count) + 1)
        accepted = 1 if bool(evidence.accepted) else 0
        rejected = 1 - accepted
        real_utility = float(evidence.utility)
        utility = weighted(edge.utility, real_utility, edge.evidence_count, 1)
        confidence_delta = 0.08 if evidence.accepted else -0.12
        confidence = clamp01(edge.confidence + confidence_delta)
        updated = WorldEdge(
            edge_id=edge.edge_id,
            source=edge.source,
            relation=edge.relation,
            target=edge.target,
            effect_size=edge.effect_size,
            confidence=confidence,
            utility=utility,
            sample_cost=weighted(edge.sample_cost, evidence.cost, edge.evidence_count, 1),
            evidence_count=count,
            success_count=int(edge.success_count) + accepted,
            failure_count=int(edge.failure_count) + rejected,
            metadata=dict(edge.metadata),
        )
        self._edges[edge_key] = updated
        return updated

    def node(self, node_id: str) -> WorldNode | None:
        return self._nodes.get(node_id)

    def edge(self, edge_key: str) -> WorldEdge | None:
        return self._edges.get(edge_key)

    def nodes(self) -> list[WorldNode]:
        return list(self._nodes.values())

    def edges(self) -> list[WorldEdge]:
        return list(self._edges.values())

    def outgoing(self, source: str, *, relation: str | None = None) -> list[WorldEdge]:
        edges = [edge for edge in self._edges.values() if edge.source == source]
        if relation is not None:
            edges = [edge for edge in edges if edge.relation == relation]
        return sorted(edges, key=edge_score, reverse=True)

    def ranked_edges(self, *, relation: str | None = None, count: int = 16) -> list[WorldEdge]:
        rows = self.edges() if relation is None else [edge for edge in self.edges() if edge.relation == relation]
        return sorted(rows, key=edge_score, reverse=True)[: max(0, int(count))]

    def summary(self, *, prefix: str = "worldmap") -> dict[str, float]:
        edges = self.edges()
        validated = [edge for edge in edges if edge.evidence_count > 0]
        return {
            f"{prefix}_node_count": float(len(self._nodes)),
            f"{prefix}_edge_count": float(len(edges)),
            f"{prefix}_validated_edge_count": float(len(validated)),
            f"{prefix}_edge_confidence_mean": mean([edge.confidence for edge in edges]),
            f"{prefix}_causal_lift_mean": mean([abs(edge.effect_size) for edge in edges]),
            f"{prefix}_sample_utility": sample_utility(edges),
        }


def merge_node(left: WorldNode, right: WorldNode) -> WorldNode:
    support = float(left.support) + float(right.support)
    return WorldNode(
        node_id=left.node_id,
        kind=left.kind,
        label=left.label or right.label,
        vector=left.vector or right.vector,
        support=support,
        confidence=weighted(left.confidence, right.confidence, left.support, right.support),
        utility=weighted(left.utility, right.utility, left.support, right.support),
        metadata={**right.metadata, **left.metadata},
    )


def merge_edge(left: WorldEdge, right: WorldEdge) -> WorldEdge:
    left_count = max(1, int(left.evidence_count))
    right_count = max(1, int(right.evidence_count))
    return WorldEdge(
        edge_id=left.edge_id,
        source=left.source,
        relation=left.relation,
        target=left.target,
        effect_size=weighted(left.effect_size, right.effect_size, left_count, right_count),
        confidence=clamp01(weighted(left.confidence, right.confidence, left_count, right_count)),
        utility=weighted(left.utility, right.utility, left_count, right_count),
        sample_cost=max(1e-6, weighted(left.sample_cost, right.sample_cost, left_count, right_count)),
        evidence_count=int(left.evidence_count) + int(right.evidence_count),
        success_count=int(left.success_count) + int(right.success_count),
        failure_count=int(left.failure_count) + int(right.failure_count),
        metadata={**right.metadata, **left.metadata},
    )


def edge_score(edge: WorldEdge) -> float:
    return float(edge.confidence * (abs(edge.effect_size) + max(0.0, edge.utility)) / max(edge.sample_cost, 1e-6))


def sample_utility(edges: list[WorldEdge]) -> float:
    value = sum(edge.confidence * max(0.0, edge.utility) for edge in edges)
    cost = sum(max(1e-6, edge.sample_cost) for edge in edges)
    return float(value / max(1.0, cost))


def weighted(left: float, right: float, left_weight: float, right_weight: float) -> float:
    total = max(1e-6, float(left_weight) + float(right_weight))
    return float((float(left) * float(left_weight) + float(right) * float(right_weight)) / total)


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


__all__ = ["WorldMap", "edge_score"]
