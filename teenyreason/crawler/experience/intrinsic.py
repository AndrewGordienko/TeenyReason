"""Intrinsic practice drive for crawler-side world understanding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...cognition.imagination import Proposal, Target, TargetBank
from ...cognition.skills import SkillMemory
from ...cognition.worldmap import WorldMap
from .mindmap import CrawlerMindMap, ResidualCorrection, vector_distance


@dataclass(frozen=True)
class DriveWeights:
    """Soft weights for self-directed practice scoring."""

    learning_progress: float = 0.22
    controllability: float = 0.18
    recoverability: float = 0.15
    prediction_error_reducibility: float = 0.13
    novelty: float = 0.10
    expected_utility: float = 0.16
    composability: float = 0.10
    sample_cost: float = 0.10
    danger: float = 0.14
    residual_penalty: float = 0.20


@dataclass(frozen=True)
class PracticeSignal:
    """Factorized score explaining why the crawler should practice a target."""

    learning_progress: float = 0.0
    controllability: float = 0.0
    recoverability: float = 0.0
    prediction_error_reducibility: float = 0.0
    novelty: float = 0.0
    expected_utility: float = 0.0
    composability: float = 0.0
    sample_cost: float = 0.0
    danger: float = 0.0
    residual_penalty: float = 0.0


@dataclass(frozen=True)
class PracticeTarget:
    """A ranked thing the crawler can try to understand or control next."""

    target_id: str
    kind: str
    latent: tuple[float, ...]
    source: str
    score: float
    signal: PracticeSignal
    metadata: dict[str, Any]


class IntrinsicDrive:
    """Self-directed practice ranking over target, graph, skill, and residual memory."""

    def __init__(self, weights: DriveWeights | None = None):
        self.weights = weights or DriveWeights()
        self._ranked: list[PracticeTarget] = []
        self._selected: list[PracticeTarget] = []

    def refresh(
        self,
        targets: TargetBank,
        mindmap: CrawlerMindMap,
        skill_memory: SkillMemory,
        world_map: WorldMap | None,
    ) -> list[PracticeTarget]:
        rows: list[PracticeTarget] = []
        for target in targets.targets():
            rows.append(self._from_target(target, mindmap, skill_memory, world_map))
        if world_map is not None:
            for edge in world_map.ranked_edges(count=16):
                rows.append(self._from_world_edge(edge, mindmap, skill_memory))
        for node in mindmap.nodes(kind_prefix="residual")[:16]:
            rows.append(self._from_residual_node(node))
        rows.sort(key=lambda item: item.score, reverse=True)
        self._ranked = dedupe_targets(rows)
        return list(self._ranked)

    def top(self, *, count: int, kind: str | None = None) -> list[PracticeTarget]:
        rows = self._ranked if kind is None else [item for item in self._ranked if item.kind == kind]
        selected = rows[: max(0, int(count))]
        self._selected = selected
        return list(selected)

    def score_proposal(self, proposal: Proposal, correction: ResidualCorrection) -> float:
        utility = squash(float(proposal.expected_solver_utility) + float(proposal.predicted_lift))
        confidence = mean([proposal.support_confidence, proposal.reachability, proposal.consistency, proposal.trust_score])
        uncertainty = squash(float(proposal.uncertainty))
        cost = squash(float(proposal.generation_cost))
        residual = squash(float(correction.penalty))
        reducible = float(correction.support > 0.0) * (1.0 - residual)
        signal = PracticeSignal(
            learning_progress=max(0.0, float(correction.acceptance_rate)),
            controllability=confidence,
            recoverability=float(proposal.reachability),
            prediction_error_reducibility=reducible,
            novelty=max(0.0, 1.0 - confidence),
            expected_utility=utility,
            composability=float(proposal.metadata.get("composability", 0.0)),
            sample_cost=cost,
            danger=uncertainty,
            residual_penalty=residual,
        )
        return score_signal(signal, self.weights)

    def summary(self, *, prefix: str = "intrinsic_drive") -> dict[str, float]:
        ranked = list(self._ranked)
        selected = list(self._selected)
        return {
            f"{prefix}_target_count": float(len(ranked)),
            f"{prefix}_selected_count": float(len(selected)),
            f"{prefix}_score_max": max_or_zero([item.score for item in ranked]),
            f"{prefix}_score_mean": mean([item.score for item in ranked]),
            f"{prefix}_learning_progress_mean": mean([item.signal.learning_progress for item in ranked]),
            f"{prefix}_controllability_mean": mean([item.signal.controllability for item in ranked]),
            f"{prefix}_recoverability_mean": mean([item.signal.recoverability for item in ranked]),
            f"{prefix}_residual_penalty_mean": mean([item.signal.residual_penalty for item in ranked]),
        }

    def _from_target(
        self,
        target: Target,
        mindmap: CrawlerMindMap,
        skill_memory: SkillMemory,
        world_map: WorldMap | None,
    ) -> PracticeTarget:
        residual_penalty = nearest_residual_penalty(mindmap, target.latent)
        skill_support = nearby_skill_support(skill_memory, target.latent)
        graph_support = world_support(world_map, target)
        signal = PracticeSignal(
            learning_progress=squash(float(target.metadata.get("recent_lift", target.utility))),
            controllability=max(graph_support, float(target.metadata.get("controllability", 0.0))),
            recoverability=max(float(target.stability), skill_support),
            prediction_error_reducibility=max(0.0, 1.0 - residual_penalty),
            novelty=novelty_against_map(mindmap, target.latent),
            expected_utility=squash(float(target.utility)),
            composability=skill_support,
            sample_cost=target_sample_cost(target),
            danger=target_danger(target),
            residual_penalty=residual_penalty,
        )
        return PracticeTarget(
            target_id=f"target:{target.target_id}",
            kind=str(target.kind),
            latent=target.latent,
            source=str(target.source),
            score=score_signal(signal, self.weights),
            signal=signal,
            metadata=dict(target.metadata),
        )

    def _from_world_edge(self, edge, mindmap: CrawlerMindMap, skill_memory: SkillMemory) -> PracticeTarget:
        residual_penalty = 0.0
        if edge.relation == "predicts" and edge.target == "event:terminal":
            residual_penalty = 0.10
        utility = max(0.0, float(edge.utility))
        skill_support = nearby_skill_support(skill_memory, ())
        signal = PracticeSignal(
            learning_progress=squash(utility),
            controllability=float(edge.confidence),
            recoverability=1.0 if edge.relation in {"prevents", "causes"} else 0.35,
            prediction_error_reducibility=0.5,
            novelty=novelty_against_map(mindmap, (float(edge.effect_size), float(edge.utility))),
            expected_utility=squash(utility + abs(float(edge.effect_size))),
            composability=skill_support,
            sample_cost=squash(float(edge.sample_cost) / 100.0),
            danger=1.0 if edge.target == "event:terminal" and edge.relation == "predicts" else 0.0,
            residual_penalty=residual_penalty,
        )
        return PracticeTarget(
            target_id=f"world:{edge.edge_id}",
            kind=f"worldmap:{edge.relation}",
            latent=(float(edge.effect_size), float(edge.utility), float(edge.confidence)),
            source="worldmap",
            score=score_signal(signal, self.weights),
            signal=signal,
            metadata={"edge_id": edge.edge_id, "source": edge.source, "target": edge.target},
        )

    def _from_residual_node(self, node) -> PracticeTarget:
        error = abs(float(node.utility))
        signal = PracticeSignal(
            learning_progress=squash(error),
            controllability=0.25,
            recoverability=0.20,
            prediction_error_reducibility=0.80,
            novelty=0.25,
            expected_utility=0.10,
            composability=0.0,
            sample_cost=0.20,
            danger=0.25,
            residual_penalty=0.0,
        )
        return PracticeTarget(
            target_id=f"residual:{node.node_id}",
            kind="residual_repair",
            latent=node.vector,
            source="mindmap",
            score=score_signal(signal, self.weights),
            signal=signal,
            metadata=dict(node.metadata),
        )


def score_signal(signal: PracticeSignal, weights: DriveWeights) -> float:
    positive = (
        weights.learning_progress * signal.learning_progress
        + weights.controllability * signal.controllability
        + weights.recoverability * signal.recoverability
        + weights.prediction_error_reducibility * signal.prediction_error_reducibility
        + weights.novelty * signal.novelty
        + weights.expected_utility * signal.expected_utility
        + weights.composability * signal.composability
    )
    negative = (
        weights.sample_cost * signal.sample_cost
        + weights.danger * signal.danger
        + weights.residual_penalty * signal.residual_penalty
    )
    return float(positive - negative)


def nearest_residual_penalty(mindmap: CrawlerMindMap, latent: tuple[float, ...]) -> float:
    residuals = mindmap.nearest_residuals(latent, (), k=4)
    if not residuals:
        return 0.0
    errors = [
        (1.0 / (1.0 + float(distance))) * max(0.0, record.raw_error) / max(1.0, abs(record.predicted_lift))
        for distance, record in residuals
    ]
    return float(np.clip(mean(errors), 0.0, 1.0))


def nearby_skill_support(skill_memory: SkillMemory, latent: tuple[float, ...]) -> float:
    records = skill_memory.records()
    if not records:
        return 0.0
    if not latent:
        return mean([record.reliability for record in records])
    distances = [vector_distance(latent, tuple(float(x) for x in record.initiation_observation[: len(latent)])) for record in records]
    best = min(distances) if distances else 1.0
    return float(1.0 / (1.0 + best))


def world_support(world_map: WorldMap | None, target: Target) -> float:
    if world_map is None:
        return 0.0
    edge_id = target.metadata.get("edge_id")
    if edge_id is None:
        return 0.0
    edge = world_map.edge(str(edge_id))
    return 0.0 if edge is None else float(edge.confidence)


def novelty_against_map(mindmap: CrawlerMindMap, latent: tuple[float, ...]) -> float:
    if not latent:
        return 0.25
    distances = [vector_distance(latent, node.vector) for node in mindmap.nodes() if node.vector]
    if not distances:
        return 1.0
    return float(np.clip(min(distances), 0.0, 1.0))


def target_sample_cost(target: Target) -> float:
    cost = float(target.metadata.get("sample_cost", 1.0))
    return squash(cost)


def target_danger(target: Target) -> float:
    if "terminal" in str(target.kind).lower():
        return 1.0
    return float(np.clip(1.0 - float(target.stability), 0.0, 1.0))


def dedupe_targets(rows: list[PracticeTarget]) -> list[PracticeTarget]:
    out: list[PracticeTarget] = []
    seen: set[str] = set()
    for item in rows:
        key = f"{item.kind}:{tuple(round(float(x), 3) for x in item.latent[:8])}:{item.source}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def squash(value: float) -> float:
    value = float(value)
    return float(value / (1.0 + abs(value))) if value != 0.0 else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


def max_or_zero(values: list[float]) -> float:
    return float(np.max(np.asarray(values, dtype=np.float32))) if values else 0.0


__all__ = ["DriveWeights", "IntrinsicDrive", "PracticeSignal", "PracticeTarget", "score_signal"]
