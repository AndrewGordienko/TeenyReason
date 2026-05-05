"""Mindmap-style memory for crawler predictions, corrections, and evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...cognition.imagination import Proposal
from .mindmap_schema import (
    MindEdge,
    MindNode,
    ResidualCorrection,
    ResidualRecord,
    action_node_id,
    context_node_id,
    intervention_latent,
    mean,
    merge_edge,
    merge_node,
    residual_distance,
    skill_node_id,
    vector_distance,
    vector_tuple,
)


class CrawlerMindMap:
    """Structured memory: concepts, relations, episodes, and residuals."""

    def __init__(self):
        self._nodes: dict[str, MindNode] = {}
        self._edges: dict[str, MindEdge] = {}
        self._residuals: list[ResidualRecord] = []
        self._last_corrections: list[ResidualCorrection] = []

    def nodes(self, *, kind_prefix: str | None = None) -> list[MindNode]:
        rows = list(self._nodes.values())
        if kind_prefix is None:
            return rows
        return [node for node in rows if node.kind.startswith(str(kind_prefix))]

    def edges(self, *, relation: str | None = None) -> list[MindEdge]:
        rows = list(self._edges.values())
        if relation is None:
            return rows
        return [edge for edge in rows if edge.relation == str(relation)]

    def neighbors(self, node_id: str, *, relation: str | None = None) -> list[MindEdge]:
        rows = [edge for edge in self._edges.values() if edge.source == str(node_id)]
        if relation is not None:
            rows = [edge for edge in rows if edge.relation == str(relation)]
        return sorted(rows, key=lambda edge: edge.confidence * (abs(edge.strength) + max(0.0, edge.utility)), reverse=True)

    def add_node(
        self,
        node_id: str,
        kind: str,
        *,
        vector: tuple[float, ...] = (),
        label: str = "",
        support: float = 1.0,
        confidence: float = 0.5,
        utility: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> MindNode:
        incoming = MindNode(
            node_id=str(node_id),
            kind=str(kind),
            vector=tuple(float(item) for item in vector),
            label=str(label),
            support=float(support),
            confidence=float(confidence),
            utility=float(utility),
            metadata=dict(metadata or {}),
        )
        current = self._nodes.get(incoming.node_id)
        if current is None:
            self._nodes[incoming.node_id] = incoming
            return incoming
        merged = merge_node(current, incoming)
        self._nodes[merged.node_id] = merged
        return merged

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        *,
        strength: float,
        confidence: float,
        utility: float = 0.0,
        cost: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MindEdge:
        key = f"{source}->{relation}->{target}"
        incoming = MindEdge(
            edge_id=key,
            source=str(source),
            relation=str(relation),
            target=str(target),
            strength=float(strength),
            confidence=float(confidence),
            utility=float(utility),
            cost=max(1e-6, float(cost)),
            evidence_count=1,
            metadata=dict(metadata or {}),
        )
        current = self._edges.get(key)
        edge = incoming if current is None else merge_edge(current, incoming)
        self._edges[key] = edge
        return edge

    def add_target(
        self,
        target_id: str,
        kind: str,
        *,
        latent: tuple[float, ...],
        utility: float,
        stability: float,
        source: str,
    ) -> None:
        node_id = f"target:{target_id}"
        self.add_node(
            node_id,
            f"target:{kind}",
            vector=latent,
            support=1.0,
            confidence=float(stability),
            utility=float(utility),
            metadata={"source": str(source)},
        )

    def add_factor_node(
        self,
        factor_id: str,
        factor_kind: str,
        *,
        vector: tuple[float, ...] = (),
        confidence: float,
        utility: float = 0.0,
        support: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.add_node(
            f"factor:{factor_id}",
            f"factor:{factor_kind}",
            vector=vector,
            support=support,
            confidence=confidence,
            utility=utility,
            metadata=metadata,
        )

    def add_episode_node(
        self,
        episode_id: str,
        episode_kind: str,
        *,
        vector: tuple[float, ...] = (),
        utility: float = 0.0,
        support: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.add_node(
            f"episode:{episode_id}",
            f"episode:{episode_kind}",
            vector=vector,
            support=support,
            confidence=1.0,
            utility=utility,
            metadata=metadata,
        )

    def add_skill_node(self, skill: Any) -> None:
        skill_id = skill_node_id(skill)
        goal = getattr(skill, "goal", None)
        initiation = vector_tuple(getattr(skill, "initiation_observation", ()))
        outcome = vector_tuple(getattr(skill, "outcome_delta", ()))
        utility = float(getattr(skill, "real_return_lift", 0.0)) + 0.1 * float(getattr(skill, "survival_lift", 0.0))
        reliability = float(getattr(skill, "reliability", 0.0))
        self.add_node(
            skill_id,
            "skill",
            vector=initiation + outcome[:16],
            support=max(1.0, float(getattr(skill, "duration", 1))),
            confidence=reliability,
            utility=utility,
            metadata={
                "goal_kind": str(getattr(goal, "goal_kind", "")),
                "source": str(getattr(skill, "source", "real_validated")),
            },
        )
        if goal is not None:
            goal_id = f"goal:{getattr(goal, 'goal_id', 'unknown')}:{getattr(goal, 'goal_kind', 'unknown')}"
            self.add_node(goal_id, "target:skill_goal", vector=vector_tuple(getattr(goal, "target_delta", ())), confidence=1.0, utility=utility)
            self.add_edge(skill_id, "leads_to", goal_id, strength=utility, confidence=reliability, utility=utility, cost=max(1.0, float(getattr(skill, "duration", 1))))
        if float(getattr(skill, "survival_lift", 0.0)) > 0.0:
            self.add_factor_node("survival", "stability", confidence=reliability, utility=float(getattr(skill, "survival_lift", 0.0)))
            self.add_edge(skill_id, "stabilizes", "factor:survival", strength=float(getattr(skill, "survival_lift", 0.0)), confidence=reliability, utility=utility)
        if float(getattr(skill, "terminal_avoid", 0.0)) > 0.0:
            self.add_factor_node("terminal", "danger", confidence=1.0, utility=-1.0)
            self.add_edge(skill_id, "prevents", "factor:terminal", strength=float(getattr(skill, "terminal_avoid", 0.0)), confidence=reliability, utility=utility)

    def add_skill_compositions(self, skills: list[Any], *, max_distance: float = 1.25) -> None:
        for left in skills:
            left_end = vector_tuple(getattr(left, "termination_observation", ()))
            if not left_end:
                continue
            for right in skills:
                if getattr(left, "skill_id", None) == getattr(right, "skill_id", None):
                    continue
                right_start = vector_tuple(getattr(right, "initiation_observation", ()))
                distance = vector_distance(left_end, right_start)
                if distance > float(max_distance):
                    continue
                reliability = mean([float(getattr(left, "reliability", 0.0)), float(getattr(right, "reliability", 0.0))])
                utility = mean([float(getattr(left, "real_return_lift", 0.0)), float(getattr(right, "real_return_lift", 0.0))])
                self.add_edge(
                    skill_node_id(left),
                    "composes_with",
                    skill_node_id(right),
                    strength=1.0 / (1.0 + distance),
                    confidence=reliability,
                    utility=utility,
                    cost=max(1.0, float(getattr(left, "duration", 1)) + float(getattr(right, "duration", 1))),
                    metadata={"distance": distance},
                )

    def add_value_evidence(
        self,
        source_id: str,
        metric: str,
        *,
        value: float,
        cost: float,
        accepted: bool,
    ) -> None:
        metric_id = f"value:{metric}"
        self.add_node(metric_id, "value_metric", confidence=1.0, utility=value)
        self.add_edge(
            str(source_id),
            "improved" if bool(accepted) else "failed_to_improve",
            metric_id,
            strength=float(value),
            confidence=1.0,
            utility=float(value),
            cost=max(1.0, float(cost)),
        )

    def add_proposal(self, proposal: Proposal, correction: ResidualCorrection | None = None) -> None:
        context_id = context_node_id(proposal.domain, proposal.context_latent)
        action_id = action_node_id(proposal.domain, proposal.intervention)
        proposal_id = f"proposal:{proposal.proposal_id}"
        self.add_node(context_id, "context", vector=proposal.context_latent, support=proposal.support_confidence)
        self.add_node(action_id, "intervention", vector=intervention_latent(proposal.intervention))
        self.add_node(
            proposal_id,
            "prediction",
            vector=proposal.predicted_latent,
            confidence=proposal.trust_score,
            utility=proposal.predicted_lift,
            metadata={"domain": proposal.domain},
        )
        lift = proposal.predicted_lift if correction is None else correction.corrected_predicted_lift
        penalty = 0.0 if correction is None else correction.penalty
        self.add_edge(context_id, "uses", action_id, strength=1.0, confidence=proposal.support_confidence)
        self.add_edge(
            action_id,
            "predicts",
            proposal_id,
            strength=lift,
            confidence=max(0.0, proposal.trust_score - penalty),
            utility=lift,
            cost=max(1e-6, proposal.generation_cost),
        )

    def correction_for_proposal(self, proposal: Proposal, *, k: int = 8) -> ResidualCorrection:
        context = vector_tuple(proposal.context_latent)
        action = intervention_latent(proposal.intervention)
        records = self.nearest_residuals(context, action, k=k)
        if not records:
            correction = ResidualCorrection(
                penalty=0.0,
                corrected_predicted_lift=float(proposal.predicted_lift),
                support=0.0,
                acceptance_rate=0.0,
                nearest_distance=float("inf"),
            )
            self._last_corrections.append(correction)
            return correction
        distances = np.asarray([distance for distance, _record in records], dtype=np.float32)
        weights = np.exp(-distances / max(1e-6, float(np.median(distances) + 1e-4))).astype(np.float32)
        weights = weights / max(float(np.sum(weights)), 1e-6)
        residuals = [record for _distance, record in records]
        over_errors = np.asarray([max(0.0, record.raw_error) for record in residuals], dtype=np.float32)
        accepted = np.asarray([1.0 if record.accepted else 0.0 for record in residuals], dtype=np.float32)
        penalty = float(np.sum(weights * over_errors))
        correction = ResidualCorrection(
            penalty=penalty,
            corrected_predicted_lift=float(proposal.predicted_lift - penalty),
            support=float(len(residuals)),
            acceptance_rate=float(np.sum(weights * accepted)),
            nearest_distance=float(np.min(distances)),
        )
        self._last_corrections.append(correction)
        return correction

    def add_residual(
        self,
        proposal: Proposal,
        *,
        real_lift: float,
        accepted: bool,
        validation_cost: float,
        correction: ResidualCorrection,
        metadata: dict[str, Any] | None = None,
    ) -> ResidualRecord:
        record = ResidualRecord(
            residual_id=f"residual:{len(self._residuals)}",
            proposal_id=str(proposal.proposal_id),
            domain=str(proposal.domain),
            context_vector=vector_tuple(proposal.context_latent),
            intervention_vector=intervention_latent(proposal.intervention),
            predicted_lift=float(proposal.predicted_lift),
            corrected_predicted_lift=float(correction.corrected_predicted_lift),
            real_lift=float(real_lift),
            accepted=bool(accepted),
            validation_cost=float(validation_cost),
            nearest_distance=float(correction.nearest_distance),
            metadata=dict(metadata or {}),
        )
        self._residuals.append(record)
        residual_node = record.residual_id
        self.add_node(
            residual_node,
            "residual",
            vector=record.context_vector,
            confidence=1.0,
            utility=-abs(record.corrected_error),
            metadata={"proposal_id": record.proposal_id, "accepted": record.accepted},
        )
        self.add_edge(
            context_node_id(record.domain, record.context_vector),
            "has_prediction_error",
            residual_node,
            strength=abs(record.raw_error),
            confidence=1.0,
            utility=-abs(record.raw_error),
            cost=max(1.0, record.validation_cost),
        )
        if record.raw_error > 0.0:
            self.add_edge(
                residual_node,
                "corrects_overestimate",
                action_node_id(record.domain, record.intervention_vector),
                strength=record.raw_error,
                confidence=1.0,
                utility=record.corrected_predicted_lift,
                cost=max(1.0, record.validation_cost),
            )
        return record

    def nearest_residuals(
        self,
        context_vector: tuple[float, ...],
        intervention_vector: tuple[float, ...],
        *,
        k: int,
    ) -> list[tuple[float, ResidualRecord]]:
        scored = [
            (
                residual_distance(
                    context_vector,
                    intervention_vector,
                    record.context_vector,
                    record.intervention_vector,
                ),
                record,
            )
            for record in self._residuals
        ]
        scored.sort(key=lambda item: item[0])
        return scored[: max(0, int(k))]

    def summary(self, *, prefix: str = "crawler_mindmap") -> dict[str, float]:
        residuals = list(self._residuals)
        corrections = list(self._last_corrections[-128:])
        corrected_positive = [record for record in residuals if record.corrected_predicted_lift > 0.0]
        accepted_after = [1.0 if record.accepted else 0.0 for record in corrected_positive]
        nodes = list(self._nodes.values())
        edges = list(self._edges.values())
        return {
            f"{prefix}_node_count": float(len(nodes)),
            f"{prefix}_edge_count": float(len(edges)),
            f"{prefix}_factor_count": float(sum(1 for node in nodes if node.kind.startswith("factor"))),
            f"{prefix}_skill_count": float(sum(1 for node in nodes if node.kind == "skill")),
            f"{prefix}_episode_count": float(sum(1 for node in nodes if node.kind.startswith("episode"))),
            f"{prefix}_value_edge_count": float(sum(1 for edge in edges if edge.relation in {"improved", "failed_to_improve"})),
            f"{prefix}_skill_composition_count": float(sum(1 for edge in edges if edge.relation == "composes_with")),
            f"{prefix}_residual_memory_count": float(len(residuals)),
            f"{prefix}_nearest_residual_penalty_mean": mean([item.penalty for item in corrections]),
            f"{prefix}_correction_support_mean": mean([item.support for item in corrections]),
            f"{prefix}_corrected_predicted_lift_mean": mean(
                [item.corrected_predicted_lift for item in corrections]
            ),
            f"{prefix}_raw_vs_real_error_mean": mean([abs(item.raw_error) for item in residuals]),
            f"{prefix}_corrected_vs_real_error_mean": mean(
                [abs(item.corrected_error) for item in residuals]
            ),
            f"{prefix}_acceptance_rate_after_correction": mean(accepted_after),
        }

__all__ = [
    "CrawlerMindMap",
    "MindEdge",
    "MindNode",
    "ResidualCorrection",
    "ResidualRecord",
    "skill_node_id",
    "vector_distance",
]
