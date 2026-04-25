"""Compatibility helpers between the generic crawler API and the RL stack."""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import (
    BeliefState,
    CrawlerMessage,
    CrawlerStep,
    EnvExpression,
    EvidenceSlice,
    LegacyCrawlerStepResult,
)


def crawler_message_to_env_expression(
    message: CrawlerMessage,
    *,
    compressed: bool = False,
    metadata: dict[str, Any] | None = None,
) -> EnvExpression:
    """Project a generic crawler message into the legacy RL env-expression type."""
    merged_metadata = dict(message.metadata or {})
    if metadata:
        merged_metadata.update(metadata)
    return EnvExpression(
        vector=np.asarray(message.vector, dtype=np.float32).reshape(-1),
        confidence=float(message.confidence),
        ready=bool(message.ready),
        uncertainty_scalar=float(message.uncertainty),
        compressed=bool(compressed),
        metadata=merged_metadata,
    )


def env_expression_to_crawler_message(env_expression: EnvExpression) -> CrawlerMessage:
    """Project the legacy RL env-expression into the generic crawler message type."""
    return CrawlerMessage(
        vector=np.asarray(env_expression.vector, dtype=np.float32).reshape(-1),
        confidence=float(env_expression.confidence),
        ready=bool(env_expression.ready),
        uncertainty=float(env_expression.uncertainty_scalar),
        metadata=dict(env_expression.metadata or {}),
    )


def legacy_step_result_to_crawler_step(
    step_result: LegacyCrawlerStepResult,
    *,
    query_name: str | None = None,
    source_id: str = "legacy",
) -> CrawlerStep:
    """Wrap one legacy crawler step in the generic step view."""
    evidence = None
    if query_name is not None:
        evidence = EvidenceSlice(
            query_name=str(query_name),
            source_id=str(source_id),
            payload={},
            metadata={},
        )
    belief_state = BeliefState(
        latent=np.asarray(step_result.predictive_belief.mean_raw, dtype=np.float32).reshape(-1),
        uncertainty=float(step_result.uncertainty.scalar),
        support_size=int(step_result.predictive_belief.support_count),
        metadata={
            "future_probe_error": float(step_result.predictive_belief.future_probe_error),
            "support_diversity_ratio": float(step_result.predictive_belief.support_diversity_ratio),
        },
    )
    return CrawlerStep(
        query_name=query_name,
        evidence=evidence,
        belief_state=belief_state,
        message=env_expression_to_crawler_message(step_result.env_expression),
        stop_reason=step_result.stop_reason,
        metadata={
            "expected_family_gain": dict(step_result.expected_family_gain),
            "realized_family_gain": dict(step_result.realized_family_gain),
        },
    )
