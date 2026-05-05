"""Shared evidence payload helpers for crawler recipes."""

from __future__ import annotations

from typing import Any

import numpy as np


STANDARD_EVIDENCE_FIELDS = (
    "modality",
    "query_family",
    "source_id",
    "intervention_cost",
    "hidden_target",
    "local_state",
    "outcome",
)


def evidence_payload(
    *,
    modality: str,
    query_family: str,
    source_id: str,
    intervention_cost: float,
    hidden_target: dict[str, Any] | None,
    local_state: dict[str, Any],
    outcome: dict[str, Any],
    vector: np.ndarray,
    belief_source: str = "learned",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one standardized opaque payload for the generic crawler core."""
    payload: dict[str, Any] = {
        "modality": str(modality),
        "query_family": str(query_family),
        "source_id": str(source_id),
        "intervention_cost": float(intervention_cost),
        "hidden_target": {} if hidden_target is None else dict(hidden_target),
        "local_state": dict(local_state),
        "outcome": dict(outcome),
        "vector": np.asarray(vector, dtype=np.float32).reshape(-1),
        "belief_source": str(belief_source),
    }
    if extra:
        payload.update(extra)
    return payload


def evidence_metadata(
    *,
    payload: dict[str, Any],
    query_index: int,
) -> dict[str, Any]:
    """Mirror the standardized payload fields into lightweight slice metadata."""
    return {
        "query_index": int(query_index),
        "modality": str(payload.get("modality", "unknown")),
        "query_family": str(payload.get("query_family", "")),
        "source_id": str(payload.get("source_id", "")),
        "intervention_cost": float(payload.get("intervention_cost", 0.0)),
        "belief_source": str(payload.get("belief_source", "learned")),
    }
