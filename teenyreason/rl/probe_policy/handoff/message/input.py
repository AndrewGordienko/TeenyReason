"""Solver-facing vector construction and keep-scale helpers."""

import numpy as np

from .....crawler.types import EnvExpression
from ....core import sanitize_numpy


def solver_expression_input_from_env_expression(
    env_expression: EnvExpression,
    *,
    confidence_scale: float | None = None,
) -> np.ndarray:
    """Build the canonical solver-side env-expression input.

    The controller sees the compact expression vector plus two explicit scalar
    slots: confidence and uncertainty.
    """
    vector = sanitize_numpy(np.asarray(env_expression.vector, dtype=np.float32).reshape(-1))
    confidence = (
        float(env_expression.confidence)
        if confidence_scale is None
        else float(confidence_scale)
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))
    uncertainty = float(env_expression.uncertainty_scalar)
    return sanitize_numpy(
        np.concatenate(
            [
                vector,
                np.asarray([confidence, uncertainty], dtype=np.float32),
            ],
            axis=0,
        )
    )


def solver_belief_input_from_message(
    belief_message: np.ndarray,
    uncertainty_scalar: float,
    message_scale: float = 1.0,
) -> np.ndarray:
    """Compatibility wrapper around the env-expression controller contract."""
    env_expression = EnvExpression(
        vector=sanitize_numpy(np.asarray(belief_message, dtype=np.float32).reshape(-1)),
        confidence=float(np.clip(message_scale, 0.0, 1.0)),
        ready=False,
        uncertainty_scalar=float(uncertainty_scalar),
        compressed=False,
        metadata={},
    )
    return solver_expression_input_from_env_expression(env_expression)


def apply_solver_expression_keep_scale(
    solver_expression: np.ndarray,
    keep_scale: float,
) -> np.ndarray:
    """Scale the expression content while preserving the uncertainty slot."""
    expression = sanitize_numpy(np.asarray(solver_expression, dtype=np.float32).reshape(-1))
    keep_scale = float(np.clip(keep_scale, 0.0, 1.0))
    if expression.size == 0:
        return expression
    if expression.size == 1:
        return sanitize_numpy(expression * keep_scale)
    scaled = expression.copy()
    if expression.size > 2:
        scaled[:-2] = sanitize_numpy(scaled[:-2] * keep_scale)
    scaled[-2] = float(scaled[-2] * keep_scale)
    return sanitize_numpy(scaled)


def apply_solver_message_keep_scale(
    solver_belief: np.ndarray,
    keep_scale: float,
) -> np.ndarray:
    """Compatibility alias for the older message-oriented name."""
    return apply_solver_expression_keep_scale(solver_belief, keep_scale)
