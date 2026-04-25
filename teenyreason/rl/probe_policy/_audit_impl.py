"""Small controller-input audits for solver-facing env expressions."""

from __future__ import annotations

import numpy as np

from ...crawler.types import EnvExpression
from .messages import build_solver_episode_expression


def solver_message_content(solver_input: np.ndarray) -> np.ndarray:
    """Return just the message portion of a solver input."""
    value = np.asarray(solver_input, dtype=np.float32).reshape(-1)
    if value.size <= 2:
        return np.zeros((0,), dtype=np.float32)
    return value[:-2].astype(np.float32, copy=False)


def build_solver_expression_audit(
    *,
    env_expression: EnvExpression,
    current_episode: int,
    total_episodes: int,
    strict_fair_mode: bool = False,
    shadow_expression_mode: bool = False,
) -> dict[str, float | bool]:
    """Compare the live env-expression path against an explicitly muted path."""
    enabled_input, enabled_scale = build_solver_episode_expression(
        env_expression=env_expression,
        current_episode=current_episode,
        total_episodes=total_episodes,
        disable_env_expression=False,
        strict_fair_mode=strict_fair_mode,
        shadow_expression_mode=shadow_expression_mode,
    )
    muted_input, muted_scale = build_solver_episode_expression(
        env_expression=env_expression,
        current_episode=current_episode,
        total_episodes=total_episodes,
        disable_env_expression=True,
        strict_fair_mode=strict_fair_mode,
        shadow_expression_mode=shadow_expression_mode,
    )
    enabled_message = solver_message_content(enabled_input)
    muted_message = solver_message_content(muted_input)
    enabled_message_norm = float(np.linalg.norm(enabled_message))
    muted_message_norm = float(np.linalg.norm(muted_message))
    return {
        "enabled_scale": float(enabled_scale),
        "muted_scale": float(muted_scale),
        "input_delta": float(
            np.linalg.norm(
                np.asarray(enabled_input, dtype=np.float32)
                - np.asarray(muted_input, dtype=np.float32)
            )
        ),
        "message_delta": float(np.linalg.norm(enabled_message - muted_message)),
        "enabled_message_norm": enabled_message_norm,
        "muted_message_norm": muted_message_norm,
        "enabled_has_message": bool(enabled_message_norm > 1e-6),
        "muted_has_message": bool(muted_message_norm > 1e-6),
    }
