"""Solver-facing probe handoff helpers."""

__all__ = [
    "build_solver_episode_belief",
    "build_solver_episode_expression",
    "build_solver_expression_audit",
    "compute_online_future_diagnostics",
    "solver_belief_input_from_message",
    "solver_expression_input_from_env_expression",
    "solver_message_content",
]


def __getattr__(name: str):
    """Load handoff helpers lazily to avoid crawler import cycles."""
    if name in {
        "build_solver_episode_belief",
        "build_solver_episode_expression",
        "solver_belief_input_from_message",
        "solver_expression_input_from_env_expression",
    }:
        from . import message

        return getattr(message, name)
    if name in {"build_solver_expression_audit", "solver_message_content"}:
        from . import audit

        return getattr(audit, name)
    if name == "compute_online_future_diagnostics":
        from .diagnostics import compute_online_future_diagnostics

        return compute_online_future_diagnostics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
