"""Probe-conditioned PPO entrypoints and result types."""

__all__ = [
    "MatchedEvalSummary",
    "TrainingRunResult",
    "train_plain_ppo",
    "train_probe_conditioned_ppo",
]


def __getattr__(name: str):
    """Load training entrypoints lazily to avoid crawler/message import cycles."""
    if name == "train_plain_ppo":
        from .train_plain import train_plain_ppo

        return train_plain_ppo
    if name == "train_probe_conditioned_ppo":
        from .train_probe import train_probe_conditioned_ppo

        return train_probe_conditioned_ppo
    if name in {"MatchedEvalSummary", "TrainingRunResult"}:
        from .types import MatchedEvalSummary, TrainingRunResult

        return {
            "MatchedEvalSummary": MatchedEvalSummary,
            "TrainingRunResult": TrainingRunResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
