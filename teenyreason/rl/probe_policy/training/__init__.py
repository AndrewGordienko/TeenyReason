"""Probe-policy training loop and support helpers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loops.plain import train_plain_ppo
    from .loops.probe import train_probe_conditioned_ppo

__all__ = [
    "train_plain_ppo",
    "train_probe_conditioned_ppo",
]


def __getattr__(name: str):
    """Load trainer entrypoints lazily."""
    if name == "train_plain_ppo":
        from .loops.plain import train_plain_ppo

        return train_plain_ppo
    if name == "train_probe_conditioned_ppo":
        from .loops.probe import train_probe_conditioned_ppo

        return train_probe_conditioned_ppo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
