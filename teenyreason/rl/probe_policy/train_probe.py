"""Probe-conditioned PPO entrypoint routed through the probe-policy package."""

from ._train_impl import train_probe_conditioned_ppo

__all__ = ["train_probe_conditioned_ppo"]
