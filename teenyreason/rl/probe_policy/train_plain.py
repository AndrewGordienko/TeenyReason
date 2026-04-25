"""Plain PPO entrypoint routed through the probe-policy package."""

from ._train_impl import train_plain_ppo

__all__ = ["train_plain_ppo"]
