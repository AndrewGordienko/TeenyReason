"""Small crawler setup object for app composition."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..envs import CONTINUOUS_CARTPOLE_NAME
from ..envs.continuous_cartpole import ContinuousCartPoleEnv


@dataclass(frozen=True)
class CrawlerSetup:
    """A reusable crawler configuration before it is loaded or trained."""

    env_name: str = CONTINUOUS_CARTPOLE_NAME
    checkpoint_path: Path | None = None
    runtime_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        source: str | object | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        **runtime_overrides: Any,
    ) -> "CrawlerSetup":
        """Build a crawler setup from a Gym id, Gym env, or local env class."""
        if isinstance(source, CrawlerSetup):
            extra = dict(source.runtime_overrides)
            extra.update(runtime_overrides)
            return replace(
                source,
                checkpoint_path=(
                    source.checkpoint_path if checkpoint_path is None else Path(checkpoint_path)
                ),
                runtime_overrides=extra,
            )
        return cls(
            env_name=_resolve_env_name(source),
            checkpoint_path=None if checkpoint_path is None else Path(checkpoint_path),
            runtime_overrides=dict(runtime_overrides),
        )

    def load(self, **overrides: Any):
        """Load the latest trained runtime crawler for this setup."""
        from .runtime import best_crawler

        runtime_overrides = dict(self.runtime_overrides)
        runtime_overrides.update(overrides)
        return best_crawler(
            self.env_name,
            checkpoint_path=self.checkpoint_path,
            **runtime_overrides,
        )

    def run(self, env=None, **kwargs: Any):
        """Load the runtime crawler and collect one belief expression."""
        runtime = self.load()
        return runtime.run(env, **kwargs)

    def __call__(self, env=None, **kwargs: Any):
        return self.run(env, **kwargs)


def _resolve_env_name(source: str | object | None) -> str:
    if source is None:
        return CONTINUOUS_CARTPOLE_NAME
    if isinstance(source, str):
        return str(source)
    if source is ContinuousCartPoleEnv:
        return CONTINUOUS_CARTPOLE_NAME
    unwrapped = getattr(source, "unwrapped", None)
    if isinstance(unwrapped, ContinuousCartPoleEnv):
        return CONTINUOUS_CARTPOLE_NAME
    spec = getattr(source, "spec", None)
    spec_id = getattr(spec, "id", None)
    if isinstance(spec_id, str) and spec_id:
        return spec_id
    for attr_name in ("gym_id", "env_name"):
        value = getattr(source, attr_name, None)
        if isinstance(value, str) and value:
            return value
    return CONTINUOUS_CARTPOLE_NAME


def crawler_setup(source: str | object | None = None, **kwargs: Any) -> CrawlerSetup:
    """Public helper for `tr.crawler(...)`."""
    return CrawlerSetup.from_env(source, **kwargs)


__all__ = ["CrawlerSetup", "crawler_setup"]
