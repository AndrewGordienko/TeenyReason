"""Generic crawler recipe objects.

Recipes are user-facing compositions. They choose concrete capabilities for
one use case without making those choices part of the crawler core API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..crawler.core import (
    BeliefBackend,
    Crawler,
    MessageProjector,
    QueryPolicy,
    StopPolicy,
    WorldAdapter,
)


WorldAdapterFactory = Callable[[], WorldAdapter]
BeliefBackendFactory = Callable[[], BeliefBackend]
QueryPolicyFactory = Callable[[], QueryPolicy]
StopPolicyFactory = Callable[[], StopPolicy]
MessageProjectorFactory = Callable[[], MessageProjector]


@dataclass(frozen=True)
class BenchmarkSpec:
    """Small explicit app-layer benchmark payload for compatibility consumers."""

    kind: str
    env_name: str | None = None
    config: Any = None
    config_override: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrawlerRecipe:
    """Composition object for one crawler setup."""

    name: str
    description: str
    world_adapter_factory: WorldAdapterFactory
    belief_backend_factory: BeliefBackendFactory
    query_policy_factory: QueryPolicyFactory
    stop_policy_factory: StopPolicyFactory
    message_projector_factory: MessageProjectorFactory
    max_steps: int = 4
    metadata: dict[str, Any] = field(default_factory=dict)
    benchmark: BenchmarkSpec | None = None

    def build_crawler(self, *, max_steps: int | None = None) -> Crawler:
        """Instantiate a fresh crawler from the recipe."""
        return Crawler(
            world=self.world_adapter_factory(),
            belief_backend=self.belief_backend_factory(),
            query_policy=self.query_policy_factory(),
            stop_policy=self.stop_policy_factory(),
            message_projector=self.message_projector_factory(),
            max_steps=int(self.max_steps if max_steps is None else max_steps),
        )
