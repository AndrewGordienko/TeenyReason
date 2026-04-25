"""Downstream consumer interfaces for crawler recipes."""

from __future__ import annotations

from typing import Any, Protocol

from ..recipes import CrawlerRecipe


class DownstreamConsumer(Protocol):
    """Minimal protocol for app-layer consumers of crawler recipes."""

    def run(self, recipe: CrawlerRecipe, **kwargs: Any) -> Any:
        """Run one downstream algorithm against the supplied recipe."""
