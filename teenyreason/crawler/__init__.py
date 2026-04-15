"""Crawler library entrypoints.

Keep bundle imports lazy so predictive helpers can be imported from
``teenyreason.crawler.predictive`` without pulling the whole training/library
stack back into ``belief_world_model`` during module import.
"""

from typing import TYPE_CHECKING

from .predictive import (
    group_window_targets_numpy,
    group_window_targets_torch,
    masked_group_average_numpy,
    masked_group_average_torch,
)

if TYPE_CHECKING:
    from .library import (
        CrawlerModelBundle,
        load_crawler_bundle_from_checkpoint,
        train_crawler_library,
    )

__all__ = [
    "CrawlerModelBundle",
    "group_window_targets_numpy",
    "group_window_targets_torch",
    "load_crawler_bundle_from_checkpoint",
    "masked_group_average_numpy",
    "masked_group_average_torch",
    "train_crawler_library",
]


def __getattr__(name: str):
    if name in {
        "CrawlerModelBundle",
        "load_crawler_bundle_from_checkpoint",
        "train_crawler_library",
    }:
        from .library import (
            CrawlerModelBundle,
            load_crawler_bundle_from_checkpoint,
            train_crawler_library,
        )

        exports = {
            "CrawlerModelBundle": CrawlerModelBundle,
            "load_crawler_bundle_from_checkpoint": load_crawler_bundle_from_checkpoint,
            "train_crawler_library": train_crawler_library,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
