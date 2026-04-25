"""Crawler library entrypoints.

The generic crawler API is now the canonical public surface. The older
RL-facing objects are still exported as compatibility adapters while the PPO
benchmark and dashboard stack migrates onto the slimmer library contract.
"""

from typing import TYPE_CHECKING

from .compat import (
    crawler_message_to_env_expression,
    env_expression_to_crawler_message,
    legacy_step_result_to_crawler_step,
)
from .core import (
    BeliefBackend,
    Crawler,
    LinearMessageProjector,
    MessageProjector,
    QueryPolicy,
    RoundRobinQueryPolicy,
    ScriptedWorldAdapter,
    StopPolicy,
    SupportLimitStopPolicy,
    VectorBeliefBackend,
    WorldAdapter,
)
from .predictive import (
    group_window_targets_numpy,
    group_window_targets_torch,
    masked_group_average_numpy,
    masked_group_average_torch,
)
from .types import (
    BeliefMessage,
    BeliefState,
    ControllerBeliefContext,
    CrawlerMessage,
    CrawlerRunResult,
    CrawlerStep,
    CrawlerStepResult,
    EvidenceBatch,
    EvidenceSlice,
    EvidenceWindow,
    EnvExpression,
    LegacyCrawlerRunResult,
    LegacyCrawlerStepResult,
    MetricBelief,
    PredictiveBelief,
    UncertaintyEstimate,
)

if TYPE_CHECKING:
    from .library import (
        CrawlerModelBundle,
        load_crawler_bundle_from_checkpoint,
        train_crawler_library,
    )

__all__ = [
    "BeliefBackend",
    "BeliefMessage",
    "BeliefState",
    "ControllerBeliefContext",
    "Crawler",
    "CrawlerMessage",
    "CrawlerModelBundle",
    "CrawlerRunResult",
    "CrawlerStep",
    "CrawlerStepResult",
    "EvidenceBatch",
    "EvidenceSlice",
    "EvidenceWindow",
    "EnvExpression",
    "LegacyCrawlerRunResult",
    "LegacyCrawlerStepResult",
    "LinearMessageProjector",
    "MessageProjector",
    "MetricBelief",
    "PredictiveBelief",
    "QueryPolicy",
    "RoundRobinQueryPolicy",
    "ScriptedWorldAdapter",
    "StopPolicy",
    "SupportLimitStopPolicy",
    "UncertaintyEstimate",
    "VectorBeliefBackend",
    "WorldAdapter",
    "crawler_message_to_env_expression",
    "env_expression_to_crawler_message",
    "group_window_targets_numpy",
    "group_window_targets_torch",
    "legacy_step_result_to_crawler_step",
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
