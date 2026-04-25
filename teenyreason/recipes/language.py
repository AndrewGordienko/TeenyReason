"""Language recipe composition for the generic crawler library."""

from __future__ import annotations

from functools import partial

import numpy as np

from ..crawler.core import (
    LinearMessageProjector,
    RoundRobinQueryPolicy,
    ScriptedWorldAdapter,
    SupportLimitStopPolicy,
    VectorBeliefBackend,
)
from ..multidomain.language_benchmark import LanguageProbeBenchmarkConfig
from .base import BenchmarkSpec, CrawlerRecipe


def build_language_recipe(
    config: LanguageProbeBenchmarkConfig | None = None,
) -> CrawlerRecipe:
    """Build the Tiny Shakespeare benchmark recipe."""
    config = config or LanguageProbeBenchmarkConfig()
    query_payloads = {
        "continue_context": {"vector": np.asarray([1.0, 0.0, 0.2, 0.2], dtype=np.float32)},
        "swap_continuation": {"vector": np.asarray([0.2, 1.0, 0.1, 0.3], dtype=np.float32)},
        "mask_span": {"vector": np.asarray([0.1, 0.2, 1.0, 0.4], dtype=np.float32)},
    }
    return CrawlerRecipe(
        name="language",
        description="Generic crawler composition for the Tiny Shakespeare benchmark.",
        world_adapter_factory=partial(
            ScriptedWorldAdapter,
            query_payloads=query_payloads,
            source_prefix="language",
        ),
        belief_backend_factory=partial(VectorBeliefBackend, vector_key="vector"),
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(SupportLimitStopPolicy, min_support=2),
        message_projector_factory=LinearMessageProjector,
        max_steps=3,
        metadata={
            "modality": "language",
            "recipe_family": "language",
        },
        benchmark=BenchmarkSpec(kind="language_probe", config=config),
    )
