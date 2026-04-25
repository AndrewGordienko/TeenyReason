"""MNIST recipe composition for the generic crawler library."""

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
from ..multidomain.image_benchmark import ImageProbeBenchmarkConfig
from .base import BenchmarkSpec, CrawlerRecipe


def build_mnist_recipe(
    config: ImageProbeBenchmarkConfig | None = None,
) -> CrawlerRecipe:
    """Build the MNIST probe benchmark recipe."""
    config = config or ImageProbeBenchmarkConfig()
    query_payloads = {
        "rotate_quadrants": {"vector": np.asarray([1.0, 0.0, 0.4, 0.1], dtype=np.float32)},
        "mask_center": {"vector": np.asarray([0.2, 1.0, 0.3, 0.2], dtype=np.float32)},
        "invert_contrast": {"vector": np.asarray([0.1, 0.3, 1.0, 0.4], dtype=np.float32)},
    }
    return CrawlerRecipe(
        name="mnist",
        description="Generic crawler composition for the MNIST sample-efficiency benchmark.",
        world_adapter_factory=partial(
            ScriptedWorldAdapter,
            query_payloads=query_payloads,
            source_prefix="mnist",
        ),
        belief_backend_factory=partial(VectorBeliefBackend, vector_key="vector"),
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(SupportLimitStopPolicy, min_support=2),
        message_projector_factory=LinearMessageProjector,
        max_steps=3,
        metadata={
            "modality": "image",
            "recipe_family": "mnist",
        },
        benchmark=BenchmarkSpec(kind="image_probe", config=config),
    )
