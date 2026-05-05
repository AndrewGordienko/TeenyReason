"""Recipe builders for the existing benchmark environments."""

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
from ..envs import (
    BIPEDAL_WALKER_NAME,
    CONTINUOUS_CARTPOLE_NAME,
    CONTINUOUS_LUNAR_LANDER_NAME,
)
from .base import BenchmarkSpec, CrawlerRecipe
from .domains import build_bipedal_recipe, build_cartpole_recipe
from .domains import register_bipedal_recipe_targets, register_cartpole_recipe_targets


def build_generic_rl_recipe(env_name: str) -> CrawlerRecipe:
    """Build a generic compatibility recipe for an RL benchmark environment."""
    query_payloads = {
        "scan": {"vector": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)},
        "stress": {"vector": np.asarray([0.0, 1.0, 0.0], dtype=np.float32)},
        "recover": {"vector": np.asarray([0.0, 0.0, 1.0], dtype=np.float32)},
    }
    return CrawlerRecipe(
        name=env_name.replace("_", "-"),
        description=f"Generic crawler composition for the {env_name} benchmark.",
        world_adapter_factory=partial(
            ScriptedWorldAdapter,
            query_payloads=query_payloads,
            source_prefix=env_name,
        ),
        belief_backend_factory=partial(VectorBeliefBackend, vector_key="vector"),
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(SupportLimitStopPolicy, min_support=2),
        message_projector_factory=LinearMessageProjector,
        max_steps=2,
        metadata={
            "modality": "rl",
            "recipe_family": "generic_rl",
        },
        benchmark=BenchmarkSpec(kind="ppo", env_name=env_name),
    )


def build_benchmark_recipe(env_name: str) -> CrawlerRecipe:
    """Build the user-facing recipe for one benchmark environment."""
    if env_name == CONTINUOUS_CARTPOLE_NAME:
        register_cartpole_recipe_targets()
        return build_cartpole_recipe()
    if env_name == BIPEDAL_WALKER_NAME:
        register_bipedal_recipe_targets()
        return build_bipedal_recipe()
    if env_name == CONTINUOUS_LUNAR_LANDER_NAME:
        return build_generic_rl_recipe(env_name)
    return build_generic_rl_recipe(env_name)
