"""Small public entrypoints for the crawler library."""

from __future__ import annotations

from typing import Any

from .algos import ImageProbeBenchmarkConsumer, LanguageProbeBenchmarkConsumer, PPOBenchmarkConsumer
from .crawler import Crawler
from .crawler.types import (
    BeliefState,
    CrawlerMessage,
    CrawlerRunResult,
    CrawlerStep,
    EvidenceSlice,
)
from .envs import (
    BIPEDAL_WALKER_NAME,
    CONTINUOUS_CARTPOLE_NAME,
    CONTINUOUS_LUNAR_LANDER_NAME,
)
from .envs.continuous_cartpole import ContinuousCartPoleEnv
from .recipes import (
    CrawlerRecipe,
    build_benchmark_recipe,
    build_bipedal_recipe,
    build_cartpole_recipe,
    build_language_recipe,
    build_mnist_recipe,
    register_default_recipe_targets,
)


Evidence = EvidenceSlice
Belief = BeliefState
Message = CrawlerMessage
Step = CrawlerStep
Run = CrawlerRunResult


def _normalize_profile_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    """Map the short public `profile` name onto the internal config field."""
    normalized = dict(values)
    if "profile" in normalized and "benchmark_profile" not in normalized:
        normalized["benchmark_profile"] = normalized.pop("profile")
    return normalized


def _normalize_seed_input(seeds: int | list[int] | tuple[int, ...] | None) -> list[int] | None:
    if seeds is None:
        return None
    if isinstance(seeds, int):
        return list(range(max(1, int(seeds))))
    return [int(seed) for seed in seeds]


def recipe(name: str, **kwargs: Any) -> CrawlerRecipe:
    """Build one small user-facing recipe by name."""
    register_default_recipe_targets()
    key = str(name).strip()
    lowered = key.lower()
    if lowered in {"cartpole", "continuous_cartpole", CONTINUOUS_CARTPOLE_NAME.lower()}:
        return build_cartpole_recipe()
    if lowered in {"bipedal", "bipedal_walker", BIPEDAL_WALKER_NAME.lower()}:
        return build_bipedal_recipe()
    if lowered in {"mnist", "image"}:
        return build_mnist_recipe(**kwargs)
    if lowered in {"language", "tinyshakespeare", "tiny_shakespeare", "shakespeare"}:
        return build_language_recipe(**kwargs)
    if lowered in {"lunar", "lunarlander", "continuous_lunar_lander", CONTINUOUS_LUNAR_LANDER_NAME.lower()}:
        return build_benchmark_recipe(CONTINUOUS_LUNAR_LANDER_NAME)
    return build_benchmark_recipe(key)


def ppo(**kwargs: Any) -> PPOBenchmarkConsumer:
    """Build the small PPO benchmark consumer."""
    return PPOBenchmarkConsumer(
        default_config_override=_normalize_profile_kwargs(kwargs)
    )


def _default_algo(recipe_obj: CrawlerRecipe):
    """Pick the obvious benchmark consumer for a recipe."""
    spec = recipe_obj.benchmark
    kind = None if spec is None else spec.kind
    if kind == "image_probe":
        return ImageProbeBenchmarkConsumer()
    if kind == "language_probe":
        return LanguageProbeBenchmarkConsumer()
    return PPOBenchmarkConsumer()


def _coerce_recipe(source: str | CrawlerRecipe | type | object, **kwargs: Any) -> CrawlerRecipe:
    if isinstance(source, CrawlerRecipe):
        return source
    if isinstance(source, str):
        return recipe(source, **kwargs)
    if source is ContinuousCartPoleEnv:
        return recipe(CONTINUOUS_CARTPOLE_NAME, **kwargs)
    env_name = getattr(source, "gym_id", None) or getattr(source, "env_name", None)
    if isinstance(env_name, str) and env_name:
        return recipe(env_name, **kwargs)
    raise ValueError("run(...) expects a gym id string, a crawler recipe, or a custom env class with env_name/gym_id.")


def run(
    env,
    algo=None,
    *,
    seeds: int | list[int] | tuple[int, ...] | None = 2,
    profile: str | None = "fast",
    overrides: dict[str, Any] | None = None,
):
    """Run one gym-like environment or recipe through the chosen algorithm."""
    recipe_obj = _coerce_recipe(env)
    consumer = _default_algo(recipe_obj) if algo is None else algo
    config_override = _normalize_profile_kwargs(overrides or {})
    if profile is not None:
        config_override["benchmark_profile"] = profile
    run_kwargs: dict[str, Any] = {}
    seed_list = _normalize_seed_input(seeds)
    if seed_list is not None:
        run_kwargs["seeds"] = seed_list
    if config_override:
        run_kwargs["config_override"] = config_override
    return consumer.run(recipe_obj, **run_kwargs)


def bench(
    recipe_obj: CrawlerRecipe,
    *,
    algo=None,
    seeds: list[int] | None = None,
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
):
    """Compatibility wrapper around the smaller public run(...) entrypoint."""
    return run(recipe_obj, algo=algo, seeds=seeds, profile=profile, overrides=overrides)


__all__ = [
    "Belief",
    "Crawler",
    "Evidence",
    "Message",
    "Run",
    "Step",
    "bench",
    "ppo",
    "recipe",
    "run",
]
