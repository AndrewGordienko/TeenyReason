"""App-layer benchmark consumers for crawler recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..app.benchmark import run_training_pipeline
from ..multidomain import run_mnist_probe_benchmark, run_shakespeare_probe_benchmark
from ..recipes import CrawlerRecipe


def _benchmark_spec(recipe: CrawlerRecipe, expected_kind: str | None = None):
    """Return the explicit benchmark payload for a recipe."""
    spec = recipe.benchmark
    if spec is None:
        raise ValueError(f"Recipe {recipe.name!r} does not define a benchmark payload.")
    if expected_kind is not None and spec.kind != expected_kind:
        raise ValueError(
            f"Recipe {recipe.name!r} is wired for {spec.kind!r}, not {expected_kind!r}."
        )
    return spec


def _normalize_rl_config_override(values: dict[str, Any]) -> dict[str, Any]:
    """Accept the short public `profile` spelling at the benchmark edge."""
    normalized = dict(values)
    if "profile" in normalized and "benchmark_profile" not in normalized:
        normalized["benchmark_profile"] = normalized.pop("profile")
    return normalized


@dataclass(frozen=True)
class PPOBenchmarkConsumer:
    """Compatibility consumer that runs the existing PPO benchmark harness."""

    default_config_override: dict[str, Any] = field(default_factory=dict)

    def run(
        self,
        recipe: CrawlerRecipe,
        *,
        seeds: list[int] | None = None,
        config_override: dict[str, Any] | None = None,
    ):
        spec = _benchmark_spec(recipe, expected_kind="ppo")
        env_name = spec.env_name
        if not isinstance(env_name, str) or not env_name:
            raise ValueError(
                f"Recipe {recipe.name!r} does not define an RL benchmark env_name."
            )
        merged_override = _normalize_rl_config_override(spec.config_override)
        merged_override.update(
            _normalize_rl_config_override(self.default_config_override)
        )
        if config_override:
            merged_override.update(_normalize_rl_config_override(config_override))
        return run_training_pipeline(
            env_name=env_name,
            seeds=seeds,
            config_override=merged_override or None,
        )


@dataclass(frozen=True)
class ImageProbeBenchmarkConsumer:
    """Compatibility consumer for the MNIST sample-efficiency benchmark."""

    def run(self, recipe: CrawlerRecipe):
        config = _benchmark_spec(recipe, expected_kind="image_probe").config
        return run_mnist_probe_benchmark(config)


@dataclass(frozen=True)
class LanguageProbeBenchmarkConsumer:
    """Compatibility consumer for the Tiny Shakespeare sample-efficiency benchmark."""

    def run(self, recipe: CrawlerRecipe):
        config = _benchmark_spec(recipe, expected_kind="language_probe").config
        return run_shakespeare_probe_benchmark(config)
