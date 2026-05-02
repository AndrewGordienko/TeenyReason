"""App-layer benchmark consumers for crawler recipes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..app.benchmark import run_training_pipeline
from ..envs import CONTINUOUS_CARTPOLE_NAME
from ..envs.continuous_cartpole import ContinuousCartPoleEnv
from ..multidomain import run_mnist_probe_benchmark, run_shakespeare_probe_benchmark
from ..recipes import CrawlerRecipe


SeedInput = int | list[int] | tuple[int, ...] | None


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


def _normalize_seed_input(seeds: SeedInput) -> list[int] | None:
    if seeds is None:
        return None
    if isinstance(seeds, int):
        return list(range(max(1, int(seeds))))
    return [int(seed) for seed in seeds]


def _env_name_from_source(source) -> str:
    """Resolve a training env name from a crawler setup, recipe, or Gym env."""
    if isinstance(source, CrawlerRecipe):
        return str(_benchmark_spec(source, expected_kind="ppo").env_name)
    benchmark = getattr(source, "benchmark", None)
    if benchmark is not None and getattr(benchmark, "kind", None) == "ppo":
        return str(benchmark.env_name)
    config = getattr(source, "config", None)
    config_env_name = getattr(config, "env_name", None)
    if isinstance(config_env_name, str) and config_env_name:
        return config_env_name
    env_name = getattr(source, "env_name", None)
    if isinstance(env_name, str) and env_name:
        return env_name
    if source is ContinuousCartPoleEnv:
        return CONTINUOUS_CARTPOLE_NAME
    unwrapped = getattr(source, "unwrapped", None)
    if isinstance(unwrapped, ContinuousCartPoleEnv):
        return CONTINUOUS_CARTPOLE_NAME
    spec = getattr(source, "spec", None)
    spec_id = getattr(spec, "id", None)
    if isinstance(spec_id, str) and spec_id:
        return spec_id
    if isinstance(source, str):
        return str(source)
    return CONTINUOUS_CARTPOLE_NAME


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
class ProbeConditionedPPO:
    """Train PPO with a separate crawler setup as its context source."""

    profile: str | None = "fast"
    seeds: SeedInput = 1
    config_override: dict[str, Any] = field(default_factory=dict)

    def train(
        self,
        crawler_or_env=CONTINUOUS_CARTPOLE_NAME,
        *,
        seeds: SeedInput = None,
        profile: str | None = None,
        overrides: dict[str, Any] | None = None,
    ):
        env_name = _env_name_from_source(crawler_or_env)
        return self._run_probe_pipeline(
            env_name,
            seeds=self.seeds if seeds is None else seeds,
            profile=self.profile if profile is None else profile,
            overrides=overrides,
        )

    def run(
        self,
        recipe: CrawlerRecipe,
        *,
        seeds: list[int] | None = None,
        config_override: dict[str, Any] | None = None,
    ):
        spec = _benchmark_spec(recipe, expected_kind="ppo")
        merged = _normalize_rl_config_override(spec.config_override)
        merged.update(self.config_override)
        if config_override:
            merged.update(_normalize_rl_config_override(config_override))
        active_profile = None if "benchmark_profile" in merged else self.profile
        return self._run_probe_pipeline(
            str(spec.env_name),
            seeds=self.seeds if seeds is None else seeds,
            profile=active_profile,
            overrides=merged,
        )

    def _run_probe_pipeline(
        self,
        env_name: str,
        *,
        seeds: SeedInput,
        profile: str | None,
        overrides: dict[str, Any] | None,
    ):
        config_override = _normalize_rl_config_override(self.config_override)
        if overrides:
            config_override.update(_normalize_rl_config_override(overrides))
        if profile is not None:
            config_override["benchmark_profile"] = profile
        from ..app.probe_ppo import run_probe_conditioned_pipeline

        return run_probe_conditioned_pipeline(
            env_name,
            seeds=_normalize_seed_input(seeds),
            config_override=config_override or None,
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
