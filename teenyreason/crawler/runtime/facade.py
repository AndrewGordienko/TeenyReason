"""Runtime facade for using the trained crawler as its own entity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch

from ...app.config import build_experiment_config
from ...envs import CONTINUOUS_CARTPOLE_NAME, get_action_values, make_env
from ..probes.data import default_env_params
from ..library import CrawlerModelBundle, load_crawler_bundle_from_checkpoint
from ..types import ControllerBeliefContext, EnvExpression, LegacyCrawlerStepResult


DEFAULT_ARTIFACT_DIR = Path("artifacts")
CHECKPOINT_GLOB = "*_probe_ppo_checkpoint.pt"


@dataclass(frozen=True)
class CrawlerRuntimeConfig:
    """Concrete knobs for one runtime crawler pass."""

    env_name: str = CONTINUOUS_CARTPOLE_NAME
    action_bins: int = 9
    window_size: int = 16
    base_probe_episodes: int = 2
    max_probe_episodes: int = 3
    probe_adaptive_budget: bool = False
    uncertainty_probe_threshold: float = 0.18
    surprise_probe_threshold: float = 0.75
    belief_bits_per_dim: int = 0
    belief_use_residual_sketch: bool = False
    checkpoint_path: Path | None = None
    variant_label: str = "latent-crawler"

    @classmethod
    def from_env(
        cls,
        env_name: str = CONTINUOUS_CARTPOLE_NAME,
        *,
        checkpoint_path: str | Path | None = None,
    ) -> "CrawlerRuntimeConfig":
        """Build the current best runtime defaults for a named environment."""
        try:
            config = build_experiment_config(env_name)
        except ValueError:
            return cls(
                env_name=str(env_name),
                checkpoint_path=None if checkpoint_path is None else Path(checkpoint_path),
            )
        return cls(
            env_name=str(env_name),
            action_bins=int(config.action_bins),
            window_size=int(config.window_size),
            base_probe_episodes=int(config.base_probe_episodes),
            max_probe_episodes=int(config.max_probe_episodes),
            probe_adaptive_budget=bool(config.probe_adaptive_budget),
            uncertainty_probe_threshold=float(config.uncertainty_probe_threshold),
            surprise_probe_threshold=float(config.surprise_probe_threshold),
            belief_bits_per_dim=int(config.belief_bits_per_dim),
            belief_use_residual_sketch=bool(config.belief_use_residual_sketch),
            checkpoint_path=None if checkpoint_path is None else Path(checkpoint_path),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict[str, Any],
        *,
        checkpoint_path: str | Path | None = None,
        env_name: str | None = None,
    ) -> "CrawlerRuntimeConfig":
        """Read runtime probe settings from a saved probe-policy checkpoint."""
        resolved_env_name = str(env_name or checkpoint.get("env_name") or CONTINUOUS_CARTPOLE_NAME)
        base = cls.from_env(resolved_env_name, checkpoint_path=checkpoint_path)
        return replace(
            base,
            action_bins=int(checkpoint.get("action_bins", base.action_bins)),
            window_size=int(checkpoint.get("window_size", base.window_size)),
            base_probe_episodes=int(
                checkpoint.get("base_probe_episodes", base.base_probe_episodes)
            ),
            max_probe_episodes=int(
                checkpoint.get("max_probe_episodes", base.max_probe_episodes)
            ),
            probe_adaptive_budget=bool(
                checkpoint.get("probe_adaptive_budget", base.probe_adaptive_budget)
            ),
            belief_bits_per_dim=int(
                checkpoint.get("belief_bits_per_dim", base.belief_bits_per_dim)
            ),
            belief_use_residual_sketch=bool(
                checkpoint.get(
                    "belief_use_residual_sketch",
                    base.belief_use_residual_sketch,
                )
            ),
        )


@dataclass(frozen=True)
class CrawlerExpressionResult:
    """One completed crawler pass and the algorithm-facing payload it produced."""

    env_expression: EnvExpression
    controller_context: ControllerBeliefContext | None
    step_result: LegacyCrawlerStepResult
    belief: np.ndarray
    probe_windows: tuple[dict[str, object], ...]
    probe_count: int
    probe_steps: int
    metadata: dict[str, Any]

    @property
    def vector(self) -> np.ndarray:
        """Return the solver-facing env-expression vector."""
        return np.asarray(self.env_expression.vector, dtype=np.float32).reshape(-1)

    @property
    def ready(self) -> bool:
        """Whether the crawler thinks this expression is ready for control."""
        return bool(self.env_expression.ready)

    @property
    def confidence(self) -> float:
        """Expression confidence on the crawler side."""
        return float(self.env_expression.confidence)

    @property
    def uncertainty(self) -> float:
        """Expression uncertainty scalar on the crawler side."""
        return float(self.env_expression.uncertainty_scalar)

    def as_algo_context(self) -> dict[str, Any]:
        """Return the explicit payload downstream algorithms should consume."""
        return {
            "env_expression": self.env_expression,
            "controller_context": self.controller_context,
            "crawler_step": self.step_result,
            "expression_vector": self.vector,
            "confidence": self.confidence,
            "ready": self.ready,
            "uncertainty": self.uncertainty,
            "probe_count": int(self.probe_count),
            "probe_steps": int(self.probe_steps),
            "metadata": dict(self.metadata),
        }

    def summary(self) -> str:
        """Return one compact Gym-style status line."""
        return (
            "Belief("
            f"ready={self.ready}, "
            f"confidence={self.confidence:.3f}, "
            f"uncertainty={self.uncertainty:.3f}, "
            f"dim={self.vector.shape[0]}, "
            f"probe_steps={self.probe_steps}"
            ")"
        )

    def __repr__(self) -> str:
        return self.summary()

    def feed(self, algo):
        """Feed this result into an algorithm object with an explicit crawler hook."""
        context = self.as_algo_context()
        if hasattr(algo, "with_crawler_context"):
            return algo.with_crawler_context(context)
        if hasattr(algo, "set_crawler_context"):
            algo.set_crawler_context(context)
            return algo
        if hasattr(algo, "with_env_expression"):
            return algo.with_env_expression(self.env_expression)
        if hasattr(algo, "set_env_expression"):
            algo.set_env_expression(self.env_expression)
            return algo
        if callable(algo):
            return algo(context)
        raise TypeError(
            "Algorithm does not expose with_crawler_context, set_crawler_context, "
            "with_env_expression, set_env_expression, or a callable context hook."
        )


class LatentCrawler:
    """A trained crawler bundle that can be run before any downstream algorithm."""

    def __init__(
        self,
        *,
        bundle: CrawlerModelBundle,
        config: CrawlerRuntimeConfig,
    ) -> None:
        self.bundle = bundle
        self.config = config

    @classmethod
    def from_bundle(
        cls,
        bundle: CrawlerModelBundle,
        *,
        config: CrawlerRuntimeConfig | None = None,
        env_name: str = CONTINUOUS_CARTPOLE_NAME,
    ) -> "LatentCrawler":
        """Wrap an already trained crawler bundle."""
        runtime_config = config or CrawlerRuntimeConfig.from_env(env_name)
        return cls(bundle=bundle, config=runtime_config)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        env=None,
        env_name: str | None = None,
        device: str | torch.device = "cpu",
    ) -> "LatentCrawler":
        """Load a trained crawler from a saved probe-policy checkpoint."""
        path = Path(checkpoint_path)
        checkpoint = load_runtime_checkpoint(path)
        resolved_env_name = str(env_name or checkpoint.get("env_name") or CONTINUOUS_CARTPOLE_NAME)
        config = CrawlerRuntimeConfig.from_checkpoint(
            checkpoint,
            checkpoint_path=path,
            env_name=resolved_env_name,
        )
        owned_env = env is None
        runtime_env = make_env(resolved_env_name) if owned_env else env
        try:
            action_values = get_action_values(
                runtime_env,
                int(config.action_bins),
                env_name=resolved_env_name,
            )
            action_vocab_size = _action_vocab_size(runtime_env, action_values)
            state_dim = _flat_observation_dim(runtime_env)
            bundle = load_crawler_bundle_from_checkpoint(
                checkpoint=checkpoint,
                state_dim=state_dim,
                action_vocab_size=action_vocab_size,
                device=torch.device(device),
            )
        finally:
            if owned_env and hasattr(runtime_env, "close"):
                runtime_env.close()
        return cls(bundle=bundle, config=config)

    def describe(self) -> str:
        """Return a short human-readable runtime summary."""
        path = self.config.checkpoint_path
        checkpoint_label = "uncheckpointed" if path is None else str(path)
        families = ", ".join(self.bundle.family_names) if self.bundle.family_names else "unlabeled"
        return (
            f"LatentCrawler(env={self.config.env_name}, "
            f"checkpoint={checkpoint_label}, "
            f"probes={self.config.base_probe_episodes}-{self.config.max_probe_episodes}, "
            f"families={families})"
        )

    def __call__(self, env=None, **kwargs) -> CrawlerExpressionResult:
        """Run the crawler with call syntax: result = crawler(env)."""
        return self.run(env, **kwargs)

    def run(
        self,
        env=None,
        *,
        seed: int | None = None,
        episode_physics=None,
        rng: np.random.Generator | None = None,
        action_values: np.ndarray | None = None,
        trace_writer=None,
        episode: int = 0,
    ) -> CrawlerExpressionResult:
        """Run the crawler into an environment and return its latent expression."""
        from .support_context import collect_support_context

        owned_env = env is None
        runtime_env = make_env(self.config.env_name) if owned_env else env
        runtime_rng = rng or np.random.default_rng(seed)
        if seed is not None and hasattr(runtime_env.action_space, "seed"):
            runtime_env.action_space.seed(int(seed))
        try:
            support_physics = (
                default_env_params(self.config.env_name, runtime_env)
                if episode_physics is None
                else episode_physics
            )
            support_action_values = (
                get_action_values(
                    runtime_env,
                    int(self.config.action_bins),
                    env_name=self.config.env_name,
                )
                if action_values is None
                else action_values
            )
            support = collect_support_context(
                probe_env=runtime_env,
                crawler_bundle=self.bundle,
                encoder=self.bundle.encoder,
                belief_aggregator=self.bundle.belief_aggregator,
                env_param_predictor=self.bundle.env_param_predictor,
                env_future_predictor=self.bundle.env_future_predictor,
                predictor=self.bundle.predictor,
                rng=runtime_rng,
                env_name=self.config.env_name,
                episode_physics=support_physics,
                action_values=support_action_values,
                window_size=int(self.config.window_size),
                base_probe_episodes=int(self.config.base_probe_episodes),
                max_probe_episodes=int(self.config.max_probe_episodes),
                probe_adaptive_budget=bool(self.config.probe_adaptive_budget),
                uncertainty_probe_threshold=float(self.config.uncertainty_probe_threshold),
                surprise_probe_threshold=float(self.config.surprise_probe_threshold),
                trace_writer=trace_writer,
                episode=int(episode),
                variant_label=self.config.variant_label,
                belief_bits_per_dim=int(self.config.belief_bits_per_dim),
                belief_use_residual_sketch=bool(self.config.belief_use_residual_sketch),
            )
        finally:
            if owned_env and hasattr(runtime_env, "close"):
                runtime_env.close()
        if support is None:
            raise RuntimeError("Crawler failed to collect a complete support context.")
        step_result = support["step_result"]
        return CrawlerExpressionResult(
            env_expression=step_result.env_expression,
            controller_context=step_result.controller_context,
            step_result=step_result,
            belief=np.asarray(support["belief"], dtype=np.float32),
            probe_windows=tuple(support.get("probe_windows", ())),
            probe_count=int(support.get("probe_count", 0)),
            probe_steps=int(support.get("probe_steps_total", 0)),
            metadata={
                "env_name": self.config.env_name,
                "checkpoint_path": None
                if self.config.checkpoint_path is None
                else str(self.config.checkpoint_path),
                "variant_label": self.config.variant_label,
                "probe_windows_total": int(support.get("probe_windows_total", 0)),
            },
        )

    def run_into(self, env, algo=None, **kwargs):
        """Run the crawler, then optionally hand the result into an algorithm."""
        result = self.run(env, **kwargs)
        if algo is None:
            return result
        return result.feed(algo)


def best_crawler(
    env_name: str | object | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    artifacts_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    env=None,
    device: str | torch.device = "cpu",
    bundle: CrawlerModelBundle | None = None,
    **config_overrides: Any,
) -> LatentCrawler:
    """Return the best available runtime crawler for a Gym-like environment."""
    source_env = None if env_name is None or isinstance(env_name, str) else env_name
    runtime_env = env if env is not None else source_env
    resolved_env_name = _resolve_env_name(env_name)
    if bundle is not None:
        config = CrawlerRuntimeConfig.from_env(resolved_env_name)
        config = _replace_config(config, config_overrides)
        return LatentCrawler.from_bundle(bundle, config=config, env_name=resolved_env_name)

    path = Path(checkpoint_path) if checkpoint_path is not None else latest_crawler_checkpoint(
        resolved_env_name,
        artifacts_dir=artifacts_dir,
    )
    if path is None:
        raise FileNotFoundError(
            f"No probe crawler checkpoint found for {resolved_env_name!r}. "
            "Pass checkpoint_path=... or train one with run(env, ppo(), profile='fast')."
        )
    crawler = LatentCrawler.from_checkpoint(
        path,
        env=runtime_env,
        env_name=resolved_env_name,
        device=device,
    )
    if config_overrides:
        crawler = LatentCrawler(
            bundle=crawler.bundle,
            config=_replace_config(crawler.config, config_overrides),
        )
    return crawler


def crawler_for(*args, **kwargs) -> LatentCrawler:
    """Alias for best_crawler, named for app composition code."""
    return best_crawler(*args, **kwargs)


def latest_crawler_checkpoint(
    env_name: str = CONTINUOUS_CARTPOLE_NAME,
    *,
    artifacts_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> Path | None:
    """Find the newest saved probe checkpoint that matches an environment."""
    artifact_path = Path(artifacts_dir)
    if not artifact_path.exists():
        return None
    candidates = _pattern_candidates(env_name, artifact_path)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return _latest_checkpoint_by_metadata(env_name, artifact_path)


def load_runtime_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load both current safe checkpoints and older local checkpoints."""
    try:
        return torch.load(Path(path), map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(Path(path), map_location="cpu", weights_only=False)


def _replace_config(
    config: CrawlerRuntimeConfig,
    overrides: dict[str, Any],
) -> CrawlerRuntimeConfig:
    allowed = set(CrawlerRuntimeConfig.__dataclass_fields__)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unknown crawler runtime config keys: {unknown}")
    return replace(config, **overrides)


def _pattern_candidates(env_name: str, artifact_path: Path) -> list[Path]:
    try:
        tag = build_experiment_config(env_name).benchmark_tag
    except ValueError:
        return []
    return list(artifact_path.glob(f"{tag}_seed_*_probe_ppo_checkpoint.pt"))


def _latest_checkpoint_by_metadata(env_name: str, artifact_path: Path) -> Path | None:
    matches: list[Path] = []
    for path in artifact_path.glob(CHECKPOINT_GLOB):
        try:
            checkpoint = load_runtime_checkpoint(path)
        except (OSError, RuntimeError, pickle.UnpicklingError):
            continue
        if str(checkpoint.get("env_name", "")) == str(env_name):
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _flat_observation_dim(env) -> int:
    shape = getattr(env.observation_space, "shape", None)
    if not shape:
        raise ValueError("Crawler runtime expects a vector observation space.")
    return int(np.prod(shape))


def _action_vocab_size(env, action_values: np.ndarray | None) -> int:
    if action_values is None:
        return int(env.action_space.n)
    return int(len(action_values))


def _resolve_env_name(env_name: str | object | None) -> str:
    if env_name is None:
        return CONTINUOUS_CARTPOLE_NAME
    if isinstance(env_name, str):
        return str(env_name)
    spec = getattr(env_name, "spec", None)
    spec_id = getattr(spec, "id", None)
    if isinstance(spec_id, str) and spec_id:
        return spec_id
    for attr_name in ("gym_id", "env_name"):
        value = getattr(env_name, attr_name, None)
        if isinstance(value, str) and value:
            return value
    return CONTINUOUS_CARTPOLE_NAME


__all__ = [
    "CrawlerExpressionResult",
    "CrawlerRuntimeConfig",
    "LatentCrawler",
    "best_crawler",
    "crawler_for",
    "latest_crawler_checkpoint",
    "load_runtime_checkpoint",
]
