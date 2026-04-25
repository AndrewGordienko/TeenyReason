"""Side-by-side sample-efficiency suite for RL, images, and language."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..algos import (
    ImageProbeBenchmarkConsumer,
    LanguageProbeBenchmarkConsumer,
    PPOBenchmarkConsumer,
)
from ..envs import CONTINUOUS_CARTPOLE_NAME, CONTINUOUS_LUNAR_LANDER_NAME
from ..multidomain import ImageProbeBenchmarkConfig, LanguageProbeBenchmarkConfig
from ..recipes import build_benchmark_recipe, build_language_recipe, build_mnist_recipe


@dataclass(frozen=True)
class MultidomainSuiteConfig:
    """Configuration for the cross-domain sample-efficiency suite."""

    artifact_dir: Path = Path("artifacts")
    rl_envs: tuple[str, ...] = (CONTINUOUS_CARTPOLE_NAME, CONTINUOUS_LUNAR_LANDER_NAME)
    rl_seeds: tuple[int, ...] = (0, 1, 2)
    lunar_lander_randomize_physics: bool = True
    run_image_benchmark: bool = True
    run_language_benchmark: bool = True
    image: ImageProbeBenchmarkConfig = field(default_factory=ImageProbeBenchmarkConfig)
    language: LanguageProbeBenchmarkConfig = field(default_factory=LanguageProbeBenchmarkConfig)


def summarize_rows(
    rows: list[dict[str, float | int]],
    *,
    baseline_key: str,
    probe_key: str,
) -> dict[str, float]:
    """Compute a compact mean summary from a small benchmark curve."""
    if not rows:
        return {
            "baseline_mean": 0.0,
            "probe_mean": 0.0,
            "probe_minus_baseline": 0.0,
        }
    baseline_mean = sum(float(row[baseline_key]) for row in rows) / float(len(rows))
    probe_mean = sum(float(row[probe_key]) for row in rows) / float(len(rows))
    return {
        "baseline_mean": baseline_mean,
        "probe_mean": probe_mean,
        "probe_minus_baseline": probe_mean - baseline_mean,
    }


def run_multidomain_suite(
    config: MultidomainSuiteConfig | None = None,
) -> dict[str, object]:
    """Run the current RL benchmark plus image and language sample-efficiency checks."""
    config = config or MultidomainSuiteConfig()
    config.artifact_dir.mkdir(exist_ok=True)
    results: dict[str, object] = {
        "rl": {},
        "image": None,
        "language": None,
    }

    print("\n=== Multi-Domain Suite | RL ===")
    rl_consumer = PPOBenchmarkConsumer()
    for env_name in config.rl_envs:
        override = None
        if env_name == CONTINUOUS_LUNAR_LANDER_NAME and config.lunar_lander_randomize_physics:
            override = {"randomize_physics": True}
        print(f"\n--- RL benchmark | env={env_name} ---")
        rl_recipe = build_benchmark_recipe(env_name)
        rl_result = rl_consumer.run(
            rl_recipe,
            seeds=list(config.rl_seeds),
            config_override=override,
        )
        results["rl"][env_name] = rl_result

    if config.run_image_benchmark:
        print("\n=== Multi-Domain Suite | Images ===")
        image_recipe = build_mnist_recipe(config.image)
        image_result = ImageProbeBenchmarkConsumer().run(image_recipe)
        image_summary = summarize_rows(
            image_result["rows"],
            baseline_key="baseline_accuracy",
            probe_key="probe_accuracy",
        )
        print(
            "MNIST probe benchmark | "
            f"baseline_mean_acc={image_summary['baseline_mean']:.4f} | "
            f"probe_mean_acc={image_summary['probe_mean']:.4f} | "
            f"delta={image_summary['probe_minus_baseline']:.4f}"
        )
        results["image"] = image_result

    if config.run_language_benchmark:
        print("\n=== Multi-Domain Suite | Language ===")
        language_recipe = build_language_recipe(config.language)
        language_result = LanguageProbeBenchmarkConsumer().run(language_recipe)
        language_summary = summarize_rows(
            language_result["rows"],
            baseline_key="baseline_bpc",
            probe_key="probe_bpc",
        )
        print(
            "Shakespeare probe benchmark | "
            f"baseline_mean_bpc={language_summary['baseline_mean']:.4f} | "
            f"probe_mean_bpc={language_summary['probe_mean']:.4f} | "
            f"delta={language_summary['probe_minus_baseline']:.4f}"
        )
        results["language"] = language_result

    summary_path = config.artifact_dir / "multidomain_probe_suite_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved multi-domain suite summary to {summary_path}")
    return results
