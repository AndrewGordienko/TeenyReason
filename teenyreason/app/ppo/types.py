"""Result types for PPO comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...envs import (
    BIPEDAL_WALKER_NAME,
    CONTINUOUS_CARTPOLE_NAME,
    CONTINUOUS_LUNAR_LANDER_NAME,
)


DEFAULT_COMPARISON_ENVS = (
    CONTINUOUS_CARTPOLE_NAME,
    CONTINUOUS_LUNAR_LANDER_NAME,
    BIPEDAL_WALKER_NAME,
)

GYM_SOLVE_AVG_WINDOW = 10
GYM_SOLVE_GRACE_WINDOWS = 3
GYM_SOLVED_RETURNS = {
    CONTINUOUS_CARTPOLE_NAME: 475.0,
    CONTINUOUS_LUNAR_LANDER_NAME: 200.0,
    BIPEDAL_WALKER_NAME: 300.0,
}


@dataclass(frozen=True)
class PPOComparisonSeedResult:
    """One baseline-vs-probe comparison for a single env seed."""

    seed: int
    encoder_probe_steps: int
    baseline_best_return: float
    baseline_best_episode: int | None
    baseline_best_env_steps: int | None
    baseline_solved_episode: int | None
    baseline_solved_env_steps: int | None
    baseline_total_env_steps: int
    probe_best_return: float
    probe_best_episode: int | None
    probe_best_env_steps: int | None
    probe_best_env_steps_with_encoder: int | None
    probe_solved_episode: int | None
    probe_solved_env_steps: int | None
    probe_solved_env_steps_with_encoder: int | None
    probe_total_env_steps: int
    probe_total_env_steps_with_encoder: int


@dataclass(frozen=True)
class PPOComparisonEnvResult:
    """All comparison seeds for one environment."""

    env_name: str
    benchmark_tag: str
    profile: str
    solved_return: float
    solve_avg_window: int
    seed_results: tuple[PPOComparisonSeedResult, ...]


@dataclass(frozen=True)
class PPOComparisonSuiteResult:
    """Result object returned by the live comparison runner."""

    env_results: tuple[PPOComparisonEnvResult, ...]
    summary_path: Path
