"""Shared helpers for benchmark reporting and profile selection."""

from __future__ import annotations

import statistics
from typing import Iterable

import numpy as np

from .config import ExperimentConfig


def print_return_summary(name: str, returns):
    """Report short-horizon and medium-horizon return averages."""
    if not returns:
        print(f"{name}: not run")
        return
    print(
        f"{name}: "
        f"avg10={np.mean(returns[-10:]):.2f} | "
        f"avg50={np.mean(returns[-50:]):.2f}"
    )


def print_solve_summary(name: str, solves, unsolved_caps):
    """Summarize solve speed with an explicit per-run cap for unsolved seeds."""
    if not any(value is not None for value in solves) and not any(
        int(cap) > 0 for cap in unsolved_caps
    ):
        print(f"{name}: not run")
        return
    success_count = sum(1 for value in solves if value is not None)
    capped_solves = [
        value if value is not None else cap
        for value, cap in zip(solves, unsolved_caps)
    ]
    display_solves = [value if value is not None else -1 for value in solves]
    print(
        f"{name}: "
        f"success_rate={success_count}/{len(solves)} | "
        f"capped_median={statistics.median(capped_solves):.2f} | "
        f"capped_mean={sum(capped_solves) / len(capped_solves):.2f} | "
        f"solves={display_solves}"
    )


def matched_eval_summary_dict(summary) -> dict | None:
    """Convert one optional matched-eval dataclass into a plain dict."""
    if summary is None:
        return None
    return summary.to_dict()


def print_matched_eval_summary(name: str, summary: dict | None):
    """Report one matched controller ablation without pretending it is a solve run."""
    if not summary:
        print(f"{name}: not run")
        return
    print(
        f"{name}: "
        f"mean_return={float(summary['mean_return']):.2f} | "
        f"mean_total_steps={float(summary['mean_total_env_steps']):.2f} | "
        f"solved={int(summary['solved_count'])}/{int(summary['fixture_count'])}"
    )


def aggregate_matched_eval_summary(results: list[dict], key: str) -> dict | None:
    """Aggregate one matched-eval summary key across seeds."""
    if len(results) == 1:
        return results[0].get(key)
    available = [item[key] for item in results if item.get(key) is not None]
    if not available:
        return None
    return {
        "mean_return": float(np.mean([float(item["mean_return"]) for item in available])),
        "mean_total_env_steps": float(
            np.mean([float(item["mean_total_env_steps"]) for item in available])
        ),
        "solved_count": int(sum(int(item["solved_count"]) for item in available)),
        "fixture_count": int(sum(int(item["fixture_count"]) for item in available)),
    }


def solve_rank_tuple(episode_solve: int | None, step_solve: int | None) -> tuple[float, float]:
    """Rank solve outcomes with episode count first and env steps second."""
    return (
        float("inf") if episode_solve is None else float(episode_solve),
        float("inf") if step_solve is None else float(step_solve),
    )


def clip_unit_interval(value: float | None) -> float:
    """Clamp one optional scalar into a stable [0, 1] fraction."""
    if value is None:
        return 0.0
    return float(np.clip(float(value), 0.0, 1.0))


def finite_values(values: Iterable[float | None]) -> list[float]:
    """Keep only finite numeric values from one optional sequence."""
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            cleaned.append(numeric)
    return cleaned


def optional_mean(values: Iterable[float | None], default: float = 0.0) -> float:
    """Average one optional metric sequence with a predictable empty fallback."""
    cleaned = finite_values(values)
    if not cleaned:
        return float(default)
    return float(np.mean(np.asarray(cleaned, dtype=np.float32)))


def optional_median(values: Iterable[float | None], default: float = 0.0) -> float:
    """Take the median of one optional metric sequence with a predictable fallback."""
    cleaned = finite_values(values)
    if not cleaned:
        return float(default)
    return float(np.median(np.asarray(cleaned, dtype=np.float32)))


def summarize_capped_solve_stats(
    solves: Iterable[int | None],
    unsolved_caps: Iterable[int],
) -> dict[str, float]:
    """Mirror the console solve-summary math for gate computation and payloads."""
    solve_list = list(solves)
    cap_list = [int(cap) for cap in unsolved_caps]
    if not solve_list:
        return {
            "success_rate": 0.0,
            "capped_median": 0.0,
            "capped_mean": 0.0,
        }
    success_count = sum(1 for value in solve_list if value is not None)
    capped_values = np.asarray(
        [
            float(value if value is not None else cap)
            for value, cap in zip(solve_list, cap_list)
        ],
        dtype=np.float32,
    )
    return {
        "success_rate": float(success_count) / float(max(len(solve_list), 1)),
        "capped_median": float(np.median(capped_values)) if capped_values.size else 0.0,
        "capped_mean": float(np.mean(capped_values)) if capped_values.size else 0.0,
    }


def compute_belief_progress_index(
    *,
    mechanics_fit: float | None,
    neighbor_alignment: float | None,
    split_retrieval: float | None,
    heldout_probe_error: float | None,
    uncert_error_corr: float | None,
    probe_leakage: float | None,
) -> float:
    """Collapse the current latent-health signals into one fast-loop score."""
    mechanics_fit_score = clip_unit_interval(mechanics_fit)
    neighbor_alignment_score = clip_unit_interval(neighbor_alignment)
    split_retrieval_score = clip_unit_interval(split_retrieval)
    heldout_probe_score = 1.0 - clip_unit_interval(heldout_probe_error)
    uncertainty_alignment_score = clip_unit_interval(max(float(uncert_error_corr or 0.0), 0.0))
    probe_leakage_score = 1.0 - clip_unit_interval(probe_leakage)
    return float(
        0.30 * mechanics_fit_score
        + 0.20 * neighbor_alignment_score
        + 0.15 * split_retrieval_score
        + 0.15 * heldout_probe_score
        + 0.10 * uncertainty_alignment_score
        + 0.10 * probe_leakage_score
    )


def evaluate_latent_win_gate(
    *,
    benchmark_profile: str,
    seed_count: int,
    baseline_episode_solves: Iterable[int | None],
    baseline_completed_episodes: Iterable[int],
    probe_episode_solves: Iterable[int | None],
    probe_completed_episodes: Iterable[int],
    baseline_step_solves: Iterable[int | None],
    baseline_total_env_steps: Iterable[int],
    probe_step_solves: Iterable[int | None],
    probe_total_env_steps: Iterable[int],
    probe_env_expression_delta: Iterable[float | None],
    probe_ready_fraction: Iterable[float | None],
    probe_muted_fraction: Iterable[float | None],
    latent_mechanics_fit: Iterable[float | None],
    latent_neighbor_alignment: Iterable[float | None],
    latent_split_retrieval: Iterable[float | None],
    latent_gap_ratio: Iterable[float | None],
    latent_probe_leakage: Iterable[float | None],
    latent_uncert_error_corr: Iterable[float | None],
    full_system_state_only_ablation_delta: Iterable[float | None],
    full_system_zero_context_ablation_delta: Iterable[float | None],
    full_system_shuffled_context_ablation_delta: Iterable[float | None],
    full_system_stale_context_ablation_delta: Iterable[float | None],
) -> dict[str, object]:
    """Decide whether the benchmark has genuinely cracked latent utility."""
    baseline_episode_stats = summarize_capped_solve_stats(
        baseline_episode_solves,
        baseline_completed_episodes,
    )
    probe_episode_stats = summarize_capped_solve_stats(
        probe_episode_solves,
        probe_completed_episodes,
    )
    baseline_step_stats = summarize_capped_solve_stats(
        baseline_step_solves,
        baseline_total_env_steps,
    )
    probe_step_stats = summarize_capped_solve_stats(
        probe_step_solves,
        probe_total_env_steps,
    )

    probe_delta_median = optional_median(probe_env_expression_delta, default=0.0)
    probe_ready_mean = optional_mean(probe_ready_fraction, default=0.0)
    probe_muted_mean = optional_mean(probe_muted_fraction, default=0.0)
    mechanics_fit_median = optional_median(latent_mechanics_fit, default=0.0)
    neighbor_alignment_median = optional_median(latent_neighbor_alignment, default=0.0)
    split_retrieval_median = optional_median(latent_split_retrieval, default=0.0)
    gap_ratio_median = optional_median(latent_gap_ratio, default=float("inf"))
    probe_leakage_median = optional_median(latent_probe_leakage, default=1.0)
    uncert_corr_median = optional_median(latent_uncert_error_corr, default=0.0)
    state_only_delta_median = optional_median(full_system_state_only_ablation_delta, default=float("-inf"))
    zero_delta_median = optional_median(full_system_zero_context_ablation_delta, default=float("-inf"))
    shuffled_delta_median = optional_median(full_system_shuffled_context_ablation_delta, default=float("-inf"))
    stale_delta_median = optional_median(full_system_stale_context_ablation_delta, default=float("-inf"))

    checks = {
        "full_benchmark": str(benchmark_profile) == "full" and int(seed_count) >= 5,
        "probe_success_rate": probe_episode_stats["success_rate"] >= baseline_episode_stats["success_rate"],
        "probe_episode_speed": probe_episode_stats["capped_median"] <= 0.90 * baseline_episode_stats["capped_median"],
        "probe_step_speed": probe_step_stats["capped_median"] < baseline_step_stats["capped_median"],
        "env_expression_beats_noexpr": probe_delta_median > 0.0,
        "probe_ready_fraction": probe_ready_mean >= 0.50,
        "probe_muted_fraction": probe_muted_mean <= 0.50,
        "mechanics_fit": mechanics_fit_median >= 0.50,
        "neighbor_alignment": neighbor_alignment_median >= 0.25,
        "split_retrieval": split_retrieval_median >= 0.20,
        "gap_ratio": gap_ratio_median <= 1.0,
        "probe_leakage": probe_leakage_median <= 0.15,
        "uncert_error_corr": uncert_corr_median >= 0.20,
        "full_system_state_only_delta": state_only_delta_median >= 50.0,
        "full_system_zero_delta": zero_delta_median >= 50.0,
        "full_system_shuffled_delta": shuffled_delta_median >= 50.0,
        "full_system_stale_delta": stale_delta_median >= 50.0,
    }
    failure_reason_map = {
        "full_benchmark": "benchmark_not_full",
        "probe_success_rate": "probe_success_rate_below_baseline",
        "probe_episode_speed": "probe_episode_speed_below_target",
        "probe_step_speed": "probe_step_cost_below_target",
        "env_expression_beats_noexpr": "env_expression_not_beating_noexpr",
        "probe_ready_fraction": "probe_ready_fraction_too_low",
        "probe_muted_fraction": "probe_muted_fraction_too_high",
        "mechanics_fit": "mechanics_fit_too_low",
        "neighbor_alignment": "neighbor_alignment_too_low",
        "split_retrieval": "split_retrieval_too_low",
        "gap_ratio": "gap_ratio_too_high",
        "probe_leakage": "probe_leakage_too_high",
        "uncert_error_corr": "uncertainty_alignment_too_low",
        "full_system_state_only_delta": "full_system_state_only_delta_too_low",
        "full_system_zero_delta": "full_system_zero_delta_too_low",
        "full_system_shuffled_delta": "full_system_shuffled_delta_too_low",
        "full_system_stale_delta": "full_system_stale_delta_too_low",
    }
    failure_reasons = [
        failure_reason_map[name]
        for name, passed in checks.items()
        if not bool(passed)
    ]
    return {
        "pass": bool(all(checks.values())),
        "failure_reasons": failure_reasons,
        "checks": checks,
        "metrics": {
            "baseline_episode_success_rate": float(baseline_episode_stats["success_rate"]),
            "probe_episode_success_rate": float(probe_episode_stats["success_rate"]),
            "baseline_episode_median": float(baseline_episode_stats["capped_median"]),
            "probe_episode_median": float(probe_episode_stats["capped_median"]),
            "baseline_step_median": float(baseline_step_stats["capped_median"]),
            "probe_step_median": float(probe_step_stats["capped_median"]),
            "probe_env_expression_delta_median": float(probe_delta_median),
            "probe_ready_fraction_mean": float(probe_ready_mean),
            "probe_muted_fraction_mean": float(probe_muted_mean),
            "mechanics_fit_median": float(mechanics_fit_median),
            "neighbor_alignment_median": float(neighbor_alignment_median),
            "split_retrieval_median": float(split_retrieval_median),
            "gap_ratio_median": float(gap_ratio_median),
            "probe_leakage_median": float(probe_leakage_median),
            "uncert_error_corr_median": float(uncert_corr_median),
            "full_system_state_only_delta_median": float(state_only_delta_median),
            "full_system_zero_delta_median": float(zero_delta_median),
            "full_system_shuffled_delta_median": float(shuffled_delta_median),
            "full_system_stale_delta_median": float(stale_delta_median),
        },
    }


def classify_probe_run(
    *,
    baseline_episode: int | None,
    baseline_steps: int | None,
    probe_episode: int | None,
    probe_steps: int | None,
    probe_no_expression_episode: int | None,
    probe_no_expression_steps: int | None,
    probe_env_expression_delta: float | None = None,
    probe_fair_ready_handoff_fraction: float | None = None,
    probe_fair_expression_enabled_fraction: float | None = None,
) -> str:
    """Classify whether a run is latent-driven or mostly a protocol/controller win."""
    baseline_rank = solve_rank_tuple(baseline_episode, baseline_steps)
    probe_rank = solve_rank_tuple(probe_episode, probe_steps)
    probe_no_expression_rank = solve_rank_tuple(
        probe_no_expression_episode,
        probe_no_expression_steps,
    )
    fair_ready_fraction = (
        None
        if probe_fair_ready_handoff_fraction is None
        else float(probe_fair_ready_handoff_fraction)
    )
    fair_expression_enabled_fraction = (
        None
        if probe_fair_expression_enabled_fraction is None
        else float(probe_fair_expression_enabled_fraction)
    )
    latent_was_used = True
    if fair_ready_fraction is not None and fair_ready_fraction <= 0.0:
        latent_was_used = False
    if (
        fair_expression_enabled_fraction is not None
        and fair_expression_enabled_fraction <= 0.0
    ):
        latent_was_used = False
    if not latent_was_used:
        if probe_rank < baseline_rank:
            return "protocol_win"
        return "controller_compensation"
    if probe_no_expression_episode is None and probe_no_expression_steps is None:
        if (
            probe_rank < baseline_rank
            and probe_env_expression_delta is not None
            and float(probe_env_expression_delta) > 0.0
        ):
            return "latent_win"
        if probe_rank < baseline_rank:
            return "protocol_win"
        return "controller_compensation"
    if probe_rank < baseline_rank and probe_rank < probe_no_expression_rank:
        return "latent_win"
    if probe_rank < baseline_rank:
        return "protocol_win"
    return "controller_compensation"


def probe_strict_usage_status(
    probe_fair_expression_enabled_fraction: float | None,
) -> str:
    """Bucket strict fair-mode usage into one compact honesty label."""
    if probe_fair_expression_enabled_fraction is None:
        return "unused"
    enabled_fraction = float(probe_fair_expression_enabled_fraction)
    if enabled_fraction <= 0.0:
        return "unused"
    if enabled_fraction < 0.50:
        return "intermittent"
    return "active"


def resolve_benchmark_profile(config: ExperimentConfig) -> str:
    """Return one validated benchmark profile string."""
    profile = str(config.benchmark_profile)
    if profile not in {"fast", "full", "archived_planner"}:
        raise ValueError(f"Unsupported benchmark profile: {profile}")
    return profile


def default_seeds_for_profile(profile: str) -> list[int]:
    """Choose the default seed list for this profile."""
    return [0, 1] if str(profile) == "fast" else [0, 1, 2, 3, 4]


def solve_eval_episodes_for_profile(config: ExperimentConfig) -> int:
    """Keep the quick profile cheap while preserving the configured full eval depth."""
    if resolve_benchmark_profile(config) == "fast":
        return 1
    return int(config.solve_eval_episodes)


def benchmark_profile_flags(profile: str) -> dict[str, bool]:
    """Centralize which variants each profile is allowed to run."""
    profile = str(profile)
    return {
        "run_probe_shadow": profile != "fast",
        "run_probe_no_expression_training": profile == "full",
        "run_belief_controller": profile in {"fast", "full"},
        "run_belief_controller_oracle": profile == "full",
        "run_sim_fanout": profile in {"fast", "full"},
        "run_archived_planner": profile == "archived_planner",
    }


def optional_solve_rank(value: int | None) -> int:
    """Encode an optional solve integer for artifact storage."""
    return -1 if value is None else int(value)


def full_system_display_label(result) -> str:
    """Map one full-system run result to a human-readable benchmark label."""
    style = "" if result is None else str(getattr(result, "controller_style", ""))
    if "belief_planner" in style:
        return "Belief-planner"
    if "sim_fanout" in style:
        return "Sim-fanout"
    return "Belief-controller"
