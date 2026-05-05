"""Shared helpers for benchmark reporting and profile selection."""

from __future__ import annotations

import statistics
from typing import Iterable

import numpy as np

from ..config import ExperimentConfig


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


def compute_system_id_progress_index(
    *,
    trusted: bool,
    validation_top1: float | None,
    validation_margin: float | None,
    particle_entropy_norm: float | None,
    particle_ess_ratio: float | None,
    particle_leaveout_shift: float | None,
) -> float:
    """Score particle sysid health without reusing legacy latent geometry."""
    trust_score = 1.0 if trusted else 0.0
    top1_score = clip_unit_interval(validation_top1)
    margin_score = clip_unit_interval(float(validation_margin or 0.0) / 0.25)
    sharpness_score = 1.0 - clip_unit_interval(particle_entropy_norm)
    concentration_score = 1.0 - clip_unit_interval(particle_ess_ratio)
    stability_score = 1.0 - clip_unit_interval(float(particle_leaveout_shift or 0.0) / 0.55)
    return float(
        0.25 * trust_score
        + 0.20 * top1_score
        + 0.15 * margin_score
        + 0.15 * sharpness_score
        + 0.10 * concentration_score
        + 0.15 * stability_score
    )


def belief_source_from_mode(belief_mode: str, *, context_source: str | None = None) -> str:
    """Map internal belief modes and eval arms onto dashboard source labels."""
    if context_source in {"oracle", "zero", "shuffled", "stale"}:
        return str(context_source)
    if str(belief_mode) == "particle_sysid":
        return "sysid"
    return "learned"


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
    probe_expression_enabled_fraction: Iterable[float | None] | None = None,
    probe_no_expression_available: bool | None = True,
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
    probe_enabled_values = [] if probe_expression_enabled_fraction is None else probe_expression_enabled_fraction
    probe_enabled_mean = optional_mean(probe_enabled_values, default=0.0)
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
        "probe_no_expression_available": bool(probe_no_expression_available),
        "probe_success_rate": probe_episode_stats["success_rate"] >= baseline_episode_stats["success_rate"],
        "probe_episode_speed": probe_episode_stats["capped_median"] <= 0.90 * baseline_episode_stats["capped_median"],
        "probe_step_speed": probe_step_stats["capped_median"] < baseline_step_stats["capped_median"],
        "env_expression_beats_noexpr": probe_delta_median > 0.0,
        "probe_ready_fraction": probe_ready_mean >= 0.50,
        "probe_expression_enabled_fraction": probe_enabled_mean >= 0.20,
        "probe_muted_fraction": probe_muted_mean <= 0.50,
        "mechanics_fit": mechanics_fit_median >= 0.60,
        "neighbor_alignment": neighbor_alignment_median >= 0.20,
        "split_retrieval": split_retrieval_median >= (0.45 if str(benchmark_profile) == "full" else 0.30),
        "gap_ratio": gap_ratio_median <= 1.0,
        "probe_leakage": probe_leakage_median <= 0.15,
        "uncert_error_corr": uncert_corr_median >= 0.30,
        "full_system_state_only_delta": state_only_delta_median >= 50.0,
        "full_system_zero_delta": zero_delta_median >= 50.0,
        "full_system_shuffled_delta": shuffled_delta_median >= 50.0,
        "full_system_stale_delta": stale_delta_median >= 50.0,
    }
    failure_reason_map = {
        "full_benchmark": "benchmark_not_full",
        "probe_no_expression_available": "probe_no_expression_missing",
        "probe_success_rate": "probe_success_rate_below_baseline",
        "probe_episode_speed": "probe_episode_speed_below_target",
        "probe_step_speed": "probe_step_cost_below_target",
        "env_expression_beats_noexpr": "env_expression_not_beating_noexpr",
        "probe_ready_fraction": "probe_ready_fraction_too_low",
        "probe_expression_enabled_fraction": "probe_strict_enabled_fraction_too_low",
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
            "probe_expression_enabled_fraction_mean": float(probe_enabled_mean),
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


def evaluate_representation_gate(
    *,
    paired_split_top1: float,
    cross_split_top1: float,
    neighbor_alignment: float,
    paired_gap_ratio: float,
    belief_norm_std: float,
    nearest_between_median: float,
    min_paired_top1: float = 0.20,
    min_cross_top1: float = 0.01,
    min_neighbor_alignment: float = 0.20,
    max_paired_gap_ratio: float = 1.0,
    min_belief_norm_std: float = 1e-3,
    min_nearest_between: float = 1e-4,
) -> dict[str, object]:
    """Gate downstream control on learned belief geometry, not particle sysid."""
    metrics = {
        "paired_split_top1": float(paired_split_top1),
        "cross_split_top1": float(cross_split_top1),
        "neighbor_alignment": float(neighbor_alignment),
        "paired_gap_ratio": float(paired_gap_ratio),
        "belief_norm_std": float(belief_norm_std),
        "nearest_between_median": float(nearest_between_median),
    }
    thresholds = {
        "min_paired_top1": float(min_paired_top1),
        "min_cross_top1": float(min_cross_top1),
        "min_neighbor_alignment": float(min_neighbor_alignment),
        "max_paired_gap_ratio": float(max_paired_gap_ratio),
        "min_belief_norm_std": float(min_belief_norm_std),
        "min_nearest_between": float(min_nearest_between),
    }
    checks = {
        "paired_split_top1": metrics["paired_split_top1"] >= thresholds["min_paired_top1"],
        "cross_split_top1": metrics["cross_split_top1"] >= thresholds["min_cross_top1"],
        "neighbor_alignment": metrics["neighbor_alignment"] >= thresholds["min_neighbor_alignment"],
        "paired_gap_ratio": metrics["paired_gap_ratio"] <= thresholds["max_paired_gap_ratio"],
        "belief_norm_std": metrics["belief_norm_std"] >= thresholds["min_belief_norm_std"],
        "nearest_between_median": metrics["nearest_between_median"] >= thresholds["min_nearest_between"],
    }
    failure_reason_map = {
        "paired_split_top1": "paired_split_retrieval_too_low",
        "cross_split_top1": "cross_split_retrieval_too_low",
        "neighbor_alignment": "neighbor_alignment_too_low",
        "paired_gap_ratio": "paired_gap_ratio_too_high",
        "belief_norm_std": "belief_norm_std_too_low",
        "nearest_between_median": "nearest_between_too_low",
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
        "metrics": metrics,
        "thresholds": thresholds,
    }


def apply_system_id_representation_override(
    representation_gate: dict[str, object],
    *,
    belief_mode: str,
    sysid_trusted: bool,
    sysid_validation_top1: float,
    sysid_validation_margin: float,
    min_top1: float = 0.75,
    min_margin: float = 1.0,
) -> dict[str, object]:
    """Allow trusted particle sysid to satisfy the operational representation gate."""
    gate = dict(representation_gate)
    metrics = dict(gate.get("metrics", {}))
    checks = dict(gate.get("checks", {}))
    thresholds = dict(gate.get("thresholds", {}))
    latent_failure_reasons = list(gate.get("failure_reasons", []))

    metrics["sysid_validation_top1"] = float(sysid_validation_top1)
    metrics["sysid_validation_margin"] = float(sysid_validation_margin)
    metrics["sysid_trusted"] = 1.0 if bool(sysid_trusted) else 0.0
    thresholds["min_sysid_top1"] = float(min_top1)
    thresholds["min_sysid_margin"] = float(min_margin)

    sysid_ok = (
        str(belief_mode) == "particle_sysid"
        and bool(sysid_trusted)
        and float(sysid_validation_top1) >= float(min_top1)
        and float(sysid_validation_margin) >= float(min_margin)
    )
    checks["trusted_particle_sysid"] = bool(sysid_ok)
    gate["metrics"] = metrics
    gate["checks"] = checks
    gate["thresholds"] = thresholds
    gate["latent_pass"] = bool(gate.get("pass", False))
    gate["latent_failure_reasons"] = latent_failure_reasons
    gate["override_reason"] = ""
    if sysid_ok and not bool(gate.get("pass", False)):
        gate["pass"] = True
        gate["failure_reasons"] = []
        gate["override_reason"] = "trusted_particle_sysid"
    return gate


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
    full_system_zero_context_ablation_delta: float | None = None,
    full_system_shuffled_context_ablation_delta: float | None = None,
    full_system_stale_context_ablation_delta: float | None = None,
    benchmark_profile: str | None = None,
    seed_count: int | None = None,
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
        and fair_expression_enabled_fraction < 0.20
    ):
        latent_was_used = False
    if not latent_was_used:
        if probe_rank < baseline_rank:
            return "protocol_win"
        return "controller_compensation"
    if probe_no_expression_episode is None and probe_no_expression_steps is None:
        if probe_rank < baseline_rank:
            return "protocol_win"
        return "controller_compensation"
    context_deltas = [
        full_system_zero_context_ablation_delta,
        full_system_shuffled_context_ablation_delta,
        full_system_stale_context_ablation_delta,
    ]
    context_identity_matters = all(
        value is not None and float(value) > 0.0
        for value in context_deltas
    )
    expression_beats_noexpr = probe_rank < probe_no_expression_rank
    if probe_env_expression_delta is not None:
        expression_beats_noexpr = expression_beats_noexpr and float(probe_env_expression_delta) > 0.0
    full_benchmark = True
    if benchmark_profile is not None:
        full_benchmark = str(benchmark_profile) == "full" and int(seed_count or 0) >= 5
    if (
        probe_rank < baseline_rank
        and expression_beats_noexpr
        and context_identity_matters
        and full_benchmark
    ):
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
        "run_probe_no_expression_training": profile in {"fast", "full"},
        "run_belief_controller": False,
        "run_belief_controller_oracle": False,
        "run_sim_fanout": False,
        "run_archived_planner": False,
    }


def optional_solve_rank(value: int | None) -> int:
    """Encode an optional solve integer for artifact storage."""
    return -1 if value is None else int(value)
