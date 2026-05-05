"""Research metric summaries for dashboard benchmark artifacts."""

from __future__ import annotations

import numpy as np


def finite_values(values: np.ndarray | list | tuple) -> list[float]:
    """Return finite numeric values from a dashboard artifact field."""
    result: list[float] = []
    for item in np.asarray(values).reshape(-1).tolist():
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            result.append(value)
    return result


def finite_nonnegative_values(values: np.ndarray | list | tuple) -> list[float]:
    """Return finite non-negative values, filtering solve sentinel values."""
    return [value for value in finite_values(values) if value >= 0.0]


def median_or_none(values: np.ndarray | list | tuple) -> float | None:
    """Return a JSON-friendly median when at least one finite value exists."""
    finite = finite_values(values)
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=np.float32)))


def median_nonnegative_or_none(values: np.ndarray | list | tuple) -> float | None:
    """Return a JSON-friendly median after filtering negative sentinels."""
    finite = finite_nonnegative_values(values)
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=np.float32)))


def mean_nonnegative_or_none(values: np.ndarray | list | tuple) -> float | None:
    """Return a JSON-friendly mean after filtering negative sentinels."""
    finite = finite_nonnegative_values(values)
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float32)))


def median_savings_or_none(
    reference_values: np.ndarray | list | tuple,
    candidate_values: np.ndarray | list | tuple,
) -> float | None:
    """Positive values mean the candidate used fewer samples than reference."""
    reference = median_nonnegative_or_none(reference_values)
    candidate = median_nonnegative_or_none(candidate_values)
    if reference is None or candidate is None:
        return None
    return float(reference - candidate)


def sample_savings_or_none(reference_value: int, candidate_value: int) -> int | None:
    """Positive values mean the candidate used fewer samples than reference."""
    if reference_value < 0 or candidate_value < 0:
        return None
    return int(reference_value - candidate_value)


def median_regret_or_none(
    candidate_values: np.ndarray | list | tuple,
    ceiling_values: np.ndarray | list | tuple,
) -> float | None:
    """Positive values mean the candidate spent more samples than the ceiling."""
    candidate = median_nonnegative_or_none(candidate_values)
    ceiling = median_nonnegative_or_none(ceiling_values)
    if candidate is None or ceiling is None:
        return None
    return float(candidate - ceiling)


def median_ratio_or_none(
    numerators: np.ndarray | list | tuple,
    denominators: np.ndarray | list | tuple,
) -> float | None:
    """Return median finite ratio for aligned arrays with positive denominators."""
    numerator_array = np.asarray(numerators).reshape(-1)
    denominator_array = np.asarray(denominators).reshape(-1)
    ratios: list[float] = []
    for numerator, denominator in zip(numerator_array.tolist(), denominator_array.tolist()):
        try:
            numerator_value = float(numerator)
            denominator_value = float(denominator)
        except (TypeError, ValueError):
            continue
        if (
            np.isfinite(numerator_value)
            and np.isfinite(denominator_value)
            and numerator_value >= 0.0
            and denominator_value > 0.0
        ):
            ratios.append(numerator_value / denominator_value)
    if not ratios:
        return None
    return float(np.median(np.asarray(ratios, dtype=np.float32)))


def benchmark_research_arm_summary(
    episode_solves: np.ndarray,
    step_solves: np.ndarray,
    total_env_steps: np.ndarray,
    completed_episodes: np.ndarray,
) -> dict:
    """Summarize one regular-vs-crawler benchmark arm for research graphs."""
    run_count = max(
        int(np.asarray(episode_solves).size),
        int(np.asarray(step_solves).size),
        int(np.asarray(total_env_steps).size),
        int(np.asarray(completed_episodes).size),
    )
    episode_solve_count = len(finite_nonnegative_values(episode_solves))
    step_solve_count = len(finite_nonnegative_values(step_solves))
    solve_count = max(episode_solve_count, step_solve_count)
    return {
        "available": solve_count > 0 or bool(finite_nonnegative_values(total_env_steps)),
        "run_count": run_count,
        "solve_count": solve_count,
        "solve_rate": float(solve_count) / float(max(run_count, 1)),
        "solve_episodes_median": median_nonnegative_or_none(episode_solves),
        "solve_episodes_mean": mean_nonnegative_or_none(episode_solves),
        "solve_steps_median": median_nonnegative_or_none(step_solves),
        "solve_steps_mean": mean_nonnegative_or_none(step_solves),
        "total_env_steps_median": median_nonnegative_or_none(total_env_steps),
        "completed_episodes_median": median_nonnegative_or_none(completed_episodes),
    }


def research_benchmark_status(
    *,
    arms: dict[str, dict],
    deltas: dict[str, float | None],
    peak: dict[str, float | None],
    probe_cost: dict[str, float | None],
) -> list[dict[str, object]]:
    """Return the benchmark checklist the dashboard should make visible."""
    baseline_solve = arms["baseline"]["solve_steps_median"] is not None
    probe_solve = arms["probe"]["solve_steps_median"] is not None
    return [
        {
            "name": "samples_to_solve",
            "available": bool(baseline_solve and probe_solve),
            "question": "How many environment samples does each arm need to solve?",
        },
        {
            "name": "samples_to_peak_score",
            "available": bool(
                peak["baseline_steps_to_peak_median"] is not None
                and peak["probe_steps_to_peak_median"] is not None
            ),
            "question": "How many environment samples does each arm need to reach its best score?",
        },
        {
            "name": "peak_score",
            "available": bool(
                peak["baseline_best_return_median"] is not None
                and peak["probe_best_return_median"] is not None
            ),
            "question": "Does the crawler improve the best return, not just learning speed?",
        },
        {
            "name": "belief_ablation_savings",
            "available": deltas["probe_step_savings_vs_no_expression"] is not None,
            "question": "Does the learned belief beat the same probe protocol with belief muted?",
        },
        {
            "name": "probe_cost_fraction",
            "available": probe_cost["online_probe_fraction_of_total_median"] is not None,
            "question": "How much of the crawler path is spent gathering evidence instead of controlling?",
        },
        {
            "name": "learning_curve_auc",
            "available": False,
            "question": "What is the area under the return-vs-samples curve?",
        },
    ]


def build_benchmark_research_metrics(
    *,
    baseline_episode_solves: np.ndarray,
    baseline_step_solves: np.ndarray,
    baseline_total_env_steps: np.ndarray,
    baseline_completed_episodes: np.ndarray,
    probe_episode_solves: np.ndarray,
    probe_step_solves: np.ndarray,
    probe_total_env_steps: np.ndarray,
    probe_completed_episodes: np.ndarray,
    probe_no_expression_episode_solves: np.ndarray,
    probe_no_expression_step_solves: np.ndarray,
    probe_no_expression_total_env_steps: np.ndarray,
    probe_no_expression_completed_episodes: np.ndarray,
    full_system_episode_solves: np.ndarray,
    full_system_step_solves: np.ndarray,
    full_system_total_env_steps: np.ndarray,
    full_system_completed_episodes: np.ndarray,
    sim_fanout_episode_solves: np.ndarray,
    sim_fanout_step_solves: np.ndarray,
    sim_fanout_total_env_steps: np.ndarray,
    sim_fanout_completed_episodes: np.ndarray,
    probe_probe_env_steps: np.ndarray,
    probe_encoder_steps: np.ndarray,
    probe_control_env_steps: np.ndarray,
    probe_post_expression_env_steps: np.ndarray,
    baseline_best_returns: np.ndarray,
    probe_best_returns: np.ndarray,
    baseline_peak_env_steps: np.ndarray,
    probe_peak_env_steps_with_encoder: np.ndarray,
) -> dict:
    """Build stable benchmark metrics for sample-efficiency research plots."""
    arms = {
        "baseline": benchmark_research_arm_summary(
            baseline_episode_solves,
            baseline_step_solves,
            baseline_total_env_steps,
            baseline_completed_episodes,
        ),
        "probe": benchmark_research_arm_summary(
            probe_episode_solves,
            probe_step_solves,
            probe_total_env_steps,
            probe_completed_episodes,
        ),
        "probe_no_expression": benchmark_research_arm_summary(
            probe_no_expression_episode_solves,
            probe_no_expression_step_solves,
            probe_no_expression_total_env_steps,
            probe_no_expression_completed_episodes,
        ),
        "full_system": benchmark_research_arm_summary(
            full_system_episode_solves,
            full_system_step_solves,
            full_system_total_env_steps,
            full_system_completed_episodes,
        ),
        "sim_fanout": benchmark_research_arm_summary(
            sim_fanout_episode_solves,
            sim_fanout_step_solves,
            sim_fanout_total_env_steps,
            sim_fanout_completed_episodes,
        ),
    }
    deltas = {
        "probe_episode_savings_vs_baseline": median_savings_or_none(
            baseline_episode_solves,
            probe_episode_solves,
        ),
        "probe_step_savings_vs_baseline": median_savings_or_none(
            baseline_step_solves,
            probe_step_solves,
        ),
        "probe_episode_savings_vs_no_expression": median_savings_or_none(
            probe_no_expression_episode_solves,
            probe_episode_solves,
        ),
        "probe_step_savings_vs_no_expression": median_savings_or_none(
            probe_no_expression_step_solves,
            probe_step_solves,
        ),
        "full_system_episode_savings_vs_no_expression": median_savings_or_none(
            probe_no_expression_episode_solves,
            full_system_episode_solves,
        ),
        "full_system_step_savings_vs_no_expression": median_savings_or_none(
            probe_no_expression_step_solves,
            full_system_step_solves,
        ),
        "full_system_episode_regret_vs_sim_fanout": median_regret_or_none(
            full_system_episode_solves,
            sim_fanout_episode_solves,
        ),
        "full_system_step_regret_vs_sim_fanout": median_regret_or_none(
            full_system_step_solves,
            sim_fanout_step_solves,
        ),
    }
    probe_cost = {
        "online_probe_step_median": median_nonnegative_or_none(probe_probe_env_steps),
        "encoder_step_median": median_nonnegative_or_none(probe_encoder_steps),
        "control_step_median": median_nonnegative_or_none(probe_control_env_steps),
        "post_expression_step_median": median_nonnegative_or_none(
            probe_post_expression_env_steps
        ),
        "online_probe_fraction_of_total_median": median_ratio_or_none(
            probe_probe_env_steps,
            probe_total_env_steps,
        ),
        "encoder_fraction_of_total_median": median_ratio_or_none(
            probe_encoder_steps,
            probe_total_env_steps,
        ),
    }
    peak = {
        "baseline_best_return_median": median_or_none(baseline_best_returns),
        "probe_best_return_median": median_or_none(probe_best_returns),
        "baseline_steps_to_peak_median": median_nonnegative_or_none(
            baseline_peak_env_steps
        ),
        "probe_steps_to_peak_median": median_nonnegative_or_none(
            probe_peak_env_steps_with_encoder
        ),
        "probe_steps_to_peak_savings_vs_baseline": median_savings_or_none(
            baseline_peak_env_steps,
            probe_peak_env_steps_with_encoder,
        ),
    }
    return {
        "arms": arms,
        "deltas": deltas,
        "probe_cost": probe_cost,
        "peak": peak,
        "benchmarks": research_benchmark_status(
            arms=arms,
            deltas=deltas,
            peak=peak,
            probe_cost=probe_cost,
        ),
        "auc": {
            "available": False,
            "reason": "Solve benchmark artifacts do not store per-episode return histories; live comparison runs provide those curves.",
        },
    }
