"""Reporting tables for CartPole planner comparisons."""

from __future__ import annotations

import numpy as np


def planner_comparison_row(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    """Return the dashboard/API comparison row for one domain."""
    metrics = domain.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    if domain_name != "cartpole" or not metrics.get("planner_comparison_profile"):
        return _empty_row(domain_name)
    ppo_solve = _optional_float(metrics.get("baseline_solve_steps"))
    ppo_peak = _optional_float(metrics.get("baseline_steps_to_peak"))
    crawler_solve = _optional_float(metrics.get("planner_comparison_crawler_samples_to_solve"))
    fallback_solve = _optional_float(metrics.get("planner_comparison_fallback_samples_to_solve"))
    no_belief_solve = _optional_float(metrics.get("planner_comparison_no_belief_samples_to_solve"))
    persistent_solve = _optional_float(
        metrics.get("planner_comparison_persistent_affordance_samples_to_solve")
    )
    persistent_amortized_solve = _optional_float(
        metrics.get("planner_comparison_persistent_affordance_amortized_samples_to_solve")
    )
    row = {
        "domain": domain_name,
        "profile": metrics.get("planner_comparison_profile", ""),
        "ppo_samples_to_peak": ppo_peak,
        "ppo_samples_to_solve": ppo_solve,
        "ppo_peak_return": _optional_float(metrics.get("baseline_best_return")),
        "no_belief_mpc_samples_to_solve": no_belief_solve,
        "crawler_belief_mpc_samples_to_solve": crawler_solve,
        "oracle_mpc_samples_to_solve": _optional_float(
            metrics.get("planner_comparison_oracle_samples_to_solve")
        ),
        "cheap_fallback_samples_to_solve": fallback_solve,
        "persistent_affordance_samples_to_solve": persistent_solve,
        "persistent_affordance_amortized_samples_to_solve": persistent_amortized_solve,
        "crawler_vs_ppo_sample_savings": _sample_savings(ppo_solve, crawler_solve),
        "fallback_vs_ppo_sample_savings": _sample_savings(ppo_solve, fallback_solve),
        "persistent_affordance_vs_ppo_sample_savings": _sample_savings(
            ppo_solve,
            persistent_solve,
        ),
        "persistent_affordance_amortized_vs_ppo_sample_savings": _sample_savings(
            ppo_solve,
            persistent_amortized_solve,
        ),
        "crawler_vs_no_belief_mpc_sample_savings": _sample_savings(no_belief_solve, crawler_solve),
        "fallback_vs_no_belief_mpc_sample_savings": _sample_savings(no_belief_solve, fallback_solve),
        "persistent_affordance_vs_no_belief_mpc_sample_savings": _sample_savings(
            no_belief_solve,
            persistent_solve,
        ),
        "persistent_affordance_amortized_vs_no_belief_mpc_sample_savings": _sample_savings(
            no_belief_solve,
            persistent_amortized_solve,
        ),
        "planner_return_gain": _float(metrics.get("planner_comparison_solver_gain", 0.0)),
        "planner_content_lift": _float(metrics.get("planner_comparison_content_lift", 0.0)),
        "action_match_oracle": _float(metrics.get("planner_comparison_action_match_oracle", 0.0)),
        "action_regret_reduction": _float(
            metrics.get("planner_comparison_action_regret_reduction", 0.0)
        ),
        "probe_roi": _float(metrics.get("planner_comparison_probe_roi", 0.0)),
        "fallback_probe_roi": _float(metrics.get("planner_comparison_fallback_probe_roi", 0.0)),
        "persistent_affordance_probe_roi": _float(
            metrics.get("planner_comparison_persistent_affordance_probe_roi", 0.0)
        ),
        "persistent_affordance_probe_cost": _float(
            metrics.get("planner_comparison_persistent_affordance_probe_cost", 0.0)
        ),
        "persistent_affordance_amortized_probe_cost": _float(
            metrics.get("planner_comparison_persistent_affordance_amortized_probe_cost", 0.0)
        ),
        "persistent_affordance_reuse_horizon": _float(
            metrics.get("planner_comparison_persistent_affordance_reuse_horizon", 0.0)
        ),
        "persistent_affordance_regret_reduction": _float(
            metrics.get("planner_comparison_persistent_affordance_regret_reduction", 0.0)
        ),
        "persistent_affordance_probe_value": _float(
            metrics.get("planner_comparison_persistent_affordance_probe_value", 0.0)
        ),
        "fallback_wake_rate": _float(metrics.get("planner_comparison_fallback_wake_rate", 0.0)),
        "belief_beats_no_belief_fraction": _float(
            metrics.get("planner_comparison_belief_beats_no_belief_fraction", 0.0)
        ),
        "belief_beats_all_ablation_fraction": _float(
            metrics.get("planner_comparison_belief_beats_all_ablation_fraction", 0.0)
        ),
        "diagnostic_state": metrics.get("planner_comparison_diagnostic_state", ""),
    }
    row["verdict"] = _row_verdict(row)
    return row


def planner_arm_rows(
    expensive: dict[str, object],
    cheap: dict[str, object],
    persistent: dict[str, object],
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return arm-level planner comparison metrics."""
    return [
        _arm_row("mpc_no_belief", expensive, "no_belief"),
        _arm_row("mpc_crawler_belief", expensive, "belief"),
        _arm_row("mpc_oracle", expensive, "oracle"),
        _arm_row("mpc_cheap_belief", cheap, "belief"),
        {
            "arm": "mpc_persistent_affordance",
            "mean_return": _float(persistent.get("affordance_mpc_return", 0.0)),
            "median_return": _median(
                [
                    {"value": _float(row.get("affordance_mpc_return", 0.0))}
                    for row in persistent.get("rows", [])
                    if isinstance(row, dict)
                ],
                "value",
            ),
            "solve_rate": _float(persistent.get("affordance_solve_rate", 0.0)),
            "samples_to_peak_return": _float(
                persistent.get("affordance_env_samples_strict", 0.0)
            ),
            "samples_to_solve": _optional_float(
                persistent.get("affordance_samples_to_solve_strict")
            ),
            "amortized_samples_to_solve": _optional_float(
                persistent.get("affordance_samples_to_solve_amortized")
            ),
            "probe_env_steps": _float(persistent.get("probe_cost", 0.0)),
            "amortized_probe_env_steps": _float(
                persistent.get("amortized_probe_cost", 0.0)
            ),
            "reuse_horizon": _float(persistent.get("reuse_horizon", 0.0)),
            "control_env_steps": _float(expensive.get("control_steps", 0.0)),
            "total_env_steps": _float(persistent.get("affordance_env_samples_strict", 0.0)),
        },
        {
            "arm": "mpc_cheap_then_fallback",
            "mean_return": _mean(rows, "selected_return"),
            "median_return": _median(rows, "selected_return"),
            "solve_rate": _mean(rows, "selected_solved"),
            "samples_to_peak_return": _mean(rows, "selected_total_env_samples"),
            "samples_to_solve": _nullable_mean(rows, "selected_samples_to_solve"),
            "probe_env_steps": _mean_selected_probe_steps(rows, expensive, cheap),
            "control_env_steps": _float(expensive.get("control_steps", 0.0)),
            "total_env_steps": _mean(rows, "selected_total_env_samples"),
        },
    ]


def _arm_row(result_name: str, result: dict[str, object], prefix: str) -> dict[str, object]:
    return {
        "arm": result_name,
        "mean_return": _float(result.get(f"{prefix}_mpc_return", result.get(f"{prefix}_return", 0.0))),
        "median_return": _median(
            [
                {"value": _float(row.get(f"{prefix}_mpc_return", row.get(f"{prefix}_return", 0.0)))}
                for row in result.get("rows", [])
                if isinstance(row, dict)
            ],
            "value",
        ),
        "solve_rate": _float(result.get(f"{prefix}_solve_rate", 0.0)),
        "samples_to_peak_return": _optional_float(result.get(f"{prefix}_samples_to_peak_return")),
        "samples_to_solve": _optional_float(result.get(f"{prefix}_samples_to_solve")),
        "probe_env_steps": _float(result.get("probe_steps", 0.0)) if prefix == "belief" else 0.0,
        "control_env_steps": _float(result.get("control_steps", 0.0)),
        "total_env_steps": _optional_float(result.get(f"{prefix}_env_samples")),
    }


def _mean_selected_probe_steps(
    rows: list[dict[str, object]],
    expensive: dict[str, object],
    cheap: dict[str, object],
) -> float:
    expensive_steps = _float(expensive.get("probe_steps", 0.0))
    cheap_steps = _float(cheap.get("probe_steps", 0.0))
    costs = []
    for row in rows:
        selected = str(row.get("selected_arm", ""))
        if selected == "mpc_crawler_belief":
            costs.append(cheap_steps + expensive_steps)
        else:
            costs.append(cheap_steps)
    return float(np.mean(costs)) if costs else 0.0


def _row_verdict(row: dict[str, object]) -> str:
    ppo_savings = _optional_float(row.get("crawler_vs_ppo_sample_savings"))
    no_belief_savings = _optional_float(row.get("crawler_vs_no_belief_mpc_sample_savings"))
    persistent_ppo_savings = _optional_float(
        row.get("persistent_affordance_amortized_vs_ppo_sample_savings")
    )
    persistent_mpc_savings = _optional_float(
        row.get("persistent_affordance_amortized_vs_no_belief_mpc_sample_savings")
    )
    content = _float(row.get("planner_content_lift", 0.0))
    regret = _float(row.get("action_regret_reduction", 0.0))
    persistent_regret = _float(row.get("persistent_affordance_regret_reduction", 0.0))
    if (
        persistent_ppo_savings is not None
        and persistent_ppo_savings > 0.0
        and persistent_regret > 0.0
    ):
        return "persistent_affordance_wins_vs_ppo"
    if (
        persistent_mpc_savings is not None
        and persistent_mpc_savings > 0.0
        and persistent_regret > 0.0
    ):
        return "persistent_affordance_wins_vs_mpc"
    if ppo_savings is not None and ppo_savings > 0.0 and content > 0.0 and regret > 0.0:
        return "planner_belief_wins_vs_ppo"
    if no_belief_savings is not None and no_belief_savings > 0.0 and content > 0.0:
        return "planner_belief_wins_vs_mpc"
    if persistent_regret > 0.0:
        return "persistent_affordance_predictive_but_costly"
    if regret > 0.0 and content > 0.0:
        return "planner_belief_predictive_but_costly"
    if regret > 0.0:
        return "planner_action_regret_better_but_ablation_dirty"
    return "planner_belief_not_helping"


def _empty_row(domain_name: str) -> dict[str, object]:
    return {
        "domain": domain_name,
        "profile": "",
        "ppo_samples_to_peak": None,
        "ppo_samples_to_solve": None,
        "no_belief_mpc_samples_to_solve": None,
        "crawler_belief_mpc_samples_to_solve": None,
        "oracle_mpc_samples_to_solve": None,
        "cheap_fallback_samples_to_solve": None,
        "persistent_affordance_samples_to_solve": None,
        "persistent_affordance_amortized_samples_to_solve": None,
        "crawler_vs_ppo_sample_savings": None,
        "fallback_vs_ppo_sample_savings": None,
        "persistent_affordance_vs_ppo_sample_savings": None,
        "persistent_affordance_amortized_vs_ppo_sample_savings": None,
        "crawler_vs_no_belief_mpc_sample_savings": None,
        "fallback_vs_no_belief_mpc_sample_savings": None,
        "persistent_affordance_vs_no_belief_mpc_sample_savings": None,
        "persistent_affordance_amortized_vs_no_belief_mpc_sample_savings": None,
        "planner_return_gain": 0.0,
        "planner_content_lift": 0.0,
        "action_match_oracle": 0.0,
        "action_regret_reduction": 0.0,
        "probe_roi": 0.0,
        "fallback_probe_roi": 0.0,
        "persistent_affordance_probe_roi": 0.0,
        "persistent_affordance_probe_cost": 0.0,
        "persistent_affordance_amortized_probe_cost": 0.0,
        "persistent_affordance_reuse_horizon": 0.0,
        "persistent_affordance_regret_reduction": 0.0,
        "persistent_affordance_probe_value": 0.0,
        "fallback_wake_rate": 0.0,
        "belief_beats_no_belief_fraction": 0.0,
        "belief_beats_all_ablation_fraction": 0.0,
        "diagnostic_state": "",
        "verdict": "",
    }


def _sample_savings(baseline: float | None, crawler: float | None) -> float | None:
    if baseline is None or crawler is None:
        return None
    return float(baseline - crawler)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([_float(row.get(key, 0.0)) for row in rows]))


def _nullable_mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [_optional_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _median(rows: list[dict[str, object]], key: str) -> float:
    values = [_optional_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return 0.0
    return float(np.median(clean))
