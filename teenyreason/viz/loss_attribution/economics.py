"""Probe and sample economics for loss attribution."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .common import _capped_solve_values, _median, _ratio, _row_values


def _probe_family_economics(rows: list[dict]) -> dict[str, object]:
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    second_probe_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    metric_names = (
        "selection_score",
        "predicted_marginal_value",
        "value_per_probe_step",
        "sample_efficiency_score",
        "control_utility_value",
        "stability_confidence",
        "stability_adjusted_value",
        "estimated_probe_cost",
        "future_gain_for_choice",
        "predicted_split_reduction",
        "predicted_mechanics_reduction",
    )
    for row in rows:
        for family, count in dict(row.get("probe_second_probe_selection_count", {})).items():
            second_probe_counts[str(family)] += int(float(count))
        for family, count in dict(row.get("probe_family_selection_count", {})).items():
            selected_counts[str(family)] += int(float(count))
        family_rows = row.get("probe_family_expected_gain", {})
        if not isinstance(family_rows, dict):
            continue
        for family, metrics in family_rows.items():
            if not isinstance(metrics, dict):
                continue
            family_key = str(family)
            counts[family_key] = counts.get(family_key, 0) + 1
            target = totals.setdefault(family_key, {name: 0.0 for name in metric_names})
            for name in metric_names:
                try:
                    value = float(metrics.get(name, 0.0))
                except (TypeError, ValueError):
                    value = 0.0
                if np.isfinite(value):
                    target[name] += value

    families = []
    for family, metrics in totals.items():
        denom = float(max(counts.get(family, 0), 1))
        averaged = {name: float(value) / denom for name, value in metrics.items()}
        clears_second = (
            averaged["predicted_marginal_value"] >= 0.08
            and averaged["value_per_probe_step"] >= 0.06
            and averaged["selection_score"] >= 0.08
        )
        families.append(
            {
                "family": family,
                **averaged,
                "clears_fair_second_probe_floor": bool(clears_second),
                "selected_count": int(selected_counts.get(family, 0)),
                "second_probe_selected_count": int(second_probe_counts.get(family, 0)),
            }
        )
    families.sort(
        key=lambda row: (
            float(row["sample_efficiency_score"]),
            float(row["selection_score"]),
            -float(row["estimated_probe_cost"]),
        ),
        reverse=True,
    )
    return {
        "families": families,
        "families_clearing_second_probe_floor": int(
            sum(1 for row in families if row["clears_fair_second_probe_floor"])
        ),
        "second_probe_selection_count": dict(second_probe_counts),
        "first_or_only_probe_selection_count": dict(selected_counts),
        "best_family": families[0]["family"] if families else "",
    }

def _sample_economics(rows: list[dict]) -> dict[str, object]:
    baseline_steps = _median(_row_values(rows, "baseline_step_solve", nonnegative=True))
    probe_steps = _median(_row_values(rows, "probe_step_solve", nonnegative=True))
    noexpr_steps = _median(_row_values(rows, "probe_no_expression_step_solve", nonnegative=True))
    baseline_capped_steps = _median(
        _capped_solve_values(rows, "baseline_step_solve", "baseline_total_env_steps")
    )
    probe_capped_steps = _median(
        _capped_solve_values(rows, "probe_step_solve", "probe_total_env_steps")
    )
    noexpr_capped_steps = _median(
        _capped_solve_values(
            rows,
            "probe_no_expression_step_solve",
            "probe_no_expression_total_env_steps",
        )
    )
    probe_total = _median(_row_values(rows, "probe_total_env_steps", nonnegative=True))
    encoder_steps = _median(_row_values(rows, "probe_encoder_steps", nonnegative=True))
    online_probe_steps = _median(_row_values(rows, "probe_probe_env_steps", nonnegative=True))
    control_steps = _median(_row_values(rows, "probe_control_env_steps", nonnegative=True))
    return {
        "baseline_solve_steps_median": baseline_steps,
        "probe_solve_steps_median": probe_steps,
        "probe_no_expression_solve_steps_median": noexpr_steps,
        "baseline_capped_steps_median": baseline_capped_steps,
        "probe_capped_steps_median": probe_capped_steps,
        "probe_no_expression_capped_steps_median": noexpr_capped_steps,
        "probe_step_savings_vs_baseline": baseline_steps - probe_steps,
        "probe_step_savings_vs_no_expression": noexpr_steps - probe_steps,
        "probe_capped_step_savings_vs_baseline": baseline_capped_steps - probe_capped_steps,
        "probe_capped_step_savings_vs_no_expression": noexpr_capped_steps - probe_capped_steps,
        "probe_total_steps_median": probe_total,
        "encoder_steps_median": encoder_steps,
        "online_probe_steps_median": online_probe_steps,
        "control_steps_median": control_steps,
        "encoder_fraction_of_probe_total": _ratio(encoder_steps, probe_total),
        "online_probe_fraction_of_probe_total": _ratio(online_probe_steps, probe_total),
        "control_fraction_of_probe_total": _ratio(control_steps, probe_total),
    }

