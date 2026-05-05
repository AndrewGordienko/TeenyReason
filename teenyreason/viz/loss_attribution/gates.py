"""Benchmark gate summaries for loss attribution."""

from __future__ import annotations

from collections import Counter

from .common import _capped_solve_values, _mean, _median, _row_values, _threshold_row


def _representation_gate_summary(rows: list[dict]) -> dict[str, object]:
    gate_rows = [
        row.get("representation_gate", {})
        for row in rows
        if isinstance(row.get("representation_gate"), dict)
        and (
            row.get("representation_gate", {}).get("metrics")
            or row.get("representation_gate", {}).get("checks")
            or row.get("representation_gate", {}).get("thresholds")
        )
    ]
    if not gate_rows:
        return {"available": False, "pass_fraction": 0.0, "metrics": [], "failure_reasons": {}}

    metric_names = sorted(
        {
            name
            for gate in gate_rows
            for name in dict(gate.get("metrics", {})).keys()
        }
    )
    threshold_aliases = {
        "paired_split_top1": ("min_paired_top1", ">="),
        "cross_split_top1": ("min_cross_top1", ">="),
        "neighbor_alignment": ("min_neighbor_alignment", ">="),
        "paired_gap_ratio": ("max_paired_gap_ratio", "<="),
        "belief_norm_std": ("min_belief_norm_std", ">="),
        "nearest_between_median": ("min_nearest_between", ">="),
        "sysid_validation_top1": ("min_sysid_top1", ">="),
        "sysid_validation_margin": ("min_sysid_margin", ">="),
    }
    metrics: list[dict[str, object]] = []
    for name in metric_names:
        observed = _median(dict(gate.get("metrics", {})).get(name) for gate in gate_rows)
        target_key, direction = threshold_aliases.get(name, ("", ">="))
        target_values = [
            dict(gate.get("thresholds", {})).get(target_key)
            for gate in gate_rows
            if target_key
        ]
        if target_values:
            metrics.append(
                _threshold_row(
                    name=name,
                    observed=observed,
                    target=_median(target_values),
                    direction=direction,
                )
            )
        else:
            metrics.append(
                {
                    "name": name,
                    "observed": observed,
                    "target": None,
                    "direction": "",
                    "margin": None,
                    "ratio_to_target": None,
                    "passed": bool(observed > 0.0),
                    "unit": "",
                }
            )

    failures: Counter[str] = Counter()
    for gate in gate_rows:
        for reason in gate.get("latent_failure_reasons", gate.get("failure_reasons", [])):
            failures[str(reason)] += 1
    return {
        "available": True,
        "pass_fraction": float(
            sum(1 for gate in gate_rows if bool(gate.get("pass", False)))
        )
        / float(max(len(gate_rows), 1)),
        "latent_pass_fraction": float(
            sum(1 for gate in gate_rows if bool(gate.get("latent_pass", gate.get("pass", False))))
        )
        / float(max(len(gate_rows), 1)),
        "override_count": int(sum(1 for gate in gate_rows if str(gate.get("override_reason", "")))),
        "metrics": metrics,
        "failure_reasons": dict(failures),
    }

def _latent_win_gate_summary(
    *,
    rows: list[dict],
    benchmark_profile: str | None,
    latent_win_gate: dict,
    seed_count: int,
) -> dict[str, object]:
    target_split = 0.45 if str(benchmark_profile) == "full" else 0.30
    baseline_episode = _median(
        _capped_solve_values(rows, "baseline_episode_solve", "baseline_completed_episodes")
    )
    probe_episode = _median(
        _capped_solve_values(rows, "probe_episode_solve", "probe_completed_episodes")
    )
    baseline_steps = _median(
        _capped_solve_values(rows, "baseline_step_solve", "baseline_total_env_steps")
    )
    probe_steps = _median(
        _capped_solve_values(rows, "probe_step_solve", "probe_total_env_steps")
    )
    targets = [
        {
            "name": "full_benchmark_seed_count",
            "observed": float(seed_count),
            "target": 5.0,
            "direction": ">=",
            "margin": float(seed_count - 5),
            "ratio_to_target": float(seed_count) / 5.0,
            "passed": str(benchmark_profile) == "full" and seed_count >= 5,
            "unit": "seeds",
        },
        _threshold_row(
            name="probe_episode_speed",
            observed=probe_episode,
            target=0.90 * baseline_episode,
            direction="<=",
            unit="episodes",
        ),
        _threshold_row(
            name="probe_step_speed",
            observed=probe_steps,
            target=baseline_steps,
            direction="<",
            unit="env_steps",
        ),
        _threshold_row(
            name="env_expression_delta",
            observed=_median(_row_values(rows, "probe_env_expression_delta")),
            target=0.0,
            direction=">",
            unit="return",
        ),
        _threshold_row(
            name="ready_handoff_fraction",
            observed=_mean(_row_values(rows, "probe_fair_ready_handoff_fraction")),
            target=0.50,
            direction=">=",
        ),
        _threshold_row(
            name="expression_enabled_fraction",
            observed=_mean(_row_values(rows, "probe_fair_expression_enabled_fraction")),
            target=0.20,
            direction=">=",
        ),
        _threshold_row(
            name="muted_fraction",
            observed=_mean(_row_values(rows, "probe_fair_expression_force_muted_fraction")),
            target=0.50,
            direction="<=",
        ),
        _threshold_row(
            name="mechanics_fit",
            observed=_median(_row_values(rows, "latent_mechanics_fit")),
            target=0.60,
            direction=">=",
        ),
        _threshold_row(
            name="neighbor_alignment",
            observed=_median(_row_values(rows, "latent_neighbor_alignment")),
            target=0.20,
            direction=">=",
        ),
        _threshold_row(
            name="split_retrieval_top1",
            observed=_median(_row_values(rows, "latent_paired_split_top1")),
            target=target_split,
            direction=">=",
        ),
        _threshold_row(
            name="gap_ratio",
            observed=_median(_row_values(rows, "latent_gap_ratio")),
            target=1.0,
            direction="<=",
        ),
        _threshold_row(
            name="uncert_error_corr",
            observed=_median(_row_values(rows, "latent_uncert_error_corr")),
            target=0.30,
            direction=">=",
        ),
    ]
    return {
        "available": bool(rows),
        "pass": bool(latent_win_gate.get("pass", False)),
        "failure_reasons": list(latent_win_gate.get("failure_reasons", [])),
        "metrics": targets,
    }
