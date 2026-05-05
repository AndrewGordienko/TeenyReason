"""Sample-performance accounting for crawler solver handoffs."""

from __future__ import annotations

import math
from typing import Any


IMAGE_SOLVE_ACCURACY = 0.90
BOARD_SOLVE_ACCURACY = 1.0


def attach_sample_performance_blocks(domains: dict[str, object]) -> None:
    """Attach peak and solve sample accounting to every suite domain."""
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            domain["sample_performance"] = build_sample_performance_block(domain_name, domain)


def sample_performance_row(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    """Return a compact dashboard/API row for one domain."""
    block = _dict(domain.get("sample_performance"))
    peak = _dict(block.get("peak"))
    solve = _dict(block.get("solve"))
    decision = _dict(block.get("decision"))
    return {
        "domain": domain_name,
        "sample_axis": block.get("sample_axis", ""),
        "score_name": block.get("score_name", ""),
        "lower_is_better": bool(block.get("lower_is_better", False)),
        "baseline_samples_to_peak": peak.get("baseline_samples_to_peak"),
        "crawler_samples_to_peak": peak.get("crawler_samples_to_peak"),
        "peak_sample_savings": peak.get("sample_savings"),
        "baseline_peak_score": peak.get("baseline_score"),
        "crawler_peak_score": peak.get("crawler_score"),
        "peak_score_delta": peak.get("score_delta", 0.0),
        "best_solver_gain": peak.get("best_solver_gain", 0.0),
        "best_solver_gain_sample": peak.get("best_solver_gain_sample"),
        "baseline_samples_to_solve": solve.get("baseline_samples_to_solve"),
        "crawler_samples_to_solve": solve.get("crawler_samples_to_solve"),
        "solve_sample_savings": solve.get("sample_savings"),
        "solve_threshold": solve.get("threshold"),
        "solve_available": bool(solve.get("available", False)),
        "crawler_wins_peak_samples": bool(peak.get("crawler_wins_samples", False)),
        "crawler_wins_peak_score": bool(peak.get("crawler_wins_score", False)),
        "crawler_wins_solve_samples": bool(solve.get("crawler_wins_samples", False)),
        "state": decision.get("state", ""),
        "next_action": decision.get("next_action", ""),
    }


def build_sample_performance_block(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    """Build sample-to-peak and sample-to-solve metrics from domain rows."""
    if domain_name == "cartpole":
        return _cartpole_block(domain)
    if domain_name == "language":
        return _curve_block(
            domain_name=domain_name,
            domain=domain,
            sample_axis="train_chars",
            score_name="validation_bpc",
            lower_is_better=True,
            sample_keys=("train_char_budget",),
            baseline_keys=("baseline_bpc",),
            crawler_keys=("belief_bpc", "probe_bpc"),
            solve_threshold=None,
        )
    if domain_name == "image":
        return _curve_block(
            domain_name=domain_name,
            domain=domain,
            sample_axis="labels",
            score_name="accuracy",
            lower_is_better=False,
            sample_keys=("label_budget",),
            baseline_keys=("baseline_accuracy",),
            crawler_keys=("belief_accuracy", "probe_accuracy"),
            solve_threshold=IMAGE_SOLVE_ACCURACY,
        )
    if domain_name == "board":
        return _curve_block(
            domain_name=domain_name,
            domain=domain,
            sample_axis="probe_queries",
            score_name="best_move_accuracy",
            lower_is_better=False,
            sample_keys=("query_count", "probe_queries"),
            baseline_keys=("baseline_move_accuracy",),
            crawler_keys=("belief_move_accuracy",),
            solve_threshold=BOARD_SOLVE_ACCURACY,
        )
    return _empty_block(domain_name, "samples", "score", False)


def _cartpole_block(domain: dict[str, object]) -> dict[str, object]:
    metrics = _dict(domain.get("metrics"))
    baseline_peak_samples = _optional_float(metrics.get("baseline_steps_to_peak"))
    crawler_peak_samples = _optional_float(metrics.get("probe_steps_to_peak"))
    baseline_solve_samples = _optional_float(metrics.get("baseline_solve_steps"))
    crawler_solve_samples = _optional_float(metrics.get("probe_solve_steps"))
    baseline_peak_score = _optional_float(metrics.get("baseline_best_return"))
    crawler_peak_score = _optional_float(metrics.get("probe_best_return"))
    peak_available = baseline_peak_samples is not None and crawler_peak_samples is not None
    solve_available = baseline_solve_samples is not None and crawler_solve_samples is not None
    peak_score_delta = _score_delta(
        baseline_peak_score,
        crawler_peak_score,
        lower_is_better=False,
    )
    peak_savings = _savings(baseline_peak_samples, crawler_peak_samples)
    solve_savings = _savings(baseline_solve_samples, crawler_solve_samples)
    block = {
        "schema_version": 1,
        "domain": "cartpole",
        "sample_axis": "env_steps",
        "score_name": "return",
        "lower_is_better": False,
        "peak": {
            "available": peak_available,
            "baseline_samples_to_peak": baseline_peak_samples,
            "crawler_samples_to_peak": crawler_peak_samples,
            "sample_savings": peak_savings,
            "baseline_score": baseline_peak_score,
            "crawler_score": crawler_peak_score,
            "score_delta": peak_score_delta,
            "crawler_wins_samples": _positive(peak_savings),
            "crawler_wins_score": _positive(peak_score_delta),
            "best_solver_gain": _float(metrics.get("probe_step_savings_vs_baseline", 0.0)),
            "best_solver_gain_sample": crawler_peak_samples,
        },
        "solve": {
            "available": solve_available,
            "threshold": metrics.get("solved_return"),
            "baseline_samples_to_solve": baseline_solve_samples,
            "crawler_samples_to_solve": crawler_solve_samples,
            "sample_savings": solve_savings,
            "crawler_wins_samples": _positive(solve_savings),
        },
        "raw": {
            "probe_no_expression_solve_steps": _optional_float(
                metrics.get("probe_no_expression_solve_steps")
            ),
            "probe_step_savings_vs_no_expression": _optional_float(
                metrics.get("probe_step_savings_vs_no_expression")
            ),
        },
    }
    block["decision"] = _decision_for_block(block)
    return block


def _curve_block(
    *,
    domain_name: str,
    domain: dict[str, object],
    sample_axis: str,
    score_name: str,
    lower_is_better: bool,
    sample_keys: tuple[str, ...],
    baseline_keys: tuple[str, ...],
    crawler_keys: tuple[str, ...],
    solve_threshold: float | None,
) -> dict[str, object]:
    rows = _curve_rows(
        domain.get("rows"),
        sample_keys=sample_keys,
        baseline_keys=baseline_keys,
        crawler_keys=crawler_keys,
        lower_is_better=lower_is_better,
    )
    baseline_peak = _peak_row(rows, "baseline_score", lower_is_better=lower_is_better)
    crawler_peak = _peak_row(rows, "crawler_score", lower_is_better=lower_is_better)
    baseline_solve = _threshold_row(
        rows,
        "baseline_score",
        threshold=solve_threshold,
        lower_is_better=lower_is_better,
    )
    crawler_solve = _threshold_row(
        rows,
        "crawler_score",
        threshold=solve_threshold,
        lower_is_better=lower_is_better,
    )
    best_gain_row = _peak_row(rows, "solver_gain", lower_is_better=False)
    peak_savings = _savings(
        _row_sample(baseline_peak),
        _row_sample(crawler_peak),
    )
    solve_savings = _savings(
        _row_sample(baseline_solve),
        _row_sample(crawler_solve),
    )
    peak_score_delta = _score_delta(
        _row_value(baseline_peak, "baseline_score"),
        _row_value(crawler_peak, "crawler_score"),
        lower_is_better=lower_is_better,
    )
    block = {
        "schema_version": 1,
        "domain": domain_name,
        "sample_axis": sample_axis,
        "score_name": score_name,
        "lower_is_better": lower_is_better,
        "peak": {
            "available": baseline_peak is not None and crawler_peak is not None,
            "baseline_samples_to_peak": _row_sample(baseline_peak),
            "crawler_samples_to_peak": _row_sample(crawler_peak),
            "sample_savings": peak_savings,
            "baseline_score": _row_value(baseline_peak, "baseline_score"),
            "crawler_score": _row_value(crawler_peak, "crawler_score"),
            "score_delta": peak_score_delta,
            "crawler_wins_samples": _positive(peak_savings),
            "crawler_wins_score": _positive(peak_score_delta),
            "best_solver_gain": _row_value(best_gain_row, "solver_gain") or 0.0,
            "best_solver_gain_sample": _row_sample(best_gain_row),
            "mean_solver_gain": _mean(row["solver_gain"] for row in rows),
        },
        "solve": {
            "available": baseline_solve is not None and crawler_solve is not None,
            "threshold": solve_threshold,
            "baseline_samples_to_solve": _row_sample(baseline_solve),
            "crawler_samples_to_solve": _row_sample(crawler_solve),
            "sample_savings": solve_savings,
            "crawler_wins_samples": _positive(solve_savings),
        },
        "curve": rows,
    }
    block["decision"] = _decision_for_block(block)
    return block


def _curve_rows(
    value: object,
    *,
    sample_keys: tuple[str, ...],
    baseline_keys: tuple[str, ...],
    crawler_keys: tuple[str, ...],
    lower_is_better: bool,
) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, float]] = []
    fallback_sample = 1.0
    for row in value:
        if not isinstance(row, dict):
            continue
        sample = _first_number(row, sample_keys)
        if sample is None:
            sample = fallback_sample
        baseline = _first_number(row, baseline_keys)
        crawler = _first_number(row, crawler_keys)
        if baseline is None or crawler is None:
            fallback_sample += 1.0
            continue
        rows.append(
            {
                "sample": float(sample),
                "baseline_score": float(baseline),
                "crawler_score": float(crawler),
                "solver_gain": baseline - crawler if lower_is_better else crawler - baseline,
            }
        )
        fallback_sample += 1.0
    return rows


def _peak_row(
    rows: list[dict[str, float]],
    key: str,
    *,
    lower_is_better: bool,
) -> dict[str, float] | None:
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: (row["sample"],))
    if lower_is_better:
        return min(ordered, key=lambda row: (row[key], row["sample"]))
    return max(ordered, key=lambda row: (row[key], -row["sample"]))


def _threshold_row(
    rows: list[dict[str, float]],
    key: str,
    *,
    threshold: float | None,
    lower_is_better: bool,
) -> dict[str, float] | None:
    if threshold is None:
        return None
    ordered = sorted(rows, key=lambda row: row["sample"])
    for row in ordered:
        if lower_is_better and row[key] <= threshold:
            return row
        if not lower_is_better and row[key] >= threshold:
            return row
    return None


def _decision_for_block(block: dict[str, object]) -> dict[str, object]:
    peak = _dict(block.get("peak"))
    solve = _dict(block.get("solve"))
    peak_score_delta = _float(peak.get("score_delta", 0.0))
    peak_savings = _optional_float(peak.get("sample_savings"))
    solve_savings = _optional_float(solve.get("sample_savings"))
    best_gain = _float(peak.get("best_solver_gain", 0.0))
    if solve.get("available") and solve_savings is not None and solve_savings > 0.0:
        return {"state": "crawler_solves_with_fewer_samples", "next_action": "scale_current_handoff"}
    if solve.get("available") and solve_savings is not None and solve_savings < 0.0:
        return {"state": "crawler_loses_solve_samples", "next_action": "repair_solver_handoff_cost"}
    if peak_score_delta > 0.0 and peak_savings is not None and peak_savings >= 0.0:
        return {"state": "crawler_peak_win", "next_action": "scale_current_handoff"}
    if best_gain > 0.0:
        return {"state": "local_gain_not_global_peak", "next_action": "gate_belief_by_budget"}
    if peak_score_delta < 0.0:
        return {"state": "crawler_peak_score_loss", "next_action": "repair_solver_handoff"}
    return {"state": "no_sample_win_measured", "next_action": "add_matched_sample_curve"}


def _empty_block(
    domain_name: str,
    sample_axis: str,
    score_name: str,
    lower_is_better: bool,
) -> dict[str, object]:
    block = {
        "schema_version": 1,
        "domain": domain_name,
        "sample_axis": sample_axis,
        "score_name": score_name,
        "lower_is_better": lower_is_better,
        "peak": {"available": False},
        "solve": {"available": False},
    }
    block["decision"] = _decision_for_block(block)
    return block


def _first_number(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _row_sample(row: dict[str, float] | None) -> float | None:
    if row is None:
        return None
    return row.get("sample")


def _row_value(row: dict[str, float] | None, key: str) -> float | None:
    if row is None:
        return None
    return row.get(key)


def _score_delta(
    baseline_value: float | None,
    crawler_value: float | None,
    *,
    lower_is_better: bool,
) -> float:
    if baseline_value is None or crawler_value is None:
        return 0.0
    if lower_is_better:
        return baseline_value - crawler_value
    return crawler_value - baseline_value


def _savings(baseline_samples: float | None, crawler_samples: float | None) -> float | None:
    if baseline_samples is None or crawler_samples is None:
        return None
    return baseline_samples - crawler_samples


def _positive(value: float | None) -> bool:
    return value is not None and value > 0.0


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _float(value: object) -> float:
    number = _optional_float(value)
    return 0.0 if number is None else number


def _mean(values: object) -> float:
    numbers = [_float(value) for value in values]
    if not numbers:
        return 0.0
    return float(sum(numbers) / float(len(numbers)))


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
