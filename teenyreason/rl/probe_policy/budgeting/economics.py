"""Probe-family economics thresholds and small scalar helpers."""

MIN_SELECTABLE_MARGINAL_VALUE = 0.02
MIN_SELECTABLE_VALUE_PER_STEP = 0.02
FAIR_MODE_PROBE_CAP = 2
FAIR_MODE_FIRST_PROBE_SHORTLIST = (
    "boundary_push",
    "cart_brake",
    "chirp",
    "impulse_right",
    "impulse_left",
    "counter_balance",
)
FAIR_MODE_FIRST_PROBE_PREFERENCE = {
    "chirp": 0.08,
    "cart_brake": 0.07,
    "impulse_right": 0.05,
    "impulse_left": 0.05,
    "counter_balance": 0.04,
    "boundary_push": 0.00,
}
FAIR_MODE_SCALAR_BOX_FIRST_PROBE_PREFERENCE = {
    "neg_1": 0.08,
    "pos_1": 0.08,
    "neg_2": 0.05,
    "pos_2": 0.05,
    "neg_3": 0.03,
    "pos_3": 0.03,
    "center": -0.10,
}
MIN_THIRD_PROBE_MARGINAL_VALUE = 0.08
MIN_THIRD_PROBE_VALUE_PER_STEP = 0.05
MIN_THIRD_PROBE_PASSIVE_MARGINAL_VALUE = 0.22
MIN_THIRD_PROBE_PASSIVE_VALUE_PER_STEP = 0.35
MIN_FAIR_SECOND_MARGINAL_VALUE = 0.08
MIN_FAIR_SECOND_VALUE_PER_STEP = 0.06


def default_probe_stop_reasons() -> dict[str, int]:
    """Create one mutable stop-reason counter with stable keys."""
    return {
        "expression_ready": 0,
        "fair_two_probe_handoff": 0,
        "low_uncertainty_low_gain": 0,
        "low_economic_value": 0,
        "nonpositive_value_per_step": 0,
        "passive_repeat_cap": 0,
        "stalled_realized_gain": 0,
        "adaptive_expand_cap": 0,
        "fixed_cap_reached": 0,
        "adaptive_continue": 0,
        "probe_failure": 0,
    }


def probe_family_selection_metrics(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    """Read the crawler's comparable family-value scalars with stable fallbacks."""
    predicted_marginal_value = float(
        metrics.get(
            "predicted_marginal_value",
            metrics.get(
                "future_gain_for_choice",
                metrics.get(
                    "predicted_future_error_reduction",
                    metrics.get("value_per_probe_step", metrics.get("selection_score", metrics.get("score", 0.0))),
                ),
            ),
        )
    )
    estimated_probe_cost = float(metrics.get("estimated_probe_cost", 1.0))
    value_per_probe_step = float(
        metrics.get(
            "value_per_probe_step",
            predicted_marginal_value / max(estimated_probe_cost, 1e-6),
        )
    )
    selection_score = float(metrics.get("selection_score", metrics.get("score", predicted_marginal_value)))
    score = float(metrics.get("score", selection_score))
    return predicted_marginal_value, value_per_probe_step, selection_score, score


def probe_family_has_positive_economics(metrics: dict[str, float]) -> bool:
    """Treat a family as selectable only when it clears a real economics margin."""
    predicted_marginal_value, value_per_probe_step, _selection_score, _score = probe_family_selection_metrics(metrics)
    return (
        predicted_marginal_value > MIN_SELECTABLE_MARGINAL_VALUE
        and value_per_probe_step > MIN_SELECTABLE_VALUE_PER_STEP
    )


def probe_family_clears_fair_second_probe_floor(metrics: dict[str, float]) -> bool:
    """Require a real value-per-step margin before fair mode buys probe two."""
    predicted_marginal_value, value_per_probe_step, selection_score, _score = probe_family_selection_metrics(metrics)
    return (
        predicted_marginal_value >= MIN_FAIR_SECOND_MARGINAL_VALUE
        and value_per_probe_step >= MIN_FAIR_SECOND_VALUE_PER_STEP
        and selection_score >= MIN_FAIR_SECOND_MARGINAL_VALUE
    )


def third_probe_economic_floors(
    *,
    only_passive_family_viable: bool,
) -> tuple[float, float]:
    """Return the stricter economics floor required before buying probe three."""
    if only_passive_family_viable:
        return MIN_THIRD_PROBE_PASSIVE_VALUE_PER_STEP, MIN_THIRD_PROBE_PASSIVE_MARGINAL_VALUE
    return MIN_THIRD_PROBE_VALUE_PER_STEP, MIN_THIRD_PROBE_MARGINAL_VALUE


def probe_family_is_suppressed(
    *,
    family: str,
    metrics: dict[str, float],
    global_family_counts: dict[str, int] | None = None,
    family_realized_gain_history: dict[str, float] | None = None,
    family_bad_streaks: dict[str, int] | None = None,
) -> bool:
    """Hide families that keep looking bad until their economics recover clearly."""
    global_family_counts = global_family_counts or {}
    family_realized_gain_history = family_realized_gain_history or {}
    family_bad_streaks = family_bad_streaks or {}
    predicted_marginal_value, value_per_probe_step, selection_score, _score = probe_family_selection_metrics(metrics)
    global_family_count = int(global_family_counts.get(family, 0))
    bad_streak = int(family_bad_streaks.get(family, 0))
    realized_gain = float(family_realized_gain_history.get(family, 0.0))
    if family == "passive_decay":
        if (
            global_family_count > 0
            and bad_streak >= 1
            and realized_gain <= 0.02
            and value_per_probe_step <= 0.35
        ):
            return True
        return False
    if bad_streak < 2:
        return False
    if (
        predicted_marginal_value > 0.12
        and value_per_probe_step > 0.12
        and selection_score > 0.12
    ):
        return False
    return True
