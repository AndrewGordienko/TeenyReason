"""Fair-mode probe-family selection and stopping rules."""

from .economics import (
    FAIR_MODE_FIRST_PROBE_PREFERENCE,
    FAIR_MODE_FIRST_PROBE_SHORTLIST,
    FAIR_MODE_PROBE_CAP,
    FAIR_MODE_SCALAR_BOX_FIRST_PROBE_PREFERENCE,
    probe_family_clears_fair_second_probe_floor,
    probe_family_is_suppressed,
    probe_family_selection_metrics,
)


def _fair_probe_two_stage_sort_key(
    family: str,
    metrics: dict[str, float],
    *,
    family_names: tuple[str, ...],
    family_counts: dict[str, int],
    global_family_counts: dict[str, int] | None = None,
    family_realized_gain_history: dict[str, float] | None = None,
    recent_families: tuple[str, ...] = (),
    probe_surprise: float = 0.0,
) -> tuple[float, float, float, float, float, float, float, float, float, float, str]:
    """Order active probe-two candidates by the child-like fair identification rule."""
    global_family_counts = global_family_counts or {}
    family_realized_gain_history = family_realized_gain_history or {}
    entropy_reduction = float(metrics.get("predicted_entropy_reduction", 0.0))
    future_probe_gain = float(
        metrics.get(
            "future_gain_for_choice",
            metrics.get(
                "predicted_future_error_reduction",
                metrics.get("future_error_estimate", 0.0),
            ),
        )
    )
    predicted_split_reduction = float(metrics.get("predicted_split_reduction", 0.0))
    predicted_mechanics_reduction = float(metrics.get("predicted_mechanics_reduction", 0.0))
    hypothesis_separation = float(metrics.get("predicted_hypothesis_separation", 0.0))
    estimated_probe_cost = float(metrics.get("estimated_probe_cost", 1.0))
    predicted_marginal_value, value_per_probe_step, selection_score, score = (
        probe_family_selection_metrics(metrics)
    )
    selection_tiebreak = float(metrics.get("selection_score", metrics.get("score", 0.0)))
    surprise_tiebreak = 1e-6 * float(max(probe_surprise, 0.0))
    global_count = max(0, int(global_family_counts.get(family, 0)))
    realized_gain = float(family_realized_gain_history.get(family, 0.0))
    recent_repeat_penalty = 0.08 * sum(
        1 for recent_family in recent_families[-2:] if recent_family == family
    )
    candidate_bucket = _fair_probe_bucket(family)
    bucket_global_count = _fair_bucket_global_count(
        family_names=family_names,
        global_family_counts=global_family_counts,
        bucket=candidate_bucket,
    )
    bucket_coverage_bonus = _fair_cross_bucket_bonus(
        candidate_bucket,
        family_counts=family_counts,
    )
    bucket_overuse_penalty = 0.02 * max(0, bucket_global_count - 1)
    # After the first probe, fair mode should still prefer unseen or complementary
    # families, but it should no longer converge toward a flat global count table.
    global_overuse_penalty = 0.01 * float(max(0, global_count))
    seen_families = {
        name
        for name, count in family_counts.items()
        if int(count) > 0
    }
    complement_bonus = 0.0
    if seen_families:
        if any(name in {"boundary_push", "cart_brake"} for name in seen_families):
            if family in {"chirp", "impulse_left", "impulse_right", "counter_balance"}:
                complement_bonus = 0.10
        elif any(name in {"chirp", "impulse_left", "impulse_right"} for name in seen_families):
            if family in {"boundary_push", "cart_brake", "counter_balance"}:
                complement_bonus = 0.10
    value_score = (
        0.50 * future_probe_gain
        + 0.32 * predicted_split_reduction
        + 0.22 * predicted_mechanics_reduction
        + 0.18 * value_per_probe_step
        + 0.12 * predicted_marginal_value
        + 0.18 * realized_gain
        + complement_bonus
        + bucket_coverage_bonus
        - recent_repeat_penalty
        - global_overuse_penalty
        - bucket_overuse_penalty
    )
    sample_efficiency_score = value_score / max(estimated_probe_cost, 1e-6) ** 0.5
    return (
        -sample_efficiency_score,
        -value_per_probe_step,
        -future_probe_gain,
        -predicted_split_reduction,
        -predicted_mechanics_reduction,
        -hypothesis_separation,
        -entropy_reduction,
        estimated_probe_cost,
        -(
            selection_tiebreak
            + surprise_tiebreak
            - 0.5 * global_overuse_penalty
        ),
        -score,
        family,
    )


def _fair_probe_bucket(family: str) -> str:
    """Keep the fair first probe spread across a few simple response families."""
    family_name = str(family)
    if family_name == "center":
        return "center"
    if family_name.startswith("neg_"):
        return "negative"
    if family_name.startswith("pos_"):
        return "positive"
    if family in {"boundary_push", "cart_brake"}:
        return "boundary"
    if family in {"chirp", "impulse_left", "impulse_right"}:
        return "impulse"
    if family == "counter_balance":
        return "center"
    return "other"


def _is_scalar_box_probe_family(family: str) -> bool:
    """Return whether a family name came from a one-dimensional Box action grid."""
    family_name = str(family)
    return family_name == "center" or family_name.startswith("neg_") or family_name.startswith("pos_")


def _is_scalar_directional_probe_family(family: str) -> bool:
    """Return whether a scalar Box probe actually pushes away from center."""
    family_name = str(family)
    return family_name.startswith("neg_") or family_name.startswith("pos_")


def _has_scalar_box_probe_families(family_names: tuple[str, ...]) -> bool:
    """Detect scalar Box family vocabularies without tying policy to an env name."""
    active_families = [
        family for family in family_names if family != "passive_decay"
    ]
    return bool(active_families) and all(
        _is_scalar_box_probe_family(family) for family in active_families
    )


def _fair_first_probe_sort_key(
    family: str,
    metrics: dict[str, float],
    *,
    family_names: tuple[str, ...],
    global_family_counts: dict[str, int] | None = None,
    probe_surprise: float = 0.0,
) -> tuple[float, float, float, float, float, str]:
    """Rank first fair probes by response quality with a small anti-collapse prior."""
    global_family_counts = global_family_counts or {}
    predicted_marginal_value, value_per_probe_step, selection_score, score = (
        probe_family_selection_metrics(metrics)
    )
    entropy_reduction = float(metrics.get("predicted_entropy_reduction", 0.0))
    hypothesis_separation = float(metrics.get("predicted_hypothesis_separation", 0.0))
    estimated_probe_cost = float(metrics.get("estimated_probe_cost", 1.0))
    future_probe_gain = float(
        metrics.get(
            "future_gain_for_choice",
            metrics.get(
                "predicted_future_error_reduction",
                metrics.get("future_error_estimate", 0.0),
            ),
        )
    )
    family_preference = float(
        FAIR_MODE_FIRST_PROBE_PREFERENCE.get(family, 0.0)
        + FAIR_MODE_SCALAR_BOX_FIRST_PROBE_PREFERENCE.get(family, 0.0)
    )
    family_count_penalty = 0.05 * max(0, int(global_family_counts.get(family, 0)))
    bucket = _fair_probe_bucket(family)
    bucket_global_count = _fair_bucket_global_count(
        family_names=family_names,
        global_family_counts=global_family_counts,
        bucket=bucket,
    )
    bucket_count_penalty = 0.04 * max(0, bucket_global_count - 1)
    probe_surprise_bonus = 1e-6 * float(max(probe_surprise, 0.0))
    response_score = (
        0.40 * value_per_probe_step
        + 0.25 * selection_score
        + 0.20 * predicted_marginal_value
        + 0.10 * hypothesis_separation
        + 0.05 * score
    )
    return (
        -(response_score + family_preference - family_count_penalty - bucket_count_penalty),
        -entropy_reduction,
        estimated_probe_cost,
        -(future_probe_gain + probe_surprise_bonus),
        -selection_score,
        family,
    )


def _fair_bucket_global_count(
    *,
    family_names: tuple[str, ...],
    global_family_counts: dict[str, int] | None,
    bucket: str,
) -> int:
    """Count how heavily one semantic family bucket has already been used globally."""
    if global_family_counts is None or bucket == "other":
        return 0
    return sum(
        max(0, int(global_family_counts.get(name, 0)))
        for name in family_names
        if _fair_probe_bucket(name) == bucket
    )


def _fair_cross_bucket_bonus(
    bucket: str,
    *,
    family_counts: dict[str, int],
) -> float:
    """Reward the second fair probe for covering a new response family bucket."""
    if bucket == "other":
        return 0.0
    seen_buckets = {
        _fair_probe_bucket(name)
        for name, count in family_counts.items()
        if int(count) > 0 and _fair_probe_bucket(name) != "other"
    }
    if not seen_buckets:
        return 0.0

    bonus = 0.12 if bucket not in seen_buckets else 0.0
    if len(seen_buckets) < 2 and bucket in seen_buckets:
        bonus -= 0.08
    if seen_buckets == {"positive"} and bucket == "negative":
        bonus += 0.08
    elif seen_buckets == {"negative"} and bucket == "positive":
        bonus += 0.08
    elif seen_buckets == {"center"} and bucket in {"negative", "positive"}:
        bonus += 0.05
    return bonus


def choose_fair_probe_family(
    *,
    family_names: tuple[str, ...],
    expected_family_gain: dict[str, dict[str, float]],
    family_counts: dict[str, int],
    probe_count: int,
    global_family_counts: dict[str, int] | None = None,
    family_realized_gain_history: dict[str, float] | None = None,
    recent_families: tuple[str, ...] = (),
    probe_surprise: float = 0.0,
) -> str | None:
    """Choose the fair-mode probe family with a deterministic two-stage policy.

    Fair mode is not allowed to chase the learned family-value head. The first
    probe must come from a fixed high-response shortlist rather than passive
    evidence gathering. If one probe is not enough, the second probe is a
    different active family ranked by:

    1. largest predicted held-out future gain
    2. largest hypothesis separation
    3. largest predicted entropy reduction
    4. lowest estimated probe cost
    """
    if probe_count <= 0:
        scalar_box_families = _has_scalar_box_probe_families(family_names)
        shortlist = [
            family
            for family in FAIR_MODE_FIRST_PROBE_SHORTLIST
            if family in family_names and int(family_counts.get(family, 0)) <= 0
        ]
        if not shortlist:
            active_unseen_families = [
                family
                for family in family_names
                if family != "passive_decay" and int(family_counts.get(family, 0)) <= 0
            ]
            if scalar_box_families:
                directional_families = [
                    family
                    for family in active_unseen_families
                    if _is_scalar_directional_probe_family(family)
                ]
                shortlist = directional_families or active_unseen_families
            else:
                shortlist = active_unseen_families
        if not shortlist:
            return None
        shortlist.sort(
            key=lambda family: _fair_first_probe_sort_key(
                family,
                expected_family_gain.get(family, {}),
                family_names=family_names,
                global_family_counts=global_family_counts,
                probe_surprise=probe_surprise,
            )
        )
        return shortlist[0]
    if probe_count >= FAIR_MODE_PROBE_CAP:
        return None

    if not expected_family_gain:
        return None
    active_families = [
        family
        for family in family_names
        if family != "passive_decay" and int(family_counts.get(family, 0)) <= 0
    ]
    if _has_scalar_box_probe_families(family_names):
        directional_families = [
            family
            for family in active_families
            if _is_scalar_directional_probe_family(family)
        ]
        if directional_families:
            active_families = directional_families
    active_families = [
        family
        for family in active_families
        if probe_family_clears_fair_second_probe_floor(expected_family_gain.get(family, {}))
        and not probe_family_is_suppressed(
            family=family,
            metrics=expected_family_gain.get(family, {}),
            global_family_counts=global_family_counts,
            family_realized_gain_history=family_realized_gain_history,
        )
    ]
    if not active_families:
        return None
    active_families.sort(
        key=lambda family: _fair_probe_two_stage_sort_key(
            family,
            expected_family_gain.get(family, {}),
            family_names=family_names,
            family_counts=family_counts,
            global_family_counts=global_family_counts,
            family_realized_gain_history=family_realized_gain_history,
            recent_families=recent_families,
            probe_surprise=probe_surprise,
        )
    )
    return active_families[0]


def should_stop_probing_fair(
    *,
    probe_count: int,
    min_seed_support: int,
    max_probe_episodes: int,
    uncertainty_scalar: float,
    uncertainty_probe_threshold: float,
    posterior_entropy: float,
    best_expected_gain: float,
    best_entropy_reduction: float,
    best_hypothesis_separation: float,
    best_value_per_probe_step: float,
    best_marginal_value: float | None = None,
    best_selection_score: float | None = None,
    best_realized_gain: float | None = None,
    recent_realized_gain: float | None = None,
    future_probe_error: float,
    support_diversity_ratio: float,
    family_coverage_ratio: float = 1.0,
    min_family_coverage_ratio: float = 0.75,
    has_selectable_family: bool | None = None,
    best_selectable_value_per_probe_step: float | None = None,
    best_selectable_marginal_value: float | None = None,
    best_selectable_selection_score: float | None = None,
    best_selectable_realized_gain: float | None = None,
    best_selectable_family_count: int = 0,
    best_selectable_bad_streak: int = 0,
    only_passive_family_viable: bool = False,
    fair_stop_ready: bool | None = None,
    expression_ready: bool = False,
) -> tuple[bool, str | None]:
    """Apply the fair-mode stop rule: one ready check, one fallback probe, then hand off.

    Fair mode is intentionally conservative and simple. It never buys a third
    probe. After probe one, it can hand off early when the env expression looks
    ready. Otherwise it allows one more probe, then hands off even if the
    resulting expression stays low-confidence.
    """
    fair_probe_cap = min(max(int(max_probe_episodes), 1), FAIR_MODE_PROBE_CAP)
    stop_ready = bool(expression_ready) if fair_stop_ready is None else bool(fair_stop_ready)
    if stop_ready:
        return True, "expression_ready"
    if has_selectable_family is False and probe_count >= 1:
        return True, "fair_two_probe_handoff"
    if probe_count >= fair_probe_cap:
        return True, "fair_two_probe_handoff"
    return False, None
