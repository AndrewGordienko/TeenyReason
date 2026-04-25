"""Probe-family selection and budget-stopping helpers.

This module keeps the crawler's family-ranking and probe-budget logic separate
from the PPO training loop so the controller path stays easy to inspect.
"""

from ...crawler import CrawlerModelBundle


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
            metrics.get("value_per_probe_step", metrics.get("selection_score", metrics.get("score", 0.0))),
        )
    )
    value_per_probe_step = float(metrics.get("value_per_probe_step", predicted_marginal_value))
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


def rank_probe_family_candidates(
    crawler_bundle: CrawlerModelBundle,
    expected_family_gain: dict[str, dict[str, float]],
    family_counts: dict[str, int],
    global_family_counts: dict[str, int] | None,
    family_realized_gain_history: dict[str, float] | None = None,
    family_bad_streaks: dict[str, int] | None = None,
    recent_families: tuple[str, ...] = (),
) -> list[str]:
    """Return selectable families ordered by the same penalties the picker uses."""
    global_family_counts = global_family_counts or {}
    family_realized_gain_history = family_realized_gain_history or {}
    family_bad_streaks = family_bad_streaks or {}
    if not expected_family_gain:
        return []
    min_family_count = min(
        (int(family_counts.get(family, 0)) for family in crawler_bundle.family_names),
        default=0,
    )
    total_global_picks = sum(
        max(0, int(global_family_counts.get(family, 0)))
        for family in crawler_bundle.family_names
    )

    def family_priority(item: tuple[str, dict[str, float]]) -> tuple[float, float, float, float, float]:
        family, metrics = item
        predicted_marginal_value, value_per_probe_step, selection_score, score = probe_family_selection_metrics(metrics)
        family_count = int(family_counts.get(family, 0))
        global_count = int(global_family_counts.get(family, 0))
        repeat_penalty = 0.12 * sum(1 for recent_family in recent_families[-2:] if recent_family == family)
        overuse_penalty = 0.04 * max(0, family_count - min_family_count)
        low_realized_penalty = 0.0
        if family_count > 0 and float(family_realized_gain_history.get(family, 0.0)) <= 0.03:
            low_realized_penalty = 0.10
        bad_streak_penalty = 0.10 * max(0, int(family_bad_streaks.get(family, 0)) - 1)
        passive_penalty = 0.0
        if family == "passive_decay":
            passive_penalty += 0.40 if family_count > 0 else 0.0
            passive_target = max(
                1.0,
                float(total_global_picks) / float(max(8 * max(len(crawler_bundle.family_names), 1), 1)),
            )
            passive_penalty += 0.08 * max(0.0, float(global_count) - passive_target)
        primary_value = predicted_marginal_value
        if family_count <= min_family_count:
            primary_value += 0.10
        primary_value -= repeat_penalty + overuse_penalty + low_realized_penalty + bad_streak_penalty + passive_penalty
        return (
            primary_value,
            value_per_probe_step - passive_penalty - bad_streak_penalty,
            selection_score - passive_penalty - bad_streak_penalty,
            score - passive_penalty - bad_streak_penalty,
            -float(family_count),
        )

    candidates: list[tuple[str, dict[str, float]]] = []
    for family, metrics in expected_family_gain.items():
        if not probe_family_has_positive_economics(metrics):
            continue
        if probe_family_is_suppressed(
            family=family,
            metrics=metrics,
            global_family_counts=global_family_counts,
            family_realized_gain_history=family_realized_gain_history,
            family_bad_streaks=family_bad_streaks,
        ):
            continue
        candidates.append((family, metrics))
    candidates.sort(key=family_priority, reverse=True)
    return [family for family, _metrics in candidates]


def selectable_unseen_active_probe_families(
    family_names: tuple[str, ...],
    expected_family_gain: dict[str, dict[str, float]],
    family_counts: dict[str, int],
    global_family_counts: dict[str, int] | None = None,
    family_realized_gain_history: dict[str, float] | None = None,
    family_bad_streaks: dict[str, int] | None = None,
) -> list[str]:
    """Return unseen active families that are still economically worth testing."""
    global_family_counts = global_family_counts or {}
    family_realized_gain_history = family_realized_gain_history or {}
    family_bad_streaks = family_bad_streaks or {}
    families: list[str] = []
    for family in family_names:
        if family == "passive_decay" or int(family_counts.get(family, 0)) > 0:
            continue
        metrics = expected_family_gain.get(family, {})
        if not probe_family_has_positive_economics(metrics):
            continue
        if probe_family_is_suppressed(
            family=family,
            metrics=metrics,
            global_family_counts=global_family_counts,
            family_realized_gain_history=family_realized_gain_history,
            family_bad_streaks=family_bad_streaks,
        ):
            continue
        families.append(family)
    return families


def should_require_seed_probe_family(
    *,
    probe_count: int,
    family_coverage_budget: int,
    family_names: tuple[str, ...],
    expected_family_gain: dict[str, dict[str, float]],
    family_counts: dict[str, int],
    global_family_counts: dict[str, int] | None = None,
    family_realized_gain_history: dict[str, float] | None = None,
    family_bad_streaks: dict[str, int] | None = None,
) -> bool:
    """Keep seed-coverage pressure only while there are useful unseen active families left."""
    if probe_count >= int(family_coverage_budget):
        return False
    return bool(
        selectable_unseen_active_probe_families(
            family_names,
            expected_family_gain,
            family_counts,
            global_family_counts=global_family_counts,
            family_realized_gain_history=family_realized_gain_history,
            family_bad_streaks=family_bad_streaks,
        )
    )


def choose_seed_probe_family(
    family_names: tuple[str, ...],
    family_counts: dict[str, int],
    expected_family_gain: dict[str, dict[str, float]],
    global_family_counts: dict[str, int] | None = None,
    family_realized_gain_history: dict[str, float] | None = None,
    family_bad_streaks: dict[str, int] | None = None,
) -> str | None:
    """Prefer unseen active families only when their economics are actually positive."""
    global_family_counts = global_family_counts or {}
    family_realized_gain_history = family_realized_gain_history or {}
    family_bad_streaks = family_bad_streaks or {}
    unseen_active_families = [
        family
        for family in family_names
        if family != "passive_decay" and family_counts.get(family, 0) <= 0
    ]
    if unseen_active_families and expected_family_gain:
        ranked = sorted(
            [
                family
                for family in unseen_active_families
                if probe_family_has_positive_economics(expected_family_gain.get(family, {}))
                and not probe_family_is_suppressed(
                    family=family,
                    metrics=expected_family_gain.get(family, {}),
                    global_family_counts=global_family_counts,
                    family_realized_gain_history=family_realized_gain_history,
                    family_bad_streaks=family_bad_streaks,
                )
            ],
            key=lambda family: (
                float(expected_family_gain.get(family, {}).get("predicted_marginal_value", 0.0)),
                float(expected_family_gain.get(family, {}).get("value_per_probe_step", 0.0)),
                float(expected_family_gain.get(family, {}).get("score", 0.0)),
            ),
            reverse=True,
        )
        if ranked:
            return ranked[0]
    passive_metrics = expected_family_gain.get("passive_decay", {})
    if (
        "passive_decay" not in family_names
        or family_counts.get("passive_decay", 0) > 0
        or not probe_family_has_positive_economics(passive_metrics)
        or probe_family_is_suppressed(
            family="passive_decay",
            metrics=passive_metrics,
            global_family_counts=global_family_counts,
            family_realized_gain_history=family_realized_gain_history,
            family_bad_streaks=family_bad_streaks,
        )
    ):
        return None
    predicted_marginal_value, value_per_probe_step, _selection_score, _score = probe_family_selection_metrics(passive_metrics)
    total_global_picks = sum(max(0, int(global_family_counts.get(family, 0))) for family in family_names)
    passive_global_count = max(0, int(global_family_counts.get("passive_decay", 0)))
    passive_share = (
        float(passive_global_count) / float(max(total_global_picks, 1))
        if total_global_picks > 0
        else 0.0
    )
    if total_global_picks == 0:
        return "passive_decay"
    if predicted_marginal_value >= 0.04 and value_per_probe_step >= 0.04 and passive_share <= 0.12:
        return "passive_decay"
    return None


def choose_quota_probe_family(
    family_names: tuple[str, ...],
    expected_family_gain: dict[str, dict[str, float]],
    family_counts: dict[str, int],
    global_family_counts: dict[str, int],
    recent_families: tuple[str, ...] = (),
    family_realized_gain_history: dict[str, float] | None = None,
    family_bad_streaks: dict[str, int] | None = None,
) -> str | None:
    """Force specialist coverage only when that family is still worth the spend."""
    family_realized_gain_history = family_realized_gain_history or {}
    family_bad_streaks = family_bad_streaks or {}
    target_families = tuple(
        family
        for family in ("boundary_push", "cart_brake")
        if family in family_names
    )
    if not target_families:
        return None
    total_global_picks = sum(max(0, int(global_family_counts.get(family, 0))) for family in family_names)
    quota_floor = max(1, total_global_picks // max(3 * max(len(family_names), 1), 1))
    eligible_targets = []
    for family in target_families:
        if int(family_counts.get(family, 0)) > 0:
            continue
        if any(recent_family == family for recent_family in recent_families[-2:]):
            continue
        global_count = int(global_family_counts.get(family, 0))
        metrics = expected_family_gain.get(family, {})
        if global_count >= quota_floor:
            continue
        if not probe_family_has_positive_economics(metrics):
            continue
        if probe_family_is_suppressed(
            family=family,
            metrics=metrics,
            global_family_counts=global_family_counts,
            family_realized_gain_history=family_realized_gain_history,
            family_bad_streaks=family_bad_streaks,
        ):
            continue
        predicted_value, value_per_probe_step, _selection_score, _score = probe_family_selection_metrics(metrics)
        eligible_targets.append((family, global_count, predicted_value, value_per_probe_step))
    if not eligible_targets:
        return None
    eligible_targets.sort(key=lambda item: (item[1], -item[3], -item[2], item[0]))
    return eligible_targets[0][0]


def choose_next_probe_family(
    crawler_bundle: CrawlerModelBundle,
    expected_family_gain: dict[str, dict[str, float]],
    family_counts: dict[str, int],
    global_family_counts: dict[str, int] | None,
    require_seed_family: bool,
    allow_quota_family: bool = True,
    family_realized_gain_history: dict[str, float] | None = None,
    family_bad_streaks: dict[str, int] | None = None,
    recent_families: tuple[str, ...] = (),
) -> str | None:
    """Choose the next probe family from viable seed/quota traffic, then ranked value."""
    global_family_counts = global_family_counts or {}
    family_realized_gain_history = family_realized_gain_history or {}
    family_bad_streaks = family_bad_streaks or {}
    if require_seed_family:
        seed_family = choose_seed_probe_family(
            crawler_bundle.family_names,
            family_counts,
            expected_family_gain,
            global_family_counts=global_family_counts,
            family_realized_gain_history=family_realized_gain_history,
            family_bad_streaks=family_bad_streaks,
        )
        if seed_family is not None:
            return seed_family
    if allow_quota_family:
        quota_family = choose_quota_probe_family(
            crawler_bundle.family_names,
            expected_family_gain,
            family_counts,
            global_family_counts,
            recent_families=recent_families,
            family_realized_gain_history=family_realized_gain_history,
            family_bad_streaks=family_bad_streaks,
        )
        if quota_family is not None:
            return quota_family
    ranked_families = rank_probe_family_candidates(
        crawler_bundle,
        expected_family_gain,
        family_counts,
        global_family_counts,
        family_realized_gain_history=family_realized_gain_history,
        family_bad_streaks=family_bad_streaks,
        recent_families=recent_families,
    )
    if not ranked_families:
        return None
    return ranked_families[0]


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
) -> tuple[float, float, float, float, float, float, float, str]:
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
        future_probe_gain
        + 0.32 * predicted_split_reduction
        + 0.22 * predicted_mechanics_reduction
        + 0.18 * realized_gain
        + complement_bonus
        + bucket_coverage_bonus
        - recent_repeat_penalty
        - global_overuse_penalty
        - bucket_overuse_penalty
    )
    return (
        -value_score,
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


def desired_family_coverage_budget(
    family_names: tuple[str, ...],
    max_probe_episodes: int,
    min_seed_support: int,
) -> int:
    """Decide how many early probes should prioritize family coverage."""
    if not family_names:
        return min_seed_support
    max_probe_episodes = max(1, int(max_probe_episodes))
    if max_probe_episodes <= 3:
        return min(
            max_probe_episodes,
            max(int(min_seed_support), min(len(family_names), 2)),
        )
    exploratory_budget = 3 if max_probe_episodes <= 4 else 4
    return min(
        max_probe_episodes,
        max(int(min_seed_support), min(len(family_names), exploratory_budget)),
    )


def minimum_family_coverage_ratio(
    *,
    family_coverage_budget: int,
    min_seed_support: int,
) -> float:
    """Return the minimum family-coverage ratio that still allows early stopping."""
    return min(
        0.75,
        float(max(1, int(min_seed_support))) / float(max(1, int(family_coverage_budget))),
    )


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


def should_continue_probing_adaptive(
    *,
    probe_count: int,
    max_probe_episodes: int,
    uncertainty_scalar: float,
    uncertainty_probe_threshold: float,
    probe_surprise: float,
    surprise_probe_threshold: float,
    best_expected_gain: float,
    future_probe_error: float,
    best_marginal_value: float | None = None,
    family_coverage_ratio: float = 1.0,
    min_family_coverage_ratio: float = 0.75,
) -> tuple[bool, str]:
    """Apply the adaptive budget rule after the seed support set is gathered."""
    if probe_count >= max_probe_episodes:
        return False, "adaptive_expand_cap"
    high_uncertainty = uncertainty_scalar >= uncertainty_probe_threshold
    high_surprise = probe_surprise >= surprise_probe_threshold
    high_expected_gain = best_expected_gain > max(0.20, future_probe_error)
    high_marginal_value = (
        best_marginal_value is not None
        and best_marginal_value > max(0.05, 0.10 * max(float(future_probe_error), 0.0))
    )
    if family_coverage_ratio < min_family_coverage_ratio:
        return True, "adaptive_continue"
    if high_uncertainty or high_surprise or high_expected_gain or high_marginal_value:
        return True, "adaptive_continue"
    return False, "low_uncertainty_low_gain"
