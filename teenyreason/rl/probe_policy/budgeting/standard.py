"""Standard probe-family ranking and non-fair selection."""

from ....crawler import CrawlerModelBundle
from .economics import (
    probe_family_has_positive_economics,
    probe_family_is_suppressed,
    probe_family_selection_metrics,
)


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
