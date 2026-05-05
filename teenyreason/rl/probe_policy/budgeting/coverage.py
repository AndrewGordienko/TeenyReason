"""Probe budget coverage and adaptive-continuation helpers."""

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
