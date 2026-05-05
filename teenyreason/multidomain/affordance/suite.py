"""Four-domain persistent affordance crawler suite."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..decision_crawler import (
    BoardDecisionLocalAdapter,
    CartPoleDecisionLocalAdapter,
    ImageDecisionLocalAdapter,
    LanguageDecisionLocalAdapter,
)
from .core import PersistentAffordanceConfig, run_persistent_affordance_crawler


@dataclass(frozen=True)
class AffordanceCrawlerSuiteConfig:
    """Domain horizons for amortized belief reuse checks."""

    seeds: tuple[int, ...] = tuple(range(8))
    cartpole: PersistentAffordanceConfig = field(
        default_factory=lambda: PersistentAffordanceConfig(
            reuse_horizon=24,
            max_expensive_probes=1,
            cost_weight=1.0,
        )
    )
    language: PersistentAffordanceConfig = field(
        default_factory=lambda: PersistentAffordanceConfig(
            reuse_horizon=16,
            max_expensive_probes=1,
            cost_weight=1.0,
        )
    )
    image: PersistentAffordanceConfig = field(
        default_factory=lambda: PersistentAffordanceConfig(
            reuse_horizon=64,
            max_expensive_probes=1,
            cost_weight=1.0,
        )
    )
    board: PersistentAffordanceConfig = field(
        default_factory=lambda: PersistentAffordanceConfig(
            reuse_horizon=9,
            max_expensive_probes=1,
            cost_weight=1.0,
        )
    )


def run_affordance_crawler_suite(
    config: AffordanceCrawlerSuiteConfig | None = None,
) -> dict[str, object]:
    """Run persistent affordance economics over the standard four domains."""
    config = config or AffordanceCrawlerSuiteConfig()
    pairs = (
        (CartPoleDecisionLocalAdapter(), config.cartpole),
        (LanguageDecisionLocalAdapter(), config.language),
        (ImageDecisionLocalAdapter(), config.image),
        (BoardDecisionLocalAdapter(), config.board),
    )
    results = {
        adapter.domain: run_persistent_affordance_crawler(
            adapter,
            seeds=config.seeds,
            config=domain_config,
        )
        for adapter, domain_config in pairs
    }
    return {
        "schema_version": 1,
        "runner": "run_affordance_crawler_suite",
        "seeds": list(config.seeds),
        **results,
        "summary_rows": [
            affordance_crawler_row(domain_name, result)
            for domain_name, result in results.items()
        ],
    }


def affordance_crawler_row(
    domain_name: str,
    result: dict[str, object] | None,
) -> dict[str, object]:
    """Return a dashboard-ready persistent affordance row."""
    if not isinstance(result, dict):
        return {
            "domain": domain_name,
            "hidden_target": "",
            "modality": "",
            "reuse_horizon": 0,
            "reuse_count": 0.0,
            "baseline_decision_score": 0.0,
            "affordance_decision_score": 0.0,
            "regret_reduction": 0.0,
            "total_probe_cost": 0.0,
            "amortized_probe_cost": 0.0,
            "net_value_after_reuse": 0.0,
            "break_even_reuse_count": None,
            "passive_update_fraction": 0.0,
            "dedicated_probe_fraction": 0.0,
            "verdict": "",
        }
    return {
        "domain": domain_name,
        "hidden_target": str(result.get("hidden_target", "")),
        "modality": str(result.get("modality", "")),
        "reuse_horizon": int(result.get("reuse_horizon", 0)),
        "reuse_count": float(result.get("reuse_count", 0.0)),
        "baseline_decision_score": float(result.get("baseline_decision_score", 0.0)),
        "affordance_decision_score": float(result.get("affordance_decision_score", 0.0)),
        "zero_score": float(result.get("zero_score", 0.0)),
        "shuffled_score": float(result.get("shuffled_score", 0.0)),
        "stale_score": float(result.get("stale_score", 0.0)),
        "oracle_score": float(result.get("oracle_score", 0.0)),
        "regret_before": float(result.get("regret_before", 0.0)),
        "regret_after": float(result.get("regret_after", 0.0)),
        "regret_reduction": float(result.get("regret_reduction", 0.0)),
        "content_lift": float(result.get("content_lift", 0.0)),
        "total_probe_cost": float(result.get("total_probe_cost", 0.0)),
        "amortized_probe_cost": float(result.get("amortized_probe_cost", 0.0)),
        "net_value_after_reuse": float(result.get("net_value_after_reuse", 0.0)),
        "future_adjusted_value": float(result.get("future_adjusted_value", 0.0)),
        "break_even_reuse_count": result.get("break_even_reuse_count"),
        "passive_update_fraction": float(result.get("passive_update_fraction", 0.0)),
        "dedicated_probe_fraction": float(result.get("dedicated_probe_fraction", 0.0)),
        "expensive_probe_count": float(result.get("expensive_probe_count", 0.0)),
        "surprise_mean": float(result.get("surprise_mean", 0.0)),
        "claim_allowed_rate": float(result.get("claim_allowed_rate", 0.0)),
        "verdict": str(result.get("verdict", "")),
    }
