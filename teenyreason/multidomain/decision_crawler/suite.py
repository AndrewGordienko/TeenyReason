"""Suite wrapper for the decision-local curiosity crawler."""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import BoardDecisionLocalAdapter
from .cartpole import CartPoleDecisionLocalAdapter
from .core import DecisionLocalCrawlerConfig, run_decision_local_crawler
from .image import ImageDecisionLocalAdapter
from .language import LanguageDecisionLocalAdapter


@dataclass(frozen=True)
class DecisionLocalCrawlerSuiteConfig:
    """Configuration for the four-domain decision-local crawler check."""

    seeds: tuple[int, ...] = tuple(range(8))
    crawler: DecisionLocalCrawlerConfig = field(
        default_factory=lambda: DecisionLocalCrawlerConfig(
            max_interventions=2,
            stability_margin=0.12,
            disagreement_floor=0.10,
            min_probe_value=0.0,
            cost_weight=0.01,
        )
    )


def run_decision_local_crawler_suite(
    config: DecisionLocalCrawlerSuiteConfig | None = None,
) -> dict[str, object]:
    """Run the same crawler loop over RL, language, image, and board adapters."""
    config = config or DecisionLocalCrawlerSuiteConfig()
    adapters = (
        CartPoleDecisionLocalAdapter(),
        LanguageDecisionLocalAdapter(),
        ImageDecisionLocalAdapter(),
        BoardDecisionLocalAdapter(),
    )
    results = {
        adapter.domain: run_decision_local_crawler(
            adapter,
            seeds=config.seeds,
            config=config.crawler,
        )
        for adapter in adapters
    }
    return {
        "schema_version": 1,
        "runner": "run_decision_local_crawler_suite",
        "seeds": list(config.seeds),
        "config": {
            "max_interventions": config.crawler.max_interventions,
            "stability_margin": config.crawler.stability_margin,
            "disagreement_floor": config.crawler.disagreement_floor,
            "min_probe_value": config.crawler.min_probe_value,
            "cost_weight": config.crawler.cost_weight,
        },
        **results,
        "summary_rows": [
            decision_local_crawler_row(domain_name, result)
            for domain_name, result in results.items()
        ],
    }


def decision_local_crawler_row(
    domain_name: str,
    result: dict[str, object] | None,
) -> dict[str, object]:
    """Return the dashboard row for one decision-local crawler result."""
    if not isinstance(result, dict):
        return {
            "domain": domain_name,
            "hidden_target": "",
            "modality": "",
            "baseline_decision_score": 0.0,
            "crawler_decision_score": 0.0,
            "regret_reduction": 0.0,
            "content_lift": 0.0,
            "voi": 0.0,
            "intervention_count": 0.0,
            "intervention_cost": 0.0,
            "net_sample_savings": 0.0,
            "entropy_reduction": 0.0,
            "claim_allowed_rate": 0.0,
            "verdict": "",
        }
    return {
        "domain": domain_name,
        "hidden_target": str(result.get("hidden_target", "")),
        "modality": str(result.get("modality", "")),
        "baseline_decision_score": float(result.get("baseline_decision_score", 0.0)),
        "crawler_decision_score": float(result.get("crawler_decision_score", 0.0)),
        "zero_score": float(result.get("zero_score", 0.0)),
        "shuffled_score": float(result.get("shuffled_score", 0.0)),
        "stale_score": float(result.get("stale_score", 0.0)),
        "regret_before": float(result.get("regret_before", 0.0)),
        "regret_after": float(result.get("regret_after", 0.0)),
        "regret_reduction": float(result.get("regret_reduction", 0.0)),
        "content_lift": float(result.get("content_lift", 0.0)),
        "voi": float(result.get("voi", 0.0)),
        "intervention_count": float(result.get("intervention_count", 0.0)),
        "intervention_cost": float(result.get("intervention_cost", 0.0)),
        "net_sample_savings": float(result.get("net_sample_savings", 0.0)),
        "decision_changed_fraction": float(result.get("decision_changed_fraction", 0.0)),
        "entropy_reduction": float(result.get("belief_entropy_reduction", 0.0)),
        "claim_allowed_rate": float(result.get("claim_allowed_rate", 0.0)),
        "stable_decision_rate": float(result.get("stable_decision_rate", 0.0)),
        "probe_worth_it_rate": float(result.get("probe_worth_it_rate", 0.0)),
        "verdict": str(result.get("verdict", "")),
    }
