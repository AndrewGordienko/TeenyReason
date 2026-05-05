"""Generic crawler adapter contract and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ...crawler.causal import (
    CausalWorldSpec,
    CounterfactualPrediction,
    Intervention,
    ObservedOutcome,
    WorldBelief,
    run_causal_crawler,
)


@dataclass(frozen=True)
class AdapterSpec:
    """Static contract between a domain and the generic crawler evaluator."""

    domain: str
    modality: str
    dataset: str
    model_family: str
    hidden_target: str
    metric_name: str
    query_families: tuple[str, ...]
    message_dim: int


@dataclass(frozen=True)
class CrawlScore:
    """One scalar score where higher is better."""

    value: float
    metric_name: str


class CrawlableAdapter(Protocol):
    """Minimal domain boundary for a crawler-driven solver check."""

    spec: AdapterSpec

    def world_for_seed(self, seed: int) -> Any:
        ...

    def collect_evidence(
        self,
        world: Any,
        *,
        seed: int,
        families: tuple[str, ...],
    ) -> Any:
        ...

    def infer_message(
        self,
        evidence: Any,
        *,
        seed: int,
        families: tuple[str, ...],
    ) -> Any:
        ...

    def ablation_messages(self, message: Any, world: Any, *, seed: int) -> dict[str, Any]:
        ...

    def score_baseline(self, world: Any, *, seed: int) -> CrawlScore:
        ...

    def score_with_message(self, world: Any, message: Any, *, seed: int) -> CrawlScore:
        ...

    def world_label(self, world: Any) -> str:
        ...

    def message_label(self, message: Any) -> str:
        ...


def run_crawler_adapter(
    adapter: CrawlableAdapter,
    *,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    """Run a domain adapter through the shared crawler-message evaluation."""
    rows = [_run_seed(adapter, int(seed)) for seed in seeds]
    causal = run_causal_crawler(
        CrawlableCausalAdapter(adapter),
        seeds=tuple(int(seed) for seed in seeds),
    )
    return {
        "domain": adapter.spec.domain,
        "dataset": adapter.spec.dataset,
        "model_family": adapter.spec.model_family,
        "hidden_target": adapter.spec.hidden_target,
        "metric_name": adapter.spec.metric_name,
        "rows": rows,
        "adapter_contract": {
            "domain": adapter.spec.domain,
            "modality": adapter.spec.modality,
            "hidden_target": adapter.spec.hidden_target,
            "query_families": list(adapter.spec.query_families),
            "message_dim": int(adapter.spec.message_dim),
            "metric_name": adapter.spec.metric_name,
            "runner": "run_crawler_adapter",
        },
        "decode_accuracy": _mean(rows, "decode_accuracy"),
        "subset_agreement": _mean(rows, "subset_agreement"),
        "baseline_value": _mean(rows, "baseline_value"),
        "belief_value": _mean(rows, "belief_value"),
        "zero_value": _mean(rows, "zero_value"),
        "shuffled_value": _mean(rows, "shuffled_value"),
        "stale_value": _mean(rows, "stale_value"),
        "solver_gain": _mean(rows, "solver_gain"),
        "content_lift": _mean(rows, "content_lift"),
        "causal_world_model": causal,
    }


class CrawlableCausalAdapter:
    """Expose the legacy bridge adapter through the general causal contract."""

    def __init__(self, adapter: CrawlableAdapter):
        self.adapter = adapter
        self.spec = CausalWorldSpec(
            domain=adapter.spec.domain,
            modality=adapter.spec.modality,
            hidden_target=adapter.spec.hidden_target,
            outcome_name=adapter.spec.metric_name,
        )

    def world_for_seed(self, seed: int) -> Any:
        return self.adapter.world_for_seed(seed)

    def intervention_space(self, world: Any, *, seed: int) -> tuple[Intervention, ...]:
        return tuple(
            Intervention(name=str(family), family=str(family), cost=1.0)
            for family in self.adapter.spec.query_families
        )

    def observe(self, world: Any, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        value = self.adapter.collect_evidence(world, seed=seed, families=(intervention.family,))
        return ObservedOutcome(
            intervention=intervention,
            value=value,
            cost=float(intervention.cost),
            metadata={"family": intervention.family},
        )

    def infer_belief(
        self,
        world: Any,
        observations: tuple[ObservedOutcome, ...],
        *,
        seed: int,
    ) -> WorldBelief:
        families = tuple(item.intervention.family for item in observations)
        evidence = self.adapter.collect_evidence(world, seed=seed, families=families)
        message = self.adapter.infer_message(evidence, seed=seed, families=families)
        label = self.adapter.message_label(message)
        confidence = 1.0 if label == self.adapter.world_label(world) else 0.0
        return WorldBelief(
            label=label,
            message=message,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            metadata={"families": list(families)},
        )

    def predict_outcome(
        self,
        world: Any,
        belief: WorldBelief,
        intervention: Intervention,
        *,
        seed: int,
    ) -> CounterfactualPrediction:
        score = self.adapter.score_with_message(world, belief.message, seed=seed)
        return CounterfactualPrediction(
            intervention=intervention,
            value={
                "label": self.adapter.message_label(belief.message),
                "score": float(score.value),
            },
            confidence=float(belief.confidence),
        )

    def true_outcome(self, world: Any, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        truth_message = self._truth_message(world, seed)
        score = self.adapter.score_with_message(world, truth_message, seed=seed)
        return ObservedOutcome(
            intervention=intervention,
            value={
                "label": self.adapter.world_label(world),
                "score": float(score.value),
            },
            cost=0.0,
            metadata={"truth_message": self.adapter.message_label(truth_message)},
        )

    def score_prediction(
        self,
        prediction: CounterfactualPrediction,
        truth: ObservedOutcome,
    ) -> float:
        pred = prediction.value if isinstance(prediction.value, dict) else {}
        target = truth.value if isinstance(truth.value, dict) else {}
        label_score = float(str(pred.get("label", "")) == str(target.get("label", "")))
        pred_score = float(pred.get("score", 0.0))
        truth_score = float(target.get("score", 0.0))
        scale = max(abs(pred_score), abs(truth_score), 1.0)
        value_score = max(0.0, 1.0 - abs(pred_score - truth_score) / scale)
        return 0.65 * label_score + 0.35 * value_score

    def world_label(self, world: Any) -> str:
        return self.adapter.world_label(world)

    def _truth_message(self, world: Any, seed: int) -> Any:
        label = self.adapter.world_label(world)
        try:
            self.adapter.score_with_message(world, label, seed=seed)
            return label
        except Exception:
            return world


def _run_seed(adapter: CrawlableAdapter, seed: int) -> dict[str, object]:
    world = adapter.world_for_seed(seed)
    families = adapter.spec.query_families
    evidence = adapter.collect_evidence(world, seed=seed, families=families)
    message = adapter.infer_message(evidence, seed=seed, families=families)
    baseline = adapter.score_baseline(world, seed=seed)
    belief = adapter.score_with_message(world, message, seed=seed)
    ablation_scores = {
        name: adapter.score_with_message(world, ablated, seed=seed).value
        for name, ablated in adapter.ablation_messages(message, world, seed=seed).items()
    }
    best_ablation = max(ablation_scores.values()) if ablation_scores else belief.value
    first_message = _subset_message(adapter, world, seed, families[::2])
    second_message = _subset_message(adapter, world, seed, families[1::2])
    decoded = adapter.message_label(message)
    return {
        "seed": seed,
        "hidden_rule": adapter.world_label(world),
        "decoded_rule": decoded,
        "decode_accuracy": float(decoded == adapter.world_label(world)),
        "subset_agreement": float(
            adapter.message_label(first_message) == decoded
            and adapter.message_label(second_message) == decoded
        ),
        "baseline_value": baseline.value,
        "belief_value": belief.value,
        "zero_value": float(ablation_scores.get("zero", 0.0)),
        "shuffled_value": float(ablation_scores.get("shuffled", 0.0)),
        "stale_value": float(ablation_scores.get("stale", 0.0)),
        "solver_gain": belief.value - baseline.value,
        "content_lift": belief.value - best_ablation,
        "metric_name": belief.metric_name,
        "query_count": int(len(families)),
    }


def _subset_message(
    adapter: CrawlableAdapter,
    world: Any,
    seed: int,
    families: tuple[str, ...],
) -> Any:
    if not families:
        families = adapter.spec.query_families
    evidence = adapter.collect_evidence(world, seed=seed, families=families)
    return adapter.infer_message(evidence, seed=seed, families=families)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))
