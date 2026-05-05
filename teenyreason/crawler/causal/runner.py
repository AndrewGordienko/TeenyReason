"""Modality-independent causal crawler evaluator."""

from __future__ import annotations

from .types import CausalWorldAdapter, ObservedOutcome


def run_causal_crawler(
    adapter: CausalWorldAdapter,
    *,
    seeds: tuple[int, ...],
) -> dict[str, object]:
    """Evaluate factor discovery and counterfactual prediction for an adapter."""
    rows = [_run_seed(adapter, int(seed)) for seed in seeds]
    return {
        "schema_version": 1,
        "runner": "run_causal_crawler",
        "domain": adapter.spec.domain,
        "modality": adapter.spec.modality,
        "hidden_target": adapter.spec.hidden_target,
        "outcome_name": adapter.spec.outcome_name,
        "rows": rows,
        "factor_decode_accuracy": _mean(rows, "factor_decode"),
        "counterfactual_accuracy": _mean(rows, "counterfactual_accuracy"),
        "intervention_coverage": _mean(rows, "intervention_coverage"),
        "mean_total_cost": _mean(rows, "total_cost"),
        "mean_cost_per_intervention": _mean(rows, "cost_per_intervention"),
        "understanding_score": _mean(rows, "understanding_score"),
    }


def _run_seed(adapter: CausalWorldAdapter, seed: int) -> dict[str, object]:
    world = adapter.world_for_seed(seed)
    interventions = adapter.intervention_space(world, seed=seed)
    observations = tuple(adapter.observe(world, item, seed=seed) for item in interventions)
    belief = adapter.infer_belief(world, observations, seed=seed)
    truth_label = adapter.world_label(world)
    prediction_scores = _prediction_scores(adapter, world, belief, interventions, seed)
    total_cost = sum(max(0.0, float(item.cost)) for item in observations)
    factor_decode = float(str(belief.label) == str(truth_label))
    counterfactual_accuracy = _mean_values(prediction_scores)
    return {
        "seed": seed,
        "world_label": truth_label,
        "belief_label": str(belief.label),
        "factor_decode": factor_decode,
        "belief_confidence": float(belief.confidence),
        "belief_uncertainty": float(belief.uncertainty),
        "counterfactual_accuracy": counterfactual_accuracy,
        "intervention_coverage": float(len(observations) > 0),
        "intervention_count": int(len(observations)),
        "total_cost": total_cost,
        "cost_per_intervention": total_cost / max(float(len(observations)), 1.0),
        "understanding_score": 0.55 * factor_decode + 0.45 * counterfactual_accuracy,
    }


def _prediction_scores(
    adapter: CausalWorldAdapter,
    world: object,
    belief,
    interventions,
    seed: int,
) -> list[float]:
    scores: list[float] = []
    for intervention in interventions:
        prediction = adapter.predict_outcome(world, belief, intervention, seed=seed)
        truth = adapter.true_outcome(world, intervention, seed=seed)
        scores.append(_clip01(adapter.score_prediction(prediction, truth)))
    return scores


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return _mean_values([float(row.get(key, 0.0)) for row in rows])


def _mean_values(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
