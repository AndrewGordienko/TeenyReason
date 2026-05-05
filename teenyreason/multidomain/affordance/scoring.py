"""Scoring helpers for persistent affordance crawling."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..decision_crawler import (
    BeliefParticle,
    DecisionIntervention,
    DecisionOption,
)


def score_intervention(
    adapter: Any,
    state: Any,
    belief: list[BeliefParticle],
    intervention: DecisionIntervention,
    seed: int,
    remaining_reuse: int,
    config: Any,
) -> dict[str, object]:
    current_regret = expected_regret(adapter, state, belief)
    expected_after = 0.0
    for observation, probability in observation_distribution(adapter, state, belief, intervention, seed):
        posterior = normalize(adapter.update_belief(state, belief, intervention, observation, seed=seed))
        expected_after += probability * expected_regret(adapter, state, posterior)
    reduction = current_regret - expected_after
    future_value = reduction * float(remaining_reuse) - float(config.cost_weight) * float(intervention.cost)
    return {
        "intervention": intervention,
        "expected_regret_reduction": reduction,
        "future_adjusted_value": future_value,
    }


def passive_outcome_update(
    adapter: Any,
    state: Any,
    belief: list[BeliefParticle],
    choice: DecisionOption,
    observed_score: float,
    config: Any,
) -> list[BeliefParticle]:
    observed_bucket = bucket(observed_score, config)
    keep = [
        particle
        for particle in belief
        if bucket(utility(adapter, state, choice, particle), config) == observed_bucket
    ]
    return normalize(keep or belief)


def observation_distribution(
    adapter: Any,
    state: Any,
    belief: list[BeliefParticle],
    intervention: DecisionIntervention,
    seed: int,
) -> list[tuple[Any, float]]:
    buckets: dict[str, tuple[Any, float]] = {}
    for particle in normalize(belief):
        observation = adapter.observe_particle(state, intervention, particle, seed=seed)
        key = repr(observation)
        previous = buckets.get(key)
        weight = max(0.0, float(particle.weight))
        if previous is None:
            buckets[key] = (observation, weight)
        else:
            buckets[key] = (previous[0], previous[1] + weight)
    total = sum(value[1] for value in buckets.values())
    if total <= 0.0:
        return []
    return [(observation, weight / total) for observation, weight in buckets.values()]


def expected_regret(
    adapter: Any,
    state: Any,
    belief: list[BeliefParticle],
) -> float:
    normalized = normalize(belief)
    choice = best_option(adapter, state, normalized)
    options = adapter.decision_options(state, normalized)
    regret = 0.0
    for particle in normalized:
        best = max(utility(adapter, state, option, particle) for option in options)
        chosen = utility(adapter, state, choice, particle)
        regret += max(0.0, best - chosen) * max(0.0, float(particle.weight))
    return regret


def belief_status(
    adapter: Any,
    state: Any,
    belief: list[BeliefParticle],
) -> dict[str, float]:
    options = adapter.decision_options(state, belief)
    scores = [(option, expected_score(adapter, state, option, belief)) for option in options]
    scores.sort(key=lambda item: item[1], reverse=True)
    best = scores[0][1] if scores else 0.0
    second = scores[1][1] if len(scores) > 1 else best
    best_name = scores[0][0].name if scores else ""
    winners = [
        particle_best_option(adapter, state, particle, belief).name
        for particle in normalize(belief)
    ]
    disagreement = float(np.mean([name != best_name for name in winners])) if winners else 0.0
    return {"margin": float(best - second), "disagreement": disagreement}


def best_option(
    adapter: Any,
    state: Any,
    belief: list[BeliefParticle],
) -> DecisionOption:
    options = adapter.decision_options(state, belief)
    if not options:
        return DecisionOption("none")
    return max(options, key=lambda option: expected_score(adapter, state, option, belief))


def particle_best_option(
    adapter: Any,
    state: Any,
    particle: BeliefParticle,
    belief: list[BeliefParticle],
) -> DecisionOption:
    options = adapter.decision_options(state, belief)
    if not options:
        return DecisionOption("none")
    return max(options, key=lambda option: utility(adapter, state, option, particle))


def expected_score(
    adapter: Any,
    state: Any,
    option: DecisionOption,
    belief: list[BeliefParticle],
) -> float:
    return sum(
        utility(adapter, state, option, particle) * max(0.0, float(particle.weight))
        for particle in normalize(belief)
    )


def utility(
    adapter: Any,
    state: Any,
    option: DecisionOption,
    particle: BeliefParticle,
) -> float:
    outcome = adapter.predict_decision(state, option, particle)
    return float(outcome.utility) - float(outcome.risk)


def truth_score(
    adapter: Any,
    state: Any,
    option: DecisionOption,
    world: Any,
) -> float:
    truth = BeliefParticle(adapter.world_label(world), world, 1.0)
    return utility(adapter, state, option, truth)


def oracle_score(
    adapter: Any,
    state: Any,
    world: Any,
    belief: list[BeliefParticle],
) -> float:
    options = adapter.decision_options(state, belief)
    if not options:
        return 0.0
    return max(truth_score(adapter, state, option, world) for option in options)


def ablation_scores(
    adapter: Any,
    state: Any,
    belief: list[BeliefParticle],
    world: Any,
    seed: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, ablation in adapter.ablation_beliefs(state, belief, world, seed=seed).items():
        choice = best_option(adapter, state, normalize(ablation))
        scores[name] = truth_score(adapter, state, choice, world)
    return scores


def normalize(belief: list[BeliefParticle]) -> list[BeliefParticle]:
    if not belief:
        return []
    total = sum(max(0.0, float(particle.weight)) for particle in belief)
    if total <= 0.0:
        weight = 1.0 / float(len(belief))
        return [
            BeliefParticle(particle.label, particle.message, weight, particle.metadata)
            for particle in belief
        ]
    return [
        BeliefParticle(
            particle.label,
            particle.message,
            max(0.0, float(particle.weight)) / total,
            particle.metadata,
        )
        for particle in belief
    ]


def entropy(belief: list[BeliefParticle]) -> float:
    weights = [max(0.0, float(particle.weight)) for particle in normalize(belief)]
    return float(-sum(weight * np.log2(max(weight, 1e-12)) for weight in weights))


def bucket(value: float, config: Any) -> int:
    size = max(float(config.outcome_bucket_size), 1e-9)
    return int(round(float(value) / size))


def new_totals() -> dict[str, float]:
    return {
        "baseline": 0.0,
        "affordance": 0.0,
        "oracle": 0.0,
        "zero": 0.0,
        "shuffled": 0.0,
        "stale": 0.0,
    }


def verdict(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "affordance_not_diagnostic"
    net_value = mean(rows, "net_value_after_reuse")
    regret = mean(rows, "regret_reduction")
    content = mean(rows, "content_lift")
    if net_value > 0.0 and regret > 0.0 and content > 0.0:
        return "persistent_affordance_economics_positive"
    if regret > 0.0 and net_value <= 0.0:
        return "belief_useful_but_unamortized"
    if content <= 0.0:
        return "not_ablation_clean"
    return "no_affordance_value"


def row_verdict(
    claim_allowed: bool,
    regret_reduction: float,
    net_value: float,
    events: list[dict[str, object]],
) -> str:
    if claim_allowed:
        return "persistent_affordance_economics_positive"
    if regret_reduction > 0.0 and net_value <= 0.0:
        return "belief_useful_but_unamortized"
    if events:
        return "probe_ran_without_net_value"
    return "no_affordance_value"


def mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0) or 0.0) for row in rows]))


def mean_defined(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def mean_values(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean([float(value) for value in values]))


def aggregate_break_even(rows: list[dict[str, object]]) -> float | None:
    reuse = mean(rows, "reuse_count")
    regret = mean(rows, "regret_reduction")
    cost = mean(rows, "total_probe_cost")
    if reuse <= 0.0 or regret <= 0.0 or cost <= 0.0:
        return None
    return float(cost / (regret / reuse))
