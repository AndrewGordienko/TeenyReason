"""Generic decision-local curiosity crawler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class DecisionOption:
    """One decision the solver could make now."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionIntervention:
    """One observation/action the crawler can buy before deciding."""

    name: str
    family: str
    cost: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeliefParticle:
    """One plausible latent-world hypothesis."""

    label: str
    message: Any
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictedDecisionOutcome:
    """Predicted utility/risk of a decision under one particle."""

    utility: float
    risk: float = 0.0
    uncertainty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionLocalCrawlerConfig:
    """Shared crawler policy knobs."""

    max_interventions: int = 2
    stability_margin: float = 0.15
    disagreement_floor: float = 0.15
    min_probe_value: float = 0.0
    cost_weight: float = 0.01


class DecisionLocalAdapter(Protocol):
    """Domain boundary for the decision-local crawler."""

    domain: str
    modality: str
    hidden_target: str
    score_name: str

    def world_for_seed(self, seed: int) -> Any:
        ...

    def initial_state(self, world: Any, *, seed: int) -> Any:
        ...

    def initial_belief(self, state: Any, *, seed: int) -> list[BeliefParticle]:
        ...

    def decision_options(self, state: Any, belief: list[BeliefParticle]) -> list[DecisionOption]:
        ...

    def candidate_interventions(
        self,
        state: Any,
        belief: list[BeliefParticle],
    ) -> list[DecisionIntervention]:
        ...

    def predict_decision(
        self,
        state: Any,
        option: DecisionOption,
        particle: BeliefParticle,
    ) -> PredictedDecisionOutcome:
        ...

    def observe_particle(
        self,
        state: Any,
        intervention: DecisionIntervention,
        particle: BeliefParticle,
        *,
        seed: int,
    ) -> Any:
        ...

    def observe_truth(
        self,
        state: Any,
        intervention: DecisionIntervention,
        world: Any,
        *,
        seed: int,
    ) -> Any:
        ...

    def update_belief(
        self,
        state: Any,
        belief: list[BeliefParticle],
        intervention: DecisionIntervention,
        observation: Any,
        *,
        seed: int,
    ) -> list[BeliefParticle]:
        ...

    def ablation_beliefs(
        self,
        state: Any,
        belief: list[BeliefParticle],
        world: Any,
        *,
        seed: int,
    ) -> dict[str, list[BeliefParticle]]:
        ...

    def world_label(self, world: Any) -> str:
        ...


def run_decision_local_crawler(
    adapter: DecisionLocalAdapter,
    *,
    seeds: tuple[int, ...],
    config: DecisionLocalCrawlerConfig | None = None,
) -> dict[str, object]:
    """Run the shared decision-local curiosity loop for one adapter."""
    config = config or DecisionLocalCrawlerConfig()
    rows = [_run_seed(adapter, int(seed), config) for seed in seeds]
    return {
        "schema_version": 1,
        "runner": "run_decision_local_crawler",
        "domain": adapter.domain,
        "modality": adapter.modality,
        "hidden_target": adapter.hidden_target,
        "score_name": adapter.score_name,
        "rows": rows,
        "baseline_decision_score": _mean(rows, "baseline_decision_score"),
        "crawler_decision_score": _mean(rows, "crawler_decision_score"),
        "zero_score": _mean(rows, "zero_score"),
        "shuffled_score": _mean(rows, "shuffled_score"),
        "stale_score": _mean(rows, "stale_score"),
        "regret_before": _mean(rows, "regret_before"),
        "regret_after": _mean(rows, "regret_after"),
        "regret_reduction": _mean(rows, "regret_reduction"),
        "voi": _mean(rows, "voi"),
        "intervention_count": _mean(rows, "intervention_count"),
        "intervention_cost": _mean(rows, "intervention_cost"),
        "net_sample_savings": _mean(rows, "net_sample_savings"),
        "decision_changed_fraction": _mean(rows, "decision_changed"),
        "belief_entropy_reduction": _mean(rows, "belief_entropy_reduction"),
        "content_lift": _mean(rows, "content_lift"),
        "claim_allowed_rate": _mean(rows, "claim_allowed"),
        "stable_decision_rate": _mean(rows, "stable_decision"),
        "probe_worth_it_rate": _mean(rows, "probe_was_worth_it"),
        "verdict": _verdict(rows),
    }


def _run_seed(
    adapter: DecisionLocalAdapter,
    seed: int,
    config: DecisionLocalCrawlerConfig,
) -> dict[str, object]:
    world = adapter.world_for_seed(seed)
    state = adapter.initial_state(world, seed=seed)
    initial = _normalize(adapter.initial_belief(state, seed=seed))
    belief = list(initial)
    baseline_choice = _best_option(adapter, state, belief)
    baseline_score = _truth_score(adapter, state, baseline_choice, world)
    initial_entropy = _entropy(belief)
    events: list[dict[str, object]] = []

    for _idx in range(int(config.max_interventions)):
        status = _belief_status(adapter, state, belief)
        if _stable(status, config):
            break
        scored = [
            _score_intervention(adapter, state, belief, intervention, seed, config)
            for intervention in adapter.candidate_interventions(state, belief)
        ]
        scored.sort(key=lambda item: item["value"], reverse=True)
        if not scored or float(scored[0]["value"]) <= float(config.min_probe_value):
            break
        chosen = scored[0]["intervention"]
        observation = adapter.observe_truth(state, chosen, world, seed=seed)
        before_entropy = _entropy(belief)
        belief = _normalize(adapter.update_belief(state, belief, chosen, observation, seed=seed))
        events.append(
            {
                "intervention": chosen.name,
                "family": chosen.family,
                "cost": float(chosen.cost),
                "predicted_value": float(scored[0]["value"]),
                "expected_regret_reduction": float(scored[0]["expected_regret_reduction"]),
                "entropy_before": before_entropy,
                "entropy_after": _entropy(belief),
            }
        )

    final_choice = _best_option(adapter, state, belief)
    crawler_score = _truth_score(adapter, state, final_choice, world)
    oracle_score = _oracle_score(adapter, state, world, belief)
    regret_before = max(0.0, oracle_score - baseline_score)
    regret_after = max(0.0, oracle_score - crawler_score)
    ablations = _ablation_scores(adapter, state, belief, world, seed)
    best_ablation = max(ablations.values()) if ablations else baseline_score
    total_cost = sum(float(event["cost"]) for event in events)
    regret_reduction = regret_before - regret_after
    voi = regret_reduction / max(total_cost, 1.0)
    content_lift = crawler_score - best_ablation
    claim_allowed = bool(regret_reduction > 0.0 and content_lift > 0.0 and voi > 0.0)
    return {
        "seed": seed,
        "world_label": adapter.world_label(world),
        "baseline_decision": baseline_choice.name,
        "crawler_decision": final_choice.name,
        "oracle_score": oracle_score,
        "baseline_decision_score": baseline_score,
        "crawler_decision_score": crawler_score,
        "zero_score": float(ablations.get("zero", baseline_score)),
        "shuffled_score": float(ablations.get("shuffled", baseline_score)),
        "stale_score": float(ablations.get("stale", baseline_score)),
        "regret_before": regret_before,
        "regret_after": regret_after,
        "regret_reduction": regret_reduction,
        "voi": voi,
        "intervention_count": int(len(events)),
        "intervention_cost": total_cost,
        "net_sample_savings": regret_reduction - total_cost,
        "decision_changed": float(final_choice.name != baseline_choice.name),
        "belief_entropy_before": initial_entropy,
        "belief_entropy_after": _entropy(belief),
        "belief_entropy_reduction": initial_entropy - _entropy(belief),
        "best_decision_margin": _belief_status(adapter, state, belief)["margin"],
        "stable_decision": float(_stable(_belief_status(adapter, state, belief), config)),
        "probe_was_worth_it": float(bool(events) and regret_reduction > 0.0 and voi > 0.0),
        "content_lift": content_lift,
        "claim_allowed": float(claim_allowed),
        "events": events,
        "verdict": _row_verdict(claim_allowed, regret_reduction, content_lift, events),
    }


def _score_intervention(
    adapter: DecisionLocalAdapter,
    state: Any,
    belief: list[BeliefParticle],
    intervention: DecisionIntervention,
    seed: int,
    config: DecisionLocalCrawlerConfig,
) -> dict[str, object]:
    current_regret = _expected_regret(adapter, state, belief)
    expected_after = 0.0
    for observation, probability in _observation_distribution(adapter, state, belief, intervention, seed):
        posterior = _normalize(adapter.update_belief(state, belief, intervention, observation, seed=seed))
        expected_after += probability * _expected_regret(adapter, state, posterior)
    reduction = current_regret - expected_after
    value = reduction - float(config.cost_weight) * float(intervention.cost)
    return {
        "intervention": intervention,
        "expected_regret_reduction": reduction,
        "value": value,
    }


def _observation_distribution(
    adapter: DecisionLocalAdapter,
    state: Any,
    belief: list[BeliefParticle],
    intervention: DecisionIntervention,
    seed: int,
) -> list[tuple[Any, float]]:
    buckets: dict[str, tuple[Any, float]] = {}
    for particle in _normalize(belief):
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


def _expected_regret(
    adapter: DecisionLocalAdapter,
    state: Any,
    belief: list[BeliefParticle],
) -> float:
    normalized = _normalize(belief)
    choice = _best_option(adapter, state, normalized)
    regret = 0.0
    for particle in normalized:
        options = adapter.decision_options(state, normalized)
        best = max(
            _utility(adapter, state, option, particle)
            for option in options
        )
        chosen = _utility(adapter, state, choice, particle)
        regret += max(0.0, best - chosen) * max(0.0, float(particle.weight))
    return regret


def _belief_status(
    adapter: DecisionLocalAdapter,
    state: Any,
    belief: list[BeliefParticle],
) -> dict[str, float]:
    options = adapter.decision_options(state, belief)
    scores = [(option, _expected_score(adapter, state, option, belief)) for option in options]
    scores.sort(key=lambda item: item[1], reverse=True)
    best = scores[0][1] if scores else 0.0
    second = scores[1][1] if len(scores) > 1 else best
    best_name = scores[0][0].name if scores else ""
    particle_winners = [
        _particle_best_option(adapter, state, particle, belief).name
        for particle in _normalize(belief)
    ]
    disagreement = 0.0
    if particle_winners:
        disagreement = float(np.mean([name != best_name for name in particle_winners]))
    return {
        "margin": float(best - second),
        "disagreement": disagreement,
    }


def _stable(status: dict[str, float], config: DecisionLocalCrawlerConfig) -> bool:
    return (
        float(status.get("margin", 0.0)) >= float(config.stability_margin)
        and float(status.get("disagreement", 0.0)) <= float(config.disagreement_floor)
    )


def _best_option(
    adapter: DecisionLocalAdapter,
    state: Any,
    belief: list[BeliefParticle],
) -> DecisionOption:
    options = adapter.decision_options(state, belief)
    if not options:
        return DecisionOption(name="none")
    return max(options, key=lambda option: _expected_score(adapter, state, option, belief))


def _particle_best_option(
    adapter: DecisionLocalAdapter,
    state: Any,
    particle: BeliefParticle,
    belief: list[BeliefParticle],
) -> DecisionOption:
    options = adapter.decision_options(state, belief)
    if not options:
        return DecisionOption(name="none")
    return max(options, key=lambda option: _utility(adapter, state, option, particle))


def _expected_score(
    adapter: DecisionLocalAdapter,
    state: Any,
    option: DecisionOption,
    belief: list[BeliefParticle],
) -> float:
    return sum(
        _utility(adapter, state, option, particle) * max(0.0, float(particle.weight))
        for particle in _normalize(belief)
    )


def _utility(
    adapter: DecisionLocalAdapter,
    state: Any,
    option: DecisionOption,
    particle: BeliefParticle,
) -> float:
    outcome = adapter.predict_decision(state, option, particle)
    return float(outcome.utility) - float(outcome.risk)


def _truth_score(
    adapter: DecisionLocalAdapter,
    state: Any,
    option: DecisionOption,
    world: Any,
) -> float:
    truth = BeliefParticle(label=adapter.world_label(world), message=world, weight=1.0)
    return _utility(adapter, state, option, truth)


def _oracle_score(
    adapter: DecisionLocalAdapter,
    state: Any,
    world: Any,
    belief: list[BeliefParticle],
) -> float:
    options = adapter.decision_options(state, belief)
    if not options:
        return 0.0
    return max(_truth_score(adapter, state, option, world) for option in options)


def _ablation_scores(
    adapter: DecisionLocalAdapter,
    state: Any,
    belief: list[BeliefParticle],
    world: Any,
    seed: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, ablation in adapter.ablation_beliefs(state, belief, world, seed=seed).items():
        choice = _best_option(adapter, state, _normalize(ablation))
        scores[name] = _truth_score(adapter, state, choice, world)
    return scores


def _normalize(belief: list[BeliefParticle]) -> list[BeliefParticle]:
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


def _entropy(belief: list[BeliefParticle]) -> float:
    weights = [max(0.0, float(particle.weight)) for particle in _normalize(belief)]
    return float(-sum(weight * np.log2(max(weight, 1e-12)) for weight in weights))


def _verdict(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "adapter_not_diagnostic"
    claim_rate = _mean(rows, "claim_allowed")
    regret = _mean(rows, "regret_reduction")
    content = _mean(rows, "content_lift")
    probe_rate = _mean(rows, "probe_was_worth_it")
    if claim_rate >= 0.6:
        return "decision_local_crawler_wins"
    if regret > 0.0 and content <= 0.0:
        return "not_ablation_clean"
    if regret > 0.0 and probe_rate > 0.0:
        return "decision_local_but_costly"
    return "no_positive_probe_value"


def _row_verdict(
    claim_allowed: bool,
    regret_reduction: float,
    content_lift: float,
    events: list[dict[str, object]],
) -> str:
    if claim_allowed:
        return "decision_local_crawler_wins"
    if regret_reduction > 0.0 and content_lift <= 0.0:
        return "not_ablation_clean"
    if events:
        return "decision_local_but_costly"
    return "no_positive_probe_value"


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))
