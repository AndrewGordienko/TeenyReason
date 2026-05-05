"""Persistent affordance crawler with amortized curiosity economics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..decision_crawler import (
    BeliefParticle,
    DecisionIntervention,
    DecisionOption,
    PredictedDecisionOutcome,
)
from .scoring import (
    ablation_scores,
    aggregate_break_even,
    belief_status,
    best_option,
    entropy,
    expected_score,
    mean,
    mean_values,
    new_totals,
    normalize,
    oracle_score,
    passive_outcome_update,
    row_verdict,
    score_intervention,
    truth_score,
    verdict,
)


class AffordanceAdapter(Protocol):
    """Minimal domain contract for persistent affordance prediction."""

    domain: str
    modality: str
    hidden_target: str
    score_name: str

    def world_for_seed(self, seed: int) -> Any:
        ...

    def world_label(self, world: Any) -> str:
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


@dataclass(frozen=True)
class PersistentAffordanceConfig:
    """Policy and accounting knobs for one persistent belief scope."""

    reuse_horizon: int = 16
    max_expensive_probes: int = 1
    cost_weight: float = 1.0
    stability_margin: float = 0.12
    disagreement_floor: float = 0.10
    min_future_value: float = 0.0
    require_instability: bool = True
    outcome_bucket_size: float = 0.05
    state_stride: int = 997


@dataclass
class PersistentBelief:
    """Belief memory reused across many decisions in the same scope."""

    particles: list[BeliefParticle]
    cost_paid: float = 0.0
    reuse_count: int = 0
    passive_update_count: int = 0
    expensive_probe_count: int = 0
    surprise_history: list[float] = field(default_factory=list)


def run_persistent_affordance_crawler(
    adapter: AffordanceAdapter,
    *,
    seeds: tuple[int, ...],
    config: PersistentAffordanceConfig | None = None,
) -> dict[str, object]:
    """Run persistent belief reuse and amortized VOI accounting."""
    config = config or PersistentAffordanceConfig()
    rows = [_run_seed(adapter, int(seed), config) for seed in seeds]
    return {
        "schema_version": 1,
        "runner": "run_persistent_affordance_crawler",
        "domain": adapter.domain,
        "modality": adapter.modality,
        "hidden_target": adapter.hidden_target,
        "score_name": adapter.score_name,
        "reuse_horizon": int(config.reuse_horizon),
        "rows": rows,
        "baseline_decision_score": mean(rows, "baseline_decision_score"),
        "affordance_decision_score": mean(rows, "affordance_decision_score"),
        "zero_score": mean(rows, "zero_score"),
        "shuffled_score": mean(rows, "shuffled_score"),
        "stale_score": mean(rows, "stale_score"),
        "oracle_score": mean(rows, "oracle_score"),
        "regret_before": mean(rows, "regret_before"),
        "regret_after": mean(rows, "regret_after"),
        "regret_reduction": mean(rows, "regret_reduction"),
        "total_probe_cost": mean(rows, "total_probe_cost"),
        "amortized_probe_cost": mean(rows, "amortized_probe_cost"),
        "net_value_after_reuse": mean(rows, "net_value_after_reuse"),
        "future_adjusted_value": mean(rows, "future_adjusted_value"),
        "break_even_reuse_count": aggregate_break_even(rows),
        "reuse_count": mean(rows, "reuse_count"),
        "passive_update_fraction": mean(rows, "passive_update_fraction"),
        "dedicated_probe_fraction": mean(rows, "dedicated_probe_fraction"),
        "expensive_probe_count": mean(rows, "expensive_probe_count"),
        "surprise_mean": mean(rows, "surprise_mean"),
        "content_lift": mean(rows, "content_lift"),
        "claim_allowed_rate": mean(rows, "claim_allowed"),
        "verdict": verdict(rows),
    }


def _run_seed(
    adapter: AffordanceAdapter,
    seed: int,
    config: PersistentAffordanceConfig,
) -> dict[str, object]:
    world = adapter.world_for_seed(seed)
    first_state = adapter.initial_state(world, seed=seed)
    memory = PersistentBelief(particles=normalize(adapter.initial_belief(first_state, seed=seed)))
    events: list[dict[str, object]] = []
    totals = new_totals()

    for step in range(int(config.reuse_horizon)):
        state_seed = seed + step * int(config.state_stride)
        state = _state_for_step(adapter, world, first_state, seed=state_seed, step=step)
        baseline_belief = normalize(adapter.initial_belief(state, seed=state_seed))
        baseline_choice = best_option(adapter, state, baseline_belief)
        baseline_score = truth_score(adapter, state, baseline_choice, world)
        oracle = oracle_score(adapter, state, world, baseline_belief)
        maybe_probe = _maybe_run_expensive_probe(
            adapter,
            state,
            world,
            state_seed,
            step,
            memory,
            config,
        )
        if maybe_probe is not None:
            events.append(maybe_probe)

        choice = best_option(adapter, state, memory.particles)
        predicted_score = expected_score(adapter, state, choice, memory.particles)
        affordance_score = truth_score(adapter, state, choice, world)
        surprise = abs(predicted_score - affordance_score)
        memory.surprise_history.append(float(surprise))
        memory.particles = passive_outcome_update(
            adapter,
            state,
            memory.particles,
            choice,
            affordance_score,
            config,
        )
        memory.reuse_count += 1
        memory.passive_update_count += 1

        ablations = ablation_scores(adapter, state, memory.particles, world, state_seed)
        totals["baseline"] += baseline_score
        totals["affordance"] += affordance_score
        totals["oracle"] += oracle
        totals["zero"] += float(ablations.get("zero", baseline_score))
        totals["shuffled"] += float(ablations.get("shuffled", baseline_score))
        totals["stale"] += float(ablations.get("stale", baseline_score))

    reuse_count = max(int(memory.reuse_count), 1)
    total_probe_cost = float(memory.cost_paid)
    regret_before = max(0.0, totals["oracle"] - totals["baseline"])
    regret_after = max(0.0, totals["oracle"] - totals["affordance"])
    regret_reduction = regret_before - regret_after
    per_decision_lift = regret_reduction / float(reuse_count)
    break_even = None
    if per_decision_lift > 0.0 and total_probe_cost > 0.0:
        break_even = total_probe_cost / per_decision_lift
    best_ablation = max(totals["zero"], totals["shuffled"], totals["stale"])
    content_lift = totals["affordance"] - best_ablation
    passive = float(memory.passive_update_count)
    dedicated = float(memory.expensive_probe_count)
    update_total = max(passive + dedicated, 1.0)
    net_value = regret_reduction - total_probe_cost
    claim_allowed = bool(net_value > 0.0 and regret_reduction > 0.0 and content_lift > 0.0)
    return {
        "seed": seed,
        "world_label": adapter.world_label(world),
        "reuse_horizon": int(config.reuse_horizon),
        "reuse_count": int(memory.reuse_count),
        "baseline_decision_score": totals["baseline"] / float(reuse_count),
        "affordance_decision_score": totals["affordance"] / float(reuse_count),
        "zero_score": totals["zero"] / float(reuse_count),
        "shuffled_score": totals["shuffled"] / float(reuse_count),
        "stale_score": totals["stale"] / float(reuse_count),
        "oracle_score": totals["oracle"] / float(reuse_count),
        "regret_before": regret_before,
        "regret_after": regret_after,
        "regret_reduction": regret_reduction,
        "per_decision_regret_reduction": per_decision_lift,
        "total_probe_cost": total_probe_cost,
        "amortized_probe_cost": total_probe_cost / float(reuse_count),
        "net_value_after_reuse": net_value,
        "future_adjusted_value": net_value,
        "break_even_reuse_count": break_even,
        "passive_update_count": int(memory.passive_update_count),
        "expensive_probe_count": int(memory.expensive_probe_count),
        "passive_update_fraction": passive / update_total,
        "dedicated_probe_fraction": dedicated / update_total,
        "surprise_mean": mean_values(memory.surprise_history),
        "content_lift": content_lift / float(reuse_count),
        "claim_allowed": float(claim_allowed),
        "events": events,
        "verdict": row_verdict(claim_allowed, regret_reduction, net_value, events),
    }


def _maybe_run_expensive_probe(
    adapter: AffordanceAdapter,
    state: Any,
    world: Any,
    seed: int,
    step: int,
    memory: PersistentBelief,
    config: PersistentAffordanceConfig,
) -> dict[str, object] | None:
    if int(memory.expensive_probe_count) >= int(config.max_expensive_probes):
        return None
    status = belief_status(adapter, state, memory.particles)
    unstable = (
        float(status["margin"]) < float(config.stability_margin)
        or float(status["disagreement"]) > float(config.disagreement_floor)
    )
    if bool(config.require_instability) and not unstable:
        return None
    remaining_reuse = max(int(config.reuse_horizon) - int(step), 1)
    scored = [
        score_intervention(adapter, state, memory.particles, intervention, seed, remaining_reuse, config)
        for intervention in adapter.candidate_interventions(state, memory.particles)
    ]
    scored.sort(key=lambda item: item["future_adjusted_value"], reverse=True)
    if not scored or float(scored[0]["future_adjusted_value"]) <= float(config.min_future_value):
        return None
    chosen = scored[0]["intervention"]
    observation = adapter.observe_truth(state, chosen, world, seed=seed)
    before_entropy = entropy(memory.particles)
    memory.particles = normalize(adapter.update_belief(state, memory.particles, chosen, observation, seed=seed))
    memory.cost_paid += float(chosen.cost)
    memory.expensive_probe_count += 1
    return {
        "intervention": chosen.name,
        "family": chosen.family,
        "cost": float(chosen.cost),
        "remaining_reuse": int(remaining_reuse),
        "expected_regret_reduction": float(scored[0]["expected_regret_reduction"]),
        "future_adjusted_value": float(scored[0]["future_adjusted_value"]),
        "entropy_before": before_entropy,
        "entropy_after": entropy(memory.particles),
    }


def _state_for_step(
    adapter: AffordanceAdapter,
    world: Any,
    first_state: Any,
    *,
    seed: int,
    step: int,
) -> Any:
    schedule = getattr(adapter, "affordance_state", None)
    if callable(schedule):
        return schedule(world, seed=seed, step=step, base_state=first_state)
    return adapter.initial_state(world, seed=seed)
