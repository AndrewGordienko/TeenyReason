"""Language adapter for the decision-local crawler."""

from __future__ import annotations

from dataclasses import dataclass

from .core import (
    BeliefParticle,
    DecisionIntervention,
    DecisionOption,
    PredictedDecisionOutcome,
)


def _single_belief(label: str) -> list[BeliefParticle]:
    return [BeliefParticle(label=label, message=label, weight=1.0)]


def _uniform_belief(labels: tuple[str, ...]) -> list[BeliefParticle]:
    weight = 1.0 / float(max(len(labels), 1))
    return [BeliefParticle(label=label, message=label, weight=weight) for label in labels]


def _filter_belief(
    belief: list[BeliefParticle],
    keep: list[BeliefParticle],
) -> list[BeliefParticle]:
    if not keep:
        return belief
    weight = 1.0 / float(len(keep))
    return [
        BeliefParticle(particle.label, particle.message, weight, particle.metadata)
        for particle in keep
    ]


@dataclass(frozen=True)
class LanguageDecisionLocalAdapter:
    """Rule-conditioned next-symbol adapter."""

    rules: tuple[str, ...] = ("previous_token", "cycle_next", "mirror_token")

    domain: str = "language"
    modality: str = "symbol_sequence"
    hidden_target: str = "sequence_rule_for_next_token"
    score_name: str = "next_token_accuracy"

    def world_for_seed(self, seed: int) -> str:
        return self.rules[int(seed) % len(self.rules)]

    def world_label(self, world: str) -> str:
        return str(world)

    def initial_state(self, world: str, *, seed: int) -> dict[str, object]:
        vocab = ("a", "b", "c", "d")
        first = vocab[int(seed) % len(vocab)]
        last = vocab[(int(seed) + 2) % len(vocab)]
        context = f"{first}{vocab[(int(seed) + 1) % len(vocab)]}{last}"
        return {"vocab": vocab, "context": context}

    def initial_belief(self, state: dict[str, object], *, seed: int) -> list[BeliefParticle]:
        return _uniform_belief(self.rules)

    def decision_options(
        self,
        state: dict[str, object],
        belief: list[BeliefParticle],
    ) -> list[DecisionOption]:
        return [DecisionOption(name=str(token)) for token in state["vocab"]]

    def candidate_interventions(
        self,
        state: dict[str, object],
        belief: list[BeliefParticle],
    ) -> list[DecisionIntervention]:
        return [
            DecisionIntervention("cheap_cloze", "cheap_context", cost=4.0),
            DecisionIntervention("support_span", "active_rule_probe", cost=8.0),
        ]

    def predict_decision(
        self,
        state: dict[str, object],
        option: DecisionOption,
        particle: BeliefParticle,
    ) -> PredictedDecisionOutcome:
        target = _next_token(str(particle.message), state)
        return PredictedDecisionOutcome(utility=1.0 if option.name == target else 0.0)

    def observe_particle(
        self,
        state: dict[str, object],
        intervention: DecisionIntervention,
        particle: BeliefParticle,
        *,
        seed: int,
    ) -> tuple[str, str]:
        return self._observation(str(particle.message), state, intervention)

    def observe_truth(
        self,
        state: dict[str, object],
        intervention: DecisionIntervention,
        world: str,
        *,
        seed: int,
    ) -> tuple[str, str]:
        return self._observation(str(world), state, intervention)

    def update_belief(
        self,
        state: dict[str, object],
        belief: list[BeliefParticle],
        intervention: DecisionIntervention,
        observation: tuple[str, str],
        *,
        seed: int,
    ) -> list[BeliefParticle]:
        keep = [
            particle
            for particle in belief
            if self.observe_particle(state, intervention, particle, seed=seed) == observation
        ]
        return _filter_belief(belief, keep)

    def ablation_beliefs(
        self,
        state: dict[str, object],
        belief: list[BeliefParticle],
        world: str,
        *,
        seed: int,
    ) -> dict[str, list[BeliefParticle]]:
        shuffled = self.rules[(self.rules.index(str(world)) + 1) % len(self.rules)]
        stale = self.world_for_seed(seed - 1)
        return {
            "zero": self.initial_belief(state, seed=seed),
            "shuffled": _single_belief(shuffled),
            "stale": _single_belief(stale),
        }

    def _observation(
        self,
        rule: str,
        state: dict[str, object],
        intervention: DecisionIntervention,
    ) -> tuple[str, str]:
        if intervention.name == "support_span":
            return ("rule", rule)
        return ("next", _next_token(rule, state))


def _next_token(rule: str, state: dict[str, object]) -> str:
    vocab = tuple(str(token) for token in state["vocab"])
    context = str(state["context"])
    if not context:
        return vocab[0]
    if rule == "previous_token":
        return context[-1]
    if rule == "cycle_next":
        idx = vocab.index(context[-1])
        return vocab[(idx + 1) % len(vocab)]
    if rule == "mirror_token":
        return context[0]
    return vocab[0]
