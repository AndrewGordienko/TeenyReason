"""Image adapter for the decision-local crawler."""

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
class ImageDecisionLocalAdapter:
    """Tiny visual-factor adapter with hidden class semantics."""

    worlds: tuple[str, ...] = ("normal_semantics", "swapped_semantics", "rotated_semantics")
    features: tuple[str, ...] = ("loop", "bar", "angle")

    domain: str = "image"
    modality: str = "visual_features"
    hidden_target: str = "feature_to_class_semantics"
    score_name: str = "class_accuracy"

    def world_for_seed(self, seed: int) -> str:
        return self.worlds[int(seed) % len(self.worlds)]

    def world_label(self, world: str) -> str:
        return str(world)

    def initial_state(self, world: str, *, seed: int) -> dict[str, object]:
        feature = self.features[(int(seed) + 1) % len(self.features)]
        contrast = 0.35 + 0.1 * (int(seed) % 4)
        return {"feature": feature, "contrast": contrast}

    def initial_belief(self, state: dict[str, object], *, seed: int) -> list[BeliefParticle]:
        return _uniform_belief(self.worlds)

    def decision_options(
        self,
        state: dict[str, object],
        belief: list[BeliefParticle],
    ) -> list[DecisionOption]:
        return [DecisionOption(name=str(label)) for label in (0, 1, 2)]

    def candidate_interventions(
        self,
        state: dict[str, object],
        belief: list[BeliefParticle],
    ) -> list[DecisionIntervention]:
        return [
            DecisionIntervention("center_crop", "cheap_global_shape", cost=4.0),
            DecisionIntervention("prototype_compare", "active_semantic_probe", cost=12.0),
        ]

    def predict_decision(
        self,
        state: dict[str, object],
        option: DecisionOption,
        particle: BeliefParticle,
    ) -> PredictedDecisionOutcome:
        label = _class_for_world(str(particle.message), str(state["feature"]))
        return PredictedDecisionOutcome(utility=1.0 if option.name == str(label) else 0.0)

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
        shuffled = self.worlds[(self.worlds.index(str(world)) + 1) % len(self.worlds)]
        stale = self.world_for_seed(seed - 1)
        return {
            "zero": self.initial_belief(state, seed=seed),
            "shuffled": _single_belief(shuffled),
            "stale": _single_belief(stale),
        }

    def _observation(
        self,
        world: str,
        state: dict[str, object],
        intervention: DecisionIntervention,
    ) -> tuple[str, str]:
        if intervention.name == "prototype_compare":
            label = _class_for_world(world, str(state["feature"]))
            return ("prototype_label", str(label))
        return ("shape", str(state["feature"]))


def _class_for_world(world: str, feature: str) -> int:
    maps = {
        "normal_semantics": {"loop": 0, "bar": 1, "angle": 2},
        "swapped_semantics": {"loop": 1, "bar": 0, "angle": 2},
        "rotated_semantics": {"loop": 2, "bar": 1, "angle": 0},
    }
    return int(maps.get(world, maps["normal_semantics"]).get(feature, 0))
