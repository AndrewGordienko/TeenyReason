"""Board-game adapter for the decision-local crawler."""

from __future__ import annotations

from dataclasses import dataclass

from ..domains.board import (
    RULE_MISERE,
    RULE_NORMAL,
    RULES,
    best_moves,
    valid_moves,
)
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
class BoardDecisionLocalAdapter:
    """Rule-conditioned minimax adapter."""

    domain: str = "board"
    modality: str = "board_game"
    hidden_target: str = "tic_tac_toe_rule_for_minimax"
    score_name: str = "optimal_move_accuracy"

    def world_for_seed(self, seed: int) -> str:
        return RULES[int(seed) % len(RULES)]

    def world_label(self, world: str) -> str:
        return str(world)

    def initial_state(self, world: str, *, seed: int) -> dict[str, object]:
        board = (-1, 0, -1, 1, 1, 0, 0, 0, 0)
        return {"board": board, "x_to_move": False}

    def initial_belief(self, state: dict[str, object], *, seed: int) -> list[BeliefParticle]:
        return _uniform_belief(RULES)

    def decision_options(
        self,
        state: dict[str, object],
        belief: list[BeliefParticle],
    ) -> list[DecisionOption]:
        board = tuple(int(value) for value in state["board"])
        return [
            DecisionOption(name=str(move), payload={"move": int(move)})
            for move in valid_moves(board)
        ]

    def candidate_interventions(
        self,
        state: dict[str, object],
        belief: list[BeliefParticle],
    ) -> list[DecisionIntervention]:
        return [
            DecisionIntervention("rule_probe", "active_rule_probe", cost=1.0),
            DecisionIntervention("line_completion_probe", "cheap_tactical_probe", cost=1.0),
        ]

    def predict_decision(
        self,
        state: dict[str, object],
        option: DecisionOption,
        particle: BeliefParticle,
    ) -> PredictedDecisionOutcome:
        board = tuple(int(value) for value in state["board"])
        x_to_move = bool(state["x_to_move"])
        optimal = best_moves(board, x_to_move, str(particle.message))
        move = int(option.payload.get("move", -1))
        return PredictedDecisionOutcome(utility=1.0 if move in optimal else 0.0)

    def observe_particle(
        self,
        state: dict[str, object],
        intervention: DecisionIntervention,
        particle: BeliefParticle,
        *,
        seed: int,
    ) -> tuple[str, str]:
        return self._observation(str(particle.message), intervention)

    def observe_truth(
        self,
        state: dict[str, object],
        intervention: DecisionIntervention,
        world: str,
        *,
        seed: int,
    ) -> tuple[str, str]:
        return self._observation(str(world), intervention)

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
        shuffled = RULE_MISERE if str(world) == RULE_NORMAL else RULE_NORMAL
        stale = self.world_for_seed(seed - 1)
        return {
            "zero": _single_belief(RULE_NORMAL),
            "shuffled": _single_belief(shuffled),
            "stale": _single_belief(stale),
        }

    def _observation(self, rule: str, intervention: DecisionIntervention) -> tuple[str, str]:
        if intervention.name == "line_completion_probe":
            value = "avoid_line" if rule == RULE_MISERE else "complete_line"
            return ("line_completion", value)
        return ("rule", rule)
