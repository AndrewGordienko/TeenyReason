"""CartPole adapter for the decision-local crawler."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domains.cartpole import (
    PROBE_FAMILIES,
    MechanicsWorld,
    _initial_state,
    _state_cost,
    _step_physics,
    candidate_worlds,
    nominal_world,
    rollout_features,
    world_for_seed,
)
from .core import (
    BeliefParticle,
    DecisionIntervention,
    DecisionOption,
    PredictedDecisionOutcome,
)


def _single_belief(label: str, message: object) -> list[BeliefParticle]:
    return [BeliefParticle(label=label, message=message, weight=1.0)]


def _filter_belief(
    belief: list[BeliefParticle],
    keep: list[BeliefParticle],
) -> list[BeliefParticle]:
    if not keep:
        return belief
    weight = 1.0 / float(len(keep))
    return [
        BeliefParticle(
            label=particle.label,
            message=particle.message,
            weight=weight,
            metadata=particle.metadata,
        )
        for particle in keep
    ]


@dataclass(frozen=True)
class CartPoleDecisionLocalAdapter:
    """Action-conditioned decision adapter over hidden CartPole mechanics."""

    probe_steps: int = 18
    decision_horizon: int = 16

    domain: str = "cartpole"
    modality: str = "rl_state"
    hidden_target: str = "cartpole_mechanics_for_action_choice"
    score_name: str = "negative_short_horizon_state_cost"

    def world_for_seed(self, seed: int) -> MechanicsWorld:
        return world_for_seed(seed)

    def world_label(self, world: MechanicsWorld) -> str:
        return world.label()

    def initial_state(self, world: MechanicsWorld, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(3100 + int(seed))
        best_state = _initial_state(seed + 3100).astype(np.float32)
        best_regret = -1.0
        for _idx in range(72):
            state = rng.uniform(
                low=(-0.18, -0.45, -0.22, -0.55),
                high=(0.18, 0.45, 0.22, 0.55),
                size=(4,),
            ).astype(np.float32)
            regret = self._expected_initial_regret(state)
            if regret > best_regret:
                best_regret = regret
                best_state = state
        return best_state.astype(np.float32)

    def affordance_state(
        self,
        world: MechanicsWorld,
        *,
        seed: int,
        step: int,
        base_state: np.ndarray,
    ) -> np.ndarray:
        rng = np.random.default_rng(7300 + int(seed) + 17 * int(step))
        state = np.asarray(base_state, dtype=np.float32).copy()
        state += rng.uniform(
            low=(-0.025, -0.06, -0.035, -0.08),
            high=(0.025, 0.06, 0.035, 0.08),
            size=(4,),
        ).astype(np.float32)
        state[0] = float(np.clip(state[0], -0.22, 0.22))
        state[1] = float(np.clip(state[1], -0.55, 0.55))
        state[2] = float(np.clip(state[2], -0.28, 0.28))
        state[3] = float(np.clip(state[3], -0.70, 0.70))
        return state.astype(np.float32)

    def initial_belief(self, state: np.ndarray, *, seed: int) -> list[BeliefParticle]:
        worlds = candidate_worlds()
        weight = 1.0 / float(len(worlds))
        return [
            BeliefParticle(label=world.label(), message=world, weight=weight)
            for world in worlds
        ]

    def decision_options(
        self,
        state: np.ndarray,
        belief: list[BeliefParticle],
    ) -> list[DecisionOption]:
        sequences = {
            "hold": (0.0,),
            "push_left": (-1.0,),
            "push_right": (1.0,),
            "left_recover": (-1.0, -1.0, 0.0, 1.0),
            "right_recover": (1.0, 1.0, 0.0, -1.0),
            "chirp_balance": (-0.65, 0.65, -0.35, 0.35),
        }
        return [
            DecisionOption(name=name, payload={"actions": actions})
            for name, actions in sequences.items()
        ]

    def candidate_interventions(
        self,
        state: np.ndarray,
        belief: list[BeliefParticle],
    ) -> list[DecisionIntervention]:
        return [
            DecisionIntervention(name=family, family=family, cost=float(self.probe_steps))
            for family in PROBE_FAMILIES
        ]

    def predict_decision(
        self,
        state: np.ndarray,
        option: DecisionOption,
        particle: BeliefParticle,
    ) -> PredictedDecisionOutcome:
        world = particle.message if isinstance(particle.message, MechanicsWorld) else nominal_world()
        actions = tuple(float(value) for value in option.payload.get("actions", (0.0,)))
        sim = np.asarray(state, dtype=np.float32).copy()
        total_cost = 0.0
        for step in range(int(self.decision_horizon)):
            action = actions[step % len(actions)]
            sim = _step_physics(sim, action, world)
            total_cost += _state_cost(sim)
        return PredictedDecisionOutcome(utility=float(-total_cost))

    def observe_particle(
        self,
        state: np.ndarray,
        intervention: DecisionIntervention,
        particle: BeliefParticle,
        *,
        seed: int,
    ) -> tuple[float, ...]:
        world = particle.message if isinstance(particle.message, MechanicsWorld) else nominal_world()
        return self._signature(world, intervention.family, seed=seed)

    def observe_truth(
        self,
        state: np.ndarray,
        intervention: DecisionIntervention,
        world: MechanicsWorld,
        *,
        seed: int,
    ) -> tuple[float, ...]:
        return self._signature(world, intervention.family, seed=seed)

    def update_belief(
        self,
        state: np.ndarray,
        belief: list[BeliefParticle],
        intervention: DecisionIntervention,
        observation: tuple[float, ...],
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
        state: np.ndarray,
        belief: list[BeliefParticle],
        world: MechanicsWorld,
        *,
        seed: int,
    ) -> dict[str, list[BeliefParticle]]:
        worlds = candidate_worlds()
        shuffled = worlds[(worlds.index(world) + 5) % len(worlds)]
        stale = world_for_seed(seed - 1)
        return {
            "zero": _single_belief("nominal", nominal_world()),
            "shuffled": _single_belief(shuffled.label(), shuffled),
            "stale": _single_belief(stale.label(), stale),
        }

    def _signature(self, world: MechanicsWorld, family: str, *, seed: int) -> tuple[float, ...]:
        features = rollout_features(world, family, seed=seed + 9100, steps=self.probe_steps)
        return tuple(float(value) for value in np.round(features, 5))

    def _expected_initial_regret(self, state: np.ndarray) -> float:
        worlds = candidate_worlds()
        options = self.decision_options(state, [])
        expected_scores = [
            float(
                np.mean(
                    [
                        self.predict_decision(
                            state,
                            option,
                            BeliefParticle(world.label(), world, 1.0),
                        ).utility
                        for world in worlds
                    ]
                )
            )
            for option in options
        ]
        baseline_idx = int(np.argmax(expected_scores))
        regret = 0.0
        for world in worlds:
            particle = BeliefParticle(world.label(), world, 1.0)
            utilities = [
                self.predict_decision(state, option, particle).utility
                for option in options
            ]
            regret += max(utilities) - utilities[baseline_idx]
        return float(regret / float(max(len(worlds), 1)))
