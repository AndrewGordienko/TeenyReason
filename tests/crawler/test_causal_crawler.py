import unittest
from dataclasses import dataclass

from teenyreason.crawler.causal import (
    CausalWorldSpec,
    CounterfactualPrediction,
    Intervention,
    ObservedOutcome,
    WorldBelief,
    run_causal_crawler,
)


@dataclass(frozen=True)
class ToyWorld:
    rule: str


class ToyCausalAdapter:
    spec = CausalWorldSpec(
        domain="toy",
        modality="symbolic",
        hidden_target="add versus subtract rule",
        outcome_name="next_value",
    )

    def world_for_seed(self, seed: int) -> ToyWorld:
        return ToyWorld(rule="add" if int(seed) % 2 == 0 else "subtract")

    def intervention_space(self, world: ToyWorld, *, seed: int) -> tuple[Intervention, ...]:
        return (
            Intervention(name="probe_zero", family="probe", payload={"x": 0}),
            Intervention(name="probe_two", family="probe", payload={"x": 2}),
        )

    def observe(self, world: ToyWorld, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        x = int(intervention.payload["x"])
        value = x + 1 if world.rule == "add" else x - 1
        return ObservedOutcome(intervention=intervention, value=value, cost=intervention.cost)

    def infer_belief(
        self,
        world: ToyWorld,
        observations: tuple[ObservedOutcome, ...],
        *,
        seed: int,
    ) -> WorldBelief:
        first = observations[0]
        x = int(first.intervention.payload["x"])
        label = "add" if int(first.value) > x else "subtract"
        return WorldBelief(label=label, message=label, confidence=1.0, uncertainty=0.0)

    def predict_outcome(
        self,
        world: ToyWorld,
        belief: WorldBelief,
        intervention: Intervention,
        *,
        seed: int,
    ) -> CounterfactualPrediction:
        x = int(intervention.payload["x"])
        value = x + 1 if belief.message == "add" else x - 1
        return CounterfactualPrediction(intervention=intervention, value=value, confidence=belief.confidence)

    def true_outcome(self, world: ToyWorld, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        return self.observe(world, intervention, seed=seed)

    def score_prediction(self, prediction: CounterfactualPrediction, truth: ObservedOutcome) -> float:
        return float(prediction.value == truth.value)

    def world_label(self, world: ToyWorld) -> str:
        return world.rule


class CausalCrawlerTests(unittest.TestCase):
    def test_causal_runner_scores_factor_and_counterfactual_accuracy(self):
        result = run_causal_crawler(ToyCausalAdapter(), seeds=(0, 1, 2, 3))

        self.assertEqual(result["runner"], "run_causal_crawler")
        self.assertEqual(result["factor_decode_accuracy"], 1.0)
        self.assertEqual(result["counterfactual_accuracy"], 1.0)
        self.assertEqual(result["intervention_coverage"], 1.0)
        self.assertGreater(result["mean_total_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
