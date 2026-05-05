import unittest

from teenyreason.cognition.imagination import AcceptanceCalibrator, ImaginationMemory, Proposal, Target, ValidationResult


class ImaginationMemoryTests(unittest.TestCase):
    def test_memory_reports_validation_economics(self):
        memory = ImaginationMemory()
        target = Target(target_id="stable-a", kind="stable_target", utility=2.0, stability=1.0)
        memory.add(
            Proposal(
                proposal_id="p0",
                domain="language",
                context_id="ctx",
                context_latent=(0.0,),
                target=target,
                intervention="rewrite",
                predicted_lift=3.0,
                uncertainty=0.1,
                support_confidence=0.8,
                reachability=0.7,
                consistency=0.9,
                generation_cost=2.0,
            ),
            ValidationResult(
                proposal_id="p0",
                accepted=True,
                real_lift=2.0,
                real_utility=2.5,
                validation_cost=4.0,
                validator="grammar_check",
            ),
        )
        memory.add(
            Proposal(
                proposal_id="p1",
                domain="language",
                context_id="ctx",
                context_latent=(1.0,),
                target=target,
                intervention="bad rewrite",
                predicted_lift=4.0,
                uncertainty=0.7,
                support_confidence=0.2,
                reachability=0.1,
                consistency=0.3,
                generation_cost=2.0,
            ),
            ValidationResult(
                proposal_id="p1",
                accepted=False,
                real_lift=-1.0,
                validation_cost=4.0,
                validator="grammar_check",
                rejected_reason="invalid",
            ),
        )

        summary = memory.summary()
        self.assertEqual(summary["imagination_generated_count"], 2.0)
        self.assertEqual(summary["imagination_validated_count"], 2.0)
        self.assertEqual(summary["imagination_accepted_count"], 1.0)
        self.assertEqual(summary["imagination_accept_rate"], 0.5)
        self.assertAlmostEqual(summary["imagination_utility_per_validation"], 0.25)

    def test_calibrator_scores_supported_low_uncertainty_proposal_higher(self):
        memory = ImaginationMemory()
        target = Target(target_id="t", kind="stable_target")
        good = Proposal(
            proposal_id="good",
            domain="image",
            context_id="ctx",
            context_latent=(0.0,),
            target=target,
            intervention="variant",
            predicted_lift=2.0,
            uncertainty=0.1,
            support_confidence=0.9,
            reachability=0.8,
            consistency=0.9,
        )
        bad = Proposal(
            proposal_id="bad",
            domain="image",
            context_id="ctx",
            context_latent=(0.0,),
            target=target,
            intervention="variant",
            predicted_lift=0.0,
            uncertainty=1.0,
            support_confidence=0.0,
            reachability=0.0,
            consistency=0.0,
        )
        memory.add(good, ValidationResult(proposal_id="good", accepted=True, real_lift=1.0, validation_cost=1.0))
        memory.add(bad, ValidationResult(proposal_id="bad", accepted=False, real_lift=-1.0, validation_cost=1.0))

        calibrator = AcceptanceCalibrator.fit(memory)
        self.assertGreater(calibrator.score(good), calibrator.score(bad))
        self.assertIn("imagination_acceptance_base_accept_rate", calibrator.summary())


if __name__ == "__main__":
    unittest.main()
