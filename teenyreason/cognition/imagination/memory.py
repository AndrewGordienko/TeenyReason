"""Memory and metrics for validated imagination."""

from __future__ import annotations

import numpy as np

from .schema import ImaginationSample, Proposal, ValidationResult


class ImaginationMemory:
    """Track generated proposals, validation outcomes, and sample economics."""

    def __init__(self):
        self._samples: dict[str, ImaginationSample] = {}

    def add_proposal(self, proposal: Proposal) -> None:
        self._samples[proposal.proposal_id] = ImaginationSample(proposal=proposal)

    def add_validation(self, validation: ValidationResult) -> None:
        sample = self._samples.get(validation.proposal_id)
        if sample is None:
            proposal = Proposal(
                proposal_id=validation.proposal_id,
                domain="unknown",
                context_id="unknown",
                context_latent=(),
                target=None,
                intervention=None,
            )
            sample = ImaginationSample(proposal=proposal)
        self._samples[validation.proposal_id] = ImaginationSample(
            proposal=sample.proposal,
            validation=validation,
            accepted_weight=accepted_weight(validation),
        )

    def add(self, proposal: Proposal, validation: ValidationResult | None = None) -> None:
        self.add_proposal(proposal)
        if validation is not None:
            self.add_validation(validation)

    def samples(self) -> list[ImaginationSample]:
        return list(self._samples.values())

    def accepted(self) -> list[ImaginationSample]:
        return [sample for sample in self.samples() if sample.accepted]

    def failed(self) -> list[ImaginationSample]:
        return [sample for sample in self.samples() if sample.validated and not sample.accepted]

    def summary(self, *, prefix: str = "imagination") -> dict[str, float]:
        samples = self.samples()
        validated = [sample for sample in samples if sample.validated]
        accepted = [sample for sample in validated if sample.accepted]
        predicted = [sample.proposal.predicted_lift for sample in validated]
        real = [float(sample.validation.real_lift) for sample in validated if sample.validation is not None]
        validation_cost = sum(float(sample.validation.validation_cost) for sample in validated if sample.validation is not None)
        accepted_utility = sum(float(sample.validation.real_lift) for sample in accepted if sample.validation is not None)
        generated_cost = sum(float(sample.proposal.generation_cost) for sample in samples)
        return {
            f"{prefix}_generated_count": float(len(samples)),
            f"{prefix}_validated_count": float(len(validated)),
            f"{prefix}_accepted_count": float(len(accepted)),
            f"{prefix}_rejected_count": float(len(validated) - len(accepted)),
            f"{prefix}_unvalidated_count": float(len(samples) - len(validated)),
            f"{prefix}_accept_rate": float(len(accepted) / max(1, len(validated))),
            f"{prefix}_predicted_vs_real_corr": corr_or_zero(predicted, real),
            f"{prefix}_overestimate_error": mean([left - right for left, right in zip(predicted, real)]),
            f"{prefix}_utility_per_validation": float(accepted_utility / max(1.0, validation_cost)),
            f"{prefix}_accepted_sample_utility": mean([float(sample.validation.real_lift) for sample in accepted if sample.validation is not None]),
            f"{prefix}_validation_cost": float(validation_cost),
            f"{prefix}_generation_cost": float(generated_cost),
            f"{prefix}_mean_uncertainty": mean([sample.proposal.uncertainty for sample in samples]),
            f"{prefix}_mean_support": mean([sample.proposal.support_confidence for sample in samples]),
            f"{prefix}_mean_reachability": mean([sample.proposal.reachability for sample in samples]),
        }


def accepted_weight(validation: ValidationResult) -> float:
    if not bool(validation.accepted):
        return 0.0
    return max(1.0, float(validation.real_lift))


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


def corr_or_zero(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if float(np.std(left_array)) < 1e-6 or float(np.std(right_array)) < 1e-6:
        return 0.0
    return float(np.corrcoef(left_array, right_array)[0, 1])


__all__ = ["ImaginationMemory"]
