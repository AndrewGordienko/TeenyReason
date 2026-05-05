"""Lightweight acceptance calibration for generated samples."""

from __future__ import annotations

import numpy as np

from .memory import ImaginationMemory, corr_or_zero, mean
from .schema import Proposal


class AcceptanceCalibrator:
    """Estimate whether an imagined proposal deserves validation budget."""

    def __init__(
        self,
        *,
        base_accept_rate: float,
        overestimate_error: float,
        utility_per_validation: float,
        feature_weights: dict[str, float],
    ):
        self.base_accept_rate = float(base_accept_rate)
        self.overestimate_error = float(overestimate_error)
        self.utility_per_validation = float(utility_per_validation)
        self.feature_weights = dict(feature_weights)

    @classmethod
    def fit(cls, memory: ImaginationMemory) -> "AcceptanceCalibrator":
        samples = memory.samples()
        validated = [sample for sample in samples if sample.validated]
        labels = [1.0 if sample.accepted else 0.0 for sample in validated]
        base = mean(labels)
        weights = {}
        for key in ("predicted_lift", "uncertainty", "support_confidence", "reachability", "consistency"):
            values = [float(getattr(sample.proposal, key)) for sample in validated]
            weights[key] = corr_or_zero(values, labels)
        predicted = [sample.proposal.predicted_lift for sample in validated]
        real = [sample.validation.real_lift for sample in validated if sample.validation is not None]
        summary = memory.summary()
        return cls(
            base_accept_rate=base,
            overestimate_error=mean([left - right for left, right in zip(predicted, real)]),
            utility_per_validation=float(summary["imagination_utility_per_validation"]),
            feature_weights=weights,
        )

    def score(self, proposal: Proposal) -> float:
        raw = self.base_accept_rate
        raw += 0.20 * bounded(proposal.predicted_lift)
        raw += 0.15 * bounded(proposal.support_confidence)
        raw += 0.15 * bounded(proposal.reachability)
        raw += 0.10 * bounded(proposal.consistency)
        raw -= 0.20 * bounded(proposal.uncertainty)
        raw -= 0.10 * bounded(self.overestimate_error)
        return float(np.clip(raw, 0.0, 1.0))

    def should_validate(self, proposal: Proposal, *, min_score: float = 0.20) -> bool:
        return self.score(proposal) >= float(min_score)

    def summary(self, *, prefix: str = "imagination_acceptance") -> dict[str, float]:
        out = {
            f"{prefix}_base_accept_rate": float(self.base_accept_rate),
            f"{prefix}_overestimate_error": float(self.overestimate_error),
            f"{prefix}_utility_per_validation": float(self.utility_per_validation),
        }
        for key, value in self.feature_weights.items():
            out[f"{prefix}_{key}_corr"] = float(value)
        return out


def bounded(value: float) -> float:
    return float(np.tanh(float(value)))


__all__ = ["AcceptanceCalibrator"]
