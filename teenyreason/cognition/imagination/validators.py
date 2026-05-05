"""Domain validation protocol for imagined proposals."""

from __future__ import annotations

from typing import Protocol

from .schema import Proposal, ValidationResult


class DomainValidator(Protocol):
    """A domain adapter that can test generated samples against truth."""

    name: str

    def validate(self, proposal: Proposal) -> ValidationResult:
        """Return a truth signal for one generated proposal."""
        raise NotImplementedError


def validate_with(validator: DomainValidator, proposals: list[Proposal]) -> list[ValidationResult]:
    return [validator.validate(proposal) for proposal in proposals]


__all__ = ["DomainValidator", "validate_with"]
