"""Target memory for directed imagination."""

from __future__ import annotations

from .schema import Target


class TargetBank:
    """Small ranked store of useful latent states to imagine toward."""

    def __init__(self, targets: list[Target] | None = None):
        self._targets = list(targets or [])

    def add(self, target: Target) -> None:
        self._targets.append(target)

    def extend(self, targets: list[Target]) -> None:
        self._targets.extend(targets)

    def targets(self) -> list[Target]:
        return list(self._targets)

    def top(self, *, count: int, kind: str | None = None) -> list[Target]:
        rows = self._targets if kind is None else [target for target in self._targets if target.kind == kind]
        ranked = sorted(rows, key=target_score, reverse=True)
        return ranked[: max(0, int(count))]

    def summary(self, *, prefix: str = "imagination_targets") -> dict[str, float]:
        targets = self.targets()
        kinds = sorted({target.kind for target in targets})
        out = {
            f"{prefix}_count": float(len(targets)),
            f"{prefix}_utility_max": max([target.utility for target in targets], default=0.0),
            f"{prefix}_stability_max": max([target.stability for target in targets], default=0.0),
        }
        for kind in kinds:
            out[f"{prefix}_{kind}_count"] = float(sum(1 for target in targets if target.kind == kind))
        return out


def target_score(target: Target) -> float:
    return float(target.utility + 0.25 * target.stability)


__all__ = ["TargetBank"]
