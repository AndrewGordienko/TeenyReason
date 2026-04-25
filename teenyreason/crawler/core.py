"""Generic crawler-core interfaces and a small default implementation set."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .types import BeliefState, CrawlerMessage, CrawlerRunResult, CrawlerStep, EvidenceSlice


def _as_vector(values: Any) -> np.ndarray:
    """Convert one payload value into a flat float32 vector."""
    return np.asarray(values, dtype=np.float32).reshape(-1)


class WorldAdapter(Protocol):
    """Generic world interface for the crawler core."""

    def reset(self, seed: int | None = None) -> None:
        """Reset the adapter before a new crawler run."""

    def available_queries(
        self,
        *,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
    ) -> Sequence[str]:
        """Return the currently available query names."""

    def execute_query(
        self,
        query_name: str,
        *,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
    ) -> EvidenceSlice:
        """Execute one query and return one evidence slice."""


class BeliefBackend(Protocol):
    """Generic belief backend for the crawler core."""

    def initialize(self) -> BeliefState:
        """Return the initial empty belief state."""

    def update(
        self,
        belief_state: BeliefState,
        evidence: EvidenceSlice,
        *,
        history: Sequence[EvidenceSlice],
    ) -> BeliefState:
        """Update the belief from one new evidence slice."""


class QueryPolicy(Protocol):
    """Choose the next query from the current belief and history."""

    def choose_query(
        self,
        *,
        world: WorldAdapter,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
    ) -> str | None:
        """Return the next query name or `None` when nothing is worth asking."""


class StopPolicy(Protocol):
    """Decide whether the crawler has enough evidence."""

    def should_stop(
        self,
        *,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
    ) -> tuple[bool, str | None]:
        """Return `(should_stop, stop_reason)`."""


class MessageProjector(Protocol):
    """Turn a belief state into a downstream-facing crawler message."""

    def build_message(
        self,
        *,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
        stop_reason: str | None,
    ) -> CrawlerMessage:
        """Build the final or intermediate crawler message."""


@dataclass
class Crawler:
    """Canonical generic crawler loop."""

    world: WorldAdapter
    belief_backend: BeliefBackend
    query_policy: QueryPolicy
    stop_policy: StopPolicy
    message_projector: MessageProjector
    max_steps: int = 4

    def run(
        self,
        *,
        seed: int | None = None,
        max_steps: int | None = None,
    ) -> CrawlerRunResult:
        """Run the generic crawl-update-stop-message loop."""
        step_limit = int(self.max_steps if max_steps is None else max_steps)
        self.world.reset(seed=seed)
        belief_state = self.belief_backend.initialize()
        history: list[EvidenceSlice] = []
        steps: list[CrawlerStep] = []
        stop_reason = "max_steps"

        for _step_idx in range(max(1, step_limit)):
            should_stop, candidate_reason = self.stop_policy.should_stop(
                belief_state=belief_state,
                history=history,
            )
            if should_stop and history:
                stop_reason = str(candidate_reason or "stop_policy")
                break
            query_name = self.query_policy.choose_query(
                world=self.world,
                belief_state=belief_state,
                history=history,
            )
            if query_name is None:
                stop_reason = "no_query"
                break
            evidence = self.world.execute_query(
                query_name,
                belief_state=belief_state,
                history=history,
            )
            history.append(evidence)
            belief_state = self.belief_backend.update(
                belief_state,
                evidence,
                history=history,
            )
            should_stop, candidate_reason = self.stop_policy.should_stop(
                belief_state=belief_state,
                history=history,
            )
            step_stop_reason = str(candidate_reason) if should_stop and candidate_reason else None
            message = self.message_projector.build_message(
                belief_state=belief_state,
                history=history,
                stop_reason=step_stop_reason,
            )
            steps.append(
                CrawlerStep(
                    query_name=str(query_name),
                    evidence=evidence,
                    belief_state=belief_state,
                    message=message,
                    stop_reason=step_stop_reason,
                    metadata={
                        "query_count": len(history),
                    },
                )
            )
            if should_stop:
                stop_reason = str(candidate_reason or "stop_policy")
                break

        final_message = self.message_projector.build_message(
            belief_state=belief_state,
            history=history,
            stop_reason=stop_reason,
        )
        return CrawlerRunResult(
            steps=tuple(steps),
            final_belief_state=belief_state,
            final_message=final_message,
            stop_reason=str(stop_reason),
            metadata={
                "query_count": len(history),
            },
        )


@dataclass
class ScriptedWorldAdapter:
    """Small generic adapter useful for examples, smoke tests, and recipes."""

    query_payloads: Mapping[str, Mapping[str, Any]]
    source_prefix: str = "world"
    _query_counter: int = field(init=False, default=0)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._query_counter = 0

    def available_queries(
        self,
        *,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
    ) -> Sequence[str]:
        del belief_state
        seen_queries = {item.query_name for item in history}
        unseen = [name for name in self.query_payloads.keys() if name not in seen_queries]
        return unseen if unseen else list(self.query_payloads.keys())

    def execute_query(
        self,
        query_name: str,
        *,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
    ) -> EvidenceSlice:
        del belief_state
        del history
        self._query_counter += 1
        payload = dict(self.query_payloads[str(query_name)])
        return EvidenceSlice(
            query_name=str(query_name),
            source_id=f"{self.source_prefix}:{self._query_counter}",
            payload=payload,
            metadata={
                "query_index": self._query_counter,
            },
        )


@dataclass
class VectorBeliefBackend:
    """Simple generic backend that pools evidence vectors by running mean."""

    vector_key: str = "vector"

    def initialize(self) -> BeliefState:
        return BeliefState(
            latent=np.zeros((0,), dtype=np.float32),
            uncertainty=1.0,
            support_size=0,
            metadata={},
        )

    def update(
        self,
        belief_state: BeliefState,
        evidence: EvidenceSlice,
        *,
        history: Sequence[EvidenceSlice],
    ) -> BeliefState:
        del belief_state
        vectors = [
            _as_vector(item.payload[self.vector_key])
            for item in history
            if self.vector_key in item.payload
        ]
        if not vectors:
            return self.initialize()
        stacked = np.stack(vectors, axis=0).astype(np.float32)
        latent = np.mean(stacked, axis=0).astype(np.float32)
        uncertainty = float(np.mean(np.linalg.norm(stacked - latent[None, :], axis=1)))
        return BeliefState(
            latent=latent,
            uncertainty=uncertainty,
            support_size=int(stacked.shape[0]),
            metadata={
                "vector_key": self.vector_key,
                "query_names": tuple(item.query_name for item in history),
            },
        )


@dataclass
class RoundRobinQueryPolicy:
    """Choose the first available query, preferring unseen evidence."""

    def choose_query(
        self,
        *,
        world: WorldAdapter,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
    ) -> str | None:
        available = list(
            world.available_queries(belief_state=belief_state, history=history)
        )
        if not available:
            return None
        return str(available[0])


@dataclass
class SupportLimitStopPolicy:
    """Stop after enough support or when uncertainty is already low."""

    min_support: int = 2
    max_uncertainty: float = 0.20

    def should_stop(
        self,
        *,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
    ) -> tuple[bool, str | None]:
        del history
        if int(belief_state.support_size) >= int(self.min_support) and float(
            belief_state.uncertainty
        ) <= float(self.max_uncertainty):
            return True, "belief_ready"
        if int(belief_state.support_size) >= int(self.min_support):
            return True, "support_limit"
        return False, None


@dataclass
class LinearMessageProjector:
    """Turn the belief state directly into a generic crawler message."""

    ready_support: int = 2
    ready_uncertainty: float = 0.20

    def build_message(
        self,
        *,
        belief_state: BeliefState,
        history: Sequence[EvidenceSlice],
        stop_reason: str | None,
    ) -> CrawlerMessage:
        confidence = float(1.0 / (1.0 + max(float(belief_state.uncertainty), 0.0)))
        ready = bool(
            int(belief_state.support_size) >= int(self.ready_support)
            and float(belief_state.uncertainty) <= float(self.ready_uncertainty)
        )
        return CrawlerMessage(
            vector=np.asarray(belief_state.latent, dtype=np.float32).reshape(-1),
            confidence=confidence,
            ready=ready,
            uncertainty=float(belief_state.uncertainty),
            metadata={
                "support_size": int(belief_state.support_size),
                "stop_reason": stop_reason,
                "query_names": tuple(item.query_name for item in history),
            },
        )
