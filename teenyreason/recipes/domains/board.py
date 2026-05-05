"""Board-game recipe composition for the generic crawler library."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

from ...crawler.core import RoundRobinQueryPolicy
from ...crawler.types import BeliefState, CrawlerMessage, EvidenceSlice
from ...multidomain.domains.board import (
    BoardProbeBenchmarkConfig,
    RULES,
    RULE_NORMAL,
    build_rule_message,
    infer_rule_from_probe_outcomes,
    query_rule_outcome,
)
from ..base import BenchmarkSpec, CrawlerRecipe
from ..evidence import evidence_metadata, evidence_payload


BOARD_QUERY_FAMILIES = (
    "x_line_completion",
    "o_line_completion",
    "fork_block",
    "endgame_choice",
)


def _board_vector(outcome: dict[str, object]) -> np.ndarray:
    """Build a compact vector for one board-rule evidence slice."""
    family = str(outcome.get("query_name", ""))
    family_onehot = np.asarray(
        [1.0 if family == name else 0.0 for name in BOARD_QUERY_FAMILIES],
        dtype=np.float32,
    )
    stats = np.asarray(
        [
            float(outcome.get("value_for_x", 0.0)),
            float(outcome.get("normal_value_for_x", 0.0)),
            float(outcome.get("misere_value_for_x", 0.0)),
            1.0 if int(outcome.get("value_for_x", 0)) == int(outcome.get("normal_value_for_x", 0)) else 0.0,
            1.0 if int(outcome.get("value_for_x", 0)) == int(outcome.get("misere_value_for_x", 0)) else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([family_onehot, stats], axis=0)


@dataclass
class TicTacToeRuleWorldAdapter:
    """Small board-game world with a hidden rule variant."""

    source_prefix: str = "board"
    _source_id: str = field(init=False, default="board:0")
    _hidden_rule: str = field(init=False, default=RULE_NORMAL)
    _query_counter: int = field(init=False, default=0)

    def reset(self, seed: int | None = None) -> None:
        seed_value = 0 if seed is None else int(seed)
        self._source_id = f"{self.source_prefix}:{seed_value}"
        self._hidden_rule = RULES[seed_value % len(RULES)]
        self._query_counter = 0

    def available_queries(
        self,
        *,
        belief_state: BeliefState,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
    ) -> tuple[str, ...]:
        del belief_state
        seen = {item.query_name for item in history}
        unseen = tuple(name for name in BOARD_QUERY_FAMILIES if name not in seen)
        return unseen if unseen else BOARD_QUERY_FAMILIES

    def execute_query(
        self,
        query_name: str,
        *,
        belief_state: BeliefState,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
    ) -> EvidenceSlice:
        del belief_state
        del history
        self._query_counter += 1
        query = str(query_name)
        outcome = query_rule_outcome(query, self._hidden_rule)
        outcome = {**outcome, "query_name": query}
        vector = _board_vector(outcome)
        payload = evidence_payload(
            modality="board_game",
            query_family=query,
            source_id=self._source_id,
            intervention_cost=1.0,
            hidden_target={"rule": self._hidden_rule, "game": "tic_tac_toe"},
            local_state={
                "board": list(outcome["board"]),
                "candidate_move": int(outcome["candidate_move"]),
                "x_to_move": bool(outcome["x_to_move"]),
            },
            outcome={
                "value_for_x": int(outcome["value_for_x"]),
                "normal_value_for_x": int(outcome["normal_value_for_x"]),
                "misere_value_for_x": int(outcome["misere_value_for_x"]),
            },
            vector=vector,
            belief_source="learned",
            extra={"rule_probe_outcome": outcome},
        )
        return EvidenceSlice(
            query_name=query,
            source_id=self._source_id,
            payload=payload,
            metadata=evidence_metadata(payload=payload, query_index=self._query_counter),
        )


@dataclass
class BoardRuleBeliefBackend:
    """Infer the hidden rule from board-rule evidence slices."""

    def initialize(self) -> BeliefState:
        return BeliefState(
            latent=np.asarray([0.5, 0.5, 0.0, 0.5], dtype=np.float32),
            uncertainty=0.5,
            support_size=0,
            metadata={"predicted_rule": RULE_NORMAL, "belief_source": "learned"},
        )

    def update(
        self,
        belief_state: BeliefState,
        evidence: EvidenceSlice,
        *,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
    ) -> BeliefState:
        del belief_state
        del evidence
        outcomes = [
            item.payload["rule_probe_outcome"]
            for item in history
            if "rule_probe_outcome" in item.payload
        ]
        predicted, confidence, latent = infer_rule_from_probe_outcomes(outcomes)
        return BeliefState(
            latent=latent,
            uncertainty=float(1.0 - confidence),
            support_size=len(outcomes),
            metadata={
                "predicted_rule": predicted,
                "belief_source": "learned",
                "query_names": tuple(item.query_name for item in history),
            },
        )


@dataclass
class BoardRuleStopPolicy:
    """Stop once the rule is identified or the probe budget is spent."""

    min_confidence: float = 0.80
    max_support: int = 2

    def should_stop(
        self,
        *,
        belief_state: BeliefState,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
    ) -> tuple[bool, str | None]:
        if history and float(1.0 - belief_state.uncertainty) >= float(self.min_confidence):
            return True, "rule_identified"
        if len(history) >= int(self.max_support):
            return True, "probe_budget"
        return False, None


@dataclass
class BoardRuleMessageProjector:
    """Build the canonical solver-facing board-game crawler message."""

    def build_message(
        self,
        *,
        belief_state: BeliefState,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
        stop_reason: str | None,
    ) -> CrawlerMessage:
        outcomes = [
            item.payload["rule_probe_outcome"]
            for item in history
            if "rule_probe_outcome" in item.payload
        ]
        if outcomes:
            message = build_rule_message(outcomes)
            metadata = dict(message.metadata)
            metadata["stop_reason"] = stop_reason
            metadata["query_names"] = tuple(item.query_name for item in history)
            return CrawlerMessage(
                vector=message.vector,
                confidence=message.confidence,
                ready=message.ready,
                uncertainty=message.uncertainty,
                metadata=metadata,
            )
        return CrawlerMessage(
            vector=np.asarray(belief_state.latent, dtype=np.float32),
            confidence=0.5,
            ready=False,
            uncertainty=0.5,
            metadata={
                "modality": "board_game",
                "game": "tic_tac_toe",
                "predicted_rule": RULE_NORMAL,
                "belief_source": "learned",
                "support_size": 0,
                "stop_reason": stop_reason,
            },
        )


def build_board_recipe(
    config: BoardProbeBenchmarkConfig | None = None,
) -> CrawlerRecipe:
    """Build the exact tic-tac-toe hidden-rule crawler recipe."""
    active_config = config or BoardProbeBenchmarkConfig()
    return CrawlerRecipe(
        name="tictactoe",
        description="Generic crawler composition for hidden board-game rule evidence.",
        world_adapter_factory=TicTacToeRuleWorldAdapter,
        belief_backend_factory=BoardRuleBeliefBackend,
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(BoardRuleStopPolicy, max_support=active_config.probe_budget),
        message_projector_factory=BoardRuleMessageProjector,
        max_steps=int(active_config.probe_budget),
        metadata={
            "modality": "board_game",
            "recipe_family": "board_game",
            "game": "tic_tac_toe",
        },
        benchmark=BenchmarkSpec(kind="board_probe", config=active_config),
    )
