"""Language recipe composition for the generic crawler library."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

from ...crawler.core import (
    LinearMessageProjector,
    RoundRobinQueryPolicy,
    SupportLimitStopPolicy,
    VectorBeliefBackend,
)
from ...crawler.types import BeliefState, EvidenceSlice
from ...multidomain.domains.language.benchmark import LanguageProbeBenchmarkConfig
from ..base import BenchmarkSpec, CrawlerRecipe
from ..evidence import evidence_metadata, evidence_payload


LANGUAGE_QUERY_FAMILIES = (
    "continuation_ranking",
    "mask_span",
    "reorder_check",
    "entity_probe",
)


def _token_ids(text: str, dim: int = 12) -> np.ndarray:
    ids = np.zeros((dim,), dtype=np.float32)
    encoded = text.encode("utf-8", errors="ignore")
    for idx, value in enumerate(encoded[:dim]):
        ids[idx] = float(value) / 255.0
    return ids


def _language_vector(
    *,
    query_name: str,
    context: str,
    outcome_score: float,
    option_count: int,
) -> np.ndarray:
    words = context.split()
    lengths = [len(word) for word in words] or [0]
    family_onehot = np.asarray(
        [1.0 if query_name == name else 0.0 for name in LANGUAGE_QUERY_FAMILIES],
        dtype=np.float32,
    )
    stats = np.asarray(
        [
            float(len(words)) / 12.0,
            float(np.mean(lengths)) / 10.0,
            float(option_count) / 4.0,
            float(outcome_score),
        ],
        dtype=np.float32,
    )
    return np.concatenate([family_onehot, stats, _token_ids(context)], axis=0)


@dataclass
class SyntheticLanguageWorldAdapter:
    """Passive language proof track with known grammar and discourse variables."""

    source_prefix: str = "language"
    _rng: np.random.Generator = field(init=False, default_factory=np.random.default_rng)
    _source_id: str = field(init=False, default="language:0")
    _hidden_target: dict[str, str] = field(init=False, default_factory=dict)
    _query_counter: int = field(init=False, default=0)
    _sentences: tuple[str, ...] = field(init=False, default=())

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._source_id = f"{self.source_prefix}:{0 if seed is None else int(seed)}"
        grammar = "svo" if int(self._rng.integers(0, 2)) == 0 else "sov"
        style = "plain" if int(self._rng.integers(0, 2)) == 0 else "formal"
        entity_rule = "hero_helps" if int(self._rng.integers(0, 2)) == 0 else "rival_blocks"
        self._hidden_target = {
            "grammar": grammar,
            "style": style,
            "entity_rule": entity_rule,
        }
        self._sentences = self._build_corpus(grammar=grammar, style=style, entity_rule=entity_rule)
        self._query_counter = 0

    def _build_corpus(self, *, grammar: str, style: str, entity_rule: str) -> tuple[str, ...]:
        subjects = ("mira", "tavo", "lena", "orin")
        verbs = ("lifts", "maps", "guards", "finds")
        objects = ("crystal", "bridge", "signal", "garden")
        suffix = "today" if style == "plain" else "with care"
        sentences: list[str] = []
        for idx, subject in enumerate(subjects):
            verb = verbs[idx % len(verbs)]
            obj = objects[(idx + 1) % len(objects)]
            if grammar == "svo":
                base = f"{subject} {verb} the {obj}"
            else:
                base = f"{subject} the {obj} {verb}"
            if entity_rule == "hero_helps":
                sentences.append(f"{base} and mira helps {suffix}")
            else:
                sentences.append(f"{base} but tavo blocks {suffix}")
        return tuple(sentences)

    def available_queries(
        self,
        *,
        belief_state: BeliefState,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
    ) -> tuple[str, ...]:
        del belief_state
        seen = {item.query_name for item in history}
        unseen = tuple(name for name in LANGUAGE_QUERY_FAMILIES if name not in seen)
        return unseen if unseen else LANGUAGE_QUERY_FAMILIES

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
        sentence = self._sentences[(self._query_counter - 1) % len(self._sentences)]
        local_state, outcome, score, option_count = self._run_query(query, sentence)
        vector = _language_vector(
            query_name=query,
            context=sentence,
            outcome_score=score,
            option_count=option_count,
        )
        payload = evidence_payload(
            modality="language",
            query_family=query,
            source_id=self._source_id,
            intervention_cost=1.0,
            hidden_target=self._hidden_target,
            local_state=local_state,
            outcome=outcome,
            vector=vector,
            belief_source="learned",
        )
        return EvidenceSlice(
            query_name=query,
            source_id=self._source_id,
            payload=payload,
            metadata=evidence_metadata(payload=payload, query_index=self._query_counter),
        )

    def _run_query(
        self,
        query_name: str,
        sentence: str,
    ) -> tuple[dict[str, object], dict[str, object], float, int]:
        grammar = self._hidden_target["grammar"]
        entity_rule = self._hidden_target["entity_rule"]
        if query_name == "continuation_ranking":
            correct = "mira helps" if entity_rule == "hero_helps" else "tavo blocks"
            candidates = (correct, "orin waits", "lena forgets")
            return (
                {"context": sentence, "candidates": candidates},
                {"correct_index": 0, "accepted": correct},
                1.0,
                len(candidates),
            )
        if query_name == "mask_span":
            words = sentence.split()
            mask_index = 2 if len(words) > 3 else max(0, len(words) - 1)
            target = words[mask_index]
            masked = " ".join(words[:mask_index] + ["<mask>"] + words[mask_index + 1 :])
            return (
                {"masked_text": masked, "mask_index": mask_index},
                {"target": target, "target_length": len(target)},
                min(1.0, len(target) / 8.0),
                1,
            )
        if query_name == "reorder_check":
            valid_order = "subject-verb-object" if grammar == "svo" else "subject-object-verb"
            words = sentence.split()[:4]
            scrambled = tuple(reversed(words))
            return (
                {"tokens": tuple(words), "scrambled": scrambled},
                {"valid_order": valid_order, "scrambled_valid": False},
                1.0,
                2,
            )
        subject = "mira" if entity_rule == "hero_helps" else "tavo"
        relation = "helper" if entity_rule == "hero_helps" else "blocker"
        return (
            {"sentence": sentence, "entity": subject},
            {"role": relation, "corefers_with_rule": True},
            1.0,
            2,
        )


def build_language_recipe(
    config: LanguageProbeBenchmarkConfig | None = None,
) -> CrawlerRecipe:
    """Build the controlled synthetic-language crawler recipe."""
    config = config or LanguageProbeBenchmarkConfig()
    return CrawlerRecipe(
        name="language",
        description="Generic crawler composition for synthetic grammar and discourse probes.",
        world_adapter_factory=SyntheticLanguageWorldAdapter,
        belief_backend_factory=partial(VectorBeliefBackend, vector_key="vector"),
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(SupportLimitStopPolicy, min_support=2),
        message_projector_factory=LinearMessageProjector,
        max_steps=4,
        metadata={
            "modality": "language",
            "recipe_family": "language",
            "belief_source": "learned",
        },
        benchmark=BenchmarkSpec(kind="language_probe", config=config),
    )
