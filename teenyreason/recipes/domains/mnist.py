"""MNIST recipe composition for the generic crawler library."""

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
from ...multidomain.domains.image.benchmark import ImageProbeBenchmarkConfig
from ..base import BenchmarkSpec, CrawlerRecipe
from ..evidence import evidence_metadata, evidence_payload


IMAGE_QUERY_FAMILIES = (
    "crop",
    "mask",
    "augment",
    "support_example",
    "contrastive_view",
)


def _blank_image(size: int = 16) -> np.ndarray:
    return np.zeros((size, size), dtype=np.float32)


def _draw_shape(shape_name: str, *, rotation: int, stroke_width: int, size: int = 16) -> np.ndarray:
    image = _blank_image(size)
    center = size // 2
    width = max(1, int(stroke_width))
    if shape_name == "vertical_bar":
        image[:, center - width : center + width] = 1.0
    elif shape_name == "horizontal_bar":
        image[center - width : center + width, :] = 1.0
    elif shape_name == "cross":
        image[:, center - width : center + width] = 1.0
        image[center - width : center + width, :] = 1.0
    else:
        for idx in range(size):
            lo = max(0, idx - width)
            hi = min(size, idx + width + 1)
            image[idx, lo:hi] = 1.0
    return np.rot90(image, k=int(rotation) % 4).astype(np.float32)


def _center_crop(image: np.ndarray, crop_size: int = 8) -> np.ndarray:
    start = (int(image.shape[0]) - int(crop_size)) // 2
    return image[start : start + crop_size, start : start + crop_size].astype(np.float32)


def _image_vector(*, query_name: str, image: np.ndarray, outcome_score: float) -> np.ndarray:
    rows, cols = np.indices(image.shape, dtype=np.float32)
    mass = float(np.sum(image))
    if mass <= 1e-6:
        center_y = 0.0
        center_x = 0.0
    else:
        center_y = float(np.sum(rows * image) / mass) / float(image.shape[0])
        center_x = float(np.sum(cols * image) / mass) / float(image.shape[1])
    family_onehot = np.asarray(
        [1.0 if query_name == name else 0.0 for name in IMAGE_QUERY_FAMILIES],
        dtype=np.float32,
    )
    stats = np.asarray(
        [
            float(np.mean(image)),
            float(np.std(image)),
            center_y,
            center_x,
            float(outcome_score),
        ],
        dtype=np.float32,
    )
    quadrants = np.asarray(
        [
            np.mean(image[:8, :8]),
            np.mean(image[:8, 8:]),
            np.mean(image[8:, :8]),
            np.mean(image[8:, 8:]),
        ],
        dtype=np.float32,
    )
    return np.concatenate([family_onehot, stats, quadrants], axis=0)


@dataclass
class ControlledShapeWorldAdapter:
    """Small visual concept proof track for the generic crawler API."""

    source_prefix: str = "image"
    _rng: np.random.Generator = field(init=False, default_factory=np.random.default_rng)
    _source_id: str = field(init=False, default="image:0")
    _hidden_target: dict[str, object] = field(init=False, default_factory=dict)
    _query_counter: int = field(init=False, default=0)
    _image: np.ndarray = field(init=False, default_factory=_blank_image)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._source_id = f"{self.source_prefix}:{0 if seed is None else int(seed)}"
        shape_names = ("vertical_bar", "horizontal_bar", "cross", "diagonal")
        shape_name = shape_names[int(self._rng.integers(0, len(shape_names)))]
        rotation = int(self._rng.integers(0, 4))
        stroke_width = int(self._rng.integers(1, 3))
        self._hidden_target = {
            "concept": shape_name,
            "rotation": rotation,
            "stroke_width": stroke_width,
        }
        self._image = _draw_shape(shape_name, rotation=rotation, stroke_width=stroke_width)
        self._query_counter = 0

    def available_queries(
        self,
        *,
        belief_state: BeliefState,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
    ) -> tuple[str, ...]:
        del belief_state
        seen = {item.query_name for item in history}
        unseen = tuple(name for name in IMAGE_QUERY_FAMILIES if name not in seen)
        return unseen if unseen else IMAGE_QUERY_FAMILIES

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
        local_state, outcome, observed_image, score = self._run_query(query)
        vector = _image_vector(query_name=query, image=observed_image, outcome_score=score)
        payload = evidence_payload(
            modality="image",
            query_family=query,
            source_id=self._source_id,
            intervention_cost=1.0,
            hidden_target=self._hidden_target,
            local_state=local_state,
            outcome=outcome,
            vector=vector,
            belief_source="learned",
            extra={"image": observed_image.astype(np.float32)},
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
    ) -> tuple[dict[str, object], dict[str, object], np.ndarray, float]:
        if query_name == "crop":
            crop = _center_crop(self._image)
            return (
                {"image": self._image, "crop_box": (4, 4, 12, 12)},
                {"crop": crop, "visible_mass": float(np.sum(crop))},
                np.pad(crop, ((4, 4), (4, 4))).astype(np.float32),
                float(np.mean(crop)),
            )
        if query_name == "mask":
            masked = self._image.copy()
            hidden_patch = masked[6:10, 6:10].copy()
            masked[6:10, 6:10] = 0.0
            return (
                {"masked_image": masked, "mask_box": (6, 6, 10, 10)},
                {"hidden_patch_mass": float(np.sum(hidden_patch))},
                masked,
                float(np.sum(hidden_patch) / 16.0),
            )
        if query_name == "augment":
            augmented = np.fliplr(self._image).astype(np.float32)
            agreement = float(np.mean(augmented == self._image))
            return (
                {"image": self._image, "transform": "horizontal_flip"},
                {"augmented": augmented, "view_agreement": agreement},
                augmented,
                agreement,
            )
        if query_name == "contrastive_view":
            rotated = np.rot90(self._image, k=1).astype(np.float32)
            contrast = 1.0 - float(np.mean(np.abs(rotated - self._image)))
            return (
                {"anchor": self._image, "candidate": rotated},
                {"same_concept": True, "contrast_score": contrast},
                rotated,
                contrast,
            )
        support = self._image + self._rng.normal(0.0, 0.02, size=self._image.shape).astype(np.float32)
        support = np.clip(support, 0.0, 1.0).astype(np.float32)
        return (
            {"support_index": int(self._query_counter), "support_image": support},
            {"class_match": True, "prototype_mass": float(np.sum(support))},
            support,
            float(np.mean(support)),
        )


def build_mnist_recipe(
    config: ImageProbeBenchmarkConfig | None = None,
) -> CrawlerRecipe:
    """Build the controlled image proof-track plus MNIST benchmark recipe."""
    config = config or ImageProbeBenchmarkConfig()
    return CrawlerRecipe(
        name="mnist",
        description="Generic crawler composition for controlled image concept evidence.",
        world_adapter_factory=ControlledShapeWorldAdapter,
        belief_backend_factory=partial(VectorBeliefBackend, vector_key="vector"),
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(SupportLimitStopPolicy, min_support=2),
        message_projector_factory=LinearMessageProjector,
        max_steps=4,
        metadata={
            "modality": "image",
            "recipe_family": "mnist",
            "belief_source": "learned",
        },
        benchmark=BenchmarkSpec(kind="image_probe", config=config),
    )
