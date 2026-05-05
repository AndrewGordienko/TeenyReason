"""Video evidence adapters for the generic crawler API.

This module keeps video/3D-world work on the same `EvidenceSlice` contract as
the RL crawler. It does not implement a vision model; it defines the payload
shape that a future adapter can fill from frame encoders, depth probes, object
tracks, camera motion, and goal-inference heads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import EvidenceSlice


@dataclass(frozen=True)
class VideoBenchmarkSpec:
    """One measurable video-world benchmark for the crawler roadmap."""

    name: str
    target: str
    lower_is_better: bool
    unit: str
    ablation: str


@dataclass(frozen=True)
class VideoBeliefTargets:
    """Belief questions a passive video crawler should learn to answer."""

    depth_consistency: bool = True
    object_persistence: bool = True
    occlusion_consistency: bool = True
    egomotion_consistency: bool = True
    contact_dynamics: bool = True
    goal_inference: bool = True

    def query_families(self) -> tuple[str, ...]:
        """Return enabled target names in stable dashboard order."""
        families: list[str] = []
        if self.depth_consistency:
            families.append("depth_consistency")
        if self.object_persistence:
            families.append("object_persistence")
        if self.occlusion_consistency:
            families.append("occlusion_consistency")
        if self.egomotion_consistency:
            families.append("egomotion_consistency")
        if self.contact_dynamics:
            families.append("contact_dynamics")
        if self.goal_inference:
            families.append("goal_inference")
        return tuple(families)


@dataclass(frozen=True)
class VideoEvidenceSpec:
    """Metadata for one video source before model-specific features are added."""

    source_id: str
    frame_count: int
    frame_shape: tuple[int, int, int] | None = None
    fps: float | None = None
    query_family: str = "passive_video"
    camera_motion: str | None = None
    targets: VideoBeliefTargets = field(default_factory=VideoBeliefTargets)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise when the evidence spec cannot describe a real video clip."""
        if not self.source_id:
            raise ValueError("source_id must be non-empty")
        if int(self.frame_count) <= 0:
            raise ValueError("frame_count must be positive")
        if self.frame_shape is not None:
            if len(self.frame_shape) != 3:
                raise ValueError("frame_shape must be (height, width, channels)")
            height, width, channels = self.frame_shape
            if height <= 0 or width <= 0 or channels <= 0:
                raise ValueError("frame_shape dimensions must be positive")
        if self.fps is not None and float(self.fps) <= 0.0:
            raise ValueError("fps must be positive when provided")


def default_video_belief_targets() -> VideoBeliefTargets:
    """Return the default video world-model target set."""
    return VideoBeliefTargets()


def default_video_benchmarks() -> tuple[VideoBenchmarkSpec, ...]:
    """Return the first benchmark contract for video/3D crawler work."""
    return (
        VideoBenchmarkSpec(
            name="frames_to_structure",
            target="infer depth, layout, object permanence, and occlusion state",
            lower_is_better=True,
            unit="frames",
            ablation="shuffle or mute structure belief",
        ),
        VideoBenchmarkSpec(
            name="frames_to_peak_task_score",
            target="reach best downstream task or imitation score",
            lower_is_better=True,
            unit="frames",
            ablation="compare regular video policy against crawler-conditioned policy",
        ),
        VideoBenchmarkSpec(
            name="future_state_prediction",
            target="predict held-out future frame or object state",
            lower_is_better=True,
            unit="error",
            ablation="remove contact and egomotion belief",
        ),
        VideoBenchmarkSpec(
            name="goal_inference_accuracy",
            target="identify the goal or task being pursued in the clip",
            lower_is_better=False,
            unit="accuracy",
            ablation="shuffle goal belief across clips",
        ),
    )


def video_evidence_slice(spec: VideoEvidenceSpec) -> EvidenceSlice:
    """Convert a video spec into the generic crawler evidence contract."""
    spec.validate()
    payload = {
        "modality": "video",
        "frame_count": int(spec.frame_count),
        "frame_shape": None if spec.frame_shape is None else tuple(int(v) for v in spec.frame_shape),
        "fps": None if spec.fps is None else float(spec.fps),
        "camera_motion": spec.camera_motion,
        "belief_targets": spec.targets.query_families(),
        "benchmark_names": tuple(item.name for item in default_video_benchmarks()),
    }
    return EvidenceSlice(
        query_name=spec.query_family,
        source_id=spec.source_id,
        payload=payload,
        metadata=dict(spec.metadata),
    )
