"""Real-environment causal adapters for the generic crawler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...crawler.causal import (
    CausalWorldSpec,
    CounterfactualPrediction,
    Intervention,
    ObservedOutcome,
    WorldBelief,
    run_causal_crawler,
)
from ..domains.cartpole import (
    MechanicsWorld,
    evidence_vector,
    infer_world,
    world_for_seed,
)
from ..domains.image.benchmark import ImageProbeBenchmarkConfig, load_mnist_tensors
from ..domains.language.benchmark import (
    LanguageProbeBenchmarkConfig,
    build_char_vocab,
    encode_text,
    ensure_tiny_shakespeare,
    split_corpus,
)


@dataclass(frozen=True)
class RealCausalAdapterConfig:
    """Small real-adapter config used by the suite diagnostics."""

    seeds: tuple[int, ...] = (0, 1, 2, 3)
    language_spans: int = 6
    language_span_length: int = 96
    image_digits: tuple[int, ...] = tuple(range(10))
    image_examples_per_digit: int = 16
    cartpole_probe_steps: int = 18


@dataclass(frozen=True)
class TextWorld:
    label: str
    encoded_train: np.ndarray
    vocab_size: int


@dataclass(frozen=True)
class ImageWorld:
    label: str
    train_images: Any
    train_labels: Any


class RealLanguageCausalAdapter:
    """Tiny Shakespeare adapter exposing text spans as interventions."""

    spec = CausalWorldSpec(
        domain="language",
        modality="text",
        hidden_target="real_character_distribution_and_local_transitions",
        outcome_name="span_statistics",
    )

    def __init__(
        self,
        config: RealCausalAdapterConfig | None = None,
        language_config: LanguageProbeBenchmarkConfig | None = None,
        *,
        text: str | None = None,
    ):
        self.config = config or RealCausalAdapterConfig()
        self.language_config = language_config or LanguageProbeBenchmarkConfig()
        if text is None:
            text = ensure_tiny_shakespeare(self.language_config).read_text(encoding="utf-8")
        stoi, _itos = build_char_vocab(text)
        encoded = encode_text(text, stoi)
        train, _validation = split_corpus(encoded, validation_chars=self.language_config.validation_chars)
        self.world = TextWorld("tiny_shakespeare", train, len(stoi))

    def world_for_seed(self, seed: int) -> TextWorld:
        return self.world

    def intervention_space(self, world: TextWorld, *, seed: int) -> tuple[Intervention, ...]:
        max_start = max(0, len(world.encoded_train) - int(self.config.language_span_length) - 1)
        rng = np.random.default_rng(9100 + int(seed))
        starts = rng.integers(0, max(max_start, 1), size=int(self.config.language_spans))
        return tuple(
            Intervention(
                name=f"span_{idx}",
                family="support_span",
                cost=float(self.config.language_span_length),
                payload={"start": int(start), "length": int(self.config.language_span_length)},
            )
            for idx, start in enumerate(starts.tolist())
        )

    def observe(self, world: TextWorld, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        return ObservedOutcome(
            intervention=intervention,
            value=_text_span_stats(world, intervention),
            cost=float(intervention.cost),
        )

    def infer_belief(
        self,
        world: TextWorld,
        observations: tuple[ObservedOutcome, ...],
        *,
        seed: int,
    ) -> WorldBelief:
        hists = [_as_array(item.value.get("histogram"), world.vocab_size) for item in observations]
        transition = [float(item.value.get("transition_mean", 0.0)) for item in observations]
        mean_hist = np.mean(np.stack(hists, axis=0), axis=0) if hists else np.zeros((world.vocab_size,), dtype=np.float32)
        message = {
            "histogram": mean_hist,
            "transition_mean": float(np.mean(transition)) if transition else 0.0,
        }
        confidence = float(1.0 - min(np.std(transition) if transition else 1.0, 1.0))
        return WorldBelief(
            label=world.label,
            message=message,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            metadata={"span_count": len(observations)},
        )

    def predict_outcome(
        self,
        world: TextWorld,
        belief: WorldBelief,
        intervention: Intervention,
        *,
        seed: int,
    ) -> CounterfactualPrediction:
        return CounterfactualPrediction(intervention=intervention, value=belief.message, confidence=belief.confidence)

    def true_outcome(self, world: TextWorld, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        return self.observe(world, intervention, seed=seed)

    def score_prediction(self, prediction: CounterfactualPrediction, truth: ObservedOutcome) -> float:
        pred = prediction.value if isinstance(prediction.value, dict) else {}
        target = truth.value if isinstance(truth.value, dict) else {}
        hist_score = _distribution_score(
            _as_array(pred.get("histogram"), 1),
            _as_array(target.get("histogram"), 1),
        )
        transition_score = 1.0 - min(abs(float(pred.get("transition_mean", 0.0)) - float(target.get("transition_mean", 0.0))), 1.0)
        return 0.75 * hist_score + 0.25 * transition_score

    def world_label(self, world: TextWorld) -> str:
        return world.label


class RealImageCausalAdapter:
    """MNIST adapter exposing digit supports as interventions."""

    spec = CausalWorldSpec(
        domain="image",
        modality="image",
        hidden_target="real_digit_pixel_bounds_and_shape_profiles",
        outcome_name="digit_profile",
    )

    def __init__(
        self,
        config: RealCausalAdapterConfig | None = None,
        image_config: ImageProbeBenchmarkConfig | None = None,
        *,
        tensors: tuple[Any, Any] | None = None,
    ):
        self.config = config or RealCausalAdapterConfig()
        self.image_config = image_config or ImageProbeBenchmarkConfig()
        if tensors is None:
            train_images, train_labels, _test_images, _test_labels = load_mnist_tensors(self.image_config.data_dir)
        else:
            train_images, train_labels = tensors
        self.world = ImageWorld("mnist", train_images, train_labels)

    def world_for_seed(self, seed: int) -> ImageWorld:
        return self.world

    def intervention_space(self, world: ImageWorld, *, seed: int) -> tuple[Intervention, ...]:
        return tuple(
            Intervention(
                name=f"digit_{digit}",
                family="digit_support",
                cost=float(self.config.image_examples_per_digit),
                payload={"digit": int(digit)},
            )
            for digit in self.config.image_digits
        )

    def observe(self, world: ImageWorld, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        digit = int(intervention.payload["digit"])
        profile = _digit_profile(
            world.train_images,
            world.train_labels,
            digit=digit,
            count=int(self.config.image_examples_per_digit),
            seed=seed,
        )
        return ObservedOutcome(intervention=intervention, value=profile, cost=float(intervention.cost))

    def infer_belief(
        self,
        world: ImageWorld,
        observations: tuple[ObservedOutcome, ...],
        *,
        seed: int,
    ) -> WorldBelief:
        profiles = {int(item.value["digit"]): item.value for item in observations}
        confidence = float(len(profiles) / max(len(self.config.image_digits), 1))
        return WorldBelief(
            label=world.label,
            message={"profiles": profiles},
            confidence=confidence,
            uncertainty=1.0 - confidence,
            metadata={"digit_count": len(profiles)},
        )

    def predict_outcome(
        self,
        world: ImageWorld,
        belief: WorldBelief,
        intervention: Intervention,
        *,
        seed: int,
    ) -> CounterfactualPrediction:
        profiles = belief.message.get("profiles", {}) if isinstance(belief.message, dict) else {}
        digit = int(intervention.payload["digit"])
        return CounterfactualPrediction(
            intervention=intervention,
            value=profiles.get(digit, {"digit": digit}),
            confidence=belief.confidence,
        )

    def true_outcome(self, world: ImageWorld, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        return self.observe(world, intervention, seed=seed)

    def score_prediction(self, prediction: CounterfactualPrediction, truth: ObservedOutcome) -> float:
        pred = prediction.value if isinstance(prediction.value, dict) else {}
        target = truth.value if isinstance(truth.value, dict) else {}
        if int(pred.get("digit", -1)) != int(target.get("digit", -2)):
            return 0.0
        mean_score = 1.0 - min(abs(float(pred.get("mean_intensity", 0.0)) - float(target.get("mean_intensity", 0.0))) * 4.0, 1.0)
        center_score = 1.0 - min(float(np.linalg.norm(_center(pred) - _center(target))) / 28.0, 1.0)
        return 0.55 * mean_score + 0.45 * center_score

    def world_label(self, world: ImageWorld) -> str:
        return world.label


class RealCartPoleCausalAdapter:
    """CartPole mechanics adapter exposing real probe families."""

    spec = CausalWorldSpec(
        domain="cartpole",
        modality="rl_state",
        hidden_target="realistic_cartpole_mechanics_bounds",
        outcome_name="rollout_feature_vector",
    )

    def __init__(self, config: RealCausalAdapterConfig | None = None):
        self.config = config or RealCausalAdapterConfig()
        self.families = ("passive_decay", "impulse_left", "impulse_right", "chirp")

    def world_for_seed(self, seed: int) -> MechanicsWorld:
        return world_for_seed(seed)

    def intervention_space(self, world: MechanicsWorld, *, seed: int) -> tuple[Intervention, ...]:
        return tuple(
            Intervention(name=family, family=family, cost=float(self.config.cartpole_probe_steps))
            for family in self.families
        )

    def observe(self, world: MechanicsWorld, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        value = evidence_vector(world, (intervention.family,), seed=seed, steps=int(self.config.cartpole_probe_steps))
        return ObservedOutcome(intervention=intervention, value=value, cost=float(intervention.cost))

    def infer_belief(
        self,
        world: MechanicsWorld,
        observations: tuple[ObservedOutcome, ...],
        *,
        seed: int,
    ) -> WorldBelief:
        families = tuple(item.intervention.family for item in observations)
        observed = np.concatenate([np.asarray(item.value, dtype=np.float32).reshape(-1) for item in observations], axis=0)
        decoded, confidence, margin = infer_world(
            observed,
            families,
            seed=seed,
            steps=int(self.config.cartpole_probe_steps),
        )
        return WorldBelief(
            label=decoded.label(),
            message=decoded,
            confidence=float(confidence),
            uncertainty=1.0 - float(confidence),
            metadata={"margin": float(margin), "families": list(families)},
        )

    def predict_outcome(
        self,
        world: MechanicsWorld,
        belief: WorldBelief,
        intervention: Intervention,
        *,
        seed: int,
    ) -> CounterfactualPrediction:
        predicted = evidence_vector(
            belief.message,
            (intervention.family,),
            seed=seed,
            steps=int(self.config.cartpole_probe_steps),
        )
        return CounterfactualPrediction(intervention=intervention, value=predicted, confidence=belief.confidence)

    def true_outcome(self, world: MechanicsWorld, intervention: Intervention, *, seed: int) -> ObservedOutcome:
        return self.observe(world, intervention, seed=seed)

    def score_prediction(self, prediction: CounterfactualPrediction, truth: ObservedOutcome) -> float:
        pred = np.asarray(prediction.value, dtype=np.float32).reshape(-1)
        target = np.asarray(truth.value, dtype=np.float32).reshape(-1)
        scale = max(float(np.linalg.norm(target)), 1.0)
        return max(0.0, 1.0 - float(np.linalg.norm(pred - target)) / scale)

    def world_label(self, world: MechanicsWorld) -> str:
        return world.label()


def run_real_causal_adapter_suite(
    config: RealCausalAdapterConfig | None = None,
    *,
    language_config: LanguageProbeBenchmarkConfig | None = None,
    image_config: ImageProbeBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Run cheap real-environment causal probes across supported domains."""
    config = config or RealCausalAdapterConfig()
    return {
        "schema_version": 1,
        "runner": "run_real_causal_adapter_suite",
        "language": run_causal_crawler(
            RealLanguageCausalAdapter(config, language_config),
            seeds=config.seeds,
        ),
        "image": run_causal_crawler(
            RealImageCausalAdapter(config, image_config),
            seeds=config.seeds,
        ),
        "cartpole": run_causal_crawler(
            RealCartPoleCausalAdapter(config),
            seeds=config.seeds,
        ),
    }


def _text_span_stats(world: TextWorld, intervention: Intervention) -> dict[str, object]:
    start = int(intervention.payload["start"])
    length = int(intervention.payload["length"])
    span = world.encoded_train[start : start + length]
    counts = np.bincount(span, minlength=world.vocab_size).astype(np.float32)
    histogram = counts / max(float(np.sum(counts)), 1.0)
    transitions = np.abs(np.diff(span.astype(np.float32))) / max(float(world.vocab_size - 1), 1.0)
    return {
        "histogram": histogram,
        "transition_mean": float(np.mean(transitions)) if len(transitions) else 0.0,
    }


def _digit_profile(images, labels, *, digit: int, count: int, seed: int) -> dict[str, object]:
    label_np = labels.detach().cpu().numpy().astype(np.int64)
    indices = np.flatnonzero(label_np == int(digit))
    rng = np.random.default_rng(9300 + int(seed) + int(digit))
    if len(indices) > int(count):
        indices = rng.choice(indices, size=int(count), replace=False)
    batch = images[indices].detach().cpu().numpy().astype(np.float32)
    if batch.ndim == 4:
        batch = batch[:, 0]
    mean_image = np.mean(batch, axis=0) if len(batch) else np.zeros((28, 28), dtype=np.float32)
    weights = np.maximum(mean_image, 0.0)
    total = max(float(np.sum(weights)), 1e-6)
    rows = np.arange(mean_image.shape[0], dtype=np.float32)
    cols = np.arange(mean_image.shape[1], dtype=np.float32)
    return {
        "digit": int(digit),
        "mean_intensity": float(np.mean(mean_image)),
        "center_row": float(np.sum(weights * rows[:, None]) / total),
        "center_col": float(np.sum(weights * cols[None, :]) / total),
    }


def _as_array(value: object, fallback_size: int) -> np.ndarray:
    if value is None:
        return np.zeros((fallback_size,), dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _distribution_score(pred: np.ndarray, target: np.ndarray) -> float:
    size = max(pred.size, target.size)
    pred = np.pad(pred, (0, max(0, size - pred.size)))
    target = np.pad(target, (0, max(0, size - target.size)))
    return max(0.0, 1.0 - float(np.sum(np.abs(pred - target))) / 2.0)


def _center(profile: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [float(profile.get("center_row", 14.0)), float(profile.get("center_col", 14.0))],
        dtype=np.float32,
    )
