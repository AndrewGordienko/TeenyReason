"""Crawler adapter implementations for bridge checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..domains.board import (
    RULES,
    RULE_NORMAL,
    build_rule_message,
    challenge_positions,
    evaluate_minimax_policy,
    query_rule_outcome,
    rule_from_message,
)
from ..domains.cartpole import (
    CartPoleControllerBridgeConfig,
    candidate_worlds,
    controller_return,
    evidence_vector,
    infer_world,
    nominal_world,
    world_for_seed,
)
from .crawler import AdapterSpec, CrawlScore, run_crawler_adapter
from .family import (
    FamilyBridgeConfig,
    IMAGE_RULES,
    LANGUAGE_RULES,
    _accuracy,
    _image_accuracy,
    _image_label_for,
    _infer_image_rule,
    _infer_language_rule,
    _language_targets,
    _shape_image,
    _visual_pattern,
)


@dataclass(frozen=True)
class GrammarWorld:
    rule: str


@dataclass(frozen=True)
class ShapeWorld:
    rule: str


@dataclass(frozen=True)
class BoardWorld:
    rule: str


class LanguageBridgeAdapter:
    """Generated text-family adapter using the shared crawler runner."""

    def __init__(self, config: FamilyBridgeConfig):
        self.config = config
        self.alphabet_size = 7
        self.spec = AdapterSpec(
            domain="language",
            modality="text",
            dataset="GeneratedGrammarFamily",
            model_family="RuleMessage+NextTokenSolver",
            hidden_target="grammar_family_next_token_rule",
            metric_name="generated_next_token_accuracy",
            query_families=("support_even", "support_odd"),
            message_dim=len(LANGUAGE_RULES),
        )

    def world_for_seed(self, seed: int) -> GrammarWorld:
        return GrammarWorld(rule=LANGUAGE_RULES[int(seed) % len(LANGUAGE_RULES)])

    def collect_evidence(self, world: GrammarWorld, *, seed: int, families: tuple[str, ...]) -> dict[str, Any]:
        rng = np.random.default_rng(3000 + int(seed))
        tokens = rng.integers(0, self.alphabet_size, size=int(self.config.support_items))
        targets = _language_targets(world.rule, tokens, self.alphabet_size)
        noisy_mask = rng.random(size=targets.shape) < float(self.config.noise)
        targets = np.where(
            noisy_mask,
            rng.integers(0, self.alphabet_size, size=targets.shape),
            targets,
        )
        if families == ("support_even",):
            tokens = tokens[::2]
            targets = targets[::2]
        elif families == ("support_odd",):
            tokens = tokens[1::2]
            targets = targets[1::2]
        return {"tokens": tokens, "targets": targets}

    def infer_message(self, evidence: dict[str, Any], *, seed: int, families: tuple[str, ...]) -> str:
        return _infer_language_rule(evidence["tokens"], evidence["targets"], self.alphabet_size)

    def ablation_messages(self, message: str, world: GrammarWorld, *, seed: int) -> dict[str, str | int]:
        evidence = self.collect_evidence(world, seed=seed, families=self.spec.query_families)
        stale = int(np.bincount(evidence["targets"], minlength=self.alphabet_size).argmax())
        shuffled = LANGUAGE_RULES[(LANGUAGE_RULES.index(message) + 1) % len(LANGUAGE_RULES)]
        return {"zero": "next_token", "shuffled": shuffled, "stale": stale}

    def score_baseline(self, world: GrammarWorld, *, seed: int) -> CrawlScore:
        return self._score_rule(world, "next_token", seed)

    def score_with_message(self, world: GrammarWorld, message: str | int, *, seed: int) -> CrawlScore:
        if isinstance(message, int):
            return self._score_constant(world, message, seed)
        return self._score_rule(world, str(message), seed)

    def world_label(self, world: GrammarWorld) -> str:
        return world.rule

    def message_label(self, message: str | int) -> str:
        return str(message)

    def _challenge(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7000 + int(seed))
        tokens = rng.integers(0, self.alphabet_size, size=int(self.config.challenge_items))
        return tokens, tokens

    def _score_rule(self, world: GrammarWorld, rule: str, seed: int) -> CrawlScore:
        tokens, _unused = self._challenge(seed)
        truth = _language_targets(world.rule, tokens, self.alphabet_size)
        predictions = _language_targets(rule, tokens, self.alphabet_size)
        return CrawlScore(_accuracy(predictions, truth), self.spec.metric_name)

    def _score_constant(self, world: GrammarWorld, label: int, seed: int) -> CrawlScore:
        tokens, _unused = self._challenge(seed)
        truth = _language_targets(world.rule, tokens, self.alphabet_size)
        return CrawlScore(_accuracy(np.full_like(truth, int(label)), truth), self.spec.metric_name)


class ImageBridgeAdapter:
    """Generated image-family adapter using the shared crawler runner."""

    def __init__(self, config: FamilyBridgeConfig):
        self.config = config
        self.spec = AdapterSpec(
            domain="image",
            modality="image",
            dataset="GeneratedShapeSemanticsFamily",
            model_family="RuleMessage+TemplateClassifier",
            hidden_target="shape_family_label_semantics",
            metric_name="generated_shape_label_accuracy",
            query_families=("support_first_half", "support_second_half"),
            message_dim=len(IMAGE_RULES),
        )

    def world_for_seed(self, seed: int) -> ShapeWorld:
        return ShapeWorld(rule=IMAGE_RULES[int(seed) % len(IMAGE_RULES)])

    def collect_evidence(self, world: ShapeWorld, *, seed: int, families: tuple[str, ...]) -> dict[str, Any]:
        rng = np.random.default_rng(4000 + int(seed))
        patterns = rng.integers(0, 4, size=int(self.config.support_items))
        images = [_shape_image(int(pattern), rng, self.config.noise) for pattern in patterns]
        labels = np.asarray([_image_label_for(int(pattern), world.rule) for pattern in patterns], dtype=np.int64)
        if families == ("support_first_half",):
            half = max(1, len(images) // 2)
            images = images[:half]
            labels = labels[:half]
        elif families == ("support_second_half",):
            half = max(1, len(images) // 2)
            images = images[half:]
            labels = labels[half:]
        return {"images": images, "labels": labels}

    def infer_message(self, evidence: dict[str, Any], *, seed: int, families: tuple[str, ...]) -> str:
        return _infer_image_rule(evidence["images"], evidence["labels"])

    def ablation_messages(self, message: str, world: ShapeWorld, *, seed: int) -> dict[str, str | int]:
        evidence = self.collect_evidence(world, seed=seed, families=self.spec.query_families)
        shuffled = IMAGE_RULES[(IMAGE_RULES.index(message) + 1) % len(IMAGE_RULES)]
        stale = int(np.bincount(evidence["labels"], minlength=4).argmax())
        return {"zero": "normal_semantics", "shuffled": shuffled, "stale": stale}

    def score_baseline(self, world: ShapeWorld, *, seed: int) -> CrawlScore:
        return self._score_rule(world, "normal_semantics", seed)

    def score_with_message(self, world: ShapeWorld, message: str | int, *, seed: int) -> CrawlScore:
        if isinstance(message, int):
            return self._score_constant(world, message, seed)
        return self._score_rule(world, str(message), seed)

    def world_label(self, world: ShapeWorld) -> str:
        return world.rule

    def message_label(self, message: str | int) -> str:
        return str(message)

    def _challenge(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(8000 + int(seed))
        patterns = rng.integers(0, 4, size=int(self.config.challenge_items))
        images = [_shape_image(int(pattern), rng, self.config.noise) for pattern in patterns]
        observed = np.asarray([_visual_pattern(image) for image in images], dtype=np.int64)
        return patterns, observed

    def _score_rule(self, world: ShapeWorld, rule: str, seed: int) -> CrawlScore:
        patterns, observed = self._challenge(seed)
        truth = np.asarray([_image_label_for(int(pattern), world.rule) for pattern in patterns], dtype=np.int64)
        return CrawlScore(_image_accuracy(observed, truth, rule), self.spec.metric_name)

    def _score_constant(self, world: ShapeWorld, label: int, seed: int) -> CrawlScore:
        patterns, _observed = self._challenge(seed)
        truth = np.asarray([_image_label_for(int(pattern), world.rule) for pattern in patterns], dtype=np.int64)
        return CrawlScore(_accuracy(np.full_like(truth, int(label)), truth), self.spec.metric_name)


class CartPoleControllerAdapter:
    """CartPole mechanics adapter using the shared crawler runner."""

    def __init__(self, config: CartPoleControllerBridgeConfig):
        self.config = config
        self.spec = AdapterSpec(
            domain="cartpole",
            modality="rl_state",
            dataset="ControlledCartPoleControllerBridge",
            model_family="MechanicsMessage+OneStepMPC",
            hidden_target="cartpole_mechanics_controller_handoff",
            metric_name="controller_return",
            query_families=tuple(config.support_families),
            message_dim=4,
        )

    def world_for_seed(self, seed: int) -> Any:
        return world_for_seed(seed)

    def collect_evidence(self, world: Any, *, seed: int, families: tuple[str, ...]) -> np.ndarray:
        return evidence_vector(world, families, seed=seed, steps=self.config.probe_steps)

    def infer_message(self, evidence: np.ndarray, *, seed: int, families: tuple[str, ...]) -> Any:
        decoded, _confidence, _margin = infer_world(evidence, families, seed=seed, steps=self.config.probe_steps)
        return decoded

    def ablation_messages(self, message: Any, world: Any, *, seed: int) -> dict[str, Any]:
        worlds = candidate_worlds()
        return {
            "zero": nominal_world(),
            "shuffled": worlds[(worlds.index(message) + 5) % len(worlds)],
            "stale": world_for_seed(int(seed) - 1),
        }

    def score_baseline(self, world: Any, *, seed: int) -> CrawlScore:
        return self.score_with_message(world, nominal_world(), seed=seed)

    def score_with_message(self, world: Any, message: Any, *, seed: int) -> CrawlScore:
        value = controller_return(
            world,
            message,
            seed=seed,
            steps=self.config.control_steps,
            action_grid=self.config.action_grid,
        )
        return CrawlScore(value, self.spec.metric_name)

    def world_label(self, world: Any) -> str:
        return str(world.label())

    def message_label(self, message: Any) -> str:
        return str(message.label())


class BoardRuleAdapter:
    """Board-game adapter proving minimax is just another solver handoff."""

    def __init__(self, challenge_count: int = 18):
        self.challenge_count = int(challenge_count)
        self.spec = AdapterSpec(
            domain="board",
            modality="board_game",
            dataset="TicTacToe hidden-rule positions",
            model_family="CrawlerMessage+ExactMinimax",
            hidden_target="normal versus misere rule",
            metric_name="best_move_accuracy",
            query_families=("x_line_completion", "o_line_completion"),
            message_dim=4,
        )

    def world_for_seed(self, seed: int) -> BoardWorld:
        return BoardWorld(rule=RULES[int(seed) % len(RULES)])

    def collect_evidence(self, world: BoardWorld, *, seed: int, families: tuple[str, ...]) -> list[dict[str, object]]:
        return [query_rule_outcome(family, world.rule) for family in families]

    def infer_message(self, evidence: list[dict[str, object]], *, seed: int, families: tuple[str, ...]) -> str:
        return rule_from_message(build_rule_message(evidence))

    def ablation_messages(self, message: str, world: BoardWorld, *, seed: int) -> dict[str, str]:
        stale = RULES[(int(seed) + 1) % len(RULES)]
        return {"zero": RULE_NORMAL, "shuffled": stale, "stale": stale}

    def score_baseline(self, world: BoardWorld, *, seed: int) -> CrawlScore:
        return self.score_with_message(world, RULE_NORMAL, seed=seed)

    def score_with_message(self, world: BoardWorld, message: str, *, seed: int) -> CrawlScore:
        positions = challenge_positions(seed=int(seed) + 101, count=self.challenge_count)
        metrics = evaluate_minimax_policy(
            positions=positions,
            hidden_rule=world.rule,
            solver_rule=str(message),
        )
        return CrawlScore(float(metrics["move_accuracy"]), self.spec.metric_name)

    def world_label(self, world: BoardWorld) -> str:
        return world.rule

    def message_label(self, message: str) -> str:
        return str(message)


def run_language_adapter_bridge(config: FamilyBridgeConfig | None = None) -> dict[str, object]:
    config = config or FamilyBridgeConfig()
    return _legacy_bridge_keys(run_crawler_adapter(LanguageBridgeAdapter(config), seeds=config.seeds))


def run_image_adapter_bridge(config: FamilyBridgeConfig | None = None) -> dict[str, object]:
    config = config or FamilyBridgeConfig()
    return _legacy_bridge_keys(run_crawler_adapter(ImageBridgeAdapter(config), seeds=config.seeds))


def run_cartpole_adapter_bridge(config: CartPoleControllerBridgeConfig | None = None) -> dict[str, object]:
    config = config or CartPoleControllerBridgeConfig()
    return _legacy_bridge_keys(run_crawler_adapter(CartPoleControllerAdapter(config), seeds=config.seeds))


def run_board_adapter_bridge(seeds: tuple[int, ...] = (0, 1, 2, 3)) -> dict[str, object]:
    return _legacy_bridge_keys(run_crawler_adapter(BoardRuleAdapter(), seeds=seeds))


def _legacy_bridge_keys(result: dict[str, object]) -> dict[str, object]:
    """Expose generic values under existing bridge field names."""
    return {
        **result,
        "baseline_accuracy": float(result.get("baseline_value", 0.0)),
        "belief_accuracy": float(result.get("belief_value", 0.0)),
        "zero_accuracy": float(result.get("zero_value", 0.0)),
        "shuffled_accuracy": float(result.get("shuffled_value", 0.0)),
        "stale_accuracy": float(result.get("stale_value", 0.0)),
        "baseline_return": float(result.get("baseline_value", 0.0)),
        "belief_return": float(result.get("belief_value", 0.0)),
        "zero_return": float(result.get("zero_value", 0.0)),
        "shuffled_return": float(result.get("shuffled_value", 0.0)),
        "stale_return": float(result.get("stale_value", 0.0)),
    }
