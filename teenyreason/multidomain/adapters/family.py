"""Family-real bridge benchmarks between mechanism checks and real tasks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FamilyBridgeConfig:
    """Small deterministic bridge benchmark settings."""

    seeds: tuple[int, ...] = tuple(range(12))
    support_items: int = 32
    challenge_items: int = 96
    noise: float = 0.04


LANGUAGE_RULES = ("previous_token", "next_token", "skip_forward", "mirror_token")
IMAGE_RULES = ("swapped_semantics", "normal_semantics", "rotated_semantics", "mirror_semantics")


def run_language_family_bridge(config: FamilyBridgeConfig | None = None) -> dict[str, object]:
    """Held-out grammar-family text where the crawler message is a rule."""
    config = config or FamilyBridgeConfig()
    alphabet_size = 7
    rows = [
        _language_bridge_world(seed, alphabet_size, config)
        for seed in config.seeds
    ]
    return _bridge_summary(
        rows,
        domain="language",
        dataset="GeneratedGrammarFamily",
        model_family="RuleMessage+NextTokenSolver",
        hidden_target="grammar_family_next_token_rule",
        learned_key="belief_next_token_accuracy",
        baseline_key="baseline_next_token_accuracy",
        zero_key="zero_next_token_accuracy",
        shuffled_key="shuffled_next_token_accuracy",
        stale_key="stale_next_token_accuracy",
    )


def run_image_family_bridge(config: FamilyBridgeConfig | None = None) -> dict[str, object]:
    """Held-out shape-family images where support labels define semantics."""
    config = config or FamilyBridgeConfig()
    rows = [_image_bridge_world(seed, config) for seed in config.seeds]
    return _bridge_summary(
        rows,
        domain="image",
        dataset="GeneratedShapeSemanticsFamily",
        model_family="RuleMessage+TemplateClassifier",
        hidden_target="shape_family_label_semantics",
        learned_key="belief_label_accuracy",
        baseline_key="baseline_label_accuracy",
        zero_key="zero_label_accuracy",
        shuffled_key="shuffled_label_accuracy",
        stale_key="stale_label_accuracy",
    )


def _language_bridge_world(
    seed: int,
    alphabet_size: int,
    config: FamilyBridgeConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(3000 + int(seed))
    hidden_rule = LANGUAGE_RULES[int(seed) % len(LANGUAGE_RULES)]
    support_tokens = rng.integers(0, alphabet_size, size=int(config.support_items))
    support_targets = _language_targets(hidden_rule, support_tokens, alphabet_size)
    noisy_mask = rng.random(size=support_targets.shape) < float(config.noise)
    support_targets = np.where(
        noisy_mask,
        rng.integers(0, alphabet_size, size=support_targets.shape),
        support_targets,
    )
    decoded_rule = _infer_language_rule(support_tokens, support_targets, alphabet_size)
    first_rule = _infer_language_rule(support_tokens[::2], support_targets[::2], alphabet_size)
    second_rule = _infer_language_rule(support_tokens[1::2], support_targets[1::2], alphabet_size)
    challenge_tokens = rng.integers(0, alphabet_size, size=int(config.challenge_items))
    truth = _language_targets(hidden_rule, challenge_tokens, alphabet_size)
    baseline_rule = "next_token"
    shuffled_rule = LANGUAGE_RULES[(LANGUAGE_RULES.index(decoded_rule) + 1) % len(LANGUAGE_RULES)]
    stale_token = int(np.bincount(support_targets, minlength=alphabet_size).argmax())
    baseline_accuracy = _accuracy(_language_targets(baseline_rule, challenge_tokens, alphabet_size), truth)
    learned_accuracy = _accuracy(_language_targets(decoded_rule, challenge_tokens, alphabet_size), truth)
    zero_accuracy = baseline_accuracy
    shuffled_accuracy = _accuracy(_language_targets(shuffled_rule, challenge_tokens, alphabet_size), truth)
    stale_accuracy = _accuracy(np.full_like(truth, stale_token), truth)
    best_ablation = max(zero_accuracy, shuffled_accuracy, stale_accuracy)
    return {
        "seed": int(seed),
        "hidden_rule": hidden_rule,
        "decoded_rule": decoded_rule,
        "decode_accuracy": float(decoded_rule == hidden_rule),
        "subset_agreement": float(first_rule == second_rule == decoded_rule),
        "baseline_next_token_accuracy": baseline_accuracy,
        "belief_next_token_accuracy": learned_accuracy,
        "zero_next_token_accuracy": zero_accuracy,
        "shuffled_next_token_accuracy": shuffled_accuracy,
        "stale_next_token_accuracy": stale_accuracy,
        "content_lift": learned_accuracy - best_ablation,
        "support_items": int(config.support_items),
        "challenge_items": int(config.challenge_items),
    }


def _image_bridge_world(seed: int, config: FamilyBridgeConfig) -> dict[str, object]:
    rng = np.random.default_rng(4000 + int(seed))
    hidden_rule = IMAGE_RULES[int(seed) % len(IMAGE_RULES)]
    support_patterns = rng.integers(0, 4, size=int(config.support_items))
    support_images = [_shape_image(int(pattern), rng, config.noise) for pattern in support_patterns]
    support_labels = np.asarray(
        [_image_label_for(int(pattern), hidden_rule) for pattern in support_patterns],
        dtype=np.int64,
    )
    decoded_rule = _infer_image_rule(support_images, support_labels)
    first_rule = _infer_image_rule(support_images[::2], support_labels[::2])
    second_rule = _infer_image_rule(support_images[1::2], support_labels[1::2])
    challenge_patterns = rng.integers(0, 4, size=int(config.challenge_items))
    challenge_images = [_shape_image(int(pattern), rng, config.noise) for pattern in challenge_patterns]
    truth = np.asarray([_image_label_for(int(pattern), hidden_rule) for pattern in challenge_patterns])
    observed_patterns = np.asarray([_visual_pattern(image) for image in challenge_images], dtype=np.int64)
    shuffled_rule = IMAGE_RULES[(IMAGE_RULES.index(decoded_rule) + 1) % len(IMAGE_RULES)]
    stale_label = int(np.bincount(support_labels, minlength=4).argmax())
    baseline_accuracy = _image_accuracy(observed_patterns, truth, "normal_semantics")
    learned_accuracy = _image_accuracy(observed_patterns, truth, decoded_rule)
    zero_accuracy = baseline_accuracy
    shuffled_accuracy = _image_accuracy(observed_patterns, truth, shuffled_rule)
    stale_accuracy = _accuracy(np.full_like(truth, stale_label), truth)
    best_ablation = max(zero_accuracy, shuffled_accuracy, stale_accuracy)
    return {
        "seed": int(seed),
        "hidden_rule": hidden_rule,
        "decoded_rule": decoded_rule,
        "decode_accuracy": float(decoded_rule == hidden_rule),
        "subset_agreement": float(first_rule == second_rule == decoded_rule),
        "baseline_label_accuracy": baseline_accuracy,
        "belief_label_accuracy": learned_accuracy,
        "zero_label_accuracy": zero_accuracy,
        "shuffled_label_accuracy": shuffled_accuracy,
        "stale_label_accuracy": stale_accuracy,
        "content_lift": learned_accuracy - best_ablation,
        "support_items": int(config.support_items),
        "challenge_items": int(config.challenge_items),
    }


def _bridge_summary(
    rows: list[dict[str, object]],
    *,
    domain: str,
    dataset: str,
    model_family: str,
    hidden_target: str,
    learned_key: str,
    baseline_key: str,
    zero_key: str,
    shuffled_key: str,
    stale_key: str,
) -> dict[str, object]:
    return {
        "domain": domain,
        "dataset": dataset,
        "model_family": model_family,
        "hidden_target": hidden_target,
        "rows": rows,
        "candidate_rules": list(LANGUAGE_RULES if domain == "language" else IMAGE_RULES),
        "decode_accuracy": _mean(rows, "decode_accuracy"),
        "subset_agreement": _mean(rows, "subset_agreement"),
        "baseline_accuracy": _mean(rows, baseline_key),
        "belief_accuracy": _mean(rows, learned_key),
        "zero_accuracy": _mean(rows, zero_key),
        "shuffled_accuracy": _mean(rows, shuffled_key),
        "stale_accuracy": _mean(rows, stale_key),
        "solver_gain": _mean(rows, learned_key) - _mean(rows, baseline_key),
        "content_lift": _mean(rows, "content_lift"),
    }


def _language_targets(rule: str, tokens: np.ndarray, alphabet_size: int) -> np.ndarray:
    if rule == "previous_token":
        return (tokens - 1) % alphabet_size
    if rule == "skip_forward":
        return (tokens + 2) % alphabet_size
    if rule == "mirror_token":
        return (alphabet_size - 1 - tokens) % alphabet_size
    return (tokens + 1) % alphabet_size


def _infer_language_rule(tokens: np.ndarray, observed: np.ndarray, alphabet_size: int) -> str:
    scores = {
        rule: _accuracy(_language_targets(rule, tokens, alphabet_size), observed)
        for rule in LANGUAGE_RULES
    }
    return max(scores, key=scores.get)


def _shape_image(pattern: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    image = rng.normal(loc=0.0, scale=float(noise), size=(8, 8)).astype(np.float32)
    if pattern == 0:
        image[:, 3] += 1.0
    elif pattern == 1:
        image[3, :] += 1.0
    elif pattern == 2:
        np.fill_diagonal(image, np.diag(image) + 1.0)
    else:
        flipped = np.fliplr(image)
        np.fill_diagonal(flipped, np.diag(flipped) + 1.0)
        image = np.fliplr(flipped)
    return image


def _visual_pattern(image: np.ndarray) -> int:
    scores = (
        float(np.sum(image[:, 3])),
        float(np.sum(image[3, :])),
        float(np.trace(image)),
        float(np.trace(np.fliplr(image))),
    )
    return int(np.argmax(scores))


def _image_label_for(pattern: int, rule: str) -> int:
    if rule == "swapped_semantics":
        return int(pattern) ^ 1
    if rule == "rotated_semantics":
        return (int(pattern) + 1) % 4
    if rule == "mirror_semantics":
        return 3 - int(pattern)
    return int(pattern)


def _infer_image_rule(images: list[np.ndarray], labels: np.ndarray) -> str:
    patterns = np.asarray([_visual_pattern(image) for image in images], dtype=np.int64)
    scores = {
        rule: _image_accuracy(patterns, labels, rule)
        for rule in IMAGE_RULES
    }
    return max(scores, key=scores.get)


def _image_accuracy(patterns: np.ndarray, truth: np.ndarray, rule: str) -> float:
    predictions = np.asarray([_image_label_for(int(pattern), rule) for pattern in patterns], dtype=np.int64)
    return _accuracy(predictions, truth)


def _accuracy(predictions: np.ndarray, truth: np.ndarray) -> float:
    if len(truth) == 0:
        return 0.0
    return float(np.mean(np.asarray(predictions) == np.asarray(truth)))


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))
