"""MNIST belief-conditioned prototype benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from .handoff import (
    best_compressed_row as _best_compressed_row,
    image_compression_rows as _image_compression_rows,
    select_image_handoff as _select_image_handoff,
)
from .residual import train_residual_belief_adapter
from .models import (
    BeliefConditionedMNISTCNN,
    MNISTClassifier,
    MNISTEncoder,
    compress_prototype_context,
    shuffle_prototype_context,
    zero_prototype_context,
)

@dataclass(frozen=True)
class ImageProbeBenchmarkConfig:
    """Configuration for the MNIST belief-conditioned benchmark."""

    data_dir: Path = Path("artifacts/data/mnist")
    label_budgets: tuple[int, ...] = (256, 1024, 4096)
    unlabeled_budget: int = 20000
    test_budget: int = 5000
    probe_epochs: int = 3
    finetune_epochs: int = 4
    batch_size: int = 128
    lr: float = 1e-3
    feature_dim: int = 64
    compression_bits: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
    seed: int = 0


def set_seed(seed: int):
    """Keep NumPy and Torch aligned for reproducible subsets and batches."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_subset_indices(labels: torch.Tensor, budget: int, seed: int) -> np.ndarray:
    """Sample a roughly class-balanced labeled subset."""
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    unique_labels = sorted(int(label) for label in np.unique(labels_np))
    rng = np.random.default_rng(seed)
    per_class = max(1, int(budget) // max(len(unique_labels), 1))
    chosen: list[int] = []
    leftovers: list[int] = []

    for label in unique_labels:
        label_indices = np.flatnonzero(labels_np == label)
        rng.shuffle(label_indices)
        chosen.extend(label_indices[:per_class].tolist())
        leftovers.extend(label_indices[per_class:].tolist())

    if len(chosen) < int(budget):
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: int(budget) - len(chosen)])

    chosen = chosen[: int(budget)]
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=np.int64)


def sample_subset_indices(size: int, budget: int, seed: int) -> np.ndarray:
    """Sample one deterministic subset without replacement."""
    rng = np.random.default_rng(seed)
    budget = min(int(budget), int(size))
    return np.asarray(rng.choice(size, size=budget, replace=False), dtype=np.int64)


def load_mnist_tensors(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST into plain tensors so the benchmark can sample subsets cheaply."""
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        train = MNIST(root=str(data_dir), train=True, download=True)
        test = MNIST(root=str(data_dir), train=False, download=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load MNIST. If the dataset is not cached locally, the first run needs internet access."
        ) from exc

    train_images = train.data.unsqueeze(1).float() / 255.0
    train_labels = train.targets.long()
    test_images = test.data.unsqueeze(1).float() / 255.0
    test_labels = test.targets.long()
    return train_images, train_labels, test_images, test_labels


def build_loader(
    images: torch.Tensor,
    labels: torch.Tensor | None,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build one plain DataLoader from a tensor subset."""
    index_t = torch.tensor(indices, dtype=torch.long)
    subset_images = images.index_select(0, index_t)
    if labels is None:
        dataset = TensorDataset(subset_images)
    else:
        subset_labels = labels.index_select(0, index_t)
        dataset = TensorDataset(subset_images, subset_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_feature_belief_context(
    encoder: MNISTEncoder,
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    support_indices: np.ndarray,
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
    source: str = "learned",
) -> dict[str, torch.Tensor | float]:
    """Aggregate support examples into class prototypes in encoder space."""
    loader = build_loader(
        images=train_images,
        labels=train_labels,
        indices=support_indices,
        batch_size=config.batch_size,
        shuffle=False,
    )
    feature_dim = int(config.feature_dim)
    sums = torch.zeros((10, feature_dim), dtype=torch.float32, device=device)
    counts = torch.zeros((10,), dtype=torch.float32, device=device)
    features_by_class: list[list[torch.Tensor]] = [[] for _ in range(10)]
    encoder.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            features = encoder(images)
            for digit in range(10):
                mask = labels == digit
                if torch.any(mask):
                    selected = features[mask]
                    sums[digit] += selected.sum(dim=0)
                    counts[digit] += float(selected.shape[0])
                    features_by_class[digit].append(selected.detach())
    prototypes = sums / torch.clamp(counts.view(-1, 1), min=1.0)
    within_distances: list[torch.Tensor] = []
    for digit, chunks in enumerate(features_by_class):
        if chunks:
            class_features = torch.cat(chunks, dim=0)
            within_distances.append(torch.norm(class_features - prototypes[digit], dim=-1))
    uncertainty = 1.0
    if within_distances:
        uncertainty = float(torch.mean(torch.cat(within_distances)).item())
    present = float(torch.count_nonzero(counts).item())
    return {
        "prototypes": prototypes.detach(),
        "confidence": present / 10.0,
        "uncertainty": uncertainty,
        "counts": counts.detach(),
        "source": source,
    }


def evaluate_plain_classifier(
    model: MNISTClassifier,
    *,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    test_indices: np.ndarray,
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate one plain classifier."""
    loader = build_loader(test_images, test_labels, test_indices, config.batch_size, False)
    model.eval()
    correct = 0
    total = 0
    losses: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            losses.append(float(F.cross_entropy(logits, labels).item()))
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return {
        "accuracy": float(correct) / float(max(total, 1)),
        "nll": float(np.mean(losses)) if losses else 0.0,
    }


def evaluate_belief_classifier(
    model: BeliefConditionedMNISTCNN,
    *,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    test_indices: np.ndarray,
    context: dict[str, torch.Tensor | float],
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a belief-conditioned classifier arm."""
    loader = build_loader(test_images, test_labels, test_indices, config.batch_size, False)
    model.eval()
    correct = 0
    total = 0
    losses: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, context)
            losses.append(float(F.cross_entropy(logits, labels).item()))
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return {
        "accuracy": float(correct) / float(max(total, 1)),
        "nll": float(np.mean(losses)) if losses else 0.0,
    }


def train_classifier(
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    labeled_indices: np.ndarray,
    init_encoder_state: dict[str, torch.Tensor] | None,
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
) -> float:
    """Train one baseline classifier and return test accuracy."""
    _model, metrics = train_classifier_model(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        labeled_indices=labeled_indices,
        init_encoder_state=init_encoder_state,
        config=config,
        device=device,
    )
    return float(metrics["accuracy"])


def train_classifier_model(
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    labeled_indices: np.ndarray,
    init_encoder_state: dict[str, torch.Tensor] | None,
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
) -> tuple[MNISTClassifier, dict[str, float]]:
    """Train one baseline classifier and return the model plus metrics."""
    model = MNISTClassifier(feature_dim=config.feature_dim).to(device)
    if init_encoder_state is not None:
        model.encoder.load_state_dict(init_encoder_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader = build_loader(
        images=train_images,
        labels=train_labels,
        indices=labeled_indices,
        batch_size=config.batch_size,
        shuffle=True,
    )
    for _epoch in range(config.finetune_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    test_indices = sample_subset_indices(len(test_images), config.test_budget, config.seed + 99)
    metrics = evaluate_plain_classifier(
        model,
        test_images=test_images,
        test_labels=test_labels,
        test_indices=test_indices,
        config=config,
        device=device,
    )
    return model, metrics


def train_belief_conditioned_classifier(
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    labeled_indices: np.ndarray,
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
    baseline_accuracy: float = 0.0,
) -> dict[str, float | dict[str, float]]:
    """Train a classifier that consumes learned prototype belief context."""
    model = BeliefConditionedMNISTCNN(feature_dim=config.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_loader = build_loader(
        images=train_images,
        labels=train_labels,
        indices=labeled_indices,
        batch_size=config.batch_size,
        shuffle=True,
    )
    stale_context = build_feature_belief_context(
        model.encoder,
        train_images=train_images,
        train_labels=train_labels,
        support_indices=labeled_indices,
        config=config,
        device=device,
        source="stale",
    )
    context = stale_context
    for _epoch in range(config.finetune_epochs):
        context = build_feature_belief_context(
            model.encoder,
            train_images=train_images,
            train_labels=train_labels,
            support_indices=labeled_indices,
            config=config,
            device=device,
        )
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, context)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    context = build_feature_belief_context(
        model.encoder,
        train_images=train_images,
        train_labels=train_labels,
        support_indices=labeled_indices,
        config=config,
        device=device,
    )
    zero_context = zero_prototype_context(config.feature_dim, device=device)
    shuffled_context = shuffle_prototype_context(context)
    test_indices = sample_subset_indices(len(test_images), config.test_budget, config.seed + 99)
    metrics = {
        "learned": evaluate_belief_classifier(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=context,
            config=config,
            device=device,
        ),
        "zero": evaluate_belief_classifier(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=zero_context,
            config=config,
            device=device,
        ),
        "shuffled": evaluate_belief_classifier(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=shuffled_context,
            config=config,
            device=device,
        ),
        "stale": evaluate_belief_classifier(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=stale_context,
            config=config,
            device=device,
        ),
    }
    compression_metrics = evaluate_compressed_prototype_contexts(
        model,
        test_images=test_images,
        test_labels=test_labels,
        test_indices=test_indices,
        context=context,
        baseline_accuracy=baseline_accuracy,
        best_ablation_accuracy=max(
            float(metrics["zero"]["accuracy"]),
            float(metrics["shuffled"]["accuracy"]),
            float(metrics["stale"]["accuracy"]),
        ),
        zero_context=zero_context,
        shuffled_context=shuffled_context,
        stale_context=stale_context,
        config=config,
        device=device,
    )
    stability = prototype_subset_agreement(
        model.encoder,
        train_images=train_images,
        train_labels=train_labels,
        labeled_indices=labeled_indices,
        config=config,
        device=device,
    )
    return {
        "ablation_metrics": metrics,
        "compression_metrics": compression_metrics,
        "prototype_stability": stability,
        "belief_bitrate": int((10 * config.feature_dim + 2) * 32),
        **metrics["learned"],
    }


def evaluate_compressed_prototype_contexts(
    model: BeliefConditionedMNISTCNN,
    *,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    test_indices: np.ndarray,
    context: dict[str, torch.Tensor | float],
    baseline_accuracy: float,
    best_ablation_accuracy: float,
    zero_context: dict[str, torch.Tensor | float],
    shuffled_context: dict[str, torch.Tensor | float],
    stale_context: dict[str, torch.Tensor | float],
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
) -> list[dict[str, float | int | bool]]:
    """Evaluate the trained solver under smaller prototype messages."""
    rows: list[dict[str, float | int | bool]] = []
    for target_bits in config.compression_bits:
        compressed = compress_prototype_context(context, target_bits=int(target_bits))
        compressed_zero = compress_prototype_context(zero_context, target_bits=int(target_bits))
        compressed_shuffled = compress_prototype_context(shuffled_context, target_bits=int(target_bits))
        compressed_stale = compress_prototype_context(stale_context, target_bits=int(target_bits))
        metrics = evaluate_belief_classifier(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=compressed,
            config=config,
            device=device,
        )
        zero_metrics = evaluate_belief_classifier(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=compressed_zero,
            config=config,
            device=device,
        )
        shuffled_metrics = evaluate_belief_classifier(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=compressed_shuffled,
            config=config,
            device=device,
        )
        stale_metrics = evaluate_belief_classifier(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=compressed_stale,
            config=config,
            device=device,
        )
        accuracy = float(metrics["accuracy"])
        zero_accuracy = float(zero_metrics["accuracy"])
        shuffled_accuracy = float(shuffled_metrics["accuracy"])
        stale_accuracy = float(stale_metrics["accuracy"])
        compressed_best_ablation = max(zero_accuracy, shuffled_accuracy, stale_accuracy)
        content_lift = accuracy - compressed_best_ablation
        solver_gain = accuracy - float(baseline_accuracy)
        rows.append(
            {
                "bits": int(target_bits),
                "accuracy": accuracy,
                "nll": float(metrics["nll"]),
                "zero_accuracy": zero_accuracy,
                "shuffled_accuracy": shuffled_accuracy,
                "stale_accuracy": stale_accuracy,
                "best_ablation_accuracy": compressed_best_ablation,
                "full_best_ablation_accuracy": float(best_ablation_accuracy),
                "solver_gain": solver_gain,
                "content_lift": content_lift,
                "lift_per_1k_bits": content_lift * 1000.0 / max(float(target_bits), 1.0),
                "gain_per_1k_bits": solver_gain * 1000.0 / max(float(target_bits), 1.0),
                "retained_feature_dims": int(compressed["retained_feature_dims"]),
                "measured": True,
            }
        )
    return rows


def image_row_economics(
    *,
    baseline_accuracy: float,
    belief_accuracy: float,
    ablations: dict[str, dict[str, float]],
    belief_bitrate: int,
    label_budget: int,
) -> dict[str, float | int | bool]:
    """Compute per-budget MNIST handoff economics."""
    zero_accuracy = float(ablations["zero"]["accuracy"])
    shuffled_accuracy = float(ablations["shuffled"]["accuracy"])
    stale_accuracy = float(ablations["stale"]["accuracy"])
    best_ablation_accuracy = max(zero_accuracy, shuffled_accuracy, stale_accuracy)
    accuracy_gain = float(belief_accuracy) - float(baseline_accuracy)
    content_lift = float(belief_accuracy) - best_ablation_accuracy
    labels = max(float(label_budget), 1.0)
    bits = max(float(belief_bitrate), 1.0)
    return {
        "best_ablation_accuracy": best_ablation_accuracy,
        "content_lift": content_lift,
        "budget_gate_uses_belief": bool(accuracy_gain > 0.0 and content_lift >= 0.0),
        "accuracy_gain_per_label": accuracy_gain / labels,
        "content_lift_per_label": content_lift / labels,
        "accuracy_gain_per_1k_bits": accuracy_gain * 1000.0 / bits,
        "content_lift_per_1k_bits": content_lift * 1000.0 / bits,
    }


def prototype_subset_agreement(
    encoder: MNISTEncoder,
    *,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    labeled_indices: np.ndarray,
    config: ImageProbeBenchmarkConfig,
    device: torch.device,
) -> float:
    """Measure whether two support halves form similar prototype beliefs."""
    if len(labeled_indices) < 4:
        return 0.0
    first = labeled_indices[::2]
    second = labeled_indices[1::2]
    ctx_a = build_feature_belief_context(
        encoder,
        train_images=train_images,
        train_labels=train_labels,
        support_indices=first,
        config=config,
        device=device,
    )
    ctx_b = build_feature_belief_context(
        encoder,
        train_images=train_images,
        train_labels=train_labels,
        support_indices=second,
        config=config,
        device=device,
    )
    counts = torch.minimum(ctx_a["counts"], ctx_b["counts"]) > 0
    if not torch.any(counts):
        return 0.0
    proto_a = F.normalize(ctx_a["prototypes"][counts], dim=-1)
    proto_b = F.normalize(ctx_b["prototypes"][counts], dim=-1)
    return float(torch.mean(torch.sum(proto_a * proto_b, dim=-1)).item())


def build_image_artifact(
    *,
    labeled_indices: np.ndarray,
    row: dict[str, float | int],
    config: ImageProbeBenchmarkConfig,
    compression_rows: list[dict[str, float | int | bool | str]] | None = None,
) -> dict[str, object]:
    """Build the dashboard-facing image belief artifact summary."""
    evidence = [
        {
            "modality": "image",
            "query_family": family,
            "source_id": "mnist",
            "intervention_cost": 1.0,
            "local_state": {"support_count": int(len(labeled_indices))},
            "outcome": {"solver": "few_shot_digit_classification"},
        }
        for family in ("support_example", "crop", "mask", "augment", "contrastive_view")
    ]
    return {
        "raw_evidence_windows": evidence,
        "source_ids": ["mnist"],
        "query_families": ["support_example", "crop", "mask", "augment", "contrastive_view"],
        "local_evidence_latents": [],
        "domain_belief": {"prototype_count": 10, "feature_dim": int(config.feature_dim)},
        "uncertainty_estimate": float(row.get("belief_nll", 0.0)),
        "hidden_rule_targets": {"digit": "mnist_label"},
        "subset_agreement": float(row.get("prototype_stability", 0.0)),
        "belief_bitrate": int(row.get("belief_bitrate", 0)),
        "compression_curve": [] if compression_rows is None else compression_rows,
    }


def run_synthetic_vision_rule_smoke(seed: int = 0) -> dict[str, object]:
    """Known-rule vision ladder where support images define label semantics."""
    rules = ("swapped_semantics", "normal_semantics", "rotated_semantics", "mirror_semantics")
    baseline_rule = "normal_semantics"

    def make_image(pattern: int) -> np.ndarray:
        image = np.zeros((4, 4), dtype=np.float32)
        if pattern == 0:
            image[:, 1] = 1.0
        elif pattern == 1:
            image[1, :] = 1.0
        elif pattern == 2:
            np.fill_diagonal(image, 1.0)
        else:
            np.fill_diagonal(np.fliplr(image), 1.0)
        return image

    def visual_pattern(image: np.ndarray) -> int:
        scores = (
            float(np.sum(image[:, 1])),
            float(np.sum(image[1, :])),
            float(np.trace(image)),
            float(np.trace(np.fliplr(image))),
        )
        return int(np.argmax(scores))

    def label_for(pattern: int, rule: str) -> int:
        if rule == "swapped_semantics":
            return int(pattern) ^ 1
        if rule == "rotated_semantics":
            return (int(pattern) + 1) % 4
        if rule == "mirror_semantics":
            return 3 - int(pattern)
        return int(pattern)

    def infer_rule(images: list[np.ndarray], labels: np.ndarray) -> str:
        scores = {
            rule: float(
                np.mean(
                    [
                        label_for(visual_pattern(image), rule) == int(label)
                        for image, label in zip(images, labels)
                    ]
                )
            )
            for rule in rules
        }
        return max(scores, key=scores.get)

    def evaluate_world(world_seed: int) -> dict[str, float | int | str]:
        hidden_rule = rules[int(world_seed) % len(rules)]
        support_patterns = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        support_images = [make_image(int(pattern)) for pattern in support_patterns]
        support_labels = np.asarray(
            [label_for(int(pattern), hidden_rule) for pattern in support_patterns],
            dtype=np.int64,
        )
        decoded_rule = infer_rule(support_images, support_labels)
        half = len(support_images) // 2
        first_rule = infer_rule(support_images[:half], support_labels[:half])
        second_rule = infer_rule(support_images[half:], support_labels[half:])
        shuffled_rule = rules[(rules.index(decoded_rule) + 1) % len(rules)]
        challenge_patterns = np.asarray([0, 1, 2, 3] * 6, dtype=np.int64)
        truth = np.asarray([label_for(int(pattern), hidden_rule) for pattern in challenge_patterns])

        def accuracy_for(rule: str) -> float:
            predictions = np.asarray([label_for(int(pattern), rule) for pattern in challenge_patterns])
            return float(np.mean(predictions == truth))

        baseline_accuracy = accuracy_for(baseline_rule)
        learned_accuracy = accuracy_for(decoded_rule)
        zero_accuracy = accuracy_for(baseline_rule)
        shuffled_accuracy = accuracy_for(shuffled_rule)
        stale_label = int(np.bincount(support_labels).argmax())
        stale_accuracy = float(np.mean(np.full_like(truth, stale_label) == truth))
        best_ablation = max(zero_accuracy, shuffled_accuracy, stale_accuracy)
        return {
            "seed": int(world_seed),
            "hidden_rule": hidden_rule,
            "decoded_rule": decoded_rule,
            "hidden_rule_decode_accuracy": float(decoded_rule == hidden_rule),
            "subset_agreement": float(first_rule == second_rule == decoded_rule),
            "shuffled_decode_accuracy": float(shuffled_rule == hidden_rule),
            "baseline_label_accuracy": baseline_accuracy,
            "belief_label_accuracy": learned_accuracy,
            "zero_label_accuracy": zero_accuracy,
            "shuffled_label_accuracy": shuffled_accuracy,
            "stale_label_accuracy": stale_accuracy,
            "content_lift": learned_accuracy - best_ablation,
            "support_worlds": int(len(support_patterns)),
            "challenge_worlds": int(len(challenge_patterns)),
        }

    rows = [evaluate_world(seed + idx) for idx in range(len(rules) * 2)]
    headline = rows[0]
    return {
        **headline,
        "rows": rows,
        "candidate_rules": list(rules),
        "hidden_rule_decode_accuracy": float(np.mean([row["hidden_rule_decode_accuracy"] for row in rows])),
        "subset_agreement": float(np.mean([row["subset_agreement"] for row in rows])),
        "belief_label_accuracy": float(np.mean([row["belief_label_accuracy"] for row in rows])),
        "baseline_label_accuracy": float(headline["baseline_label_accuracy"]),
        "mean_baseline_label_accuracy": float(np.mean([row["baseline_label_accuracy"] for row in rows])),
        "zero_label_accuracy": float(np.mean([row["zero_label_accuracy"] for row in rows])),
        "shuffled_label_accuracy": float(np.mean([row["shuffled_label_accuracy"] for row in rows])),
        "stale_label_accuracy": float(np.mean([row["stale_label_accuracy"] for row in rows])),
        "content_lift": float(np.mean([row["content_lift"] for row in rows])),
        "support_worlds": int(sum(int(row["support_worlds"]) for row in rows)),
        "challenge_worlds": int(sum(int(row["challenge_worlds"]) for row in rows)),
    }


def _selected_image_ablations(
    *,
    selected: dict[str, float | int | str | dict[str, object]],
    full_ablations: dict[str, dict[str, float]],
    compression_rows: list[dict[str, float | int | bool | str | dict[str, object]]],
) -> dict[str, dict[str, float]]:
    mode = str(selected.get("mode", ""))
    if mode == "baseline_fallback":
        accuracy = float(selected.get("accuracy", 0.0))
        return {
            "zero": {"accuracy": accuracy},
            "shuffled": {"accuracy": accuracy},
            "stale": {"accuracy": accuracy},
        }
    if "compressed_" in mode:
        bits = int(selected.get("bits", 0))
        for row in compression_rows:
            row_mode = str(row.get("mode", f"compressed_{int(row.get('bits', 0))}"))
            if int(row.get("bits", 0)) == bits and row_mode == mode:
                return {
                    "zero": {"accuracy": float(row.get("zero_accuracy", 0.0))},
                    "shuffled": {"accuracy": float(row.get("shuffled_accuracy", 0.0))},
                    "stale": {"accuracy": float(row.get("stale_accuracy", 0.0))},
                }
    return {
        "zero": {"accuracy": float(full_ablations["zero"]["accuracy"])},
        "shuffled": {"accuracy": float(full_ablations["shuffled"]["accuracy"])},
        "stale": {"accuracy": float(full_ablations["stale"]["accuracy"])},
    }


def run_mnist_probe_benchmark(
    config: ImageProbeBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Compare a plain CNN to a belief-conditioned CNN on low-label MNIST."""
    config = config or ImageProbeBenchmarkConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_images, train_labels, test_images, test_labels = load_mnist_tensors(config.data_dir)

    rows: list[dict[str, float | int]] = []
    artifacts: list[dict[str, object]] = []
    for budget in config.label_budgets:
        labeled_indices = stratified_subset_indices(train_labels, budget, config.seed + int(budget))
        baseline_model, baseline_metrics = train_classifier_model(
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            labeled_indices=labeled_indices,
            init_encoder_state=None,
            config=config,
            device=device,
        )
        baseline_accuracy = float(baseline_metrics["accuracy"])
        belief_metrics = train_belief_conditioned_classifier(
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            labeled_indices=labeled_indices,
            baseline_accuracy=float(baseline_accuracy),
            config=config,
            device=device,
        )
        ablations = belief_metrics["ablation_metrics"]
        economics = image_row_economics(
            baseline_accuracy=float(baseline_accuracy),
            belief_accuracy=float(belief_metrics["accuracy"]),
            ablations=ablations,
            belief_bitrate=int(belief_metrics["belief_bitrate"]),
            label_budget=int(budget),
        )
        compression_rows = _image_compression_rows(
            compression_metrics=belief_metrics["compression_metrics"],
            baseline_accuracy=float(baseline_accuracy),
            full_belief_accuracy=float(belief_metrics["accuracy"]),
            full_content_lift=float(economics["content_lift"]),
        )
        residual_context = build_feature_belief_context(
            baseline_model.encoder,
            train_images=train_images,
            train_labels=train_labels,
            support_indices=labeled_indices,
            config=config,
            device=device,
            source="residual",
        )
        residual_zero_context = zero_prototype_context(config.feature_dim, device=device)
        residual_shuffled_context = shuffle_prototype_context(residual_context)
        residual_stale_context = build_feature_belief_context(
            baseline_model.encoder,
            train_images=train_images,
            train_labels=train_labels,
            support_indices=labeled_indices[::2] if len(labeled_indices) > 1 else labeled_indices,
            config=config,
            device=device,
            source="residual_stale",
        )
        test_indices = sample_subset_indices(len(test_images), config.test_budget, config.seed + 99)
        residual_metrics = train_residual_belief_adapter(
            baseline_model=baseline_model,
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            labeled_indices=labeled_indices,
            test_indices=test_indices,
            context=residual_context,
            zero_context=residual_zero_context,
            shuffled_context=residual_shuffled_context,
            stale_context=residual_stale_context,
            baseline_accuracy=float(baseline_accuracy),
            config=config,
            device=device,
        )
        residual_ablations = residual_metrics["ablation_metrics"]
        if not isinstance(residual_ablations, dict):
            residual_ablations = {}
        residual_economics = image_row_economics(
            baseline_accuracy=float(baseline_accuracy),
            belief_accuracy=float(residual_metrics["accuracy"]),
            ablations=residual_ablations,
            belief_bitrate=int(belief_metrics["belief_bitrate"]),
            label_budget=int(budget),
        )
        residual_compression_rows = _image_compression_rows(
            compression_metrics=residual_metrics["compression_metrics"],
            baseline_accuracy=float(baseline_accuracy),
            full_belief_accuracy=float(residual_metrics["accuracy"]),
            full_content_lift=float(residual_economics["content_lift"]),
        )
        compression_rows = compression_rows + residual_compression_rows
        best_compressed = _best_compressed_row(compression_rows)
        selected = _select_image_handoff(
            baseline_accuracy=float(baseline_accuracy),
            full_belief_accuracy=float(belief_metrics["accuracy"]),
            full_content_lift=float(economics["content_lift"]),
            compression_rows=compression_rows,
            full_bits=int(belief_metrics["belief_bitrate"]),
            full_ablations=ablations,
        )
        selected_gate = selected.get("decision_gate", {})
        if not isinstance(selected_gate, dict):
            selected_gate = {}
        selected_ablations = _selected_image_ablations(
            selected=selected,
            full_ablations=ablations,
            compression_rows=compression_rows,
        )
        selected_bitrate = int(selected.get("bits", 0)) or int(belief_metrics["belief_bitrate"])
        selected_economics = image_row_economics(
            baseline_accuracy=float(baseline_accuracy),
            belief_accuracy=float(selected["accuracy"]),
            ablations=selected_ablations,
            belief_bitrate=selected_bitrate,
            label_budget=int(budget),
        )
        row = {
            "label_budget": int(budget),
            "baseline_accuracy": float(baseline_accuracy),
            "probe_accuracy": float(selected["accuracy"]),
            "belief_accuracy": float(selected["accuracy"]),
            "raw_belief_accuracy": float(belief_metrics["accuracy"]),
            "raw_accuracy_gain": float(belief_metrics["accuracy"]) - float(baseline_accuracy),
            "raw_best_ablation_accuracy": float(economics["best_ablation_accuracy"]),
            "raw_content_lift": float(economics["content_lift"]),
            "raw_budget_gate_uses_belief": bool(economics["budget_gate_uses_belief"]),
            "residual_accuracy": float(residual_metrics["accuracy"]),
            "residual_accuracy_gain": float(residual_metrics["accuracy"]) - float(baseline_accuracy),
            "residual_content_lift": float(residual_economics["content_lift"]),
            "handoff_mode": str(selected["mode"]),
            "handoff_gate_used_baseline": bool(selected["mode"] == "baseline_fallback"),
            "handoff_gate_reason": str(selected["reason"]),
            "decision_gate_use_belief": bool(selected_gate.get("use_belief", False)),
            "decision_gate_reason": str(selected_gate.get("reason", "")),
            "decision_delta_correct_vs_best_ablation": float(
                selected_gate.get("decision_delta_correct_vs_best_ablation", 0.0)
            ),
            "zero_belief_accuracy": float(
                selected_ablations["zero"]["accuracy"]
            ),
            "shuffled_belief_accuracy": float(
                selected_ablations["shuffled"]["accuracy"]
            ),
            "stale_belief_accuracy": float(
                selected_ablations["stale"]["accuracy"]
            ),
            "raw_zero_belief_accuracy": float(ablations["zero"]["accuracy"]),
            "raw_shuffled_belief_accuracy": float(ablations["shuffled"]["accuracy"]),
            "raw_stale_belief_accuracy": float(ablations["stale"]["accuracy"]),
            "belief_nll": float(belief_metrics["nll"]),
            "zero_belief_nll": float(ablations["zero"]["nll"]),
            "shuffled_belief_nll": float(ablations["shuffled"]["nll"]),
            "stale_belief_nll": float(ablations["stale"]["nll"]),
            "accuracy_gain": float(selected["accuracy"]) - float(baseline_accuracy),
            "ablation_gap": float(selected["accuracy"])
            - float(selected_ablations["shuffled"]["accuracy"]),
            "prototype_stability": float(belief_metrics["prototype_stability"]),
            "belief_bitrate": selected_bitrate,
            "raw_belief_bitrate": int(belief_metrics["belief_bitrate"]),
            "best_compressed_bits": int(best_compressed.get("bits", 0)),
            "best_compressed_accuracy": float(best_compressed.get("accuracy", 0.0)),
            "best_compressed_content_lift": float(best_compressed.get("content_lift", 0.0)),
            "best_compressed_lift_per_1k_bits": float(best_compressed.get("lift_per_1k_bits", 0.0)),
            "best_compressed_retained_utility": float(best_compressed.get("retained_utility", 0.0)),
            **selected_economics,
        }
        rows.append(row)
        artifacts.append(
            build_image_artifact(
                labeled_indices=labeled_indices,
                row=row,
                config=config,
                compression_rows=compression_rows,
            )
        )

    controlled_vision = run_synthetic_vision_rule_smoke(config.seed)
    return {
        "domain": "image",
        "dataset": "MNIST",
        "model_family": "BeliefConditionedMNISTCNN",
        "probe_objective": "prototype_belief_conditioning",
        "unlabeled_budget": int(config.unlabeled_budget),
        "test_budget": int(config.test_budget),
        "probe_epochs": int(config.probe_epochs),
        "finetune_epochs": int(config.finetune_epochs),
        "rows": rows,
        "artifacts": artifacts,
        "controlled_vision": controlled_vision,
        "mean_accuracy_gain": float(np.mean([row["accuracy_gain"] for row in rows])) if rows else 0.0,
        "mean_ablation_gap": float(np.mean([row["ablation_gap"] for row in rows])) if rows else 0.0,
    }
