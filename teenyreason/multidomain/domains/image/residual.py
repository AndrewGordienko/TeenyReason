"""Residual compressed-belief handoff for image solvers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import (
    MNISTClassifier,
    ResidualBeliefMNISTCNN,
    compress_prototype_context,
)


def train_residual_belief_adapter(
    *,
    baseline_model: MNISTClassifier,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    labeled_indices: np.ndarray,
    test_indices: np.ndarray,
    context: dict[str, torch.Tensor | float],
    zero_context: dict[str, torch.Tensor | float],
    shuffled_context: dict[str, torch.Tensor | float],
    stale_context: dict[str, torch.Tensor | float],
    baseline_accuracy: float,
    config: Any,
    device: torch.device,
) -> dict[str, object]:
    """Train a frozen-baseline residual adapter from compressed belief context."""
    model = ResidualBeliefMNISTCNN(
        baseline=baseline_model,
        feature_dim=int(config.feature_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=float(config.lr))
    train_loader = _build_loader(
        images=train_images,
        labels=train_labels,
        indices=labeled_indices,
        batch_size=int(config.batch_size),
        shuffle=True,
    )
    for _epoch in range(max(1, int(config.finetune_epochs))):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, context)
            loss = F.cross_entropy(logits, labels)
            loss = loss + 0.20 * _residual_ablation_penalty(
                model,
                images,
                labels,
                context=context,
                zero_context=zero_context,
                shuffled_context=shuffled_context,
                stale_context=stale_context,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    learned = _evaluate_residual(
        model,
        test_images=test_images,
        test_labels=test_labels,
        test_indices=test_indices,
        context=context,
        batch_size=int(config.batch_size),
        device=device,
    )
    zero = _evaluate_residual(
        model,
        test_images=test_images,
        test_labels=test_labels,
        test_indices=test_indices,
        context=zero_context,
        batch_size=int(config.batch_size),
        device=device,
    )
    shuffled = _evaluate_residual(
        model,
        test_images=test_images,
        test_labels=test_labels,
        test_indices=test_indices,
        context=shuffled_context,
        batch_size=int(config.batch_size),
        device=device,
    )
    stale = _evaluate_residual(
        model,
        test_images=test_images,
        test_labels=test_labels,
        test_indices=test_indices,
        context=stale_context,
        batch_size=int(config.batch_size),
        device=device,
    )
    ablations = {"zero": zero, "shuffled": shuffled, "stale": stale}
    compression = _residual_compression_rows(
        model=model,
        test_images=test_images,
        test_labels=test_labels,
        test_indices=test_indices,
        context=context,
        zero_context=zero_context,
        shuffled_context=shuffled_context,
        stale_context=stale_context,
        baseline_accuracy=float(baseline_accuracy),
        bits=tuple(int(item) for item in config.compression_bits),
        batch_size=int(config.batch_size),
        device=device,
    )
    return {
        "mode": "residual_adapter",
        "accuracy": float(learned["accuracy"]),
        "nll": float(learned["nll"]),
        "ablation_metrics": ablations,
        "compression_metrics": compression,
        "solver_gain": float(learned["accuracy"]) - float(baseline_accuracy),
        "content_lift": float(learned["accuracy"])
        - max(float(zero["accuracy"]), float(shuffled["accuracy"]), float(stale["accuracy"])),
    }


def _residual_ablation_penalty(
    model: ResidualBeliefMNISTCNN,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    context: dict[str, torch.Tensor | float],
    zero_context: dict[str, torch.Tensor | float],
    shuffled_context: dict[str, torch.Tensor | float],
    stale_context: dict[str, torch.Tensor | float],
) -> torch.Tensor:
    learned_loss = F.cross_entropy(model(images, context), labels)
    penalties: list[torch.Tensor] = []
    for ablated in (zero_context, shuffled_context, stale_context):
        ablated_loss = F.cross_entropy(model(images, ablated), labels)
        penalties.append(torch.relu(0.025 + learned_loss - ablated_loss))
    return torch.mean(torch.stack(penalties))


def _residual_compression_rows(
    *,
    model: ResidualBeliefMNISTCNN,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    test_indices: np.ndarray,
    context: dict[str, torch.Tensor | float],
    zero_context: dict[str, torch.Tensor | float],
    shuffled_context: dict[str, torch.Tensor | float],
    stale_context: dict[str, torch.Tensor | float],
    baseline_accuracy: float,
    bits: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float | int | bool | str]]:
    rows: list[dict[str, float | int | bool | str]] = []
    for target_bits in bits:
        compressed = compress_prototype_context(context, target_bits=target_bits)
        compressed_zero = compress_prototype_context(zero_context, target_bits=target_bits)
        compressed_shuffled = compress_prototype_context(shuffled_context, target_bits=target_bits)
        compressed_stale = compress_prototype_context(stale_context, target_bits=target_bits)
        learned = _evaluate_residual(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=compressed,
            batch_size=batch_size,
            device=device,
        )
        zero = _evaluate_residual(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=compressed_zero,
            batch_size=batch_size,
            device=device,
        )
        shuffled = _evaluate_residual(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=compressed_shuffled,
            batch_size=batch_size,
            device=device,
        )
        stale = _evaluate_residual(
            model,
            test_images=test_images,
            test_labels=test_labels,
            test_indices=test_indices,
            context=compressed_stale,
            batch_size=batch_size,
            device=device,
        )
        accuracy = float(learned["accuracy"])
        zero_accuracy = float(zero["accuracy"])
        shuffled_accuracy = float(shuffled["accuracy"])
        stale_accuracy = float(stale["accuracy"])
        content_lift = accuracy - max(zero_accuracy, shuffled_accuracy, stale_accuracy)
        solver_gain = accuracy - float(baseline_accuracy)
        rows.append(
            {
                "bits": int(target_bits),
                "accuracy": accuracy,
                "nll": float(learned["nll"]),
                "zero_accuracy": zero_accuracy,
                "shuffled_accuracy": shuffled_accuracy,
                "stale_accuracy": stale_accuracy,
                "solver_gain": solver_gain,
                "content_lift": content_lift,
                "lift_per_1k_bits": content_lift * 1000.0 / max(float(target_bits), 1.0),
                "gain_per_1k_bits": solver_gain * 1000.0 / max(float(target_bits), 1.0),
                "retained_feature_dims": int(compressed["retained_feature_dims"]),
                "measured": True,
                "source": "residual_adapter",
            }
        )
    return rows


def _evaluate_residual(
    model: ResidualBeliefMNISTCNN,
    *,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    test_indices: np.ndarray,
    context: dict[str, torch.Tensor | float],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    loader = _build_loader(test_images, test_labels, test_indices, batch_size, False)
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
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
    return {
        "accuracy": float(correct) / float(max(total, 1)),
        "nll": float(np.mean(losses)) if losses else 0.0,
    }


def _build_loader(
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    index_t = torch.tensor(indices, dtype=torch.long)
    dataset = TensorDataset(images.index_select(0, index_t), labels.index_select(0, index_t))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
