"""Image model modules for the multidomain benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn


class MNISTEncoder(nn.Module):
    """Small CNN backbone shared by baseline and belief-conditioned solvers."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, feature_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class MNISTClassifier(nn.Module):
    """Plain CNN baseline."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder = MNISTEncoder(feature_dim)
        self.head = nn.Linear(feature_dim, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))


class BeliefConditionedMNISTCNN(nn.Module):
    """CNN solver conditioned on class-prototype belief distances."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.encoder = MNISTEncoder(feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim + 12, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 10),
        )

    def context_features(
        self,
        images: torch.Tensor,
        context: dict[str, torch.Tensor | float],
    ) -> torch.Tensor:
        features = self.encoder(images)
        prototypes = context["prototypes"].to(device=images.device, dtype=torch.float32)
        confidence = float(context.get("confidence", 0.0))
        uncertainty = float(context.get("uncertainty", 1.0))
        distances = torch.cdist(features, prototypes) / max(features.shape[-1] ** 0.5, 1.0)
        extras = torch.tensor(
            [confidence, uncertainty],
            dtype=torch.float32,
            device=images.device,
        ).view(1, 2).expand(features.shape[0], -1)
        return torch.cat([features, -distances, extras], dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        context: dict[str, torch.Tensor | float],
    ) -> torch.Tensor:
        return self.head(self.context_features(images, context))


class ResidualBeliefMNISTCNN(nn.Module):
    """Frozen baseline CNN plus a small belief-conditioned residual head."""

    def __init__(self, baseline: MNISTClassifier, feature_dim: int, residual_scale: float = 0.5):
        super().__init__()
        self.baseline = baseline
        for parameter in self.baseline.parameters():
            parameter.requires_grad_(False)
        self.residual_scale = float(residual_scale)
        self.adapter = nn.Sequential(
            nn.Linear(12, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 10),
        )

    def context_features(
        self,
        images: torch.Tensor,
        context: dict[str, torch.Tensor | float],
    ) -> torch.Tensor:
        with torch.no_grad():
            features = self.baseline.encoder(images)
        prototypes = context["prototypes"].to(device=images.device, dtype=torch.float32)
        confidence = float(context.get("confidence", 0.0))
        uncertainty = float(context.get("uncertainty", 1.0))
        distances = torch.cdist(features, prototypes) / max(features.shape[-1] ** 0.5, 1.0)
        extras = torch.tensor(
            [confidence, uncertainty],
            dtype=torch.float32,
            device=images.device,
        ).view(1, 2).expand(features.shape[0], -1)
        return torch.cat([-distances, extras], dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        context: dict[str, torch.Tensor | float],
    ) -> torch.Tensor:
        with torch.no_grad():
            baseline_logits = self.baseline(images)
        residual_logits = self.adapter(self.context_features(images, context))
        return baseline_logits + self.residual_scale * residual_logits


def zero_prototype_context(
    feature_dim: int,
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor | float]:
    """Return an explicit no-belief prototype context."""
    return {
        "prototypes": torch.zeros((10, feature_dim), dtype=torch.float32, device=device),
        "confidence": 0.0,
        "uncertainty": 1.0,
        "counts": torch.zeros((10,), dtype=torch.float32, device=device),
        "source": "zero",
    }


def shuffle_prototype_context(context: dict[str, torch.Tensor | float]) -> dict[str, torch.Tensor | float]:
    """Return a matched shuffled-belief context."""
    return {
        **context,
        "prototypes": torch.roll(context["prototypes"], shifts=1, dims=0),
        "counts": torch.roll(context["counts"], shifts=1, dims=0),
        "source": "shuffled",
    }


def compress_prototype_context(
    context: dict[str, torch.Tensor | float],
    *,
    target_bits: int,
    scalar_bits: int = 8,
) -> dict[str, torch.Tensor | float | int]:
    """Return a fixed-shape prototype context with a measured bit budget."""
    prototypes = context["prototypes"].detach().clone().to(dtype=torch.float32)
    counts = context["counts"].detach().clone().to(dtype=torch.float32)
    feature_dim = int(prototypes.shape[-1])
    available_bits = max(int(target_bits) - 2 * int(scalar_bits), 0)
    dims_to_keep = min(feature_dim, available_bits // max(10 * int(scalar_bits), 1))
    compressed = torch.zeros_like(prototypes)
    if dims_to_keep > 0:
        scores = torch.var(prototypes, dim=0) + torch.mean(torch.abs(prototypes), dim=0)
        keep = torch.topk(scores, k=dims_to_keep).indices
        selected = prototypes.index_select(dim=1, index=keep)
        quantized = _quantize_tensor(selected, levels=2 ** int(scalar_bits))
        compressed.index_copy_(1, keep, quantized)
    return {
        "prototypes": compressed.detach(),
        "confidence": float(context.get("confidence", 0.0)),
        "uncertainty": float(context.get("uncertainty", 1.0)),
        "counts": counts.detach(),
        "source": "compressed",
        "target_bits": int(target_bits),
        "retained_feature_dims": int(dims_to_keep),
        "scalar_bits": int(scalar_bits),
    }


def _quantize_tensor(values: torch.Tensor, *, levels: int) -> torch.Tensor:
    """Uniformly quantize one tensor while preserving its original scale."""
    if values.numel() == 0 or levels <= 1:
        return torch.zeros_like(values)
    min_value = torch.min(values)
    max_value = torch.max(values)
    span = max_value - min_value
    if float(span.item()) <= 1e-8:
        return values.detach().clone()
    scaled = torch.round((values - min_value) / span * float(levels - 1))
    return scaled / float(levels - 1) * span + min_value
