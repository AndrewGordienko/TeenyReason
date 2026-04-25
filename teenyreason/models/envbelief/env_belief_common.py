"""Common helpers for env-level belief aggregation."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sanitize_tensor(values: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Replace non-finite tensor values with a finite fallback."""
    if torch.isfinite(values).all():
        return values
    return torch.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)


def safe_normalize(values: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Normalize vectors with explicit finite-value cleanup."""
    finite_values = sanitize_tensor(values)
    safe_eps = max(float(eps), 1e-3)
    norm = torch.linalg.vector_norm(finite_values, dim=dim, keepdim=True)
    norm = sanitize_tensor(norm, fill_value=safe_eps).clamp_min(safe_eps)
    return sanitize_tensor(finite_values / norm.detach())


def rescale_positive_features(feature_tensor: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Rescale positive disagreement features to preserve useful variation."""
    positive = sanitize_tensor(feature_tensor).clamp_min(0.0)
    feature_mean = positive.mean(dim=0, keepdim=True).detach()
    feature_std = positive.std(dim=0, unbiased=False, keepdim=True).detach()
    feature_scale = torch.clamp(feature_mean + feature_std, min=eps)
    return positive / feature_scale


UNCERTAINTY_FEATURE_NAMES = (
    "mechanics_posterior_std",
    "mechanics_posterior_entropy",
    "decoder_ensemble_std",
    "split_param_disagreement",
    "split_latent_disagreement",
    "split_env_shift",
    "leaveout_param_std",
    "leaveout_shift",
    "view_spread",
    "coverage_penalty",
)

UNCERTAINTY_FEATURE_INIT_WEIGHTS = (
    2.25,
    1.75,
    2.0,
    1.5,
    0.35,
    0.50,
    1.50,
    1.00,
    0.50,
    0.25,
)


def inverse_softplus(value: float) -> float:
    """Map a positive scalar into the pre-softplus domain."""
    safe_value = max(float(value), 1e-6)
    return math.log(math.expm1(safe_value))


class MonotonicUncertaintyHead(nn.Module):
    """Monotone uncertainty map so larger disagreement cannot reduce uncertainty."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.log_feature_scale = nn.Parameter(torch.zeros(input_dim))
        init_weights = torch.tensor(
            [inverse_softplus(value) for value in UNCERTAINTY_FEATURE_INIT_WEIGHTS[:input_dim]],
            dtype=torch.float32,
        )
        self.log_feature_weight = nn.Parameter(init_weights)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        positive_features = rescale_positive_features(feature_tensor).clamp_max(25.0)
        feature_scale = F.softplus(self.log_feature_scale).unsqueeze(0) + 1e-4
        feature_weight = F.softplus(self.log_feature_weight).unsqueeze(0)
        transformed = torch.log1p(positive_features * feature_scale)
        return torch.sum(transformed * feature_weight, dim=-1) + F.softplus(self.bias)

    def normalized_weights(self) -> torch.Tensor:
        feature_weight = F.softplus(self.log_feature_weight)
        total = feature_weight.sum().clamp_min(1e-6)
        return feature_weight / total
