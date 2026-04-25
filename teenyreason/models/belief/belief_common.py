"""Shared helpers for belief-world-model training."""

from __future__ import annotations

import torch


def sanitize_tensor(values: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
    """Replace non-finite tensor entries with a finite fallback."""
    if torch.isfinite(values).all():
        return values
    return torch.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)


def safe_normalize(values: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Normalize vectors without letting tiny norms create numerical junk."""
    finite_values = sanitize_tensor(values)
    safe_eps = max(float(eps), 1e-3)
    norm = torch.linalg.vector_norm(finite_values, dim=dim, keepdim=True)
    norm = sanitize_tensor(norm, fill_value=safe_eps).clamp_min(safe_eps)
    return sanitize_tensor(finite_values / norm.detach())
