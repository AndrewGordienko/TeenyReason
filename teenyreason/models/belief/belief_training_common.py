"""Shared safety helpers for belief-model training."""

from __future__ import annotations

import torch
import torch.nn as nn


def modules_are_finite(modules: list[nn.Module]) -> bool:
    """Return True when all tracked parameters remain finite."""
    for module in modules:
        for param in module.parameters():
            if not torch.isfinite(param).all():
                return False
    return True


def sanitize_modules_(modules: list[nn.Module]) -> None:
    """Repair non-finite parameters in-place so training can continue."""
    with torch.no_grad():
        for module in modules:
            for param in module.parameters():
                if torch.isfinite(param).all():
                    continue
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
                param.data.clamp_(-100.0, 100.0)
