"""Factorized belief-update helpers for env-level inference.

This module keeps the env belief closer to:

- prior over hidden mechanics
- evidence-conditioned posterior updates

instead of only attention-pooling a set of window embeddings.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_factor_slices(total_dim: int, factor_count: int) -> tuple[slice, ...]:
    """Split one latent into a small number of contiguous factor groups."""
    if total_dim <= 0:
        return tuple()
    factor_count = max(1, min(int(factor_count), int(total_dim)))
    base_width = total_dim // factor_count
    remainder = total_dim % factor_count
    slices: list[slice] = []
    start = 0
    for factor_idx in range(factor_count):
        width = base_width + (1 if factor_idx < remainder else 0)
        stop = start + max(1, width)
        slices.append(slice(start, stop))
        start = stop
    return tuple(slices)


def summarize_factor_stats(
    mean_raw: torch.Tensor,
    logvar: torch.Tensor,
    factor_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summarize posterior mean/std per factor chunk."""
    factor_slices = build_factor_slices(int(mean_raw.shape[-1]), factor_count)
    if not factor_slices:
        empty = mean_raw.new_zeros((mean_raw.shape[0], 0))
        return empty, empty

    factor_means = []
    factor_stds = []
    std = torch.exp(0.5 * logvar)
    for factor_slice in factor_slices:
        factor_means.append(mean_raw[..., factor_slice].mean(dim=-1))
        factor_stds.append(std[..., factor_slice].mean(dim=-1))
    return torch.stack(factor_means, dim=1), torch.stack(factor_stds, dim=1)


class FactorizedBeliefUpdater(nn.Module):
    """Sequentially update a small posterior-like belief from family-labeled views."""

    def __init__(
        self,
        latent_dim: int,
        *,
        hidden_dim: int = 128,
        factor_count: int = 4,
        num_families: int = 8,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.factor_count = max(1, min(int(factor_count), int(latent_dim)))
        self.num_families = max(1, int(num_families))
        family_embed_dim = max(8, hidden_dim // 4)
        self.family_embedding = nn.Embedding(self.num_families + 1, family_embed_dim)
        self.prior_mean = nn.Parameter(torch.zeros(latent_dim))
        self.prior_log_precision = nn.Parameter(torch.zeros(latent_dim))
        self.update_net = nn.Sequential(
            nn.Linear(latent_dim * 4 + family_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_shift_head = nn.Linear(hidden_dim, latent_dim)
        self.evidence_gate_head = nn.Linear(hidden_dim, latent_dim)
        self.precision_gain_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        window_mean: torch.Tensor,
        window_logvar: torch.Tensor,
        mask: torch.Tensor,
        family_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Update one posterior-like belief by processing views in sequence."""
        batch_size, max_views, latent_dim = window_mean.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, got {latent_dim}")

        current_mean = self.prior_mean.unsqueeze(0).expand(batch_size, -1)
        current_precision = F.softplus(self.prior_log_precision).unsqueeze(0).expand(batch_size, -1) + 1e-3

        if family_ids is None:
            family_ids = window_mean.new_full((batch_size, max_views), -1, dtype=torch.long)
        family_ids = family_ids.long()

        for view_idx in range(max_views):
            valid = mask[:, view_idx].unsqueeze(-1).float()
            evidence_mean = window_mean[:, view_idx, :]
            evidence_precision = torch.exp(-torch.clamp(window_logvar[:, view_idx, :], -5.0, 5.0)).clamp(1e-3, 25.0)
            safe_family_idx = torch.clamp(family_ids[:, view_idx] + 1, min=0, max=self.num_families)
            family_embed = self.family_embedding(safe_family_idx)

            update_features = torch.cat(
                [
                    current_mean,
                    torch.log(current_precision),
                    evidence_mean,
                    torch.log(evidence_precision),
                    family_embed,
                ],
                dim=-1,
            )
            hidden = self.update_net(update_features)
            mean_shift = 0.25 * torch.tanh(self.mean_shift_head(hidden))
            evidence_gate = torch.sigmoid(self.evidence_gate_head(hidden))
            precision_gain = F.softplus(self.precision_gain_head(hidden))

            proposal_mean = evidence_mean + mean_shift
            proposal_precision = evidence_gate * evidence_precision + 0.10 * precision_gain
            updated_precision = current_precision + valid * proposal_precision
            updated_mean_numerator = current_precision * current_mean + valid * proposal_precision * proposal_mean
            current_mean = updated_mean_numerator / updated_precision.clamp_min(1e-4)
            current_precision = updated_precision.clamp_min(1e-4)

        logvar = -torch.log(current_precision)
        factor_mean, factor_std = summarize_factor_stats(
            mean_raw=current_mean,
            logvar=logvar,
            factor_count=self.factor_count,
        )
        return {
            "mean_raw": current_mean,
            "logvar": logvar,
            "factor_mean": factor_mean,
            "factor_std": factor_std,
        }
