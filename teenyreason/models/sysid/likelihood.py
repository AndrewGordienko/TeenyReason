"""Likelihood model for probe-conditioned system identification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ProbeLikelihoodModel(nn.Module):
    """Predict probe outcomes for candidate normalized environment parameters."""

    def __init__(
        self,
        param_dim: int,
        query_dim: int,
        outcome_dim: int,
        num_families: int,
        hidden_dim: int = 128,
        family_emb_dim: int = 16,
    ):
        super().__init__()
        self.param_dim = int(param_dim)
        self.query_dim = int(query_dim)
        self.outcome_dim = int(outcome_dim)
        self.num_families = max(1, int(num_families))
        self.family_emb = nn.Embedding(self.num_families + 1, int(family_emb_dim))
        input_dim = self.param_dim + self.query_dim + int(family_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, self.outcome_dim)
        self.logvar_head = nn.Linear(hidden_dim, self.outcome_dim)

    def _safe_family_ids(self, family_ids: torch.Tensor) -> torch.Tensor:
        return torch.clamp(family_ids.long() + 1, min=0, max=self.num_families)

    def predict(
        self,
        params_norm: torch.Tensor,
        query_norm: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted normalized outcome mean/log variance."""
        family_emb = self.family_emb(self._safe_family_ids(family_ids))
        hidden = self.net(torch.cat([params_norm, query_norm, family_emb], dim=-1))
        mean = self.mean_head(hidden)
        logvar = torch.clamp(self.logvar_head(hidden), -4.0, 2.0)
        return mean, logvar

    def log_likelihood(
        self,
        params_norm: torch.Tensor,
        query_norm: torch.Tensor,
        family_ids: torch.Tensor,
        outcome_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Return diagonal-Gaussian log likelihood for a normalized outcome."""
        mean, logvar = self.predict(params_norm, query_norm, family_ids)
        var = torch.exp(logvar).clamp_min(1e-5)
        err = outcome_norm - mean
        return -0.5 * torch.sum(err.pow(2) / var + logvar + math.log(2.0 * math.pi), dim=-1)
