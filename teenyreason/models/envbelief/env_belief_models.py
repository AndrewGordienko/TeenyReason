"""Model classes for env-level belief aggregation and mechanics decoding."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env_belief_common import (
    MonotonicUncertaintyHead,
    UNCERTAINTY_FEATURE_NAMES,
    safe_normalize,
    sanitize_tensor,
)
from ..factor_belief import FactorizedBeliefUpdater


class EnvBeliefAggregator(nn.Module):
    """Aggregate many window posteriors into one env-level belief."""

    def __init__(
        self,
        window_z_dim: int,
        hidden_dim: int = 128,
        param_dim: int = 5,
        num_families: int = 8,
    ):
        super().__init__()
        self.window_z_dim = window_z_dim
        self.hidden_dim = hidden_dim
        self.factor_count = max(1, min(4, int(window_z_dim)))
        self.param_dim = int(param_dim)
        self.num_families = max(1, int(num_families))
        self.view_net = nn.Sequential(
            nn.Linear(window_z_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.factor_updater = FactorizedBeliefUpdater(
            latent_dim=window_z_dim,
            hidden_dim=hidden_dim,
            factor_count=self.factor_count,
            num_families=self.num_families,
        )
        self.mechanics_updater = MechanicsPosteriorUpdater(
            latent_dim=window_z_dim,
            param_dim=self.param_dim,
            hidden_dim=hidden_dim,
            num_families=self.num_families,
        )
        self.mean_head = nn.Linear(hidden_dim, window_z_dim)
        self.logvar_head = nn.Linear(hidden_dim, window_z_dim)
        self.delta_gate_logit = nn.Parameter(torch.tensor(-1.0))
        self.uncertainty_head = MonotonicUncertaintyHead(len(UNCERTAINTY_FEATURE_NAMES))

    def aggregate_stats(
        self,
        window_mean: torch.Tensor,
        window_logvar: torch.Tensor,
        mask: torch.Tensor,
        family_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate valid views for each env into raw and normalized beliefs."""
        window_mean = sanitize_tensor(window_mean)
        window_logvar = sanitize_tensor(window_logvar)
        normalized_window_mean = safe_normalize(window_mean, dim=-1)
        view_std = sanitize_tensor(torch.exp(0.5 * window_logvar), fill_value=1.0)
        view_features = sanitize_tensor(self.view_net(torch.cat([normalized_window_mean, view_std], dim=-1)))
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        pooled = (view_features * mask_f).sum(dim=1) / denom
        attended = pooled.unsqueeze(1)
        updater_stats = self.factor_updater(
            window_mean=window_mean,
            window_logvar=window_logvar,
            mask=mask,
            family_ids=family_ids,
        )
        mechanics_stats = self.mechanics_updater.update(
            window_mean=window_mean,
            window_logvar=window_logvar,
            mask=mask,
            family_ids=family_ids,
        )
        context = sanitize_tensor(self.fuse(torch.cat([attended.squeeze(1), pooled], dim=-1)))
        residual_delta = sanitize_tensor(self.mean_head(context))
        delta_gate = torch.sigmoid(self.delta_gate_logit)
        env_mean_raw = sanitize_tensor(
            0.55 * updater_stats["mean_raw"]
            + 0.35 * mechanics_stats["latent_context"]
            + delta_gate * residual_delta
        )
        env_mean = safe_normalize(env_mean_raw, dim=-1)
        context_logvar = sanitize_tensor(self.logvar_head(context))
        env_logvar = torch.clamp(
            sanitize_tensor(0.65 * updater_stats["logvar"] + 0.35 * context_logvar),
            -5.0,
            2.0,
        )

        centered = normalized_window_mean - env_mean.unsqueeze(1)
        view_var = (mask_f * centered.pow(2)).sum(dim=1) / denom
        view_spread = torch.sqrt(torch.clamp(view_var, min=1e-4))
        return {
            "env_mean": env_mean,
            "env_mean_raw": env_mean_raw,
            "env_logvar": env_logvar,
            "view_spread": view_spread,
            "factor_mean": sanitize_tensor(updater_stats["factor_mean"]),
            "factor_std": sanitize_tensor(updater_stats["factor_std"]),
            "mechanics_posterior_mean": sanitize_tensor(mechanics_stats["posterior_mean"]),
            "mechanics_posterior_std": sanitize_tensor(mechanics_stats["posterior_std"]),
            "mechanics_posterior_logvar": sanitize_tensor(mechanics_stats["posterior_logvar"]),
            "mechanics_posterior_entropy": sanitize_tensor(mechanics_stats["posterior_entropy"]),
        }

    def forward(
        self,
        window_mean: torch.Tensor,
        window_logvar: torch.Tensor,
        mask: torch.Tensor,
        family_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate valid views for each env into one normalized env belief."""
        stats = self.aggregate_stats(
            window_mean=window_mean,
            window_logvar=window_logvar,
            mask=mask,
            family_ids=family_ids,
        )
        return stats["env_mean"], stats["env_logvar"], stats["view_spread"]

    def predict_uncertainty(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Map belief disagreement features to an env-level uncertainty estimate."""
        feature_tensor = sanitize_tensor(feature_tensor)
        return sanitize_tensor(self.uncertainty_head(feature_tensor))

    def uncertainty_feature_summary(self) -> dict[str, np.ndarray]:
        """Expose learned uncertainty feature importance for diagnostics."""
        weights = self.uncertainty_head.normalized_weights().detach().cpu().numpy().astype(np.float32)
        return {
            "names": np.asarray(UNCERTAINTY_FEATURE_NAMES, dtype="U"),
            "weights": weights,
        }


class EnvParamPredictor(nn.Module):
    """One ensemble head that predicts hidden env parameters from env belief."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, env_mean: torch.Tensor) -> torch.Tensor:
        return self.net(env_mean)


class EnvParamPredictorEnsemble(nn.Module):
    """Ensemble env-parameter decoder used for mechanics uncertainty."""

    def __init__(
        self,
        ensemble_size: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.heads = nn.ModuleList(
            [EnvParamPredictor(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim) for _ in range(ensemble_size)]
        )

    def predict_all(self, env_mean: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(env_mean) for head in self.heads], dim=0)

    def forward(self, env_mean: torch.Tensor) -> torch.Tensor:
        return self.predict_all(env_mean).mean(dim=0)


class MechanicsPosteriorUpdater(nn.Module):
    """Infer a posterior over normalized env parameters from family-labeled evidence."""

    def __init__(
        self,
        latent_dim: int,
        param_dim: int,
        *,
        hidden_dim: int = 128,
        num_families: int = 8,
        family_emb_dim: int = 16,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.param_dim = int(param_dim)
        self.num_families = max(1, int(num_families))
        self.family_emb = nn.Embedding(self.num_families + 1, family_emb_dim)
        self.evidence_net = nn.Sequential(
            nn.Linear(self.latent_dim * 2 + family_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.evidence_mean_head = nn.Linear(hidden_dim, self.param_dim)
        self.evidence_log_precision_head = nn.Linear(hidden_dim, self.param_dim)
        self.family_log_precision_bias = nn.Parameter(torch.zeros(self.num_families, self.param_dim))
        self.prior_mean = nn.Parameter(torch.zeros(self.param_dim))
        self.prior_log_precision = nn.Parameter(torch.zeros(self.param_dim))
        self.posterior_to_latent = nn.Sequential(
            nn.Linear(self.param_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def infer_evidence_stats(
        self,
        window_mean: torch.Tensor,
        window_logvar: torch.Tensor,
        family_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one evidence mean/precision pair for every observed window."""
        batch_size, max_views, _latent_dim = window_mean.shape
        if family_ids is None:
            family_ids = torch.full((batch_size, max_views), -1, dtype=torch.long, device=window_mean.device)
        safe_family = torch.clamp(family_ids.long() + 1, min=0, max=self.num_families)
        family_emb = self.family_emb(safe_family)
        window_std = torch.exp(0.5 * torch.clamp(window_logvar, -5.0, 5.0))
        hidden = self.evidence_net(torch.cat([window_mean, window_std, family_emb], dim=-1))
        evidence_mean = sanitize_tensor(self.evidence_mean_head(hidden))
        evidence_log_precision = sanitize_tensor(self.evidence_log_precision_head(hidden))

        family_bias = torch.zeros_like(evidence_log_precision)
        valid_family = family_ids >= 0
        if torch.any(valid_family):
            family_bias[valid_family] = self.family_log_precision_bias[family_ids[valid_family]]
        evidence_precision = F.softplus(evidence_log_precision + family_bias) + 1e-3
        return evidence_mean, evidence_precision

    def update(
        self,
        window_mean: torch.Tensor,
        window_logvar: torch.Tensor,
        mask: torch.Tensor,
        family_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Combine window-level evidence into one mechanics posterior per env."""
        evidence_mean, evidence_precision = self.infer_evidence_stats(
            window_mean=window_mean,
            window_logvar=window_logvar,
            family_ids=family_ids,
        )
        mask_f = mask.float().unsqueeze(-1)
        prior_precision = F.softplus(self.prior_log_precision).unsqueeze(0) + 1e-3
        posterior_precision = prior_precision + (mask_f * evidence_precision).sum(dim=1)
        posterior_numerator = (
            prior_precision * self.prior_mean.unsqueeze(0)
            + (mask_f * evidence_precision * evidence_mean).sum(dim=1)
        )
        posterior_mean = posterior_numerator / posterior_precision.clamp_min(1e-4)
        posterior_std = torch.rsqrt(posterior_precision.clamp_min(1e-4))
        posterior_logvar = 2.0 * torch.log(posterior_std.clamp_min(1e-4))
        posterior_entropy = 0.5 * torch.sum(
            torch.log(2.0 * math.pi * math.e * torch.exp(posterior_logvar)),
            dim=1,
        )
        latent_context = sanitize_tensor(self.posterior_to_latent(torch.cat([posterior_mean, posterior_std], dim=-1)))
        return {
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "posterior_logvar": posterior_logvar,
            "posterior_entropy": posterior_entropy,
            "latent_context": latent_context,
            "evidence_mean": evidence_mean,
            "evidence_precision": evidence_precision,
        }

    def expected_family_information_gain(
        self,
        posterior_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate entropy reduction from one more probe of each family."""
        posterior_precision = torch.exp(-posterior_logvar).clamp_min(1e-4)
        family_precision = F.softplus(self.family_log_precision_bias).unsqueeze(0) + 1e-3
        updated_precision = posterior_precision.unsqueeze(1) + family_precision
        entropy_drop = 0.5 * torch.sum(
            torch.log(updated_precision / posterior_precision.unsqueeze(1).clamp_min(1e-4)),
            dim=-1,
        )
        return sanitize_tensor(entropy_drop)
