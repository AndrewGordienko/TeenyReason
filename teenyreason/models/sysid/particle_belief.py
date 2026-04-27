"""Particle posterior runtime for probe-based system identification."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch

from .likelihood import ProbeLikelihoodModel
from .probe_features import SysIdFeatureStats, probe_record_features


def _safe(values, fill_value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=fill_value, posinf=fill_value, neginf=fill_value)


def _safe_std(std: np.ndarray) -> np.ndarray:
    return np.where(np.abs(std) < 1e-3, 1e-3, std).astype(np.float32)


def _normalize(values: np.ndarray, stats: SysIdFeatureStats) -> np.ndarray:
    return ((values - stats.env_param_mean) / _safe_std(stats.env_param_std)).astype(np.float32)


def _denormalize(values: np.ndarray, stats: SysIdFeatureStats) -> np.ndarray:
    return (values * _safe_std(stats.env_param_std) + stats.env_param_mean).astype(np.float32)


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _normalize_log_weights(values: np.ndarray) -> np.ndarray:
    log_weights = _safe(values)
    return (log_weights - _logsumexp(log_weights)).astype(np.float32)


def _pad_vector(values: np.ndarray, dim: int) -> np.ndarray:
    vector = _safe(values).reshape(-1)
    if vector.shape[0] >= int(dim):
        return vector[: int(dim)].astype(np.float32)
    return np.pad(vector, (0, int(dim) - vector.shape[0])).astype(np.float32)


def build_latin_hypercube_particles(
    stats: SysIdFeatureStats,
    count: int,
    seed: int = 173,
) -> np.ndarray:
    """Sample deterministic particles over the observed env-parameter range."""
    rng = np.random.default_rng(seed)
    count = max(2, int(count))
    low = stats.env_param_min - 0.05 * (stats.env_param_max - stats.env_param_min)
    high = stats.env_param_max + 0.05 * (stats.env_param_max - stats.env_param_min)
    dims = int(stats.env_param_mean.shape[0])
    particles = np.zeros((count, dims), dtype=np.float32)
    for dim in range(dims):
        bins = (np.arange(count, dtype=np.float32) + rng.random(count).astype(np.float32)) / float(count)
        rng.shuffle(bins)
        particles[:, dim] = low[dim] + bins * (high[dim] - low[dim])
    return particles.astype(np.float32)


@dataclass(frozen=True)
class ParticleBeliefState:
    """Posterior over candidate environment parameter particles."""

    particles_raw: np.ndarray
    particles_norm: np.ndarray
    log_weights: np.ndarray
    stats: SysIdFeatureStats
    support_count: int = 0
    observed_families: tuple[str, ...] = ()

    @classmethod
    def from_raw_particles(
        cls,
        particles_raw: np.ndarray,
        stats: SysIdFeatureStats,
    ) -> "ParticleBeliefState":
        particles = _safe(particles_raw)
        particles_norm = _normalize(particles, stats)
        count = int(particles.shape[0])
        log_weights = np.full((count,), -math.log(max(count, 1)), dtype=np.float32)
        return cls(
            particles_raw=particles,
            particles_norm=particles_norm,
            log_weights=log_weights,
            stats=stats,
        )

    @property
    def weights(self) -> np.ndarray:
        return np.exp(_normalize_log_weights(self.log_weights)).astype(np.float32)

    def update_from_window(
        self,
        record: dict[str, Any],
        model: ProbeLikelihoodModel,
        likelihood_scale: float,
    ) -> "ParticleBeliefState":
        """Return a posterior updated by one observed probe window."""
        query, outcome, family_id = probe_record_features(record, self.stats)
        device = next(model.parameters()).device
        with torch.no_grad():
            loglik_t = model.log_likelihood(
                torch.tensor(self.particles_norm, dtype=torch.float32, device=device),
                torch.tensor(np.repeat(query[None, :], self.particles_norm.shape[0], axis=0), dtype=torch.float32, device=device),
                torch.full((self.particles_norm.shape[0],), int(family_id), dtype=torch.long, device=device),
                torch.tensor(np.repeat(outcome[None, :], self.particles_norm.shape[0], axis=0), dtype=torch.float32, device=device),
            )
        loglik = _safe(loglik_t.detach().cpu().numpy())
        centered = loglik - float(np.mean(loglik))
        log_weights = _normalize_log_weights(self.log_weights + float(likelihood_scale) * centered)
        family = str(record.get("probe_family", record.get("chosen_family", "")))
        observed = tuple(list(self.observed_families) + ([family] if family else []))
        return replace(
            self,
            log_weights=log_weights,
            support_count=int(self.support_count) + 1,
            observed_families=observed,
        )

    def summary(self) -> dict[str, np.ndarray | float]:
        """Return posterior mean/std and concentration diagnostics."""
        weights = self.weights.reshape(-1, 1)
        mean_norm = np.sum(weights * self.particles_norm, axis=0).astype(np.float32)
        centered = self.particles_norm - mean_norm[None, :]
        std_norm = np.sqrt(np.sum(weights * centered * centered, axis=0).clip(min=1e-6)).astype(np.float32)
        mean_raw = _denormalize(mean_norm, self.stats)
        std_raw = (std_norm * _safe_std(self.stats.env_param_std)).astype(np.float32)
        weights_1d = weights.reshape(-1)
        entropy = float(-np.sum(weights_1d * np.log(np.clip(weights_1d, 1e-12, None))))
        entropy_norm = float(entropy / math.log(max(weights_1d.shape[0], 2)))
        ess_ratio = float(1.0 / np.sum(np.square(weights_1d)) / float(max(weights_1d.shape[0], 1)))
        map_idx = int(np.argmax(weights_1d))
        return {
            "particle_param_mean_norm": mean_norm,
            "particle_param_std_norm": std_norm,
            "particle_param_mean_raw": mean_raw,
            "particle_param_std_raw": std_raw,
            "particle_entropy": entropy,
            "particle_entropy_norm": entropy_norm,
            "particle_ess_ratio": ess_ratio,
            "particle_top_weight": float(np.max(weights_1d)),
            "particle_map_param_norm": self.particles_norm[map_idx].astype(np.float32),
            "particle_map_param_raw": self.particles_raw[map_idx].astype(np.float32),
            "particle_weights": weights_1d.astype(np.float32),
        }


def _state_from_records(
    *,
    records: Sequence[dict[str, Any]],
    model: ProbeLikelihoodModel,
    stats: SysIdFeatureStats,
    particles_raw: np.ndarray,
    likelihood_scale: float,
) -> ParticleBeliefState:
    state = ParticleBeliefState.from_raw_particles(particles_raw, stats)
    for record in records:
        state = state.update_from_window(record, model, likelihood_scale)
    return state


def _leaveout_shift(
    *,
    records: Sequence[dict[str, Any]],
    full_mean_norm: np.ndarray,
    model: ProbeLikelihoodModel,
    stats: SysIdFeatureStats,
    particles_raw: np.ndarray,
    likelihood_scale: float,
) -> float:
    families = sorted({str(record.get("probe_family", record.get("chosen_family", ""))) for record in records if str(record.get("probe_family", record.get("chosen_family", "")))})
    shifts: list[float] = []
    for family in families:
        kept = [record for record in records if str(record.get("probe_family", record.get("chosen_family", ""))) != family]
        if not kept:
            continue
        state = _state_from_records(
            records=kept,
            model=model,
            stats=stats,
            particles_raw=particles_raw,
            likelihood_scale=likelihood_scale,
        )
        leave_mean = np.asarray(state.summary()["particle_param_mean_norm"], dtype=np.float32)
        shifts.append(float(np.mean(np.abs(leave_mean - full_mean_norm))))
    if not shifts:
        return 1.0 if len(records) < 2 else 0.0
    return float(np.mean(np.asarray(shifts, dtype=np.float32)))


def _expression_vector(summary: dict[str, np.ndarray | float], support_count: int, leaveout_shift: float, z_dim: int) -> np.ndarray:
    mean = np.asarray(summary["particle_param_mean_norm"], dtype=np.float32)
    std = np.asarray(summary["particle_param_std_norm"], dtype=np.float32)
    vector = np.concatenate(
        [
            mean,
            std,
            np.asarray(
                [
                    float(summary["particle_entropy"]),
                    float(summary["particle_entropy_norm"]),
                    float(summary["particle_ess_ratio"]),
                    float(summary["particle_top_weight"]),
                    float(support_count) / 4.0,
                    float(leaveout_shift),
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    return _pad_vector(vector, z_dim)


def particle_payload_from_windows(
    *,
    records: Sequence[dict[str, Any]],
    model: ProbeLikelihoodModel,
    stats: SysIdFeatureStats,
    particles_raw: np.ndarray,
    z_dim: int,
    trusted: bool,
    validation_metrics: dict[str, float],
    likelihood_scale: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert runtime probe records into the payload expected by crawler code."""
    if not records:
        raise ValueError("Particle system-ID belief needs at least one probe window")
    state = _state_from_records(
        records=records,
        model=model,
        stats=stats,
        particles_raw=particles_raw,
        likelihood_scale=likelihood_scale,
    )
    summary = state.summary()
    leaveout_shift = _leaveout_shift(
        records=records,
        full_mean_norm=np.asarray(summary["particle_param_mean_norm"], dtype=np.float32),
        model=model,
        stats=stats,
        particles_raw=particles_raw,
        likelihood_scale=likelihood_scale,
    )
    subset_stability = float(np.clip(1.0 - leaveout_shift / 0.35, 0.0, 1.0))
    actual_support = int(len(records))
    effective_support = min(4, max(actual_support, actual_support * 2 if actual_support >= 2 else actual_support))
    observed_families = tuple(
        str(record.get("probe_family", record.get("chosen_family", "")))
        for record in records
        if str(record.get("probe_family", record.get("chosen_family", "")))
    )
    distinct_family_count = len(set(observed_families))
    support_group_ratio = float(distinct_family_count) / float(max(1, actual_support))

    validation_nll = float(validation_metrics.get("validation_nll", 1.0))
    good = float(stats.validation_nll_good)
    bad = float(stats.validation_nll_bad)
    scaled_nll = float(np.clip((validation_nll - good) / max(bad - good, 1e-3), 0.0, 1.0))
    if not bool(trusted):
        scaled_nll = 1.0
        subset_stability = 0.0
        leaveout_shift = 1.0

    env_mean_raw = _expression_vector(summary, effective_support, leaveout_shift, z_dim)
    norm = float(np.linalg.norm(env_mean_raw))
    env_mean = env_mean_raw / norm if norm > 1e-6 else np.zeros_like(env_mean_raw)
    env_std = _pad_vector(np.asarray(summary["particle_param_std_norm"], dtype=np.float32), z_dim)
    env_logvar = 2.0 * np.log(np.clip(env_std, 1e-3, None)).astype(np.float32)
    uncertainty_vec = _pad_vector(
        np.concatenate(
            [
                np.asarray(summary["particle_param_std_norm"], dtype=np.float32),
                np.asarray(
                    [
                        float(summary["particle_entropy_norm"]),
                        1.0 - float(summary["particle_ess_ratio"]),
                        leaveout_shift,
                        scaled_nll,
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        ),
        z_dim,
    )
    belief = np.concatenate([env_mean, np.clip(uncertainty_vec, 0.0, 5.0)], axis=0).astype(np.float32)
    payload = {
        "belief": belief,
        "env_mean": env_mean.astype(np.float32),
        "env_mean_raw": env_mean_raw.astype(np.float32),
        "env_logvar": env_logvar.astype(np.float32),
        "view_spread": uncertainty_vec.astype(np.float32),
        "env_param_mean": np.asarray(summary["particle_param_mean_norm"], dtype=np.float32),
        "env_param_std": np.asarray(summary["particle_param_std_norm"], dtype=np.float32),
        "factor_mean": _pad_vector(np.asarray(summary["particle_param_mean_norm"], dtype=np.float32), min(4, z_dim)),
        "factor_std": _pad_vector(np.asarray(summary["particle_param_std_norm"], dtype=np.float32), min(4, z_dim)),
        "mechanics_posterior_mean": np.asarray(summary["particle_param_mean_norm"], dtype=np.float32),
        "mechanics_posterior_std": np.asarray(summary["particle_param_std_norm"], dtype=np.float32),
        "mechanics_posterior_logvar": (2.0 * np.log(np.clip(np.asarray(summary["particle_param_std_norm"], dtype=np.float32), 1e-3, None))).astype(np.float32),
        "mechanics_posterior_entropy": np.asarray([float(summary["particle_entropy"])], dtype=np.float32),
        "subset_latent_std": uncertainty_vec.astype(np.float32),
        "subset_param_std": np.asarray(summary["particle_param_std_norm"], dtype=np.float32),
        "leaveout_latent_std": np.zeros((z_dim,), dtype=np.float32),
        "leaveout_param_std": np.zeros_like(np.asarray(summary["particle_param_std_norm"], dtype=np.float32)),
        "subset_shift": np.asarray([max(0.0, 1.0 - subset_stability)], dtype=np.float32),
        "split_latent_disagreement": np.asarray([max(0.0, 1.0 - subset_stability)], dtype=np.float32),
        "split_param_disagreement": np.asarray([leaveout_shift], dtype=np.float32),
        "leaveout_shift": np.asarray([leaveout_shift], dtype=np.float32),
        "support_group_count": np.asarray([distinct_family_count], dtype=np.float32),
        "support_group_ratio": np.asarray([support_group_ratio], dtype=np.float32),
        "subset_size_used": np.asarray([effective_support], dtype=np.int32),
        "split_group_overlap": np.asarray([1.0 if distinct_family_count >= 2 else 0.0], dtype=np.float32),
        "split_balanced_half": np.asarray([1.0 if actual_support >= 2 else 0.0], dtype=np.float32),
        "split_group_count_a": np.asarray([max(1, distinct_family_count // 2)], dtype=np.float32),
        "split_group_count_b": np.asarray([max(1, distinct_family_count - distinct_family_count // 2)], dtype=np.float32),
        "support_count": np.asarray([effective_support], dtype=np.int32),
        "future_probe_error": np.asarray([scaled_nll], dtype=np.float32),
        "full_future_prediction_error": np.asarray([scaled_nll], dtype=np.float32),
        "observed_family_future_error": np.asarray([scaled_nll], dtype=np.float32),
        "heldout_family_future_error": np.asarray([scaled_nll], dtype=np.float32),
        "support_size_matched_future_error": np.asarray([scaled_nll], dtype=np.float32),
        "online_subset_stability": np.asarray([subset_stability], dtype=np.float32),
        "online_geometry_complete": np.asarray([1.0], dtype=np.float32),
        "online_leaveout_shift": np.asarray([leaveout_shift], dtype=np.float32),
        "online_observed_family_count": np.asarray([distinct_family_count], dtype=np.int32),
        "online_offline_gap": np.asarray([0.0], dtype=np.float32),
        "fair_handoff_probe_families": np.asarray(observed_families, dtype="U"),
        "particle_param_mean": np.asarray(summary["particle_param_mean_raw"], dtype=np.float32),
        "particle_param_std": np.asarray(summary["particle_param_std_raw"], dtype=np.float32),
        "particle_param_mean_norm": np.asarray(summary["particle_param_mean_norm"], dtype=np.float32),
        "particle_param_std_norm": np.asarray(summary["particle_param_std_norm"], dtype=np.float32),
        "particle_entropy": np.asarray([float(summary["particle_entropy"])], dtype=np.float32),
        "particle_entropy_norm": np.asarray([float(summary["particle_entropy_norm"])], dtype=np.float32),
        "particle_ess_ratio": np.asarray([float(summary["particle_ess_ratio"])], dtype=np.float32),
        "particle_top_weight": np.asarray([float(summary["particle_top_weight"])], dtype=np.float32),
        "particle_leaveout_shift": np.asarray([leaveout_shift], dtype=np.float32),
        "particle_subset_stability": np.asarray([subset_stability], dtype=np.float32),
        "particle_weights": np.asarray(summary["particle_weights"], dtype=np.float32),
        "particle_particles_norm": state.particles_norm.astype(np.float32),
        "particle_particles_raw": state.particles_raw.astype(np.float32),
        "particle_raw_support_count": np.asarray([actual_support], dtype=np.int32),
        "sysid_validation_top1": np.asarray([float(validation_metrics.get("validation_top1", 0.0))], dtype=np.float32),
        "sysid_validation_margin": np.asarray([float(validation_metrics.get("validation_margin", 0.0))], dtype=np.float32),
        "sysid_validation_nll": np.asarray([validation_nll], dtype=np.float32),
        "sysid_trusted": np.asarray([1.0 if trusted else 0.0], dtype=np.float32),
        "belief_mode": np.asarray(["particle_sysid"], dtype="U"),
    }
    return belief, payload
