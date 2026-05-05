"""Probe-family scoring for crawler bundles."""

from __future__ import annotations

import numpy as np
import torch

from ..types import PredictiveBelief, UncertaintyEstimate
from .helpers import estimate_probe_family_cost, mean_pairwise_distance, sanitize_array


def _is_passive_probe_family(family: str) -> bool:
    """Detect observation-only families without naming a specific environment."""
    label = str(family).lower()
    passive_tokens = ("passive", "observe", "idle", "noop", "no_op")
    return any(token in label for token in passive_tokens)


class ProbeScoringMixin:
    """Score probe families by useful information per interaction cost."""

    def build_mechanics_hypothesis_latents(
        self,
        predictive_belief: PredictiveBelief,
    ) -> np.ndarray:
        """Build a small set of explicit competing world hypotheses in latent space."""
        posterior_mean = sanitize_array(
            predictive_belief.metadata.get("mechanics_posterior_mean", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        posterior_std = sanitize_array(
            predictive_belief.metadata.get("mechanics_posterior_std", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        if posterior_mean.size == 0 or posterior_std.size == 0:
            return sanitize_array(predictive_belief.mean_raw).reshape(1, -1)

        candidate_means = [posterior_mean]
        top_factor_count = min(2, posterior_std.shape[0])
        ranked_factors = np.argsort(-posterior_std)[:top_factor_count]
        for factor_idx in ranked_factors.tolist():
            offset = np.zeros_like(posterior_mean)
            offset[factor_idx] = posterior_std[factor_idx]
            candidate_means.append(posterior_mean + offset)
            candidate_means.append(posterior_mean - offset)

        candidate_mean_t = torch.tensor(
            np.stack(candidate_means, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        candidate_std_t = torch.tensor(
            np.repeat(posterior_std[None, :], candidate_mean_t.shape[0], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            latent_t = self.belief_aggregator.mechanics_updater.posterior_to_latent(
                torch.cat([candidate_mean_t, candidate_std_t], dim=-1)
            )
        return sanitize_array(latent_t.cpu().numpy())

    def family_hypothesis_separation(
        self,
        predictive_belief: PredictiveBelief,
    ) -> dict[str, float]:
        """Estimate which probe families best separate the remaining world hypotheses."""
        if self.env_family_future_predictor is None or not self.family_names:
            return {family: 0.0 for family in self.family_names}

        hypothesis_latents = self.build_mechanics_hypothesis_latents(predictive_belief)
        if hypothesis_latents.shape[0] < 2:
            return {family: 0.0 for family in self.family_names}

        repeated_latents = np.repeat(
            hypothesis_latents[None, :, :],
            repeats=len(self.family_names),
            axis=0,
        )
        family_ids = np.repeat(
            np.arange(len(self.family_names), dtype=np.int64)[:, None],
            repeats=hypothesis_latents.shape[0],
            axis=1,
        )
        with torch.no_grad():
            future_t = self.env_family_future_predictor(
                torch.tensor(repeated_latents.reshape(-1, repeated_latents.shape[-1]), dtype=torch.float32, device=self.device),
                torch.tensor(family_ids.reshape(-1), dtype=torch.long, device=self.device),
            )
        future = sanitize_array(
            future_t.cpu().numpy().reshape(len(self.family_names), hypothesis_latents.shape[0], -1)
        )
        return {
            family: mean_pairwise_distance(future[idx])
            for idx, family in enumerate(self.family_names)
        }

    def score_particle_probe_families(
        self,
        predictive_belief: PredictiveBelief,
        *,
        family_counts: dict[str, int] | None = None,
        global_family_counts: dict[str, int] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Score families by how strongly they separate particle hypotheses."""
        del global_family_counts
        if self.sysid_model is None or self.sysid_stats is None or not self.family_names:
            return {}
        family_counts = family_counts or {}
        particles = sanitize_array(
            predictive_belief.metadata.get("particle_particles_norm", np.zeros((0, 0), dtype=np.float32))
        )
        weights = sanitize_array(
            predictive_belief.metadata.get("particle_weights", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        if particles.ndim != 2 or particles.shape[0] == 0 or weights.shape[0] != particles.shape[0]:
            return {}
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 1e-6:
            weights = np.full((particles.shape[0],), 1.0 / float(particles.shape[0]), dtype=np.float32)
        else:
            weights = (weights / weight_sum).astype(np.float32)

        query_means = sanitize_array(self.sysid_stats.family_query_mean_norm)
        if query_means.ndim != 2 or query_means.shape[0] < len(self.family_names):
            return {}

        self.sysid_model.eval()
        particle_t = torch.tensor(particles, dtype=torch.float32, device=self.device)
        weight_t = torch.tensor(weights, dtype=torch.float32, device=self.device).reshape(-1, 1)
        gains: dict[str, dict[str, float]] = {}
        for family_idx, family in enumerate(self.family_names):
            family_query = query_means[family_idx]
            query_t = torch.tensor(
                np.repeat(family_query[None, :], particles.shape[0], axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            family_t = torch.full((particles.shape[0],), int(family_idx), dtype=torch.long, device=self.device)
            with torch.no_grad():
                mean_t, logvar_t = self.sysid_model.predict(particle_t, query_t, family_t)
                center_t = torch.sum(weight_t * mean_t, dim=0, keepdim=True)
                between_t = torch.sum(weight_t * torch.square(mean_t - center_t), dim=0).mean()
                noise_t = torch.sum(weight_t * torch.exp(logvar_t), dim=0).mean()
            between = float(max(0.0, between_t.item()))
            noise = float(max(1e-4, noise_t.item()))
            info = between / (noise + 1e-4)
            entropy_reduction = float(np.log1p(max(0.0, info)))
            hypothesis_separation = float(np.sqrt(max(0.0, between)))
            cost = estimate_probe_family_cost(family)
            coverage_bonus = 0.08 if int(family_counts.get(family, 0)) <= 0 else 0.0
            subset_stability = float(
                predictive_belief.metadata.get(
                    "particle_subset_stability",
                    predictive_belief.metadata.get("online_subset_stability", 1.0),
                )
            )
            leaveout_shift = float(
                predictive_belief.metadata.get(
                    "particle_leaveout_shift",
                    predictive_belief.metadata.get("online_leaveout_shift", 0.0),
                )
            )
            control_utility_value = entropy_reduction + 0.35 * hypothesis_separation
            stability_confidence = float(
                np.clip(0.55 + 0.45 * subset_stability - 0.20 * leaveout_shift, 0.25, 1.10)
            )
            score = control_utility_value
            stability_adjusted_value = control_utility_value * stability_confidence
            selection_score = stability_adjusted_value + coverage_bonus - 0.35 * cost
            value_per_step = selection_score / max(cost, 1e-6)
            gains[family] = {
                "predicted_mechanics_reduction": entropy_reduction,
                "raw_predicted_future_error_reduction": entropy_reduction,
                "predicted_future_error_reduction": entropy_reduction,
                "future_gain_for_choice": entropy_reduction,
                "predicted_split_reduction": hypothesis_separation,
                "predicted_entropy_reduction": entropy_reduction,
                "predicted_hypothesis_separation": hypothesis_separation,
                "diversity_bonus": 0.0,
                "coverage_bonus": coverage_bonus,
                "quota_bonus": 0.0,
                "repeat_penalty": 0.0,
                "global_repeat_penalty": 0.0,
                "realized_gain_calibration": 1.0,
                "realized_gain_bonus": 0.0,
                "control_utility_value": float(control_utility_value),
                "stability_confidence": float(stability_confidence),
                "stability_adjusted_value": float(stability_adjusted_value),
                "raw_future_error_estimate": float(predictive_belief.future_probe_error),
                "future_error_estimate": float(predictive_belief.future_probe_error),
                "signature_norm": hypothesis_separation,
                "estimated_probe_cost": cost,
                "predicted_marginal_value": selection_score,
                "value_per_probe_step": value_per_step,
                "sample_efficiency_score": value_per_step,
                "score": score,
                "selection_score": selection_score,
            }
        return gains

    def score_probe_families(
        self,
        predictive_belief: PredictiveBelief,
        uncertainty: UncertaintyEstimate,
        *,
        family_counts: dict[str, int] | None = None,
        global_family_counts: dict[str, int] | None = None,
        family_error_history: dict[str, float] | None = None,
        family_realized_gain_history: dict[str, float] | None = None,
        use_learned_family_value: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Score probe families by expected belief improvement rather than novelty alone."""
        del uncertainty
        if str(self.belief_mode) == "particle_sysid" and str(
            predictive_belief.metadata.get("belief_mode", "")
        ) == "particle_sysid":
            particle_scores = self.score_particle_probe_families(
                predictive_belief,
                family_counts=family_counts,
                global_family_counts=global_family_counts,
            )
            if particle_scores:
                return particle_scores

        family_counts = family_counts or {}
        global_family_counts = global_family_counts or {}
        family_error_history = family_error_history or {}
        family_realized_gain_history = family_realized_gain_history or {}
        if not self.family_names:
            return {}
        min_family_count = min((int(family_counts.get(family, 0)) for family in self.family_names), default=0)
        min_global_family_count = min((int(global_family_counts.get(family, 0)) for family in self.family_names), default=0)
        total_global_family_count = sum(max(0, int(global_family_counts.get(family, 0))) for family in self.family_names)
        active_family_count = max(
            1,
            sum(0 if _is_passive_probe_family(family) else 1 for family in self.family_names),
        )
        target_quota_floor = max(
            1,
            total_global_family_count // max(3 * active_family_count, 1),
        )

        mechanics_posterior_std = sanitize_array(
            predictive_belief.metadata.get("mechanics_posterior_std", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        mechanics_posterior_logvar = sanitize_array(
            predictive_belief.metadata.get("mechanics_posterior_logvar", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        mechanics_uncertainty = float(
            np.mean(mechanics_posterior_std)
        ) if mechanics_posterior_std.size else float(np.mean(np.abs(predictive_belief.env_param_std)))
        mechanics_entropy = float(predictive_belief.metadata.get("mechanics_posterior_entropy", 0.0))
        split_uncertainty = float(predictive_belief.metadata.get("split_latent_disagreement", 0.0))
        online_subset_stability = float(
            predictive_belief.metadata.get(
                "particle_subset_stability",
                predictive_belief.metadata.get("online_subset_stability", 1.0),
            )
        )
        online_leaveout_shift = float(
            predictive_belief.metadata.get(
                "particle_leaveout_shift",
                predictive_belief.metadata.get("online_leaveout_shift", 0.0),
            )
        )
        stability_confidence = float(
            np.clip(0.55 + 0.45 * online_subset_stability - 0.20 * online_leaveout_shift, 0.25, 1.10)
        )
        factor_uncertainty = sanitize_array(
            predictive_belief.metadata.get("factor_std", np.zeros((0,), dtype=np.float32))
        ).reshape(-1)
        predictive_mean = sanitize_array(predictive_belief.mean_raw).reshape(1, -1)
        future_signature_norm = {}
        if self.env_family_future_predictor is not None:
            with torch.no_grad():
                family_idx_t = torch.arange(len(self.family_names), dtype=torch.long, device=self.device)
                repeated_belief_t = torch.tensor(
                    np.repeat(predictive_mean, len(self.family_names), axis=0),
                    dtype=torch.float32,
                    device=self.device,
                )
                family_future_t = self.env_family_future_predictor(repeated_belief_t, family_idx_t)
            family_future = sanitize_array(family_future_t.cpu().numpy())
            family_center = family_future.mean(axis=0, keepdims=True)
            family_signature_norm = {
                family: float(np.linalg.norm(family_future[idx] - family_center[0]))
                for idx, family in enumerate(self.family_names)
            }
        else:
            family_signature_norm = {family: 1.0 for family in self.family_names}

        family_hypothesis_separation = self.family_hypothesis_separation(predictive_belief)

        entropy_reduction_by_family = {family: 0.0 for family in self.family_names}
        if mechanics_posterior_logvar.size > 0:
            with torch.no_grad():
                entropy_drop_t = self.belief_aggregator.mechanics_updater.expected_family_information_gain(
                    torch.tensor(
                        mechanics_posterior_logvar[None, :],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
            entropy_drop = sanitize_array(entropy_drop_t.squeeze(0).cpu().numpy())
            for family_idx, family in enumerate(self.family_names[: entropy_drop.shape[0]]):
                entropy_reduction_by_family[family] = float(entropy_drop[family_idx])

        family_value_estimates = {
            family: {
                "mechanics": mechanics_uncertainty,
                "future": predictive_belief.future_probe_error,
                "belief_shift": split_uncertainty,
            }
            for family in self.family_names
        }
        if use_learned_family_value and self.family_value_predictor is not None:
            family_value_context = sanitize_array(
                np.concatenate(
                    [
                        predictive_belief.mean_raw.reshape(-1),
                        np.asarray(
                            [
                                float(np.mean(np.abs(predictive_belief.env_param_std))),
                                float(predictive_belief.future_probe_error),
                                float(predictive_belief.metadata.get("mechanics_posterior_entropy", 0.0)),
                                float(predictive_belief.support_diversity_ratio),
                            ],
                            dtype=np.float32,
                        ),
                    ],
                    axis=0,
                )
            )
            repeated_context = np.repeat(family_value_context[None, :], len(self.family_names), axis=0)
            with torch.no_grad():
                family_value_t = self.family_value_predictor(
                    torch.tensor(repeated_context, dtype=torch.float32, device=self.device),
                    torch.arange(len(self.family_names), dtype=torch.long, device=self.device),
                )
            family_value = sanitize_array(family_value_t.cpu().numpy())
            for family_idx, family in enumerate(self.family_names[: family_value.shape[0]]):
                family_value_estimates[family] = {
                    "mechanics": float(family_value[family_idx, 0]),
                    "future": float(family_value[family_idx, 1]),
                    "belief_shift": float(family_value[family_idx, 2]),
                }

        gains: dict[str, dict[str, float]] = {}
        for family in self.family_names:
            family_count = int(family_counts.get(family, 0))
            global_family_count = int(global_family_counts.get(family, 0))
            novelty_bonus = 1.0 / (1.0 + float(family_count))
            predictive_signature = float(family_signature_norm.get(family, 0.0))
            family_value = family_value_estimates.get(family, {})
            family_factor_gain = float(family_value.get("mechanics", mechanics_uncertainty))
            raw_future_error = float(
                family_error_history.get(
                    family,
                    family_value.get("future", predictive_belief.future_probe_error),
                )
            )
            future_error = raw_future_error
            is_passive_family = _is_passive_probe_family(family)
            is_unseen_active_family = not is_passive_family and family_count <= 0
            if is_unseen_active_family:
                future_error = max(
                    raw_future_error,
                    0.35 * float(predictive_belief.future_probe_error),
                )
            estimated_probe_cost = estimate_probe_family_cost(family)
            hypothesis_separation = float(family_hypothesis_separation.get(family, 0.0))
            realized_gain = float(family_realized_gain_history.get(family, 0.0))
            predicted_mechanics_reduction = family_factor_gain * (0.55 + 0.30 * predictive_signature)
            raw_predicted_future_error_reduction = raw_future_error * (
                0.35 + 0.25 * predictive_signature
            )
            predicted_future_error_reduction = future_error * (
                0.35 + 0.25 * predictive_signature
            )
            future_gain_for_choice = max(
                float(raw_predicted_future_error_reduction),
                0.30 * float(future_error),
            )
            predicted_split_reduction = float(family_value.get("belief_shift", split_uncertainty)) * (
                0.35 + 0.35 * hypothesis_separation
            )
            predicted_entropy_reduction = float(
                entropy_reduction_by_family.get(family, mechanics_entropy * (0.15 + 0.45 * family_factor_gain))
            )
            raw_total_gain = (
                0.22 * predicted_mechanics_reduction
                + 0.18 * predicted_future_error_reduction
                + 0.18 * predicted_split_reduction
                + 0.25 * predicted_entropy_reduction
                + 0.15 * hypothesis_separation
                + 0.02 * novelty_bonus
            )
            control_utility_value = (
                0.34 * predicted_future_error_reduction
                + 0.30 * predicted_mechanics_reduction
                + 0.22 * predicted_split_reduction
                + 0.14 * hypothesis_separation
            )
            if family in family_realized_gain_history:
                realized_gain_calibration = float(
                    np.clip(
                        0.60 + realized_gain / max(raw_total_gain, 0.10),
                        0.55,
                        1.45,
                    )
                )
            else:
                realized_gain_calibration = 1.0
            coverage_bonus = 0.08 * max(0, (min_family_count + 1) - family_count)
            repeat_penalty = 0.06 * max(0, family_count - min_family_count)
            global_repeat_penalty = 0.03 * max(0, global_family_count - min_global_family_count)
            quota_bonus = 0.0
            if not is_passive_family and global_family_count < target_quota_floor:
                undercoverage = float(target_quota_floor - global_family_count)
                utility_per_cost = control_utility_value / max(estimated_probe_cost, 1e-6)
                utility_weight = float(np.clip(0.50 + 0.50 * utility_per_cost, 0.50, 1.25))
                quota_bonus = 0.08 * undercoverage * utility_weight
            realized_gain_bonus = 0.18 * realized_gain
            stability_adjusted_value = raw_total_gain * stability_confidence
            total_gain = stability_adjusted_value * realized_gain_calibration + realized_gain_bonus
            selection_score = total_gain + coverage_bonus + quota_bonus - repeat_penalty - global_repeat_penalty
            cost_adjusted_gain = selection_score - 0.35 * estimated_probe_cost
            value_per_probe_step = cost_adjusted_gain / max(estimated_probe_cost, 1e-6)
            gains[family] = {
                "predicted_mechanics_reduction": float(predicted_mechanics_reduction),
                "raw_predicted_future_error_reduction": float(raw_predicted_future_error_reduction),
                "predicted_future_error_reduction": float(predicted_future_error_reduction),
                "future_gain_for_choice": float(future_gain_for_choice),
                "predicted_split_reduction": float(predicted_split_reduction),
                "predicted_entropy_reduction": float(predicted_entropy_reduction),
                "predicted_hypothesis_separation": float(hypothesis_separation),
                "diversity_bonus": float(novelty_bonus),
                "coverage_bonus": float(coverage_bonus),
                "quota_bonus": float(quota_bonus),
                "repeat_penalty": float(repeat_penalty),
                "global_repeat_penalty": float(global_repeat_penalty),
                "realized_gain_calibration": float(realized_gain_calibration),
                "realized_gain_bonus": float(realized_gain_bonus),
                "control_utility_value": float(control_utility_value),
                "stability_confidence": float(stability_confidence),
                "stability_adjusted_value": float(stability_adjusted_value),
                "raw_future_error_estimate": float(raw_future_error),
                "future_error_estimate": float(future_error),
                "signature_norm": float(predictive_signature),
                "estimated_probe_cost": float(estimated_probe_cost),
                "predicted_marginal_value": float(cost_adjusted_gain),
                "value_per_probe_step": float(value_per_probe_step),
                "sample_efficiency_score": float(value_per_probe_step),
                "score": float(total_gain),
                "selection_score": float(selection_score),
            }
        return gains
