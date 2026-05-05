"""Explicit phase/schedule helpers for env-belief training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class EnvBeliefPhaseConfig:
    """One boring, explicit schedule for the three env-belief training phases."""

    name: str
    predictive_scale: float
    metric_scale: float
    env_expression_scale: float
    controller_scale: float
    uncertainty_scale: float
    grad_clip_norm: float
    metric_gate_active: bool
    metric_gate_reason: str


def build_env_belief_phase_config(
    *,
    epoch_index: int,
    total_epochs: int,
    probe_leakage: float | None = None,
    split_retrieval_top1: float | None = None,
) -> EnvBeliefPhaseConfig:
    """Return the current env-belief phase schedule.

    Phase A: handoff foundation
    Phase B: add metric retrieval/contrastive pressure
    Phase C: restore stronger controller distillation and trust shaping
    """
    progress = float(epoch_index + 1) / float(max(total_epochs, 1))
    phase_a_end = 1.0 / 3.0
    phase_b_end = 2.0 / 3.0
    metric_scale_ceiling = 0.40
    metric_scale_floor = 0.0
    controller_scale_ceiling = 1.0
    env_expression_scale_ceiling = 0.95
    metric_gate_active = False
    metric_gate_reasons: list[str] = []
    leakage_value = None if probe_leakage is None else float(probe_leakage)
    if leakage_value is not None and leakage_value > 0.15:
        metric_gate_active = True
        metric_gate_reasons.append("probe_leakage")
        env_expression_scale_ceiling = min(env_expression_scale_ceiling, 0.88)
        controller_scale_ceiling = min(controller_scale_ceiling, 0.70)
    if leakage_value is not None and leakage_value > 0.25:
        metric_scale_ceiling = min(metric_scale_ceiling, 0.35)
        env_expression_scale_ceiling = min(env_expression_scale_ceiling, 0.78)
        controller_scale_ceiling = min(controller_scale_ceiling, 0.42)
    if leakage_value is not None and leakage_value > 0.40:
        metric_scale_ceiling = min(metric_scale_ceiling, 0.30)
        env_expression_scale_ceiling = min(env_expression_scale_ceiling, 0.70)
        controller_scale_ceiling = min(controller_scale_ceiling, 0.30)

    split_retrieval_value = None if split_retrieval_top1 is None else float(split_retrieval_top1)
    if split_retrieval_value is not None and split_retrieval_value < 0.18:
        metric_gate_active = True
        if "split_retrieval" not in metric_gate_reasons:
            metric_gate_reasons.append("split_retrieval")
        metric_scale_ceiling = max(metric_scale_ceiling, 0.52)
        metric_scale_floor = max(metric_scale_floor, 0.08)
        env_expression_scale_ceiling = min(env_expression_scale_ceiling, 0.82)
        controller_scale_ceiling = min(controller_scale_ceiling, 0.45)
    if split_retrieval_value is not None and split_retrieval_value < 0.10:
        metric_scale_ceiling = max(metric_scale_ceiling, 0.60)
        metric_scale_floor = max(metric_scale_floor, 0.14)
        env_expression_scale_ceiling = min(env_expression_scale_ceiling, 0.72)
        controller_scale_ceiling = min(controller_scale_ceiling, 0.28)
    metric_gate_reason = "+".join(metric_gate_reasons) if metric_gate_reasons else "ok"
    if progress <= phase_a_end:
        return EnvBeliefPhaseConfig(
            name="phase_a_handoff",
            predictive_scale=1.0,
            metric_scale=metric_scale_floor,
            env_expression_scale=min(0.46, env_expression_scale_ceiling),
            controller_scale=min(0.06, controller_scale_ceiling),
            uncertainty_scale=0.30,
            grad_clip_norm=3.0,
            metric_gate_active=metric_gate_active,
            metric_gate_reason=metric_gate_reason,
        )
    if progress <= phase_b_end:
        metric_progress = (progress - phase_a_end) / max(phase_b_end - phase_a_end, 1e-6)
        controller_phase_b_target = min(0.22, controller_scale_ceiling)
        return EnvBeliefPhaseConfig(
            name="phase_b_metric",
            predictive_scale=1.0,
            metric_scale=float(max(metric_scale_floor, metric_scale_ceiling * metric_progress)),
            env_expression_scale=min(0.66, env_expression_scale_ceiling),
            controller_scale=float(
                controller_phase_b_target * (0.35 + 0.65 * metric_progress)
            ),
            uncertainty_scale=0.60,
            grad_clip_norm=0.85,
            metric_gate_active=metric_gate_active,
            metric_gate_reason=metric_gate_reason,
        )
    distill_progress = (progress - phase_b_end) / max(1.0 - phase_b_end, 1e-6)
    controller_phase_c_floor = min(0.18, controller_scale_ceiling)
    return EnvBeliefPhaseConfig(
        name="phase_c_controller",
        predictive_scale=1.0,
        metric_scale=metric_scale_ceiling,
        env_expression_scale=min(
            float(0.78 + 0.17 * distill_progress),
            env_expression_scale_ceiling,
        ),
        controller_scale=float(
            controller_phase_c_floor
            + (controller_scale_ceiling - controller_phase_c_floor) * distill_progress
        ),
        uncertainty_scale=float(0.75 + 0.25 * distill_progress),
        grad_clip_norm=0.80,
        metric_gate_active=metric_gate_active,
        metric_gate_reason=metric_gate_reason,
    )


def build_primary_env_loss_terms(
    *,
    phase: EnvBeliefPhaseConfig,
    physics_loss_weight: float,
    env_consistency_loss_weight: float,
    contrastive_loss_weight: float,
    env_retrieval_loss_weight: float,
    env_param_loss: torch.Tensor,
    mechanics_posterior_loss: torch.Tensor,
    env_split_loss: torch.Tensor,
    env_future_loss: torch.Tensor,
    env_family_future_loss: torch.Tensor,
    env_leaveout_future_loss: torch.Tensor,
    env_leaveout_consistency_loss: torch.Tensor,
    env_split_contrastive_loss: torch.Tensor,
    env_retrieval_loss: torch.Tensor,
    raw_env_retrieval_loss: torch.Tensor,
    env_retrieval_margin_loss: torch.Tensor,
    raw_env_retrieval_margin_loss: torch.Tensor,
    env_gap_ratio_loss: torch.Tensor,
    raw_env_gap_ratio_loss: torch.Tensor,
    env_unit_gap_ratio_loss: torch.Tensor,
    env_unit_retrieval_margin_loss: torch.Tensor,
    env_metric_geometry_loss: torch.Tensor,
    env_spread_loss: torch.Tensor,
    raw_env_spread_loss: torch.Tensor,
    env_uniformity_loss: torch.Tensor,
    raw_env_uniformity_loss: torch.Tensor,
    env_vicreg_loss: torch.Tensor,
    raw_env_vicreg_loss: torch.Tensor,
    env_mode_adversary_loss: torch.Tensor,
    uncertainty_calibration_loss: torch.Tensor,
    env_expression_loss: torch.Tensor,
    controller_mechanics_loss: torch.Tensor,
    controller_affordance_loss: torch.Tensor,
    controller_successor_loss: torch.Tensor,
    oracle_mechanics_loss: torch.Tensor,
    oracle_affordance_loss: torch.Tensor,
    controller_oracle_distill_loss: torch.Tensor,
    controller_trust_loss: torch.Tensor,
    env_gaussian_loss: torch.Tensor,
    representation_repair_mode: bool = False,
) -> dict[str, torch.Tensor]:
    """Assemble the smaller primary objective used for env-belief training."""
    gate_reasons = set(str(phase.metric_gate_reason).split("+"))
    leakage_control_scale = 0.06 + phase.metric_scale * 0.08
    retrieval_pressure = 1.0
    anti_collapse_pressure = 1.0
    if "probe_leakage" in gate_reasons:
        leakage_control_scale += 0.10
    if "split_retrieval" in gate_reasons:
        retrieval_pressure = 2.0
        anti_collapse_pressure = 1.8
    if representation_repair_mode:
        repair_metric_scale = max(float(phase.metric_scale), 0.60)
        repair_retrieval_pressure = max(retrieval_pressure, 2.2)
        repair_anti_collapse_pressure = max(anti_collapse_pressure, 1.8)
        leakage_term = (
            0.14 * env_mode_adversary_loss
            if "probe_leakage" in gate_reasons
            else 0.0 * env_mode_adversary_loss
        )
        return {
            "physics": physics_loss_weight * (env_param_loss + mechanics_posterior_loss),
            "env_consistency": env_consistency_loss_weight * env_split_loss,
            "env_future": phase.predictive_scale * 0.70 * env_future_loss,
            "env_family_future": 0.0 * env_family_future_loss,
            "env_leaveout_future": phase.predictive_scale * 0.45 * env_leaveout_future_loss,
            "env_leaveout_consistency": phase.predictive_scale * 0.35 * env_leaveout_consistency_loss,
            "env_contrastive": (
                repair_metric_scale
                * 0.16
                * repair_retrieval_pressure
                * contrastive_loss_weight
                * env_split_contrastive_loss
            ),
            "env_retrieval": (
                repair_metric_scale
                * 0.34
                * repair_retrieval_pressure
                * env_retrieval_loss_weight
                * env_retrieval_loss
            ),
            "raw_env_retrieval": (
                repair_metric_scale
                * 0.24
                * repair_retrieval_pressure
                * env_retrieval_loss_weight
                * raw_env_retrieval_loss
            ),
            "env_retrieval_margin": repair_metric_scale * 0.20 * repair_retrieval_pressure * env_retrieval_margin_loss,
            "raw_env_retrieval_margin": repair_metric_scale * 0.16 * repair_retrieval_pressure * raw_env_retrieval_margin_loss,
            "env_gap_ratio": repair_metric_scale * 0.18 * repair_retrieval_pressure * env_gap_ratio_loss,
            "raw_env_gap_ratio": repair_metric_scale * 0.14 * repair_retrieval_pressure * raw_env_gap_ratio_loss,
            "env_unit_gap_ratio": repair_metric_scale * 0.22 * repair_retrieval_pressure * env_unit_gap_ratio_loss,
            "env_unit_retrieval_margin": repair_metric_scale * 0.20 * repair_retrieval_pressure * env_unit_retrieval_margin_loss,
            "env_metric_geometry": repair_metric_scale * 0.10 * repair_anti_collapse_pressure * env_metric_geometry_loss,
            "env_spread": repair_metric_scale * 0.18 * repair_anti_collapse_pressure * env_spread_loss,
            "raw_env_spread": repair_metric_scale * 0.12 * repair_anti_collapse_pressure * raw_env_spread_loss,
            "env_uniformity": repair_metric_scale * 0.10 * repair_anti_collapse_pressure * env_uniformity_loss,
            "raw_env_uniformity": repair_metric_scale * 0.08 * repair_anti_collapse_pressure * raw_env_uniformity_loss,
            "env_vicreg": repair_metric_scale * 0.16 * repair_anti_collapse_pressure * env_vicreg_loss,
            "raw_env_vicreg": repair_metric_scale * 0.10 * repair_anti_collapse_pressure * raw_env_vicreg_loss,
            "env_leakage_control": leakage_term,
            "uncertainty_calibration": 0.0 * uncertainty_calibration_loss,
            "env_expression": 0.0 * env_expression_loss,
            "controller_mechanics": 0.0 * controller_mechanics_loss,
            "controller_affordance": 0.0 * controller_affordance_loss,
            "controller_successor": 0.0 * controller_successor_loss,
            "oracle_mechanics": 0.0 * oracle_mechanics_loss,
            "oracle_affordance": 0.0 * oracle_affordance_loss,
            "controller_oracle_distill": 0.0 * controller_oracle_distill_loss,
            "controller_trust": 0.0 * controller_trust_loss,
            "env_gaussian": 0.0 * env_gaussian_loss,
        }
    return {
        "physics": physics_loss_weight * (env_param_loss + 0.85 * mechanics_posterior_loss),
        "env_consistency": env_consistency_loss_weight * env_split_loss,
        "env_future": phase.predictive_scale * 0.62 * env_future_loss,
        "env_family_future": phase.predictive_scale * 0.42 * env_family_future_loss,
        "env_leaveout_future": phase.predictive_scale * 0.40 * env_leaveout_future_loss,
        "env_leaveout_consistency": phase.predictive_scale * 0.20 * env_leaveout_consistency_loss,
        "env_contrastive": (
            phase.metric_scale
            * 0.07
            * retrieval_pressure
            * contrastive_loss_weight
            * env_split_contrastive_loss
        ),
        "env_retrieval": (
            phase.metric_scale
            * 0.18
            * retrieval_pressure
            * env_retrieval_loss_weight
            * env_retrieval_loss
        ),
        "raw_env_retrieval": (
            phase.metric_scale
            * 0.12
            * retrieval_pressure
            * env_retrieval_loss_weight
            * raw_env_retrieval_loss
        ),
        "env_retrieval_margin": phase.metric_scale * 0.12 * retrieval_pressure * env_retrieval_margin_loss,
        "raw_env_retrieval_margin": (
            phase.metric_scale
            * 0.10
            * retrieval_pressure
            * raw_env_retrieval_margin_loss
        ),
        "env_gap_ratio": phase.metric_scale * 0.12 * retrieval_pressure * env_gap_ratio_loss,
        "raw_env_gap_ratio": phase.metric_scale * 0.10 * retrieval_pressure * raw_env_gap_ratio_loss,
        "env_unit_gap_ratio": phase.metric_scale * 0.22 * retrieval_pressure * env_unit_gap_ratio_loss,
        "env_unit_retrieval_margin": (
            phase.metric_scale
            * 0.20
            * retrieval_pressure
            * env_unit_retrieval_margin_loss
        ),
        "env_metric_geometry": (
            phase.metric_scale
            * 0.08
            * anti_collapse_pressure
            * env_metric_geometry_loss
        ),
        "env_spread": phase.metric_scale * 0.16 * anti_collapse_pressure * env_spread_loss,
        "raw_env_spread": phase.metric_scale * 0.12 * anti_collapse_pressure * raw_env_spread_loss,
        "env_uniformity": phase.metric_scale * 0.08 * anti_collapse_pressure * env_uniformity_loss,
        "raw_env_uniformity": phase.metric_scale * 0.06 * anti_collapse_pressure * raw_env_uniformity_loss,
        "env_vicreg": phase.metric_scale * 0.14 * anti_collapse_pressure * env_vicreg_loss,
        "raw_env_vicreg": phase.metric_scale * 0.10 * anti_collapse_pressure * raw_env_vicreg_loss,
        "env_leakage_control": leakage_control_scale * env_mode_adversary_loss,
        "uncertainty_calibration": phase.uncertainty_scale * 0.22 * uncertainty_calibration_loss,
        "env_expression": phase.env_expression_scale * 0.34 * env_expression_loss,
        "controller_mechanics": phase.controller_scale * 0.14 * controller_mechanics_loss,
        "controller_affordance": phase.controller_scale * 0.18 * controller_affordance_loss,
        "controller_successor": phase.controller_scale * 0.16 * controller_successor_loss,
        "oracle_mechanics": phase.controller_scale * 0.10 * oracle_mechanics_loss,
        "oracle_affordance": phase.controller_scale * 0.12 * oracle_affordance_loss,
        "controller_oracle_distill": phase.controller_scale * 0.14 * controller_oracle_distill_loss,
        "controller_trust": phase.controller_scale * 0.10 * controller_trust_loss,
        "env_gaussian": phase.controller_scale * 0.03 * env_gaussian_loss,
    }


def cap_primary_env_loss_terms(
    loss_terms: dict[str, torch.Tensor],
    *,
    max_term_fraction: float = 0.15,
    exempt_names: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    """Scale down any single weighted loss term before it can dominate the update."""
    exempt = {str(name) for name in exempt_names}
    finite_terms = [
        float(torch.abs(value.detach()).item())
        for value in loss_terms.values()
        if torch.isfinite(value)
    ]
    total_magnitude = float(sum(finite_terms))
    if total_magnitude <= 1e-8:
        return loss_terms
    max_fraction = float(np.clip(max_term_fraction, 0.0, 1.0))
    max_allowed = max_fraction * total_magnitude
    capped_terms: dict[str, torch.Tensor] = {}
    for name, value in loss_terms.items():
        if name in exempt:
            capped_terms[name] = value
            continue
        if not torch.isfinite(value):
            capped_terms[name] = value
            continue
        magnitude = float(torch.abs(value.detach()).item())
        if magnitude <= max_allowed or magnitude <= 1e-8:
            capped_terms[name] = value
            continue
        capped_terms[name] = value * float(max_allowed / magnitude)
    return capped_terms
