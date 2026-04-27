"""Shared probe-training result types.

These dataclasses are used by the probe PPO trainer and the downstream
benchmark/controller code. Keeping them here trims the training file and keeps
the shape of one run result easy to inspect in one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from ..core import RunningNormalizer


@dataclass(frozen=True)
class MatchedEvalSummary:
    """Evaluation-only summary for matched controller ablations on fixed fixtures."""

    returns: list[float]
    episode_total_env_steps: list[int]
    mean_return: float
    mean_total_env_steps: float
    solved_count: int
    fixture_count: int

    def to_dict(self) -> dict[str, object]:
        """Serialize the summary into a stable JSON-friendly mapping."""
        return asdict(self)


@dataclass
class TrainingRunResult:
    """Everything the benchmark/save path needs from one PPO training run."""

    policy: nn.Module
    returns: list[float]
    state_normalizer: RunningNormalizer
    solved_episode: int | None
    solved_env_steps: int | None
    total_env_steps: int
    best_policy_state_dict: dict[str, torch.Tensor]
    best_state_normalizer_state: dict[str, np.ndarray | float]
    best_return: float
    best_episode: int | None
    solve_policy_state_dict: dict[str, torch.Tensor] | None
    solve_state_normalizer_state: dict[str, np.ndarray | float] | None
    solve_eval_returns: list[float] | None
    solve_probe_count: int | None
    probe_env_steps_total: int
    control_env_steps_total: int
    probe_windows_total: int
    probe_stop_reasons: dict[str, int] | None
    probe_family_expected_gain: dict[str, dict[str, float]] | None
    probe_family_realized_gain: dict[str, float] | None
    probe_family_future_error: dict[str, float] | None
    probe_family_selection_count: dict[str, int] | None
    last_probe_stop_reason: str | None
    solve_probe_stop_reason: str | None
    env_expression_eval_returns: list[float] | None = None
    no_env_expression_eval_returns: list[float] | None = None
    env_expression_ablation_delta: float | None = None
    forced_env_expression_eval_returns: list[float] | None = None
    forced_env_expression_ablation_delta: float | None = None
    forced_env_expression_scale: float | None = None
    post_expression_env_steps_total: int | None = None
    post_expression_episode_count: int | None = None
    expression_scale_median: float | None = None
    expression_scale_active_fraction: float | None = None
    fair_ready_handoff_fraction: float | None = None
    fair_expression_enabled_fraction: float | None = None
    fair_expression_force_muted_fraction: float | None = None
    fair_ready_confidence_median: float | None = None
    fair_muted_confidence_median: float | None = None
    fair_stop_blocker_counts: dict[str, int] | None = None
    expression_ready_but_muted_fraction: float | None = None
    readiness_reason_counts: dict[str, int] | None = None
    readiness_component_means: dict[str, float] | None = None
    shadow_expression_enabled_fraction: float | None = None
    shadow_expression_scale_median: float | None = None
    shadow_confidence_median: float | None = None
    shadow_blocker_counts: dict[str, int] | None = None
    shadow_strict_miss_fraction: float | None = None
    second_probe_family_selection_count: dict[str, int] | None = None
    second_probe_raw_future_gain_mean: float | None = None
    second_probe_future_estimate_mean: float | None = None
    second_probe_choice_future_gain_mean: float | None = None
    family_coverage_satisfied_fraction: float | None = None
    second_probe_value_driven_fraction: float | None = None
    uniformity_pressure_active_fraction: float | None = None
    fair_handoff_probe_families: list[str] | None = None
    readiness_component_timeline: list[dict[str, float]] | None = None
    online_future_quality_trace: list[float] | None = None
    online_subset_stability_trace: list[float] | None = None
    online_offline_gap_trace: list[float] | None = None
    online_offline_gap_mean: float | None = None
    online_geometry_complete_fraction: float | None = None
    online_split_latent_disagreement_mean: float | None = None
    online_split_retrieval_margin_deficit_mean: float | None = None
    online_leaveout_shift_mean: float | None = None
    message_input_delta_mean: float | None = None
    message_input_delta_max: float | None = None
    muted_message_input_delta_mean: float | None = None
    muted_message_input_delta_max: float | None = None
    actor_message_norm_mean: float | None = None
    actor_message_nonzero_fraction: float | None = None
    muted_actor_message_nonzero_fraction: float | None = None
    matched_mute_parity_fraction: float | None = None
    message_off_fraction: float | None = None
    message_diag_fraction: float | None = None
    message_on_fraction: float | None = None
    message_ablation_config_diff: dict[str, dict[str, object]] | None = None
    teacher_action_agreement: float | None = None
    controller_style: str = "matched_env_expression"
    zero_context_eval_returns: list[float] | None = None
    shuffled_context_eval_returns: list[float] | None = None
    stale_context_eval_returns: list[float] | None = None
    no_online_refinement_eval_returns: list[float] | None = None
    frozen_context_eval_returns: list[float] | None = None
    zero_context_ablation_delta: float | None = None
    shuffled_context_ablation_delta: float | None = None
    stale_context_ablation_delta: float | None = None
    online_refinement_ablation_delta: float | None = None
    frozen_context_ablation_delta: float | None = None
    actor_only_eval_returns: list[float] | None = None
    actor_only_ablation_delta: float | None = None
    state_only_eval_returns: list[float] | None = None
    state_only_ablation_delta: float | None = None
    learned_eval_summary: MatchedEvalSummary | None = None
    zero_context_eval_summary: MatchedEvalSummary | None = None
    shuffled_context_eval_summary: MatchedEvalSummary | None = None
    stale_context_eval_summary: MatchedEvalSummary | None = None
    no_online_refinement_eval_summary: MatchedEvalSummary | None = None
    frozen_context_eval_summary: MatchedEvalSummary | None = None
    actor_only_eval_summary: MatchedEvalSummary | None = None
    state_only_eval_summary: MatchedEvalSummary | None = None
    state_only_solved_episode: int | None = None
    state_only_solved_env_steps: int | None = None
    state_only_total_env_steps: int | None = None
    state_only_completed_episodes: int | None = None
    planner_trust_usage_rate: float | None = None
    actor_planner_action_divergence: float | None = None
    rollout_model_error_mean: float | None = None
    refresh_count_mean: float | None = None
    oracle_score_agreement_mean: float | None = None
    extra_checkpoint_data: dict | None = None

    @property
    def belief_message_eval_returns(self) -> list[float] | None:
        return self.env_expression_eval_returns

    @property
    def no_message_eval_returns(self) -> list[float] | None:
        return self.no_env_expression_eval_returns

    @property
    def belief_message_ablation_delta(self) -> float | None:
        return self.env_expression_ablation_delta
