"""Artifact serialization helpers for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..crawler import CrawlerModelBundle
from ..envs import get_env_display_name


def serialize_normalizer(normalizer) -> dict[str, float | torch.Tensor]:
    """Save enough running-normalizer state to reproduce evaluation preprocessing."""
    return {
        "mean": torch.tensor(normalizer.mean, dtype=torch.float32),
        "var": torch.tensor(normalizer.var, dtype=torch.float32),
        "count": float(normalizer.count),
        "clip": float(normalizer.clip),
    }


def serialize_normalizer_state(normalizer_state: dict[str, np.ndarray | float]) -> dict[str, float | torch.Tensor]:
    """Serialize a cloned normalizer snapshot into a checkpoint-safe payload."""
    return {
        "mean": torch.tensor(normalizer_state["mean"], dtype=torch.float32),
        "var": torch.tensor(normalizer_state["var"], dtype=torch.float32),
        "count": float(normalizer_state["count"]),
        "clip": float(normalizer_state["clip"]),
    }


def artifact_stem_for_controller_style(controller_style: str | None, *, oracle: bool = False) -> str:
    """Map one controller style string to a stable artifact stem."""
    style = "" if controller_style is None else str(controller_style)
    if "sim_fanout" in style:
        stem = "sim_fanout"
    elif "belief_controller" in style:
        stem = "belief_controller"
    else:
        stem = "belief_planner"
    if oracle:
        return f"{stem}_oracle"
    return stem


def artifact_label_for_controller_style(controller_style: str | None, *, oracle: bool = False) -> str:
    """Map one controller style string to a human-readable artifact label."""
    style = "" if controller_style is None else str(controller_style)
    if "sim_fanout" in style:
        label = "sim-fanout"
    elif "belief_controller" in style:
        label = "belief-controller"
    else:
        label = "belief-planner"
    if oracle:
        return f"{label} oracle"
    return label


def benchmark_summary_filename(benchmark_tag: str, benchmark_profile: str) -> str:
    """Keep archived-planner summaries separate from the default benchmark headline."""
    if str(benchmark_profile) == "archived_planner":
        return f"{benchmark_tag}_archived_planner_solve_benchmark.npz"
    return f"{benchmark_tag}_solve_benchmark.npz"


def _json_safe_value(value):
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    return value


def _maybe_save_debug_bundle(
    *,
    output_dir: Path,
    artifact_tag: str,
    artifact_stem: str,
    extra_checkpoint_data: dict | None,
) -> None:
    if not extra_checkpoint_data:
        return
    debug_bundle = extra_checkpoint_data.get("seed0_failure_debug_bundle")
    if debug_bundle is None:
        return
    debug_path = output_dir / f"{artifact_tag}_{artifact_stem}_debug_fixture.json"
    debug_path.write_text(
        json.dumps(_json_safe_value(debug_bundle), indent=2),
        encoding="utf-8",
    )


def save_training_artifacts(
    crawler_bundle: CrawlerModelBundle,
    baseline_result,
    probe_result,
    artifact_tag: str,
    env_name: str,
    benchmark_profile: str,
    benchmark_mode: str,
    probe_budget_mode: str,
    window_size: int,
    z_dim: int,
    action_bins: int,
    hidden_dim: int,
    belief_bits_per_dim: int,
    belief_use_residual_sketch: bool,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    base_probe_episodes: int,
    max_probe_episodes: int,
    probe_adaptive_budget: bool,
    probe_adaptive_policy_schedule: bool,
    randomize_physics: bool,
    solve_eval_episodes: int,
    solved_return: float,
    value_clip_ratio: float | None,
    lr_anneal: bool,
    full_system_result=None,
    full_system_oracle_result=None,
    sim_fanout_result=None,
) -> None:
    """Persist the main outputs from one benchmark configuration."""
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    encoder_path = output_dir / f"{artifact_tag}_encoder_state_dict.pt"
    baseline_returns_path = output_dir / f"{artifact_tag}_baseline_ppo_returns.npy"
    probe_returns_path = output_dir / f"{artifact_tag}_probe_ppo_returns.npy"
    baseline_checkpoint_path = output_dir / f"{artifact_tag}_baseline_ppo_checkpoint.pt"
    probe_checkpoint_path = output_dir / f"{artifact_tag}_probe_ppo_checkpoint.pt"
    full_system_stem = artifact_stem_for_controller_style(
        None if full_system_result is None else full_system_result.controller_style
    )
    full_system_oracle_stem = artifact_stem_for_controller_style(
        None if full_system_oracle_result is None else full_system_oracle_result.controller_style,
        oracle=True,
    )
    sim_fanout_stem = artifact_stem_for_controller_style(
        None if sim_fanout_result is None else sim_fanout_result.controller_style
    )
    full_system_returns_path = output_dir / f"{artifact_tag}_{full_system_stem}_ppo_returns.npy"
    full_system_checkpoint_path = output_dir / f"{artifact_tag}_{full_system_stem}_ppo_checkpoint.pt"
    full_system_oracle_returns_path = output_dir / f"{artifact_tag}_{full_system_oracle_stem}_ppo_returns.npy"
    full_system_oracle_checkpoint_path = output_dir / f"{artifact_tag}_{full_system_oracle_stem}_ppo_checkpoint.pt"
    sim_fanout_returns_path = output_dir / f"{artifact_tag}_{sim_fanout_stem}_ppo_returns.npy"
    sim_fanout_checkpoint_path = output_dir / f"{artifact_tag}_{sim_fanout_stem}_ppo_checkpoint.pt"

    torch.save(crawler_bundle.encoder.state_dict(), encoder_path)
    np.save(baseline_returns_path, np.asarray(baseline_result.returns, dtype=np.float32))
    np.save(probe_returns_path, np.asarray(probe_result.returns, dtype=np.float32))
    torch.save(
        {
            "env_name": env_name,
            "hidden_dim": hidden_dim,
            "belief_dim": int(crawler_bundle.env_expression_dim) + 2,
            "policy_style": "matched_zero_belief",
            "action_bins": action_bins,
            "benchmark_mode": benchmark_mode,
            "benchmark_profile": str(benchmark_profile),
            "probe_budget_mode": probe_budget_mode,
            "value_clip_ratio": value_clip_ratio,
            "lr_anneal": bool(lr_anneal),
            "policy_state_dict": baseline_result.policy.state_dict(),
            "state_normalizer": serialize_normalizer(baseline_result.state_normalizer),
            "best_policy_state_dict": baseline_result.best_policy_state_dict,
            "best_state_normalizer": serialize_normalizer_state(
                baseline_result.best_state_normalizer_state
            ),
            "best_return": float(baseline_result.best_return),
            "best_episode": baseline_result.best_episode,
            "solve_policy_state_dict": baseline_result.solve_policy_state_dict,
            "solve_state_normalizer": None
            if baseline_result.solve_state_normalizer_state is None
            else serialize_normalizer_state(baseline_result.solve_state_normalizer_state),
            "solve_eval_returns": None
            if baseline_result.solve_eval_returns is None
            else torch.tensor(baseline_result.solve_eval_returns, dtype=torch.float32),
            "solved_episode": baseline_result.solved_episode,
            "solved_env_steps": baseline_result.solved_env_steps,
            "total_env_steps": baseline_result.total_env_steps,
            "probe_env_steps_total": baseline_result.probe_env_steps_total,
            "control_env_steps_total": baseline_result.control_env_steps_total,
            "randomize_physics": randomize_physics,
            "solve_eval_episodes": solve_eval_episodes,
            "solved_return": solved_return,
        },
        baseline_checkpoint_path,
    )
    torch.save(
        {
            "env_name": env_name,
            "window_size": window_size,
            "z_dim": z_dim,
            "belief_dim": int(crawler_bundle.env_expression_dim) + 2,
            "belief_style": "env_expression_plus_confidence_and_uncertainty",
            "action_bins": action_bins,
            "hidden_dim": hidden_dim,
            "benchmark_mode": benchmark_mode,
            "benchmark_profile": str(benchmark_profile),
            "probe_budget_mode": probe_budget_mode,
            "belief_bits_per_dim": int(belief_bits_per_dim),
            "belief_use_residual_sketch": bool(belief_use_residual_sketch),
            "value_clip_ratio": value_clip_ratio,
            "lr_anneal": bool(lr_anneal),
            "online_z_update_alpha": online_z_update_alpha,
            "online_z_update_freq": online_z_update_freq,
            "base_probe_episodes": base_probe_episodes,
            "max_probe_episodes": max_probe_episodes,
            "probe_adaptive_budget": probe_adaptive_budget,
            "probe_adaptive_policy_schedule": probe_adaptive_policy_schedule,
            "encoder_state_dict": crawler_bundle.encoder.state_dict(),
            "predictor_state_dict": crawler_bundle.predictor.state_dict(),
            "predictor_ensemble_size": int(crawler_bundle.predictor.ensemble_size),
            "belief_aggregator_state_dict": crawler_bundle.belief_aggregator.state_dict(),
            "env_param_predictor_state_dict": crawler_bundle.env_param_predictor.state_dict(),
            "env_param_predictor_ensemble_size": int(crawler_bundle.env_param_predictor.ensemble_size),
            "env_param_dim": int(crawler_bundle.env_param_predictor.output_dim),
            "env_future_predictor_state_dict": None
            if crawler_bundle.env_future_predictor is None
            else crawler_bundle.env_future_predictor.state_dict(),
            "env_future_summary_dim": 0
            if crawler_bundle.env_future_predictor is None
            else int(crawler_bundle.env_future_predictor.output_dim),
            "env_family_future_predictor_state_dict": None
            if crawler_bundle.env_family_future_predictor is None
            else crawler_bundle.env_family_future_predictor.state_dict(),
            "env_family_count": 0
            if crawler_bundle.env_family_future_predictor is None
            else int(crawler_bundle.env_family_future_predictor.num_families),
            "family_value_predictor_state_dict": None
            if crawler_bundle.family_value_predictor is None
            else crawler_bundle.family_value_predictor.state_dict(),
            "family_value_input_dim": 0
            if crawler_bundle.family_value_predictor is None
            else int(crawler_bundle.family_value_predictor.input_dim),
            "family_value_output_dim": 0
            if crawler_bundle.family_value_predictor is None
            else int(crawler_bundle.family_value_predictor.output_dim),
            "env_metric_projector_state_dict": None
            if crawler_bundle.env_metric_projector is None
            else crawler_bundle.env_metric_projector.state_dict(),
            "env_metric_dim": 0
            if crawler_bundle.env_metric_projector is None
            else int(crawler_bundle.env_metric_projector.net[-1].out_features),
            "belief_message_projector_state_dict": None
            if crawler_bundle.env_expression_projector is None
            else crawler_bundle.env_expression_projector.state_dict(),
            "belief_message_dim": int(crawler_bundle.env_expression_dim),
            "env_expression_dim": int(crawler_bundle.env_expression_dim),
            "controller_context_projector_state_dict": None
            if crawler_bundle.controller_context_projector is None
            else crawler_bundle.controller_context_projector.state_dict(),
            "oracle_context_projector_state_dict": None
            if crawler_bundle.oracle_context_projector is None
            else crawler_bundle.oracle_context_projector.state_dict(),
            "controller_trust_predictor_state_dict": None
            if crawler_bundle.controller_trust_predictor is None
            else crawler_bundle.controller_trust_predictor.state_dict(),
            "controller_context_dim": int(crawler_bundle.full_system_controller_dim),
            "controller_mechanics_dim": int(crawler_bundle.z_dim),
            "controller_affordance_dim": int(crawler_bundle.z_dim),
            "env_param_normalizer_mean": None
            if crawler_bundle.env_param_normalizer_mean is None
            else torch.tensor(crawler_bundle.env_param_normalizer_mean, dtype=torch.float32),
            "env_param_normalizer_std": None
            if crawler_bundle.env_param_normalizer_std is None
            else torch.tensor(crawler_bundle.env_param_normalizer_std, dtype=torch.float32),
            "probe_family_names": np.asarray(crawler_bundle.family_names, dtype="U"),
            "policy_state_dict": probe_result.policy.state_dict(),
            "state_normalizer": serialize_normalizer(probe_result.state_normalizer),
            "best_policy_state_dict": probe_result.best_policy_state_dict,
            "best_state_normalizer": serialize_normalizer_state(
                probe_result.best_state_normalizer_state
            ),
            "best_return": float(probe_result.best_return),
            "best_episode": probe_result.best_episode,
            "solve_policy_state_dict": probe_result.solve_policy_state_dict,
            "solve_state_normalizer": None
            if probe_result.solve_state_normalizer_state is None
            else serialize_normalizer_state(probe_result.solve_state_normalizer_state),
            "solve_eval_returns": None
            if probe_result.solve_eval_returns is None
            else torch.tensor(probe_result.solve_eval_returns, dtype=torch.float32),
            "solve_probe_count": probe_result.solve_probe_count,
            "solved_episode": probe_result.solved_episode,
            "solved_env_steps": probe_result.solved_env_steps,
            "total_env_steps": probe_result.total_env_steps,
            "probe_env_steps_total": probe_result.probe_env_steps_total,
            "control_env_steps_total": probe_result.control_env_steps_total,
            "post_expression_env_steps_total": probe_result.post_expression_env_steps_total,
            "post_expression_episode_count": probe_result.post_expression_episode_count,
            "expression_scale_median": probe_result.expression_scale_median,
            "expression_scale_active_fraction": probe_result.expression_scale_active_fraction,
            "fair_ready_handoff_fraction": probe_result.fair_ready_handoff_fraction,
            "fair_expression_enabled_fraction": probe_result.fair_expression_enabled_fraction,
            "fair_expression_force_muted_fraction": probe_result.fair_expression_force_muted_fraction,
            "fair_ready_confidence_median": probe_result.fair_ready_confidence_median,
            "fair_muted_confidence_median": probe_result.fair_muted_confidence_median,
            "probe_windows_total": probe_result.probe_windows_total,
            "probe_stop_reasons": probe_result.probe_stop_reasons,
            "last_probe_stop_reason": probe_result.last_probe_stop_reason,
            "solve_probe_stop_reason": probe_result.solve_probe_stop_reason,
            "probe_family_expected_gain": probe_result.probe_family_expected_gain,
            "probe_family_realized_gain": probe_result.probe_family_realized_gain,
            "probe_family_future_error": probe_result.probe_family_future_error,
            "probe_family_selection_count": probe_result.probe_family_selection_count,
            "randomize_physics": randomize_physics,
            "solve_eval_episodes": solve_eval_episodes,
            "solved_return": solved_return,
        },
        probe_checkpoint_path,
    )

    print(f"Saved encoder to {encoder_path}")
    print(f"Saved baseline returns to {baseline_returns_path}")
    print(f"Saved probe-conditioned returns to {probe_returns_path}")
    print(f"Saved baseline PPO checkpoint to {baseline_checkpoint_path}")
    print(f"Saved probe-conditioned PPO checkpoint to {probe_checkpoint_path}")
    if full_system_result is not None:
        np.save(full_system_returns_path, np.asarray(full_system_result.returns, dtype=np.float32))
        torch.save(
            {
                "env_name": env_name,
                "window_size": window_size,
                "z_dim": z_dim,
                "belief_dim": int(crawler_bundle.full_system_controller_dim),
                "belief_style": (
                    "controller_context_affordance"
                    if "belief_controller" in str(full_system_result.controller_style)
                    else "planner_controller_context"
                ),
                "policy_style": str(full_system_result.controller_style),
                "action_bins": action_bins,
                "hidden_dim": hidden_dim,
                "benchmark_mode": benchmark_mode,
                "benchmark_profile": str(benchmark_profile),
                "probe_budget_mode": probe_budget_mode,
                "belief_bits_per_dim": int(belief_bits_per_dim),
                "belief_use_residual_sketch": bool(belief_use_residual_sketch),
                "value_clip_ratio": value_clip_ratio,
                "lr_anneal": bool(lr_anneal),
                "online_z_update_alpha": online_z_update_alpha,
                "online_z_update_freq": online_z_update_freq,
                "base_probe_episodes": base_probe_episodes,
                "max_probe_episodes": max_probe_episodes,
                "probe_adaptive_budget": probe_adaptive_budget,
                "probe_adaptive_policy_schedule": probe_adaptive_policy_schedule,
                "encoder_state_dict": crawler_bundle.encoder.state_dict(),
                "predictor_state_dict": crawler_bundle.predictor.state_dict(),
                "predictor_ensemble_size": int(crawler_bundle.predictor.ensemble_size),
                "belief_aggregator_state_dict": crawler_bundle.belief_aggregator.state_dict(),
                "env_param_predictor_state_dict": crawler_bundle.env_param_predictor.state_dict(),
                "env_param_predictor_ensemble_size": int(crawler_bundle.env_param_predictor.ensemble_size),
                "env_param_dim": int(crawler_bundle.env_param_predictor.output_dim),
                "env_future_predictor_state_dict": None
                if crawler_bundle.env_future_predictor is None
                else crawler_bundle.env_future_predictor.state_dict(),
                "env_future_summary_dim": 0
                if crawler_bundle.env_future_predictor is None
                else int(crawler_bundle.env_future_predictor.output_dim),
                "env_family_future_predictor_state_dict": None
                if crawler_bundle.env_family_future_predictor is None
                else crawler_bundle.env_family_future_predictor.state_dict(),
                "env_family_count": 0
                if crawler_bundle.env_family_future_predictor is None
                else int(crawler_bundle.env_family_future_predictor.num_families),
                "family_value_predictor_state_dict": None
                if crawler_bundle.family_value_predictor is None
                else crawler_bundle.family_value_predictor.state_dict(),
                "family_value_input_dim": 0
                if crawler_bundle.family_value_predictor is None
                else int(crawler_bundle.family_value_predictor.input_dim),
                "family_value_output_dim": 0
                if crawler_bundle.family_value_predictor is None
                else int(crawler_bundle.family_value_predictor.output_dim),
                "env_metric_projector_state_dict": None
                if crawler_bundle.env_metric_projector is None
                else crawler_bundle.env_metric_projector.state_dict(),
                "env_metric_dim": 0
                if crawler_bundle.env_metric_projector is None
                else int(crawler_bundle.env_metric_projector.net[-1].out_features),
                "belief_message_projector_state_dict": None
                if crawler_bundle.env_expression_projector is None
                else crawler_bundle.env_expression_projector.state_dict(),
                "belief_message_dim": int(crawler_bundle.env_expression_dim),
                "env_expression_dim": int(crawler_bundle.env_expression_dim),
                "controller_context_projector_state_dict": None
                if crawler_bundle.controller_context_projector is None
                else crawler_bundle.controller_context_projector.state_dict(),
                "oracle_context_projector_state_dict": None
                if crawler_bundle.oracle_context_projector is None
                else crawler_bundle.oracle_context_projector.state_dict(),
                "controller_trust_predictor_state_dict": None
                if crawler_bundle.controller_trust_predictor is None
                else crawler_bundle.controller_trust_predictor.state_dict(),
                "controller_context_dim": int(crawler_bundle.full_system_controller_dim),
                "controller_mechanics_dim": int(crawler_bundle.z_dim),
                "controller_affordance_dim": int(crawler_bundle.z_dim),
                "env_param_normalizer_mean": None
                if crawler_bundle.env_param_normalizer_mean is None
                else torch.tensor(crawler_bundle.env_param_normalizer_mean, dtype=torch.float32),
                "env_param_normalizer_std": None
                if crawler_bundle.env_param_normalizer_std is None
                else torch.tensor(crawler_bundle.env_param_normalizer_std, dtype=torch.float32),
                "probe_family_names": np.asarray(crawler_bundle.family_names, dtype="U"),
                "policy_state_dict": full_system_result.policy.state_dict(),
                "state_normalizer": serialize_normalizer(full_system_result.state_normalizer),
                "best_policy_state_dict": full_system_result.best_policy_state_dict,
                "best_state_normalizer": serialize_normalizer_state(
                    full_system_result.best_state_normalizer_state
                ),
                "best_return": float(full_system_result.best_return),
                "best_episode": full_system_result.best_episode,
                "solve_policy_state_dict": full_system_result.solve_policy_state_dict,
                "solve_state_normalizer": None
                if full_system_result.solve_state_normalizer_state is None
                else serialize_normalizer_state(full_system_result.solve_state_normalizer_state),
                "solve_eval_returns": None
                if full_system_result.solve_eval_returns is None
                else torch.tensor(full_system_result.solve_eval_returns, dtype=torch.float32),
                "zero_context_eval_returns": None
                if full_system_result.zero_context_eval_returns is None
                else torch.tensor(full_system_result.zero_context_eval_returns, dtype=torch.float32),
                "shuffled_context_eval_returns": None
                if full_system_result.shuffled_context_eval_returns is None
                else torch.tensor(full_system_result.shuffled_context_eval_returns, dtype=torch.float32),
                "stale_context_eval_returns": None
                if full_system_result.stale_context_eval_returns is None
                else torch.tensor(full_system_result.stale_context_eval_returns, dtype=torch.float32),
                "no_online_refinement_eval_returns": None
                if full_system_result.no_online_refinement_eval_returns is None
                else torch.tensor(full_system_result.no_online_refinement_eval_returns, dtype=torch.float32),
                "frozen_context_eval_returns": None
                if full_system_result.frozen_context_eval_returns is None
                else torch.tensor(full_system_result.frozen_context_eval_returns, dtype=torch.float32),
                "actor_only_eval_returns": None
                if full_system_result.actor_only_eval_returns is None
                else torch.tensor(full_system_result.actor_only_eval_returns, dtype=torch.float32),
                "state_only_eval_returns": None
                if full_system_result.state_only_eval_returns is None
                else torch.tensor(full_system_result.state_only_eval_returns, dtype=torch.float32),
                "zero_context_ablation_delta": full_system_result.zero_context_ablation_delta,
                "shuffled_context_ablation_delta": full_system_result.shuffled_context_ablation_delta,
                "stale_context_ablation_delta": full_system_result.stale_context_ablation_delta,
                "online_refinement_ablation_delta": full_system_result.online_refinement_ablation_delta,
                "frozen_context_ablation_delta": full_system_result.frozen_context_ablation_delta,
                "actor_only_ablation_delta": full_system_result.actor_only_ablation_delta,
                "state_only_ablation_delta": full_system_result.state_only_ablation_delta,
                "state_only_solved_episode": full_system_result.state_only_solved_episode,
                "state_only_solved_env_steps": full_system_result.state_only_solved_env_steps,
                "state_only_total_env_steps": full_system_result.state_only_total_env_steps,
                "state_only_completed_episodes": full_system_result.state_only_completed_episodes,
                "planner_trust_usage_rate": full_system_result.planner_trust_usage_rate,
                "actor_planner_action_divergence": full_system_result.actor_planner_action_divergence,
                "rollout_model_error_mean": full_system_result.rollout_model_error_mean,
                "refresh_count_mean": full_system_result.refresh_count_mean,
                "oracle_score_agreement_mean": full_system_result.oracle_score_agreement_mean,
                "solve_probe_count": full_system_result.solve_probe_count,
                "solved_episode": full_system_result.solved_episode,
                "solved_env_steps": full_system_result.solved_env_steps,
                "total_env_steps": full_system_result.total_env_steps,
                "probe_env_steps_total": full_system_result.probe_env_steps_total,
                "control_env_steps_total": full_system_result.control_env_steps_total,
                "post_expression_env_steps_total": full_system_result.post_expression_env_steps_total,
                "post_expression_episode_count": full_system_result.post_expression_episode_count,
                "randomize_physics": randomize_physics,
                "solve_eval_episodes": solve_eval_episodes,
                "solved_return": solved_return,
                "extra_checkpoint_data": full_system_result.extra_checkpoint_data,
            },
            full_system_checkpoint_path,
        )
        full_system_label = artifact_label_for_controller_style(full_system_result.controller_style)
        print(f"Saved {full_system_label} returns to {full_system_returns_path}")
        print(f"Saved {full_system_label} checkpoint to {full_system_checkpoint_path}")
        _maybe_save_debug_bundle(
            output_dir=output_dir,
            artifact_tag=artifact_tag,
            artifact_stem=full_system_stem,
            extra_checkpoint_data=full_system_result.extra_checkpoint_data,
        )
    if full_system_oracle_result is not None:
        np.save(
            full_system_oracle_returns_path,
            np.asarray(full_system_oracle_result.returns, dtype=np.float32),
        )
        torch.save(
            {
                "env_name": env_name,
                "window_size": window_size,
                "z_dim": z_dim,
                "belief_dim": int(crawler_bundle.full_system_controller_dim),
                "belief_style": (
                    "controller_context_affordance"
                    if "belief_controller" in str(full_system_oracle_result.controller_style)
                    else "planner_controller_context"
                ),
                "policy_style": str(full_system_oracle_result.controller_style),
                "action_bins": action_bins,
                "hidden_dim": hidden_dim,
                "benchmark_mode": benchmark_mode,
                "benchmark_profile": str(benchmark_profile),
                "probe_budget_mode": probe_budget_mode,
                "belief_bits_per_dim": int(belief_bits_per_dim),
                "belief_use_residual_sketch": bool(belief_use_residual_sketch),
                "value_clip_ratio": value_clip_ratio,
                "lr_anneal": bool(lr_anneal),
                "controller_context_dim": int(crawler_bundle.full_system_controller_dim),
                "policy_state_dict": full_system_oracle_result.policy.state_dict(),
                "state_normalizer": serialize_normalizer(full_system_oracle_result.state_normalizer),
                "best_policy_state_dict": full_system_oracle_result.best_policy_state_dict,
                "best_state_normalizer": serialize_normalizer_state(
                    full_system_oracle_result.best_state_normalizer_state
                ),
                "best_return": float(full_system_oracle_result.best_return),
                "best_episode": full_system_oracle_result.best_episode,
                "solve_policy_state_dict": full_system_oracle_result.solve_policy_state_dict,
                "solve_state_normalizer": None
                if full_system_oracle_result.solve_state_normalizer_state is None
                else serialize_normalizer_state(full_system_oracle_result.solve_state_normalizer_state),
                "solve_eval_returns": None
                if full_system_oracle_result.solve_eval_returns is None
                else torch.tensor(full_system_oracle_result.solve_eval_returns, dtype=torch.float32),
                "zero_context_eval_returns": None
                if full_system_oracle_result.zero_context_eval_returns is None
                else torch.tensor(full_system_oracle_result.zero_context_eval_returns, dtype=torch.float32),
                "shuffled_context_eval_returns": None
                if full_system_oracle_result.shuffled_context_eval_returns is None
                else torch.tensor(full_system_oracle_result.shuffled_context_eval_returns, dtype=torch.float32),
                "stale_context_eval_returns": None
                if full_system_oracle_result.stale_context_eval_returns is None
                else torch.tensor(full_system_oracle_result.stale_context_eval_returns, dtype=torch.float32),
                "no_online_refinement_eval_returns": None
                if full_system_oracle_result.no_online_refinement_eval_returns is None
                else torch.tensor(full_system_oracle_result.no_online_refinement_eval_returns, dtype=torch.float32),
                "frozen_context_eval_returns": None
                if full_system_oracle_result.frozen_context_eval_returns is None
                else torch.tensor(full_system_oracle_result.frozen_context_eval_returns, dtype=torch.float32),
                "actor_only_eval_returns": None
                if full_system_oracle_result.actor_only_eval_returns is None
                else torch.tensor(full_system_oracle_result.actor_only_eval_returns, dtype=torch.float32),
                "state_only_eval_returns": None
                if full_system_oracle_result.state_only_eval_returns is None
                else torch.tensor(full_system_oracle_result.state_only_eval_returns, dtype=torch.float32),
                "zero_context_ablation_delta": full_system_oracle_result.zero_context_ablation_delta,
                "shuffled_context_ablation_delta": full_system_oracle_result.shuffled_context_ablation_delta,
                "stale_context_ablation_delta": full_system_oracle_result.stale_context_ablation_delta,
                "online_refinement_ablation_delta": full_system_oracle_result.online_refinement_ablation_delta,
                "frozen_context_ablation_delta": full_system_oracle_result.frozen_context_ablation_delta,
                "actor_only_ablation_delta": full_system_oracle_result.actor_only_ablation_delta,
                "state_only_ablation_delta": full_system_oracle_result.state_only_ablation_delta,
                "state_only_solved_episode": full_system_oracle_result.state_only_solved_episode,
                "state_only_solved_env_steps": full_system_oracle_result.state_only_solved_env_steps,
                "state_only_total_env_steps": full_system_oracle_result.state_only_total_env_steps,
                "state_only_completed_episodes": full_system_oracle_result.state_only_completed_episodes,
                "planner_trust_usage_rate": full_system_oracle_result.planner_trust_usage_rate,
                "actor_planner_action_divergence": full_system_oracle_result.actor_planner_action_divergence,
                "rollout_model_error_mean": full_system_oracle_result.rollout_model_error_mean,
                "refresh_count_mean": full_system_oracle_result.refresh_count_mean,
                "oracle_score_agreement_mean": full_system_oracle_result.oracle_score_agreement_mean,
                "solve_probe_count": full_system_oracle_result.solve_probe_count,
                "solved_episode": full_system_oracle_result.solved_episode,
                "solved_env_steps": full_system_oracle_result.solved_env_steps,
                "total_env_steps": full_system_oracle_result.total_env_steps,
                "probe_env_steps_total": full_system_oracle_result.probe_env_steps_total,
                "control_env_steps_total": full_system_oracle_result.control_env_steps_total,
                "post_expression_env_steps_total": full_system_oracle_result.post_expression_env_steps_total,
                "post_expression_episode_count": full_system_oracle_result.post_expression_episode_count,
                "randomize_physics": randomize_physics,
                "solve_eval_episodes": solve_eval_episodes,
                "solved_return": solved_return,
                "extra_checkpoint_data": full_system_oracle_result.extra_checkpoint_data,
            },
            full_system_oracle_checkpoint_path,
        )
        oracle_label = artifact_label_for_controller_style(
            full_system_oracle_result.controller_style,
            oracle=True,
        )
        print(f"Saved {oracle_label} returns to {full_system_oracle_returns_path}")
        print(f"Saved {oracle_label} checkpoint to {full_system_oracle_checkpoint_path}")
        _maybe_save_debug_bundle(
            output_dir=output_dir,
            artifact_tag=artifact_tag,
            artifact_stem=full_system_oracle_stem,
            extra_checkpoint_data=full_system_oracle_result.extra_checkpoint_data,
        )
    if sim_fanout_result is not None:
        np.save(sim_fanout_returns_path, np.asarray(sim_fanout_result.returns, dtype=np.float32))
        torch.save(
            {
                "env_name": env_name,
                "window_size": window_size,
                "z_dim": z_dim,
                "belief_dim": int(crawler_bundle.full_system_controller_dim),
                "belief_style": "state_only_sim_fanout",
                "policy_style": str(sim_fanout_result.controller_style),
                "action_bins": action_bins,
                "hidden_dim": hidden_dim,
                "benchmark_mode": benchmark_mode,
                "benchmark_profile": str(benchmark_profile),
                "probe_budget_mode": probe_budget_mode,
                "value_clip_ratio": value_clip_ratio,
                "lr_anneal": bool(lr_anneal),
                "policy_state_dict": sim_fanout_result.policy.state_dict(),
                "state_normalizer": serialize_normalizer(sim_fanout_result.state_normalizer),
                "best_policy_state_dict": sim_fanout_result.best_policy_state_dict,
                "best_state_normalizer": serialize_normalizer_state(
                    sim_fanout_result.best_state_normalizer_state
                ),
                "best_return": float(sim_fanout_result.best_return),
                "best_episode": sim_fanout_result.best_episode,
                "solve_policy_state_dict": sim_fanout_result.solve_policy_state_dict,
                "solve_state_normalizer": None
                if sim_fanout_result.solve_state_normalizer_state is None
                else serialize_normalizer_state(sim_fanout_result.solve_state_normalizer_state),
                "solve_eval_returns": None
                if sim_fanout_result.solve_eval_returns is None
                else torch.tensor(sim_fanout_result.solve_eval_returns, dtype=torch.float32),
                "solved_episode": sim_fanout_result.solved_episode,
                "solved_env_steps": sim_fanout_result.solved_env_steps,
                "total_env_steps": sim_fanout_result.total_env_steps,
                "probe_env_steps_total": sim_fanout_result.probe_env_steps_total,
                "control_env_steps_total": sim_fanout_result.control_env_steps_total,
                "post_expression_env_steps_total": sim_fanout_result.post_expression_env_steps_total,
                "post_expression_episode_count": sim_fanout_result.post_expression_episode_count,
                "solve_eval_episodes": solve_eval_episodes,
                "solved_return": solved_return,
                "extra_checkpoint_data": sim_fanout_result.extra_checkpoint_data,
            },
            sim_fanout_checkpoint_path,
        )
        print(f"Saved sim-fanout returns to {sim_fanout_returns_path}")
        print(f"Saved sim-fanout checkpoint to {sim_fanout_checkpoint_path}")


def save_benchmark_results(
    env_name: str,
    benchmark_tag: str,
    benchmark_profile: str,
    benchmark_mode: str,
    probe_budget_mode: str,
    seeds,
    baseline_episode_solves,
    probe_episode_solves,
    probe_shadow_episode_solves,
    probe_no_expression_episode_solves,
    full_system_episode_solves,
    full_system_state_only_episode_solves,
    full_system_oracle_episode_solves,
    baseline_step_solves,
    probe_step_solves,
    probe_shadow_step_solves,
    probe_no_expression_step_solves,
    full_system_step_solves,
    full_system_state_only_step_solves,
    full_system_oracle_step_solves,
    baseline_total_env_steps,
    probe_total_env_steps,
    probe_shadow_total_env_steps,
    probe_no_expression_total_env_steps,
    full_system_total_env_steps,
    full_system_state_only_total_env_steps,
    full_system_oracle_total_env_steps,
    baseline_control_env_steps,
    probe_probe_env_steps,
    probe_control_env_steps,
    probe_post_expression_env_steps,
    probe_post_expression_episodes,
    probe_shadow_probe_env_steps,
    probe_shadow_control_env_steps,
    probe_shadow_post_expression_env_steps,
    probe_shadow_post_expression_episodes,
    probe_no_expression_probe_env_steps,
    probe_no_expression_control_env_steps,
    probe_no_expression_post_expression_env_steps,
    probe_no_expression_post_expression_episodes,
    full_system_probe_env_steps,
    full_system_control_env_steps,
    full_system_post_context_env_steps,
    full_system_post_context_episodes,
    full_system_oracle_probe_env_steps,
    full_system_oracle_control_env_steps,
    full_system_oracle_post_context_env_steps,
    full_system_oracle_post_context_episodes,
    baseline_completed_episodes,
    probe_completed_episodes,
    probe_shadow_completed_episodes,
    probe_no_expression_completed_episodes,
    full_system_completed_episodes,
    full_system_state_only_completed_episodes,
    full_system_oracle_completed_episodes,
    probe_encoder_steps,
    probe_windows_total,
    probe_expression_scale_median,
    probe_expression_scale_active_fraction,
    probe_fair_ready_handoff_fraction,
    probe_fair_expression_enabled_fraction,
    probe_fair_expression_force_muted_fraction,
    probe_fair_ready_confidence_median,
    probe_fair_muted_confidence_median,
    probe_expression_ready_but_muted_fraction,
    probe_shadow_expression_enabled_fraction,
    probe_shadow_expression_scale_median,
    probe_shadow_confidence_median,
    probe_shadow_strict_miss_fraction,
    probe_run_classification,
    belief_progress_index,
    latent_mechanics_fit,
    latent_split_top1,
    latent_neighbor_alignment,
    latent_gap_ratio,
    latent_heldout_probe_error,
    latent_probe_leakage,
    latent_uncert_error_corr,
    latent_support_diagnostics_json,
    latent_win_gate_json,
    latent_win_gate_failure_reasons_json,
    probe_stop_reasons_json,
    probe_final_stop_reason,
    probe_family_expected_gain_json,
    probe_family_realized_gain_json,
    probe_family_future_error_json,
    probe_family_selection_count_json,
    probe_readiness_reason_counts_json,
    probe_readiness_component_means_json,
    probe_fair_stop_blocker_counts_json,
    probe_shadow_blocker_counts_json,
    probe_second_probe_selection_count_json,
    probe_second_probe_raw_future_gain_mean,
    probe_second_probe_future_estimate_mean,
    probe_second_probe_choice_future_gain_mean,
    probe_family_coverage_satisfied_fraction,
    probe_second_probe_value_driven_fraction,
    probe_uniformity_pressure_active_fraction,
    probe_env_expression_delta,
    probe_forced_env_expression_delta,
    probe_forced_env_expression_scale,
    probe_strict_usage_status,
    probe_fair_handoff_probe_families_json,
    probe_readiness_component_timeline_json,
    probe_message_ablation_config_diff_json,
    probe_online_future_quality_trace_json,
    probe_online_subset_stability_trace_json,
    probe_online_offline_gap_trace_json,
    probe_message_input_delta_mean,
    probe_message_input_delta_max,
    probe_muted_message_input_delta_mean,
    probe_muted_message_input_delta_max,
    probe_actor_message_norm_mean,
    probe_actor_message_nonzero_fraction,
    probe_muted_actor_message_nonzero_fraction,
    probe_matched_mute_parity_fraction,
    probe_online_subset_stability_mean,
    probe_online_offline_gap_mean,
    probe_online_geometry_complete_fraction,
    probe_online_split_latent_disagreement_mean,
    probe_online_split_retrieval_margin_deficit_mean,
    probe_online_leaveout_shift_mean,
    probe_teacher_action_agreement,
    full_system_state_only_eval_returns_json,
    full_system_learned_eval_summary_json,
    full_system_state_only_eval_summary_json,
    full_system_zero_context_eval_summary_json,
    full_system_shuffled_context_eval_summary_json,
    full_system_stale_context_eval_summary_json,
    full_system_online_refinement_eval_summary_json,
    full_system_frozen_context_eval_summary_json,
    full_system_actor_only_eval_summary_json,
    full_system_state_only_ablation_delta,
    full_system_zero_context_ablation_delta,
    full_system_shuffled_context_ablation_delta,
    full_system_stale_context_ablation_delta,
    full_system_online_refinement_ablation_delta,
    full_system_frozen_context_ablation_delta,
    full_system_actor_only_ablation_delta,
    full_system_oracle_learned_eval_summary_json,
    full_system_oracle_zero_context_eval_summary_json,
    full_system_oracle_shuffled_context_eval_summary_json,
    full_system_oracle_stale_context_eval_summary_json,
    full_system_oracle_online_refinement_eval_summary_json,
    full_system_oracle_frozen_context_eval_summary_json,
    full_system_oracle_actor_only_eval_summary_json,
    full_system_oracle_zero_context_ablation_delta,
    full_system_oracle_shuffled_context_ablation_delta,
    full_system_oracle_stale_context_ablation_delta,
    full_system_oracle_online_refinement_ablation_delta,
    full_system_oracle_frozen_context_ablation_delta,
    full_system_oracle_actor_only_ablation_delta,
    sim_fanout_episode_solves,
    sim_fanout_step_solves,
    sim_fanout_total_env_steps,
    sim_fanout_probe_env_steps,
    sim_fanout_control_env_steps,
    sim_fanout_post_context_env_steps,
    sim_fanout_post_context_episodes,
    sim_fanout_completed_episodes,
    full_system_controller_style,
    full_system_oracle_controller_style,
    sim_fanout_controller_style,
) -> None:
    """Save the cross-seed solve summary in one compact file."""
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    benchmark_path = output_dir / benchmark_summary_filename(benchmark_tag, benchmark_profile)
    np.savez(
        benchmark_path,
        env_name=np.asarray(env_name),
        benchmark_profile=np.asarray(benchmark_profile),
        benchmark_mode=np.asarray(benchmark_mode),
        probe_budget_mode=np.asarray(probe_budget_mode),
        seeds=np.asarray(seeds, dtype=np.int64),
        baseline_solves=np.asarray(baseline_episode_solves, dtype=np.int64),
        probe_solves=np.asarray(probe_episode_solves, dtype=np.int64),
        probe_shadow_solves=np.asarray(probe_shadow_episode_solves, dtype=np.int64),
        probe_no_expression_solves=np.asarray(probe_no_expression_episode_solves, dtype=np.int64),
        baseline_episode_solves=np.asarray(baseline_episode_solves, dtype=np.int64),
        probe_episode_solves=np.asarray(probe_episode_solves, dtype=np.int64),
        probe_shadow_episode_solves=np.asarray(probe_shadow_episode_solves, dtype=np.int64),
        probe_no_expression_episode_solves=np.asarray(probe_no_expression_episode_solves, dtype=np.int64),
        full_system_episode_solves=np.asarray(full_system_episode_solves, dtype=np.int64),
        full_system_state_only_episode_solves=np.asarray(full_system_state_only_episode_solves, dtype=np.int64),
        full_system_oracle_episode_solves=np.asarray(full_system_oracle_episode_solves, dtype=np.int64),
        sim_fanout_episode_solves=np.asarray(sim_fanout_episode_solves, dtype=np.int64),
        baseline_step_solves=np.asarray(baseline_step_solves, dtype=np.int64),
        probe_step_solves=np.asarray(probe_step_solves, dtype=np.int64),
        probe_shadow_step_solves=np.asarray(probe_shadow_step_solves, dtype=np.int64),
        probe_no_expression_step_solves=np.asarray(probe_no_expression_step_solves, dtype=np.int64),
        full_system_step_solves=np.asarray(full_system_step_solves, dtype=np.int64),
        full_system_state_only_step_solves=np.asarray(full_system_state_only_step_solves, dtype=np.int64),
        full_system_oracle_step_solves=np.asarray(full_system_oracle_step_solves, dtype=np.int64),
        sim_fanout_step_solves=np.asarray(sim_fanout_step_solves, dtype=np.int64),
        baseline_total_env_steps=np.asarray(baseline_total_env_steps, dtype=np.int64),
        probe_total_env_steps=np.asarray(probe_total_env_steps, dtype=np.int64),
        probe_shadow_total_env_steps=np.asarray(probe_shadow_total_env_steps, dtype=np.int64),
        probe_no_expression_total_env_steps=np.asarray(probe_no_expression_total_env_steps, dtype=np.int64),
        full_system_total_env_steps=np.asarray(full_system_total_env_steps, dtype=np.int64),
        full_system_state_only_total_env_steps=np.asarray(full_system_state_only_total_env_steps, dtype=np.int64),
        full_system_oracle_total_env_steps=np.asarray(full_system_oracle_total_env_steps, dtype=np.int64),
        sim_fanout_total_env_steps=np.asarray(sim_fanout_total_env_steps, dtype=np.int64),
        baseline_control_env_steps=np.asarray(baseline_control_env_steps, dtype=np.int64),
        probe_probe_env_steps=np.asarray(probe_probe_env_steps, dtype=np.int64),
        probe_control_env_steps=np.asarray(probe_control_env_steps, dtype=np.int64),
        probe_post_expression_env_steps=np.asarray(probe_post_expression_env_steps, dtype=np.int64),
        probe_post_expression_episodes=np.asarray(probe_post_expression_episodes, dtype=np.int64),
        probe_shadow_probe_env_steps=np.asarray(probe_shadow_probe_env_steps, dtype=np.int64),
        probe_shadow_control_env_steps=np.asarray(probe_shadow_control_env_steps, dtype=np.int64),
        probe_shadow_post_expression_env_steps=np.asarray(probe_shadow_post_expression_env_steps, dtype=np.int64),
        probe_shadow_post_expression_episodes=np.asarray(probe_shadow_post_expression_episodes, dtype=np.int64),
        probe_no_expression_probe_env_steps=np.asarray(probe_no_expression_probe_env_steps, dtype=np.int64),
        probe_no_expression_control_env_steps=np.asarray(probe_no_expression_control_env_steps, dtype=np.int64),
        probe_no_expression_post_expression_env_steps=np.asarray(probe_no_expression_post_expression_env_steps, dtype=np.int64),
        probe_no_expression_post_expression_episodes=np.asarray(probe_no_expression_post_expression_episodes, dtype=np.int64),
        full_system_probe_env_steps=np.asarray(full_system_probe_env_steps, dtype=np.int64),
        full_system_control_env_steps=np.asarray(full_system_control_env_steps, dtype=np.int64),
        full_system_post_context_env_steps=np.asarray(full_system_post_context_env_steps, dtype=np.int64),
        full_system_post_context_episodes=np.asarray(full_system_post_context_episodes, dtype=np.int64),
        full_system_oracle_probe_env_steps=np.asarray(full_system_oracle_probe_env_steps, dtype=np.int64),
        full_system_oracle_control_env_steps=np.asarray(full_system_oracle_control_env_steps, dtype=np.int64),
        full_system_oracle_post_context_env_steps=np.asarray(full_system_oracle_post_context_env_steps, dtype=np.int64),
        full_system_oracle_post_context_episodes=np.asarray(full_system_oracle_post_context_episodes, dtype=np.int64),
        sim_fanout_probe_env_steps=np.asarray(sim_fanout_probe_env_steps, dtype=np.int64),
        sim_fanout_control_env_steps=np.asarray(sim_fanout_control_env_steps, dtype=np.int64),
        sim_fanout_post_context_env_steps=np.asarray(sim_fanout_post_context_env_steps, dtype=np.int64),
        sim_fanout_post_context_episodes=np.asarray(sim_fanout_post_context_episodes, dtype=np.int64),
        baseline_completed_episodes=np.asarray(baseline_completed_episodes, dtype=np.int64),
        probe_completed_episodes=np.asarray(probe_completed_episodes, dtype=np.int64),
        probe_shadow_completed_episodes=np.asarray(probe_shadow_completed_episodes, dtype=np.int64),
        probe_no_expression_completed_episodes=np.asarray(probe_no_expression_completed_episodes, dtype=np.int64),
        full_system_completed_episodes=np.asarray(full_system_completed_episodes, dtype=np.int64),
        full_system_state_only_completed_episodes=np.asarray(full_system_state_only_completed_episodes, dtype=np.int64),
        full_system_oracle_completed_episodes=np.asarray(full_system_oracle_completed_episodes, dtype=np.int64),
        sim_fanout_completed_episodes=np.asarray(sim_fanout_completed_episodes, dtype=np.int64),
        full_system_controller_style=np.asarray(full_system_controller_style, dtype="U"),
        full_system_oracle_controller_style=np.asarray(full_system_oracle_controller_style, dtype="U"),
        sim_fanout_controller_style=np.asarray(sim_fanout_controller_style, dtype="U"),
        probe_encoder_steps=np.asarray(probe_encoder_steps, dtype=np.int64),
        probe_windows_total=np.asarray(probe_windows_total, dtype=np.int64),
        probe_expression_scale_median=np.asarray(probe_expression_scale_median, dtype=np.float32),
        probe_expression_scale_active_fraction=np.asarray(probe_expression_scale_active_fraction, dtype=np.float32),
        probe_fair_ready_handoff_fraction=np.asarray(probe_fair_ready_handoff_fraction, dtype=np.float32),
        probe_fair_expression_enabled_fraction=np.asarray(probe_fair_expression_enabled_fraction, dtype=np.float32),
        probe_fair_expression_force_muted_fraction=np.asarray(probe_fair_expression_force_muted_fraction, dtype=np.float32),
        probe_fair_ready_confidence_median=np.asarray(probe_fair_ready_confidence_median, dtype=np.float32),
        probe_fair_muted_confidence_median=np.asarray(probe_fair_muted_confidence_median, dtype=np.float32),
        probe_expression_ready_but_muted_fraction=np.asarray(probe_expression_ready_but_muted_fraction, dtype=np.float32),
        probe_shadow_expression_enabled_fraction=np.asarray(probe_shadow_expression_enabled_fraction, dtype=np.float32),
        probe_shadow_expression_scale_median=np.asarray(probe_shadow_expression_scale_median, dtype=np.float32),
        probe_shadow_confidence_median=np.asarray(probe_shadow_confidence_median, dtype=np.float32),
        probe_shadow_strict_miss_fraction=np.asarray(probe_shadow_strict_miss_fraction, dtype=np.float32),
        probe_run_classification=np.asarray(probe_run_classification, dtype="U"),
        belief_progress_index=np.asarray(belief_progress_index, dtype=np.float32),
        latent_mechanics_fit=np.asarray(latent_mechanics_fit, dtype=np.float32),
        latent_split_top1=np.asarray(latent_split_top1, dtype=np.float32),
        latent_neighbor_alignment=np.asarray(latent_neighbor_alignment, dtype=np.float32),
        latent_gap_ratio=np.asarray(latent_gap_ratio, dtype=np.float32),
        latent_heldout_probe_error=np.asarray(latent_heldout_probe_error, dtype=np.float32),
        latent_probe_leakage=np.asarray(latent_probe_leakage, dtype=np.float32),
        latent_uncert_error_corr=np.asarray(latent_uncert_error_corr, dtype=np.float32),
        latent_support_diagnostics_json=np.asarray(latent_support_diagnostics_json, dtype="U"),
        latent_win_gate_json=np.asarray(latent_win_gate_json),
        latent_win_gate_failure_reasons_json=np.asarray(latent_win_gate_failure_reasons_json),
        probe_stop_reasons_json=np.asarray(probe_stop_reasons_json, dtype="U"),
        probe_final_stop_reason=np.asarray(probe_final_stop_reason, dtype="U"),
        probe_family_expected_gain_json=np.asarray(probe_family_expected_gain_json, dtype="U"),
        probe_family_realized_gain_json=np.asarray(probe_family_realized_gain_json, dtype="U"),
        probe_family_future_error_json=np.asarray(probe_family_future_error_json, dtype="U"),
        probe_family_selection_count_json=np.asarray(probe_family_selection_count_json, dtype="U"),
        probe_readiness_reason_counts_json=np.asarray(probe_readiness_reason_counts_json, dtype="U"),
        probe_readiness_component_means_json=np.asarray(probe_readiness_component_means_json, dtype="U"),
        probe_fair_stop_blocker_counts_json=np.asarray(probe_fair_stop_blocker_counts_json, dtype="U"),
        probe_shadow_blocker_counts_json=np.asarray(probe_shadow_blocker_counts_json, dtype="U"),
        probe_second_probe_selection_count_json=np.asarray(probe_second_probe_selection_count_json, dtype="U"),
        probe_second_probe_raw_future_gain_mean=np.asarray(probe_second_probe_raw_future_gain_mean, dtype=np.float32),
        probe_second_probe_future_estimate_mean=np.asarray(probe_second_probe_future_estimate_mean, dtype=np.float32),
        probe_second_probe_choice_future_gain_mean=np.asarray(probe_second_probe_choice_future_gain_mean, dtype=np.float32),
        probe_family_coverage_satisfied_fraction=np.asarray(probe_family_coverage_satisfied_fraction, dtype=np.float32),
        probe_second_probe_value_driven_fraction=np.asarray(probe_second_probe_value_driven_fraction, dtype=np.float32),
        probe_uniformity_pressure_active_fraction=np.asarray(probe_uniformity_pressure_active_fraction, dtype=np.float32),
        probe_env_expression_delta=np.asarray(probe_env_expression_delta, dtype=np.float32),
        probe_forced_env_expression_delta=np.asarray(probe_forced_env_expression_delta, dtype=np.float32),
        probe_forced_env_expression_scale=np.asarray(probe_forced_env_expression_scale, dtype=np.float32),
        probe_strict_usage_status=np.asarray(probe_strict_usage_status, dtype="U"),
        probe_fair_handoff_probe_families_json=np.asarray(probe_fair_handoff_probe_families_json, dtype="U"),
        probe_readiness_component_timeline_json=np.asarray(probe_readiness_component_timeline_json, dtype="U"),
        probe_message_ablation_config_diff_json=np.asarray(probe_message_ablation_config_diff_json, dtype="U"),
        probe_online_future_quality_trace_json=np.asarray(probe_online_future_quality_trace_json, dtype="U"),
        probe_online_subset_stability_trace_json=np.asarray(probe_online_subset_stability_trace_json, dtype="U"),
        probe_online_offline_gap_trace_json=np.asarray(probe_online_offline_gap_trace_json, dtype="U"),
        probe_message_input_delta_mean=np.asarray(probe_message_input_delta_mean, dtype=np.float32),
        probe_message_input_delta_max=np.asarray(probe_message_input_delta_max, dtype=np.float32),
        probe_muted_message_input_delta_mean=np.asarray(probe_muted_message_input_delta_mean, dtype=np.float32),
        probe_muted_message_input_delta_max=np.asarray(probe_muted_message_input_delta_max, dtype=np.float32),
        probe_actor_message_norm_mean=np.asarray(probe_actor_message_norm_mean, dtype=np.float32),
        probe_actor_message_nonzero_fraction=np.asarray(probe_actor_message_nonzero_fraction, dtype=np.float32),
        probe_muted_actor_message_nonzero_fraction=np.asarray(probe_muted_actor_message_nonzero_fraction, dtype=np.float32),
        probe_matched_mute_parity_fraction=np.asarray(probe_matched_mute_parity_fraction, dtype=np.float32),
        probe_online_subset_stability_mean=np.asarray(probe_online_subset_stability_mean, dtype=np.float32),
        probe_online_offline_gap_mean=np.asarray(probe_online_offline_gap_mean, dtype=np.float32),
        probe_online_geometry_complete_fraction=np.asarray(probe_online_geometry_complete_fraction, dtype=np.float32),
        probe_online_split_latent_disagreement_mean=np.asarray(probe_online_split_latent_disagreement_mean, dtype=np.float32),
        probe_online_split_retrieval_margin_deficit_mean=np.asarray(probe_online_split_retrieval_margin_deficit_mean, dtype=np.float32),
        probe_online_leaveout_shift_mean=np.asarray(probe_online_leaveout_shift_mean, dtype=np.float32),
        probe_teacher_action_agreement=np.asarray(probe_teacher_action_agreement, dtype=np.float32),
        full_system_state_only_eval_returns_json=np.asarray(full_system_state_only_eval_returns_json, dtype="U"),
        full_system_learned_eval_summary_json=np.asarray(full_system_learned_eval_summary_json, dtype="U"),
        full_system_state_only_eval_summary_json=np.asarray(full_system_state_only_eval_summary_json, dtype="U"),
        full_system_zero_context_eval_summary_json=np.asarray(full_system_zero_context_eval_summary_json, dtype="U"),
        full_system_shuffled_context_eval_summary_json=np.asarray(full_system_shuffled_context_eval_summary_json, dtype="U"),
        full_system_stale_context_eval_summary_json=np.asarray(full_system_stale_context_eval_summary_json, dtype="U"),
        full_system_online_refinement_eval_summary_json=np.asarray(full_system_online_refinement_eval_summary_json, dtype="U"),
        full_system_frozen_context_eval_summary_json=np.asarray(full_system_frozen_context_eval_summary_json, dtype="U"),
        full_system_actor_only_eval_summary_json=np.asarray(full_system_actor_only_eval_summary_json, dtype="U"),
        full_system_state_only_ablation_delta=np.asarray(full_system_state_only_ablation_delta, dtype=np.float32),
        full_system_zero_context_ablation_delta=np.asarray(full_system_zero_context_ablation_delta, dtype=np.float32),
        full_system_shuffled_context_ablation_delta=np.asarray(full_system_shuffled_context_ablation_delta, dtype=np.float32),
        full_system_stale_context_ablation_delta=np.asarray(full_system_stale_context_ablation_delta, dtype=np.float32),
        full_system_online_refinement_ablation_delta=np.asarray(full_system_online_refinement_ablation_delta, dtype=np.float32),
        full_system_frozen_context_ablation_delta=np.asarray(full_system_frozen_context_ablation_delta, dtype=np.float32),
        full_system_actor_only_ablation_delta=np.asarray(full_system_actor_only_ablation_delta, dtype=np.float32),
        full_system_oracle_learned_eval_summary_json=np.asarray(full_system_oracle_learned_eval_summary_json, dtype="U"),
        full_system_oracle_zero_context_eval_summary_json=np.asarray(full_system_oracle_zero_context_eval_summary_json, dtype="U"),
        full_system_oracle_shuffled_context_eval_summary_json=np.asarray(full_system_oracle_shuffled_context_eval_summary_json, dtype="U"),
        full_system_oracle_stale_context_eval_summary_json=np.asarray(full_system_oracle_stale_context_eval_summary_json, dtype="U"),
        full_system_oracle_online_refinement_eval_summary_json=np.asarray(full_system_oracle_online_refinement_eval_summary_json, dtype="U"),
        full_system_oracle_frozen_context_eval_summary_json=np.asarray(full_system_oracle_frozen_context_eval_summary_json, dtype="U"),
        full_system_oracle_actor_only_eval_summary_json=np.asarray(full_system_oracle_actor_only_eval_summary_json, dtype="U"),
        full_system_oracle_zero_context_ablation_delta=np.asarray(full_system_oracle_zero_context_ablation_delta, dtype=np.float32),
        full_system_oracle_shuffled_context_ablation_delta=np.asarray(full_system_oracle_shuffled_context_ablation_delta, dtype=np.float32),
        full_system_oracle_stale_context_ablation_delta=np.asarray(full_system_oracle_stale_context_ablation_delta, dtype=np.float32),
        full_system_oracle_online_refinement_ablation_delta=np.asarray(full_system_oracle_online_refinement_ablation_delta, dtype=np.float32),
        full_system_oracle_frozen_context_ablation_delta=np.asarray(full_system_oracle_frozen_context_ablation_delta, dtype=np.float32),
        full_system_oracle_actor_only_ablation_delta=np.asarray(full_system_oracle_actor_only_ablation_delta, dtype=np.float32),
    )
    print(f"Saved benchmark results to {benchmark_path}")


def save_dashboard_context(
    env_name: str,
    benchmark_tag: str,
    seeds: list[int],
    artifact_dir: Path,
    benchmark_profile: str | None = None,
) -> None:
    """Persist the current training selection so the dashboard can default to it."""
    artifact_dir.mkdir(exist_ok=True)
    context_path = artifact_dir / "dashboard_context.json"
    default_benchmark_summary = benchmark_summary_filename(benchmark_tag, "fast")
    if str(benchmark_profile) == "archived_planner" and context_path.exists():
        try:
            existing_context = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_context = {}
        existing_default = existing_context.get("default_benchmark_summary")
        if isinstance(existing_default, str) and existing_default:
            default_benchmark_summary = existing_default
    context = {
        "env_name": env_name,
        "env_display_name": get_env_display_name(env_name),
        "benchmark_tag": benchmark_tag,
        "default_benchmark_summary": default_benchmark_summary,
        "default_latent_snapshot": f"{benchmark_tag}_seed_{seeds[-1]}_latent_snapshot.npz",
        "seeds": list(seeds),
        "benchmark_profile": None if benchmark_profile is None else str(benchmark_profile),
    }
    context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
    print(f"Saved dashboard context to {context_path}")
