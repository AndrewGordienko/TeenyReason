"""Dashboard payload assembly for the multi-domain suite."""

from __future__ import annotations

from pathlib import Path

from ....envs import CONTINUOUS_CARTPOLE_NAME
from ... import affordance_crawler_row, decision_local_crawler_row, planner_comparison_row
from ...contracts.decision_utility import attach_decision_utility_blocks, decision_utility_row
from ...contracts.decision_local import attach_decision_local_blocks, decision_local_row
from ...diagnostics.latent_utility import attach_latent_utility_blocks, latent_utility_row, wake_up_row
from ...diagnostics.sample_performance import attach_sample_performance_blocks, sample_performance_row
from ...diagnostics.world_understanding import attach_world_understanding_blocks, world_understanding_row
from ...contracts.evidence import attach_standard_domain_blocks, causal_metric_row, mechanism_check_row, transfer_gap_row
from ...contracts.handoff import attach_belief_handoff_blocks, belief_handoff_row, rate_distortion_row
from ...contracts.handoff_repair import attach_handoff_repair_blocks, handoff_repair_row
from .acceptance import attach_suite_claim_gates, suite_acceptance_thresholds
from .cartpole import cartpole_domain_payload
from ..config import MultidomainSuiteConfig
from ..domains import _board_domain_payload, _image_domain_payload, _language_domain_payload, _load_cartpole_benchmark_payload
from .metrics import (
    _attach_cartpole_mechanism,
    _attach_family_bridge,
    _attach_latent_handoff,
    _attach_planner_comparison,
    _attach_predictive_planner,
    _attach_real_causal_understanding,
)


def build_suite_payload(
    *,
    config: MultidomainSuiteConfig,
    run_id: str,
    started_at: float,
    results: dict[str, object],
    detail_paths: dict[str, Path],
) -> dict[str, object]:
    """Build the one artifact consumed by the tri-domain dashboard."""
    rl_results = results.get("rl", {})
    cartpole_result = None
    if isinstance(rl_results, dict):
        cartpole_result = rl_results.get(CONTINUOUS_CARTPOLE_NAME)
    cartpole_payload = cartpole_result
    benchmark_payload = _load_cartpole_benchmark_payload(config, cartpole_result)
    if benchmark_payload is not None:
        cartpole_payload = benchmark_payload
    domains = {
        "cartpole": cartpole_domain_payload(cartpole_payload, detail_paths.get("cartpole")),
        "language": _language_domain_payload(results.get("language"), detail_paths.get("language")),
        "image": _image_domain_payload(results.get("image"), detail_paths.get("image")),
        "board": _board_domain_payload(results.get("board"), detail_paths.get("board")),
    }
    _attach_cartpole_mechanism(domains["cartpole"], results.get("cartpole_mechanics"))
    _attach_family_bridge(
        domains["cartpole"],
        results.get("cartpole_controller_bridge"),
        metric_name="controller_return",
    )
    _attach_family_bridge(
        domains["language"],
        results.get("language_bridge"),
        metric_name="generated_next_token_accuracy",
    )
    _attach_family_bridge(
        domains["image"],
        results.get("image_bridge"),
        metric_name="generated_shape_label_accuracy",
    )
    _attach_family_bridge(
        domains["board"],
        results.get("board_bridge"),
        metric_name="best_move_accuracy",
    )
    _attach_latent_handoff(domains["cartpole"], results.get("latent_handoff"))
    _attach_predictive_planner(domains["cartpole"], results.get("cartpole_latent_mpc"))
    _attach_planner_comparison(domains["cartpole"], results.get("cartpole_planner_comparison"))
    real_causal = results.get("real_causal")
    if isinstance(real_causal, dict):
        for domain_name in ("cartpole", "language", "image"):
            _attach_real_causal_understanding(
                domains[domain_name],
                real_causal.get(domain_name),
            )
    attach_standard_domain_blocks(domains)
    attach_latent_utility_blocks(domains)
    attach_world_understanding_blocks(domains)
    attach_belief_handoff_blocks(domains)
    attach_handoff_repair_blocks(domains)
    attach_decision_utility_blocks(domains)
    attach_decision_local_blocks(domains)
    attach_sample_performance_blocks(domains)
    acceptance = attach_suite_claim_gates(domains)
    decision_crawler = results.get("decision_local_crawler")
    affordance_crawler = results.get("affordance_crawler")
    return {
        "schema_version": 1,
        "suite_name": config.suite_name,
        "run_id": run_id,
        "created_at": started_at,
        "artifact_dir": str(config.artifact_dir),
        "domains": domains,
        "cross_domain": {
            "metric_rows": [
                {
                    "domain": domain_name,
                    "subset_consistency": domain.get("metrics", {}).get("subset_consistency", domain.get("trust", 0.0)),
                    "solver_gain": domain.get("belief_contribution_margin", 0.0),
                    "ablation_gap": domain.get("ablation_gap", 0.0),
                    "content_lift": domain.get("causal_ablation", {}).get("content_lift", 0.0),
                    "content_causal": domain.get("causal_ablation", {}).get("content_causal", False),
                    "belief_bitrate": domain.get("metrics", {}).get("belief_bitrate", 0),
                }
                for domain_name, domain in domains.items()
            ],
            "causal_rows": [
                causal_metric_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "mechanism_rows": [
                mechanism_check_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "transfer_rows": [
                transfer_gap_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "handoff_rows": [
                _handoff_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "latent_utility_rows": [
                latent_utility_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "wake_up_rows": [
                wake_up_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "world_understanding_rows": [
                world_understanding_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "belief_handoff_rows": [
                belief_handoff_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "rate_distortion_rows": [
                rate_distortion_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "handoff_repair_rows": [
                handoff_repair_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "decision_utility_rows": [
                decision_utility_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "decision_local_rows": [
                decision_local_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "decision_local_crawler_rows": [
                decision_local_crawler_row(
                    domain_name,
                    _decision_local_crawler_domain_result(decision_crawler, domain_name),
                )
                for domain_name in domains
            ],
            "affordance_crawler_rows": [
                affordance_crawler_row(
                    domain_name,
                    _domain_result(affordance_crawler, domain_name),
                )
                for domain_name in domains
            ],
            "sample_performance_rows": [
                sample_performance_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "predictive_planner_rows": [
                _predictive_planner_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "planner_comparison_rows": [
                planner_comparison_row(domain_name, domain)
                for domain_name, domain in domains.items()
            ],
            "acceptance": acceptance,
            "acceptance_thresholds": suite_acceptance_thresholds(),
        },
    }


def _decision_local_crawler_domain_result(
    result: object,
    domain_name: str,
) -> dict[str, object] | None:
    return _domain_result(result, domain_name)


def _domain_result(
    result: object,
    domain_name: str,
) -> dict[str, object] | None:
    if not isinstance(result, dict):
        return None
    domain_result = result.get(domain_name)
    return domain_result if isinstance(domain_result, dict) else None


def _handoff_row(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    metrics = domain.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "domain": domain_name,
        "hidden_target": metrics.get("handoff_hidden_target", ""),
        "cheap_decode_accuracy": metrics.get("handoff_cheap_decode_accuracy", 0.0),
        "centroid_head_decode_accuracy": metrics.get("handoff_centroid_head_decode_accuracy", 0.0),
        "cheap_content_lift": metrics.get("handoff_cheap_content_lift", 0.0),
        "action_change_fraction": metrics.get("handoff_action_change_fraction", 0.0),
        "value_delta_correct_vs_shuffled": metrics.get(
            "handoff_value_delta_correct_vs_shuffled",
            0.0,
        ),
        "cheap_dedicated_probe_steps": metrics.get("handoff_cheap_dedicated_probe_steps", 0.0),
        "expensive_dedicated_probe_steps": metrics.get(
            "handoff_expensive_dedicated_probe_steps",
            0.0,
        ),
        "dual_use_probe_fraction": metrics.get("handoff_dual_use_probe_fraction", 0.0),
        "dedicated_probe_steps_saved": metrics.get("handoff_dedicated_probe_steps_saved", 0.0),
    }


def _predictive_planner_row(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    metrics = domain.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "domain": domain_name,
        "hidden_target": metrics.get("planner_hidden_target", ""),
        "model_family": metrics.get("planner_model_family", ""),
        "decode_accuracy": metrics.get("planner_decode_accuracy", 0.0),
        "no_belief_return": metrics.get("planner_no_belief_return", 0.0),
        "belief_return": metrics.get("planner_belief_return", 0.0),
        "oracle_return": metrics.get("planner_oracle_return", 0.0),
        "solver_gain": metrics.get("planner_solver_gain", 0.0),
        "content_lift": metrics.get("planner_content_lift", 0.0),
        "oracle_gap": metrics.get("planner_oracle_gap", 0.0),
        "action_match_oracle": metrics.get("planner_action_match_oracle", 0.0),
        "no_belief_action_match_oracle": metrics.get(
            "planner_no_belief_action_match_oracle",
            0.0,
        ),
        "one_step_prediction_mse": metrics.get("planner_one_step_prediction_mse", 0.0),
        "k_step_prediction_mse": metrics.get("planner_k_step_prediction_mse", 0.0),
        "no_belief_k_step_prediction_mse": metrics.get(
            "planner_no_belief_k_step_prediction_mse",
            0.0,
        ),
        "belief_solve_rate": metrics.get("planner_belief_solve_rate", 0.0),
        "no_belief_solve_rate": metrics.get("planner_no_belief_solve_rate", 0.0),
        "oracle_solve_rate": metrics.get("planner_oracle_solve_rate", 0.0),
        "belief_samples_to_peak": metrics.get("planner_belief_samples_to_peak", 0.0),
        "no_belief_samples_to_peak": metrics.get("planner_no_belief_samples_to_peak", 0.0),
        "belief_samples_to_solve": metrics.get("planner_belief_samples_to_solve"),
        "no_belief_samples_to_solve": metrics.get("planner_no_belief_samples_to_solve"),
        "net_samples_to_solve_savings": metrics.get("planner_net_samples_to_solve_savings"),
        "net_env_sample_savings": metrics.get("planner_net_env_sample_savings", 0.0),
        "probe_steps": metrics.get("planner_probe_steps", 0.0),
        "horizon": metrics.get("planner_horizon", 0.0),
        "candidate_count": metrics.get("planner_candidate_count", 0.0),
    }
