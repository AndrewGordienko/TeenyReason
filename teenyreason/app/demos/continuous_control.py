"""CLI helpers for generic continuous-control demos."""

from __future__ import annotations

from .summary_keys import SUMMARY_DIAGNOSTIC_KEYS


def print_generic_summary(result) -> None:
    summary = result.summary()
    print_header("Generic Gym Latent MPC Summary")
    if "model_family" in summary:
        print(f"model: {summary['model_family']}")
    print(f"env: {summary['env_name']}")
    print(f"probe_samples: {summary['probe_samples']}")
    print(f"probe_return_mean: {float(summary['probe_return_mean']):.2f}")
    print(f"control_return: {float(summary['control_return']):.2f}")
    print(f"control_steps: {summary['control_steps']}")
    print(f"total_samples: {summary['total_samples']}")
    print(f"solved: {summary['solved']}")
    print(f"solve_return: {float(summary['solve_return']):.2f}")
    print(f"horizon: {summary['horizon']}")
    print(f"candidate_count: {summary['candidate_count']}")
    if "method" in summary:
        print(f"method: {summary['method']}")
        if "latest_eval_return" in summary:
            print(f"latest_eval_return: {float(summary.get('latest_eval_return', 0.0)):.2f}")
            print(f"retained_actor_return: {float(summary.get('retained_actor_return', 0.0)):.2f}")
            print(f"collector_best_return: {float(summary.get('collector_best_return', 0.0)):.2f}")
        print(f"best_return: {float(summary.get('best_return', 0.0)):.2f}")
        print_model_diagnostics(summary.get("diagnostics", {}))
    if "cem_iterations" in summary:
        print(f"cem_iterations: {summary['cem_iterations']}")
        print(f"ensemble_size: {summary['ensemble_size']}")
        print(f"train_loss: {float(summary['train_loss']):.6f}")
        print(f"uncertainty_penalty: {float(summary['uncertainty_penalty']):.4f}")
        print(f"online_refit: {summary['online_refit']}")
        print(f"online_refits: {summary['online_refits']}")
        print(f"event_weighting: {summary['event_weighting']}")
        print(f"uncertainty_execution_gate: {summary['uncertainty_execution_gate']}")
        print(f"value_bootstrap: {summary.get('value_bootstrap', False)}")
        print(f"action_value_bootstrap: {summary.get('action_value_bootstrap', False)}")
        print(f"control_preset: {summary.get('control_preset', 'default')}")
        print(f"actor_policy: {summary.get('actor_policy', False)}")
        print(f"actor_collection_prior: {summary.get('actor_collection_prior', False)}")
        print(f"actor_center_prior: {summary.get('actor_center_prior', False)}")
        print(f"pessimistic_planning: {summary.get('pessimistic_planning', False)}")
        print(f"value_calibration: {summary.get('value_calibration', False)}")
        print(f"collector: {summary.get('collector', 'random')}")
        print_model_diagnostics(summary.get("diagnostics", {}))


def print_model_diagnostics(diagnostics: object) -> None:
    if not isinstance(diagnostics, dict):
        return
    print("model_diagnostics:")
    for key in SUMMARY_DIAGNOSTIC_KEYS:
        if key in diagnostics:
            print(f"  {key}: {format_optional(diagnostics[key])}")


def print_performance_table(summary: dict[str, object]) -> None:
    diagnostics = summary.get("diagnostics") or summary.get("initial_diagnostics") or {}
    rows = summary.get("rows", [])
    best_return = float(summary.get("best_return", best_observed_return(summary, diagnostics)))
    samples_to_peak = summary.get("samples_to_peak", summary.get("total_samples", "n/a"))
    samples_to_solve = summary.get("samples_to_solve") or (summary.get("total_samples") if summary.get("solved") else None)
    repair_rate = first_metric(diagnostics, ("frontier_restart_accept_rate", "curriculum_repair_accept_rate", "replay_branch_acceptance_rate"))
    bottleneck = bottleneck_label(diagnostics)
    print_header("Performance Table")
    print("env | collector | best_return | samples_to_peak | samples_to_solve | collector_steps | repair_accept_rate")
    print(
        f"{summary.get('env_name')} | {summary.get('collector')} | "
        f"{best_return:.2f} | {samples_to_peak} | {samples_to_solve or 'n/a'} | "
        f"{diagnostics.get('collector_interaction_steps', 'n/a')} | {format_rate(repair_rate)}"
    )
    print(f"bottleneck: {bottleneck}")
    if isinstance(rows, list) and rows:
        for row in rows:
            print(
                f"round {row.get('round', 1)} | eval={float(row.get('eval_return', summary.get('control_return', 0.0))):.2f} | "
                f"best={float(row.get('best_return', best_return)):.2f} | samples={row.get('real_samples', summary.get('total_samples', 'n/a'))}"
            )


def best_observed_return(summary: dict[str, object], diagnostics: object) -> float:
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    return max(float(summary.get("control_return", 0.0)), float(diag.get("collector_best_return", 0.0)))


def first_metric(metrics: object, keys: tuple[str, ...]) -> object:
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def bottleneck_label(metrics: object) -> str:
    if not isinstance(metrics, dict):
        return "unknown"
    value_gap = abs(float(metrics.get("control_predicted_vs_real_reward_gap", 0.0)))
    off_manifold = float(metrics.get("control_plan_off_manifold_penalty_total_mean", 0.0))
    repair_rate = float(first_metric(metrics, ("frontier_restart_accept_rate", "curriculum_repair_accept_rate", "replay_branch_acceptance_rate")) or 0.0)
    if value_gap >= 3.0:
        return "planner_value_model_overtrust"
    if off_manifold >= 2.0:
        return "off_manifold_control"
    if repair_rate <= 0.05:
        return "frontier_repair_acceptance_low"
    return "solver_exploitation"


def format_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_optional(value: object) -> str:
    number = optional_float(value)
    if number is None:
        return "n/a"
    return f"{number:.2f}"


def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
