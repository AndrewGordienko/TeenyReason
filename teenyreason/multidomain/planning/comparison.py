"""Matched planner comparison rows for crawler-belief MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cartpole_latent_mpc import CartPoleLatentMPCConfig, run_cartpole_latent_mpc_benchmark
from .comparison_tables import planner_arm_rows, planner_comparison_row
from .persistent_affordance import (
    CartPolePersistentAffordanceMPCConfig,
    run_cartpole_persistent_affordance_mpc,
)


@dataclass(frozen=True)
class CartPolePlannerComparisonConfig:
    """Config for comparing predictive crawler-MPC arms."""

    profile: str = "smoke"
    seeds: tuple[int, ...] = tuple(range(16))
    matched_seeds: tuple[int, ...] = tuple(range(32))
    support_families: tuple[str, ...] = ("passive_decay", "impulse_left", "impulse_right", "chirp")
    cheap_support_families: tuple[str, ...] = ("passive_decay", "chirp")
    probe_steps: int = 18
    cheap_probe_steps: int = 6
    control_steps: int = 80
    matched_control_steps: int = 500
    horizon: int = 4
    candidate_count: int = 32
    matched_horizon: int = 12
    matched_candidate_count: int = 96
    fallback_roi_floor: float = 0.01
    cheap_confidence_floor: float = 0.45
    persistent_reuse_horizon: int = 24
    persistent_probe_cost_weight: float = 1.0


def run_cartpole_planner_comparison(
    config: CartPolePlannerComparisonConfig | None = None,
) -> dict[str, object]:
    """Compare no-belief, crawler-belief, oracle, and cheap-fallback MPC."""
    config = config or CartPolePlannerComparisonConfig()
    expensive_config = _latent_config(config, cheap=False)
    cheap_config = _latent_config(config, cheap=True)
    expensive = run_cartpole_latent_mpc_benchmark(expensive_config)
    cheap = run_cartpole_latent_mpc_benchmark(cheap_config)
    persistent = run_cartpole_persistent_affordance_mpc(
        CartPolePersistentAffordanceMPCConfig(
            mpc=expensive_config,
            reuse_horizon=int(config.persistent_reuse_horizon),
            cost_weight=float(config.persistent_probe_cost_weight),
        )
    )
    cheap_rows = {
        int(row.get("seed", -1)): row
        for row in cheap.get("rows", [])
        if isinstance(row, dict)
    }
    persistent_rows = {
        int(row.get("seed", -1)): row
        for row in persistent.get("rows", [])
        if isinstance(row, dict)
    }
    rows = [
        _comparison_row(
            row,
            cheap_rows.get(int(row.get("seed", -1)), {}),
            persistent_rows.get(int(row.get("seed", -1)), {}),
            config,
        )
        for row in expensive.get("rows", [])
        if isinstance(row, dict)
    ]
    arms = planner_arm_rows(expensive, cheap, persistent, rows)
    return {
        "domain": "cartpole",
        "dataset": "ControlledCartPolePlannerComparison",
        "model_family": "CrawlerBelief+PredictiveMPCComparison",
        "hidden_target": "cartpole_mechanics_action_conditioned_world_model",
        "profile": _profile(config),
        "rows": rows,
        "arms": arms,
        "decode_accuracy": _float(expensive.get("decode_accuracy", 0.0)),
        "cheap_decode_accuracy": _float(cheap.get("decode_accuracy", 0.0)),
        "belief_action_match_oracle": _float(expensive.get("belief_action_match_oracle", 0.0)),
        "no_belief_action_match_oracle": _float(
            expensive.get("no_belief_action_match_oracle", 0.0)
        ),
        "oracle_gap": _float(expensive.get("oracle_gap", 0.0)),
        "solver_gain": _float(expensive.get("solver_gain", 0.0)),
        "content_lift": _float(expensive.get("content_lift", 0.0)),
        "action_regret_no_belief": _mean(rows, "action_regret_no_belief"),
        "action_regret_belief": _mean(rows, "action_regret_belief"),
        "action_regret_reduction": _mean(rows, "action_regret_reduction"),
        "probe_roi": _mean(rows, "probe_roi"),
        "cheap_probe_roi": _mean(rows, "cheap_probe_roi"),
        "fallback_probe_roi": _mean(rows, "fallback_probe_roi"),
        "fallback_wake_rate": _mean(rows, "wake_expensive_probe"),
        "persistent_affordance_return": _float(persistent.get("affordance_mpc_return", 0.0)),
        "persistent_affordance_solve_rate": _float(persistent.get("affordance_solve_rate", 0.0)),
        "persistent_affordance_probe_roi": _mean(rows, "persistent_probe_roi"),
        "persistent_affordance_probe_cost": _float(persistent.get("probe_cost", 0.0)),
        "persistent_affordance_amortized_probe_cost": _float(
            persistent.get("amortized_probe_cost", 0.0)
        ),
        "persistent_affordance_reuse_horizon": _float(persistent.get("reuse_horizon", 0.0)),
        "persistent_affordance_probe_value": _float(
            persistent.get("probe_future_adjusted_value", 0.0)
        ),
        "persistent_affordance_regret_reduction": _mean(
            rows,
            "persistent_action_regret_reduction",
        ),
        "belief_beats_no_belief_fraction": _mean(rows, "belief_beats_no_belief"),
        "belief_beats_all_ablation_fraction": _mean(rows, "belief_beats_all_ablation"),
        "cheap_selected_fraction": _fraction(rows, "selected_arm", "mpc_cheap_belief"),
        "expensive_selected_fraction": _fraction(rows, "selected_arm", "mpc_crawler_belief"),
        "no_belief_selected_fraction": _fraction(rows, "selected_arm", "mpc_no_belief"),
        "no_belief_mpc_samples_to_solve": _optional_float(
            expensive.get("no_belief_samples_to_solve")
        ),
        "crawler_belief_mpc_samples_to_solve": _optional_float(
            expensive.get("belief_samples_to_solve")
        ),
        "oracle_mpc_samples_to_solve": _optional_float(expensive.get("oracle_samples_to_solve")),
        "cheap_fallback_samples_to_solve": _nullable_mean(rows, "selected_samples_to_solve"),
        "persistent_affordance_samples_to_solve": _optional_float(
            persistent.get("affordance_samples_to_solve_strict")
        ),
        "persistent_affordance_amortized_samples_to_solve": _optional_float(
            persistent.get("affordance_samples_to_solve_amortized")
        ),
        "crawler_vs_no_belief_mpc_sample_savings": _sample_savings(
            _optional_float(expensive.get("no_belief_samples_to_solve")),
            _optional_float(expensive.get("belief_samples_to_solve")),
        ),
        "cheap_fallback_vs_no_belief_mpc_sample_savings": _sample_savings(
            _optional_float(expensive.get("no_belief_samples_to_solve")),
            _nullable_mean(rows, "selected_samples_to_solve"),
        ),
        "persistent_affordance_vs_no_belief_mpc_sample_savings": _sample_savings(
            _optional_float(expensive.get("no_belief_samples_to_solve")),
            _optional_float(persistent.get("affordance_samples_to_solve_strict")),
        ),
        "persistent_affordance_amortized_vs_no_belief_mpc_sample_savings": _sample_savings(
            _optional_float(expensive.get("no_belief_samples_to_solve")),
            _optional_float(persistent.get("affordance_samples_to_solve_amortized")),
        ),
        "probe_steps": _float(expensive.get("probe_steps", 0.0)),
        "cheap_probe_steps": _float(cheap.get("probe_steps", 0.0)),
        "control_steps": _float(expensive.get("control_steps", 0.0)),
        "horizon": _float(expensive.get("horizon", 0.0)),
        "candidate_count": _float(expensive.get("candidate_count", 0.0)),
        "diagnostic_state": _diagnostic_state(expensive, rows),
    }


def _latent_config(
    config: CartPolePlannerComparisonConfig,
    *,
    cheap: bool,
) -> CartPoleLatentMPCConfig:
    profile = _profile(config)
    if profile == "matched":
        seeds = config.matched_seeds
        control_steps = config.matched_control_steps
        horizon = config.matched_horizon
        candidate_count = config.matched_candidate_count
    else:
        seeds = config.seeds
        control_steps = config.control_steps
        horizon = config.horizon
        candidate_count = config.candidate_count
    return CartPoleLatentMPCConfig(
        seeds=tuple(int(seed) for seed in seeds),
        support_families=config.cheap_support_families if cheap else config.support_families,
        probe_steps=int(config.cheap_probe_steps if cheap else config.probe_steps),
        control_steps=int(control_steps),
        horizon=int(horizon),
        candidate_count=int(candidate_count),
    )


def _comparison_row(
    expensive: dict[str, object],
    cheap: dict[str, object],
    persistent: dict[str, object],
    config: CartPolePlannerComparisonConfig,
) -> dict[str, object]:
    oracle_return = _float(expensive.get("oracle_mpc_return", 0.0))
    no_belief_return = _float(expensive.get("no_belief_return", 0.0))
    belief_return = _float(expensive.get("belief_mpc_return", 0.0))
    cheap_return = _float(cheap.get("belief_mpc_return", no_belief_return))
    no_regret = max(0.0, oracle_return - no_belief_return)
    belief_regret = max(0.0, oracle_return - belief_return)
    cheap_regret = max(0.0, oracle_return - cheap_return)
    persistent_return = _float(persistent.get("affordance_mpc_return", no_belief_return))
    persistent_regret = max(0.0, oracle_return - persistent_return)
    regret_reduction = no_regret - belief_regret
    cheap_regret_reduction = no_regret - cheap_regret
    persistent_regret_reduction = no_regret - persistent_regret
    expensive_probe_steps = _float(expensive.get("probe_steps", 0.0))
    cheap_probe_steps = _float(cheap.get("probe_steps", 0.0))
    fallback_gain = max(0.0, belief_return - max(no_belief_return, cheap_return))
    fallback_roi = fallback_gain / max(expensive_probe_steps, 1.0)
    cheap_confidence = _float(cheap.get("confidence", 0.0))
    wake = (
        cheap_confidence < float(config.cheap_confidence_floor)
        and fallback_roi >= float(config.fallback_roi_floor)
    )
    selected = _selected_arm(
        wake=wake,
        cheap_confidence=cheap_confidence,
        config=config,
    )
    selected_return, selected_samples, selected_solved = _selected_outcome(
        selected,
        expensive=expensive,
        cheap=cheap,
        cheap_probe_steps=cheap_probe_steps,
    )
    return {
        "seed": int(expensive.get("seed", 0)),
        "hidden_rule": expensive.get("hidden_rule", ""),
        "decoded_rule": expensive.get("decoded_rule", ""),
        "no_belief_return": no_belief_return,
        "belief_return": belief_return,
        "oracle_return": oracle_return,
        "cheap_return": cheap_return,
        "persistent_affordance_return": persistent_return,
        "shuffled_return": _float(expensive.get("shuffled_mpc_return", 0.0)),
        "stale_return": _float(expensive.get("stale_mpc_return", 0.0)),
        "oracle_action": expensive.get("oracle_first_action"),
        "belief_action": expensive.get("belief_first_action"),
        "no_belief_action": expensive.get("no_belief_first_action"),
        "cheap_action": cheap.get("belief_first_action"),
        "persistent_affordance_action": persistent.get("affordance_first_action"),
        "shuffled_action": expensive.get("shuffled_first_action"),
        "stale_action": expensive.get("stale_first_action"),
        "action_regret_no_belief": no_regret,
        "action_regret_belief": belief_regret,
        "action_regret_cheap": cheap_regret,
        "action_regret_persistent_affordance": persistent_regret,
        "action_regret_reduction": regret_reduction,
        "cheap_action_regret_reduction": cheap_regret_reduction,
        "persistent_action_regret_reduction": persistent_regret_reduction,
        "probe_roi": regret_reduction / max(expensive_probe_steps, 1.0),
        "cheap_probe_roi": cheap_regret_reduction / max(cheap_probe_steps, 1.0),
        "fallback_probe_roi": fallback_roi,
        "persistent_probe_roi": persistent_regret_reduction
        / max(_float(persistent.get("selected_probe_cost", 0.0)), 1.0),
        "persistent_probe_cost": _float(persistent.get("selected_probe_cost", 0.0)),
        "persistent_amortized_probe_cost": _float(
            persistent.get("amortized_probe_cost", 0.0)
        ),
        "persistent_reuse_horizon": _float(persistent.get("reuse_horizon", 0.0)),
        "persistent_probe_future_adjusted_value": _float(
            persistent.get("probe_future_adjusted_value", 0.0)
        ),
        "wake_expensive_probe": float(wake),
        "belief_beats_no_belief": float(belief_return > no_belief_return),
        "belief_beats_all_ablation": float(
            belief_return
            > max(
                no_belief_return,
                _float(expensive.get("shuffled_mpc_return", 0.0)),
                _float(expensive.get("stale_mpc_return", 0.0)),
            )
        ),
        "selected_arm": selected,
        "selected_return": selected_return,
        "selected_samples_to_solve": selected_samples,
        "selected_solved": float(selected_solved),
        "selected_total_env_samples": _selected_total_samples(
            selected,
            expensive=expensive,
            cheap=cheap,
            cheap_probe_steps=cheap_probe_steps,
        ),
        "persistent_samples_to_solve": _optional_float(
            persistent.get("affordance_samples_to_solve_strict")
        ),
        "persistent_amortized_samples_to_solve": _optional_float(
            persistent.get("affordance_samples_to_solve_amortized")
        ),
        "persistent_solved": _float(persistent.get("affordance_solved", 0.0)),
        "belief_action_match_oracle": _float(expensive.get("belief_action_match_oracle", 0.0)),
        "no_belief_action_match_oracle": _float(
            expensive.get("no_belief_action_match_oracle", 0.0)
        ),
        "decode_accuracy": _float(expensive.get("decode_accuracy", 0.0)),
        "cheap_decode_accuracy": _float(cheap.get("decode_accuracy", 0.0)),
        "cheap_confidence": cheap_confidence,
    }


def _selected_arm(
    *,
    wake: bool,
    cheap_confidence: float,
    config: CartPolePlannerComparisonConfig,
) -> str:
    if wake:
        return "mpc_crawler_belief"
    if cheap_confidence >= float(config.cheap_confidence_floor):
        return "mpc_cheap_belief"
    return "mpc_no_belief"


def _selected_outcome(
    selected: str,
    *,
    expensive: dict[str, object],
    cheap: dict[str, object],
    cheap_probe_steps: float,
) -> tuple[float, float | None, bool]:
    if selected == "mpc_crawler_belief":
        solved = bool(_float(expensive.get("belief_solved", 0.0)))
        samples = _optional_float(expensive.get("belief_samples_to_solve"))
        if samples is not None:
            samples += cheap_probe_steps
        return _float(expensive.get("belief_mpc_return", 0.0)), samples, solved
    if selected == "mpc_cheap_belief":
        solved = bool(_float(cheap.get("belief_solved", 0.0)))
        return _float(cheap.get("belief_mpc_return", 0.0)), _optional_float(
            cheap.get("belief_samples_to_solve")
        ), solved
    solved = bool(_float(expensive.get("no_belief_solved", 0.0)))
    samples = _optional_float(expensive.get("no_belief_samples_to_solve"))
    if samples is not None:
        samples += cheap_probe_steps
    return _float(expensive.get("no_belief_return", 0.0)), samples, solved


def _selected_total_samples(
    selected: str,
    *,
    expensive: dict[str, object],
    cheap: dict[str, object],
    cheap_probe_steps: float,
) -> float:
    if selected == "mpc_crawler_belief":
        return _float(expensive.get("belief_env_samples", 0.0)) + cheap_probe_steps
    if selected == "mpc_cheap_belief":
        return _float(cheap.get("belief_env_samples", 0.0))
    return _float(expensive.get("no_belief_env_samples", 0.0)) + cheap_probe_steps


def _diagnostic_state(result: dict[str, object], rows: list[dict[str, object]]) -> str:
    no_solve = _float(result.get("no_belief_solve_rate", 0.0))
    belief_solve = _float(result.get("belief_solve_rate", 0.0))
    gain = _float(result.get("solver_gain", 0.0))
    content = _float(result.get("content_lift", 0.0))
    savings = _optional_float(result.get("net_samples_to_solve_savings"))
    regret_reduction = _mean(rows, "action_regret_reduction")
    if no_solve >= 0.99 and belief_solve >= 0.99 and (savings is None or savings <= 0.0):
        return "planner_benchmark_too_easy"
    if regret_reduction <= 0.0 or gain <= 0.0:
        return "planner_belief_not_helping"
    if content <= 0.0:
        return "planner_ablation_not_clean"
    if savings is not None and savings > 0.0:
        return "planner_belief_wins"
    return "planner_belief_predictive_but_costly"


def _profile(config: CartPolePlannerComparisonConfig) -> str:
    profile = str(config.profile or "smoke")
    if profile not in {"smoke", "matched"}:
        raise ValueError(f"Unknown planner comparison profile: {profile}")
    return profile


def _sample_savings(baseline: float | None, crawler: float | None) -> float | None:
    if baseline is None or crawler is None:
        return None
    return float(baseline - crawler)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([_float(row.get(key, 0.0)) for row in rows]))


def _nullable_mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [_optional_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def _median(rows: list[dict[str, object]], key: str) -> float:
    values = [_optional_float(row.get(key)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        return 0.0
    return float(np.median(clean))


def _fraction(rows: list[dict[str, object]], key: str, value: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([str(row.get(key, "")) == value for row in rows]))
