"""Persistent-affordance CartPole MPC arm."""

from __future__ import annotations

from dataclasses import dataclass

from ..affordance.core import PersistentAffordanceConfig
from ..affordance.scoring import belief_status, normalize, score_intervention
from ..domains.cartpole import MechanicsWorld, nominal_world, world_for_seed
from ..decision_crawler import CartPoleDecisionLocalAdapter
from .cartpole_latent_mpc import (
    CartPoleLatentDynamicsModel,
    CartPoleLatentMPCConfig,
    _evaluate_planner,
    _first_action,
    _nullable_mean,
)
from .world_model import RandomShootingPlanner, dict_mean


@dataclass(frozen=True)
class CartPolePersistentAffordanceMPCConfig:
    """Config for the persistent-affordance predictive planner arm."""

    mpc: CartPoleLatentMPCConfig
    reuse_horizon: int = 24
    stability_margin: float = 0.12
    disagreement_floor: float = 0.10
    cost_weight: float = 1.0


def run_cartpole_persistent_affordance_mpc(
    config: CartPolePersistentAffordanceMPCConfig,
) -> dict[str, object]:
    """Evaluate one selected affordance probe reused by a CartPole MPC planner."""
    model = CartPoleLatentDynamicsModel()
    planner = RandomShootingPlanner(
        horizon=config.mpc.horizon,
        candidate_count=config.mpc.candidate_count,
        action_grid=config.mpc.action_grid,
    )
    adapter = CartPoleDecisionLocalAdapter(probe_steps=int(config.mpc.probe_steps))
    rows: list[dict[str, object]] = []
    for seed in config.mpc.seeds:
        truth = world_for_seed(int(seed))
        selected = _select_affordance_probe(adapter, truth, int(seed), config)
        decoded = selected["decoded_world"]
        if not isinstance(decoded, MechanicsWorld):
            decoded = nominal_world()
        eval_result = _evaluate_planner(truth, decoded, int(seed), config.mpc, model, planner)
        strict_samples = _samples_to_solve(eval_result, float(selected["probe_cost"]), config.mpc)
        amortized_samples = _samples_to_solve(
            eval_result,
            float(selected["amortized_probe_cost"]),
            config.mpc,
        )
        rows.append(
            {
                "seed": int(seed),
                "hidden_rule": truth.label(),
                "decoded_rule": decoded.label(),
                "decode_accuracy": float(decoded == truth),
                "selected_probe_family": selected["probe_family"],
                "selected_probe_cost": float(selected["probe_cost"]),
                "amortized_probe_cost": float(selected["amortized_probe_cost"]),
                "reuse_horizon": int(config.reuse_horizon),
                "probe_future_adjusted_value": float(selected["future_adjusted_value"]),
                "probe_expected_regret_reduction": float(selected["expected_regret_reduction"]),
                "affordance_mpc_return": eval_result["return"],
                "affordance_first_action": _first_action(eval_result["actions"]),
                "affordance_survival_steps": eval_result["survival_steps"],
                "affordance_solved": float(eval_result["solved"]),
                "affordance_env_samples_strict": float(selected["probe_cost"])
                + float(len(eval_result["actions"])),
                "affordance_env_samples_amortized": float(selected["amortized_probe_cost"])
                + float(len(eval_result["actions"])),
                "affordance_samples_to_solve_strict": strict_samples,
                "affordance_samples_to_solve_amortized": amortized_samples,
            }
        )
    return {
        "domain": "cartpole",
        "dataset": "ControlledCartPolePersistentAffordanceMPC",
        "model_family": "PersistentAffordanceBelief+PredictiveMPC",
        "hidden_target": "cartpole_mechanics_reused_affordance_world_model",
        "reuse_horizon": int(config.reuse_horizon),
        "rows": rows,
        "decode_accuracy": dict_mean(rows, "decode_accuracy"),
        "affordance_mpc_return": dict_mean(rows, "affordance_mpc_return"),
        "affordance_solve_rate": dict_mean(rows, "affordance_solved"),
        "probe_cost": dict_mean(rows, "selected_probe_cost"),
        "amortized_probe_cost": dict_mean(rows, "amortized_probe_cost"),
        "probe_future_adjusted_value": dict_mean(rows, "probe_future_adjusted_value"),
        "probe_expected_regret_reduction": dict_mean(rows, "probe_expected_regret_reduction"),
        "affordance_env_samples_strict": dict_mean(rows, "affordance_env_samples_strict"),
        "affordance_env_samples_amortized": dict_mean(rows, "affordance_env_samples_amortized"),
        "affordance_samples_to_solve_strict": _nullable_mean(
            rows,
            "affordance_samples_to_solve_strict",
        ),
        "affordance_samples_to_solve_amortized": _nullable_mean(
            rows,
            "affordance_samples_to_solve_amortized",
        ),
    }


def _select_affordance_probe(
    adapter: CartPoleDecisionLocalAdapter,
    truth: MechanicsWorld,
    seed: int,
    config: CartPolePersistentAffordanceMPCConfig,
) -> dict[str, object]:
    state = adapter.initial_state(truth, seed=seed)
    belief = normalize(adapter.initial_belief(state, seed=seed))
    policy = PersistentAffordanceConfig(
        reuse_horizon=int(config.reuse_horizon),
        max_expensive_probes=1,
        cost_weight=float(config.cost_weight),
        stability_margin=float(config.stability_margin),
        disagreement_floor=float(config.disagreement_floor),
    )
    status = belief_status(adapter, state, belief)
    unstable = (
        float(status["margin"]) < float(config.stability_margin)
        or float(status["disagreement"]) > float(config.disagreement_floor)
    )
    scored = [
        score_intervention(
            adapter,
            state,
            belief,
            intervention,
            seed,
            int(config.reuse_horizon),
            policy,
        )
        for intervention in adapter.candidate_interventions(state, belief)
    ]
    scored.sort(key=lambda item: item["future_adjusted_value"], reverse=True)
    if not unstable or not scored or float(scored[0]["future_adjusted_value"]) <= 0.0:
        return {
            "decoded_world": nominal_world(),
            "probe_family": "none",
            "probe_cost": 0.0,
            "amortized_probe_cost": 0.0,
            "future_adjusted_value": 0.0,
            "expected_regret_reduction": 0.0,
        }
    chosen = scored[0]["intervention"]
    observation = adapter.observe_truth(state, chosen, truth, seed=seed)
    posterior = normalize(adapter.update_belief(state, belief, chosen, observation, seed=seed))
    decoded = max(posterior, key=lambda particle: float(particle.weight)).message
    cost = float(chosen.cost)
    return {
        "decoded_world": decoded,
        "probe_family": chosen.family,
        "probe_cost": cost,
        "amortized_probe_cost": cost / float(max(int(config.reuse_horizon), 1)),
        "future_adjusted_value": float(scored[0]["future_adjusted_value"]),
        "expected_regret_reduction": float(scored[0]["expected_regret_reduction"]),
    }


def _samples_to_solve(
    result: dict[str, object],
    probe_cost: float,
    config: CartPoleLatentMPCConfig,
) -> float | None:
    if not bool(result.get("solved", False)):
        return None
    return float(probe_cost + int(config.control_steps))
