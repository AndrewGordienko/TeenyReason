"""CartPole predictive-belief MPC benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domains.cartpole import (
    MechanicsWorld,
    _initial_state,
    _state_cost,
    _step_physics,
    candidate_worlds,
    evidence_vector,
    infer_world,
    nominal_world,
    world_for_seed,
)
from .world_model import RandomShootingPlanner, dict_mean, planner_action_match_fraction


@dataclass(frozen=True)
class CartPoleLatentMPCConfig:
    """Action-conditioned world-model handoff benchmark config."""

    seeds: tuple[int, ...] = tuple(range(16))
    support_families: tuple[str, ...] = ("passive_decay", "impulse_left", "impulse_right", "chirp")
    probe_steps: int = 18
    control_steps: int = 80
    horizon: int = 4
    candidate_count: int = 32
    action_grid: tuple[float, ...] = (-1.0, 0.0, 1.0)
    failure_x: float = 2.4
    failure_theta: float = 0.2095


@dataclass(frozen=True)
class CartPoleLatentDynamicsModel:
    """Action-conditioned latent/state predictor parameterized by mechanics belief."""

    def rollout(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        *,
        belief: object,
    ) -> np.ndarray:
        world = belief if isinstance(belief, MechanicsWorld) else nominal_world()
        current = np.asarray(state, dtype=np.float32).reshape(4)
        states = [current.copy()]
        for action in np.asarray(actions, dtype=np.float32).reshape(-1):
            current = _step_physics(current, float(action), world)
            states.append(current.copy())
        return np.asarray(states, dtype=np.float32)

    def score_rollout(self, states: np.ndarray, *, belief: object) -> float:
        del belief
        costs = [_state_cost(state) for state in np.asarray(states, dtype=np.float32)[1:]]
        return float(-sum(costs))


def run_cartpole_latent_mpc_benchmark(
    config: CartPoleLatentMPCConfig | None = None,
) -> dict[str, object]:
    """Evaluate crawler belief as a predictive action-conditioned model."""
    config = config or CartPoleLatentMPCConfig()
    model = CartPoleLatentDynamicsModel()
    planner = RandomShootingPlanner(
        horizon=config.horizon,
        candidate_count=config.candidate_count,
        action_grid=config.action_grid,
    )
    worlds = candidate_worlds()
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        truth = world_for_seed(int(seed))
        decoded, confidence = _crawler_belief(truth, int(seed), config)
        shuffled = worlds[(worlds.index(decoded) + 5) % len(worlds)]
        stale = world_for_seed(int(seed) - 1)
        no_belief = nominal_world()

        no_belief_eval = _evaluate_planner(truth, no_belief, int(seed), config, model, planner)
        belief_eval = _evaluate_planner(truth, decoded, int(seed), config, model, planner)
        oracle_eval = _evaluate_planner(truth, truth, int(seed), config, model, planner)
        shuffled_eval = _evaluate_planner(truth, shuffled, int(seed), config, model, planner)
        stale_eval = _evaluate_planner(truth, stale, int(seed), config, model, planner)
        content_lift = belief_eval["return"] - max(
            no_belief_eval["return"],
            shuffled_eval["return"],
            stale_eval["return"],
        )
        probe_cost = int(len(config.support_families) * int(config.probe_steps))
        belief_samples_to_solve = _samples_to_solve(belief_eval, probe_cost, config)
        no_belief_samples_to_solve = _samples_to_solve(no_belief_eval, 0, config)
        oracle_samples_to_solve = _samples_to_solve(oracle_eval, 0, config)
        rows.append(
            {
                "seed": int(seed),
                "hidden_rule": truth.label(),
                "decoded_rule": decoded.label(),
                "decode_accuracy": float(decoded == truth),
                "confidence": float(confidence),
                "no_belief_return": no_belief_eval["return"],
                "belief_mpc_return": belief_eval["return"],
                "oracle_mpc_return": oracle_eval["return"],
                "shuffled_mpc_return": shuffled_eval["return"],
                "stale_mpc_return": stale_eval["return"],
                "no_belief_first_action": _first_action(no_belief_eval["actions"]),
                "belief_first_action": _first_action(belief_eval["actions"]),
                "oracle_first_action": _first_action(oracle_eval["actions"]),
                "shuffled_first_action": _first_action(shuffled_eval["actions"]),
                "stale_first_action": _first_action(stale_eval["actions"]),
                "solver_gain": belief_eval["return"] - no_belief_eval["return"],
                "content_lift": content_lift,
                "oracle_gap": oracle_eval["return"] - belief_eval["return"],
                "belief_action_match_oracle": planner_action_match_fraction(
                    oracle_eval["actions"],
                    belief_eval["actions"],
                ),
                "no_belief_action_match_oracle": planner_action_match_fraction(
                    oracle_eval["actions"],
                    no_belief_eval["actions"],
                ),
                "belief_one_step_prediction_mse": _prediction_mse(
                    truth,
                    decoded,
                    belief_eval["states"],
                    belief_eval["actions"],
                    one_step=True,
                ),
                "belief_k_step_prediction_mse": _prediction_mse(
                    truth,
                    decoded,
                    belief_eval["states"],
                    belief_eval["actions"],
                    one_step=False,
                ),
                "no_belief_k_step_prediction_mse": _prediction_mse(
                    truth,
                    no_belief,
                    no_belief_eval["states"],
                    no_belief_eval["actions"],
                    one_step=False,
                ),
                "belief_survival_steps": belief_eval["survival_steps"],
                "no_belief_survival_steps": no_belief_eval["survival_steps"],
                "oracle_survival_steps": oracle_eval["survival_steps"],
                "belief_solved": float(belief_eval["solved"]),
                "no_belief_solved": float(no_belief_eval["solved"]),
                "oracle_solved": float(oracle_eval["solved"]),
                "probe_steps": probe_cost,
                "control_steps": int(config.control_steps),
                "belief_env_samples": probe_cost + int(len(belief_eval["actions"])),
                "no_belief_env_samples": int(len(no_belief_eval["actions"])),
                "oracle_env_samples": int(len(oracle_eval["actions"])),
                "belief_samples_to_peak_return": probe_cost + int(len(belief_eval["actions"])),
                "no_belief_samples_to_peak_return": int(len(no_belief_eval["actions"])),
                "oracle_samples_to_peak_return": int(len(oracle_eval["actions"])),
                "belief_samples_to_solve": belief_samples_to_solve,
                "no_belief_samples_to_solve": no_belief_samples_to_solve,
                "oracle_samples_to_solve": oracle_samples_to_solve,
                "net_samples_to_solve_savings": _sample_savings(
                    no_belief_samples_to_solve,
                    belief_samples_to_solve,
                ),
                "net_env_sample_savings": int(len(no_belief_eval["actions"]))
                - (probe_cost + int(len(belief_eval["actions"]))),
            }
        )
    return {
        "domain": "cartpole",
        "dataset": "ControlledCartPoleLatentMPC",
        "model_family": "CrawlerBelief+ActionConditionedLatentMPC",
        "hidden_target": "cartpole_mechanics_action_conditioned_world_model",
        "rows": rows,
        "decode_accuracy": dict_mean(rows, "decode_accuracy"),
        "confidence": dict_mean(rows, "confidence"),
        "no_belief_return": dict_mean(rows, "no_belief_return"),
        "belief_mpc_return": dict_mean(rows, "belief_mpc_return"),
        "oracle_mpc_return": dict_mean(rows, "oracle_mpc_return"),
        "shuffled_mpc_return": dict_mean(rows, "shuffled_mpc_return"),
        "stale_mpc_return": dict_mean(rows, "stale_mpc_return"),
        "solver_gain": dict_mean(rows, "solver_gain"),
        "content_lift": dict_mean(rows, "content_lift"),
        "oracle_gap": dict_mean(rows, "oracle_gap"),
        "belief_action_match_oracle": dict_mean(rows, "belief_action_match_oracle"),
        "no_belief_action_match_oracle": dict_mean(rows, "no_belief_action_match_oracle"),
        "belief_one_step_prediction_mse": dict_mean(rows, "belief_one_step_prediction_mse"),
        "belief_k_step_prediction_mse": dict_mean(rows, "belief_k_step_prediction_mse"),
        "no_belief_k_step_prediction_mse": dict_mean(rows, "no_belief_k_step_prediction_mse"),
        "belief_solve_rate": dict_mean(rows, "belief_solved"),
        "no_belief_solve_rate": dict_mean(rows, "no_belief_solved"),
        "oracle_solve_rate": dict_mean(rows, "oracle_solved"),
        "belief_env_samples": dict_mean(rows, "belief_env_samples"),
        "no_belief_env_samples": dict_mean(rows, "no_belief_env_samples"),
        "oracle_env_samples": dict_mean(rows, "oracle_env_samples"),
        "belief_samples_to_peak_return": dict_mean(rows, "belief_samples_to_peak_return"),
        "no_belief_samples_to_peak_return": dict_mean(rows, "no_belief_samples_to_peak_return"),
        "oracle_samples_to_peak_return": dict_mean(rows, "oracle_samples_to_peak_return"),
        "belief_samples_to_solve": _nullable_mean(rows, "belief_samples_to_solve"),
        "no_belief_samples_to_solve": _nullable_mean(rows, "no_belief_samples_to_solve"),
        "oracle_samples_to_solve": _nullable_mean(rows, "oracle_samples_to_solve"),
        "net_samples_to_solve_savings": _nullable_mean(rows, "net_samples_to_solve_savings"),
        "net_env_sample_savings": dict_mean(rows, "net_env_sample_savings"),
        "probe_steps": int(len(config.support_families) * int(config.probe_steps)),
        "control_steps": int(config.control_steps),
        "horizon": int(config.horizon),
        "candidate_count": int(config.candidate_count),
        "action_grid": list(config.action_grid),
    }


def _crawler_belief(
    truth: MechanicsWorld,
    seed: int,
    config: CartPoleLatentMPCConfig,
) -> tuple[MechanicsWorld, float]:
    observed = evidence_vector(
        truth,
        config.support_families,
        seed=seed,
        steps=config.probe_steps,
    )
    decoded, confidence, _margin = infer_world(
        observed,
        config.support_families,
        seed=seed,
        steps=config.probe_steps,
    )
    return decoded, confidence


def _evaluate_planner(
    truth: MechanicsWorld,
    belief: MechanicsWorld,
    seed: int,
    config: CartPoleLatentMPCConfig,
    model: CartPoleLatentDynamicsModel,
    planner: RandomShootingPlanner,
) -> dict[str, object]:
    state = _initial_state(seed + 7_300)
    states = [state.copy()]
    actions: list[float] = []
    total_cost = 0.0
    survival_steps = 0
    for step in range(int(config.control_steps)):
        plan = planner.choose_action(model, state, belief=belief, seed=seed, step=step)
        action = float(plan.action)
        actions.append(action)
        state = _step_physics(state, action, truth)
        states.append(state.copy())
        total_cost += _state_cost(state)
        if _failed(state, config):
            break
        survival_steps += 1
    solved = survival_steps >= int(config.control_steps)
    return {
        "return": float(-total_cost),
        "states": np.asarray(states, dtype=np.float32),
        "actions": actions,
        "survival_steps": int(survival_steps),
        "solved": bool(solved),
    }


def _prediction_mse(
    truth: MechanicsWorld,
    predicted: MechanicsWorld,
    states: object,
    actions: object,
    *,
    one_step: bool,
) -> float:
    state_arr = np.asarray(states, dtype=np.float32)
    action_arr = np.asarray(actions, dtype=np.float32).reshape(-1)
    if state_arr.shape[0] < 2 or action_arr.size == 0:
        return 0.0
    if one_step:
        errors = []
        for idx, action in enumerate(action_arr[: state_arr.shape[0] - 1]):
            predicted_next = _step_physics(state_arr[idx], float(action), predicted)
            true_next = _step_physics(state_arr[idx], float(action), truth)
            errors.append(float(np.mean(np.square(predicted_next - true_next))))
        return float(np.mean(errors)) if errors else 0.0
    predicted_rollout = [state_arr[0].copy()]
    true_rollout = [state_arr[0].copy()]
    predicted_state = state_arr[0].copy()
    true_state = state_arr[0].copy()
    for action in action_arr:
        predicted_state = _step_physics(predicted_state, float(action), predicted)
        true_state = _step_physics(true_state, float(action), truth)
        predicted_rollout.append(predicted_state.copy())
        true_rollout.append(true_state.copy())
    return float(
        np.mean(
            np.square(
                np.asarray(predicted_rollout, dtype=np.float32)
                - np.asarray(true_rollout, dtype=np.float32)
            )
        )
    )


def _samples_to_solve(
    result: dict[str, object],
    probe_cost: int,
    config: CartPoleLatentMPCConfig,
) -> float | None:
    if not bool(result.get("solved", False)):
        return None
    return float(int(probe_cost) + int(config.control_steps))


def _sample_savings(baseline: float | None, belief: float | None) -> float | None:
    if baseline is None or belief is None:
        return None
    return float(baseline - belief)


def _first_action(actions: object) -> float | None:
    if not isinstance(actions, list) or not actions:
        return None
    return float(actions[0])


def _nullable_mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean([float(value) for value in values]))


def _failed(state: np.ndarray, config: CartPoleLatentMPCConfig) -> bool:
    x = float(state[0])
    theta = float(state[2])
    return abs(x) > float(config.failure_x) or abs(theta) > float(config.failure_theta)
