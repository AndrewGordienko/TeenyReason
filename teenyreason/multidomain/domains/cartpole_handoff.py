"""Cheap CartPole latent handoff checks for probe economics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contracts.decision_gate import DecisionGateInput, decision_gate_payload, evaluate_decision_delta_gate
from .cartpole import (
    CartPoleControllerBridgeConfig,
    MechanicsWorld,
    _choose_controller_action,
    _initial_state,
    _step_physics,
    candidate_worlds,
    controller_return,
    evidence_vector,
    infer_world,
    nominal_world,
    world_for_seed,
)


@dataclass(frozen=True)
class LatentControlHandoffConfig:
    """Benchmark cheap latent handoff versus dedicated probe handoff."""

    seeds: tuple[int, ...] = tuple(range(16))
    probe_steps: int = 18
    dual_use_steps: int = 24
    control_steps: int = 80
    belief_head_train_seeds: tuple[int, ...] = (101, 202, 303, 404)
    support_families: tuple[str, ...] = ("passive_decay", "impulse_left", "impulse_right", "chirp")
    action_grid: tuple[float, ...] = (-1.0, 0.0, 1.0)
    cheap_confidence_floor: float = 0.45
    action_sensitivity_floor: float = 0.05
    value_delta_floor: float = 0.10
    fallback_roi_floor: float = 0.01


@dataclass(frozen=True)
class CheapMechanicsBeliefHead:
    """Centroid belief head over passive/dual-use trajectory features."""

    labels: tuple[str, ...]
    worlds: tuple[MechanicsWorld, ...]
    centroids: np.ndarray

    def predict(self, features: np.ndarray) -> tuple[MechanicsWorld, float]:
        vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        distances = np.linalg.norm(self.centroids - vector, axis=1)
        order = np.argsort(distances)
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best
        margin = float(distances[second] - distances[best])
        confidence = float(margin / max(float(distances[second]), 1e-6))
        return self.worlds[best], confidence


def run_latent_control_handoff(
    config: LatentControlHandoffConfig | None = None,
) -> dict[str, object]:
    """Evaluate whether cheap passive evidence can replace dedicated probes."""
    config = config or LatentControlHandoffConfig()
    worlds = candidate_worlds()
    belief_head = fit_cheap_mechanics_belief_head(config)
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        truth = world_for_seed(int(seed))
        expensive = _expensive_message(truth, int(seed), config)
        cheap, cheap_confidence = _cheap_head_message(truth, int(seed), config, belief_head)
        centroid, centroid_confidence = belief_head.predict(
            _dual_use_rollout_features(truth, int(seed), config)[1]
        )
        shuffled = worlds[(worlds.index(expensive) + 5) % len(worlds)]
        stale = world_for_seed(int(seed) - 1)
        baseline = nominal_world()
        baseline_return = _return(truth, baseline, int(seed), config)
        expensive_return = _return(truth, expensive, int(seed), config)
        cheap_return = _return(truth, cheap, int(seed), config)
        shuffled_return = _return(truth, shuffled, int(seed), config)
        stale_return = _return(truth, stale, int(seed), config)
        best_ablation = max(baseline_return, shuffled_return, stale_return)
        sensitivity = _latent_sensitivity(truth, expensive, shuffled, int(seed), config)
        cheap_gate = _cartpole_context_gate(
            mode="cheap_dual_use",
            baseline_return=baseline_return,
            correct_return=cheap_return,
            shuffled_return=shuffled_return,
            stale_return=stale_return,
            solver_gain=cheap_return - baseline_return,
            content_lift=cheap_return - best_ablation,
            evidence_cost=max(float(config.dual_use_steps), 1.0),
            bits=128,
            confidence=cheap_confidence,
        )
        fallback_roi = _fallback_probe_roi(
            baseline_return=baseline_return,
            cheap_return=cheap_return,
            expensive_return=expensive_return,
            shuffled_return=shuffled_return,
            stale_return=stale_return,
            config=config,
        )
        wake_expensive = _wake_expensive_probe(
            cheap_gate=cheap_gate,
            cheap_confidence=cheap_confidence,
            fallback_roi=fallback_roi,
            sensitivity=sensitivity,
            config=config,
        )
        expensive_gate = _cartpole_context_gate(
            mode="expensive_probe",
            baseline_return=baseline_return,
            correct_return=expensive_return,
            shuffled_return=shuffled_return,
            stale_return=stale_return,
            solver_gain=expensive_return - baseline_return,
            content_lift=expensive_return - best_ablation,
            evidence_cost=max(float(_dedicated_probe_steps(config)), 1.0),
            bits=256,
            confidence=1.0,
            expensive=True,
            fallback_roi=fallback_roi,
            fallback_roi_floor=float(config.fallback_roi_floor),
        )
        selected_arm = _selected_context_arm(cheap_gate, expensive_gate, wake_expensive)
        rows.append(
            {
                "seed": int(seed),
                "hidden_rule": truth.label(),
                "expensive_decoded_rule": expensive.label(),
                "cheap_decoded_rule": cheap.label(),
                "expensive_decode_accuracy": float(expensive == truth),
                "cheap_decode_accuracy": float(cheap == truth),
                "cheap_confidence": float(cheap_confidence),
                "centroid_head_decode_accuracy": float(centroid == truth),
                "centroid_head_confidence": float(centroid_confidence),
                "baseline_return": baseline_return,
                "expensive_return": expensive_return,
                "cheap_return": cheap_return,
                "shuffled_return": shuffled_return,
                "stale_return": stale_return,
                "expensive_solver_gain": expensive_return - baseline_return,
                "cheap_solver_gain": cheap_return - baseline_return,
                "cheap_content_lift": cheap_return - best_ablation,
                "cheap_decision_gate_use_belief": bool(cheap_gate.use_belief),
                "cheap_decision_gate_reason": cheap_gate.reason,
                "cheap_decision_delta_correct_vs_best_ablation": (
                    cheap_gate.decision_delta_correct_vs_best_ablation
                ),
                "expensive_decision_gate_use_belief": bool(expensive_gate.use_belief),
                "expensive_decision_gate_reason": expensive_gate.reason,
                "expensive_decision_delta_correct_vs_best_ablation": (
                    expensive_gate.decision_delta_correct_vs_best_ablation
                ),
                "wake_expensive_probe": bool(wake_expensive),
                "fallback_probe_roi": float(fallback_roi),
                "fallback_roi_floor": float(config.fallback_roi_floor),
                "selected_context_arm": selected_arm,
                "cheap_decision_gate": decision_gate_payload(cheap_gate),
                "expensive_decision_gate": decision_gate_payload(expensive_gate),
                "expensive_dedicated_probe_steps": _dedicated_probe_steps(config),
                "cheap_dedicated_probe_steps": 0,
                "dual_use_probe_steps": int(config.dual_use_steps),
                "dual_use_probe_fraction": 1.0,
                **sensitivity,
            }
        )
    return {
        "domain": "cartpole",
        "dataset": "ControlledCartPoleLatentHandoff",
        "model_family": "CheapDualUseMechanicsBelief+OneStepMPC",
        "hidden_target": "cartpole_mechanics_policy_handoff",
        "rows": rows,
        "expensive_decode_accuracy": _mean(rows, "expensive_decode_accuracy"),
        "cheap_decode_accuracy": _mean(rows, "cheap_decode_accuracy"),
        "cheap_confidence": _mean(rows, "cheap_confidence"),
        "centroid_head_decode_accuracy": _mean(rows, "centroid_head_decode_accuracy"),
        "centroid_head_confidence": _mean(rows, "centroid_head_confidence"),
        "baseline_return": _mean(rows, "baseline_return"),
        "expensive_return": _mean(rows, "expensive_return"),
        "cheap_return": _mean(rows, "cheap_return"),
        "shuffled_return": _mean(rows, "shuffled_return"),
        "stale_return": _mean(rows, "stale_return"),
        "expensive_solver_gain": _mean(rows, "expensive_solver_gain"),
        "cheap_solver_gain": _mean(rows, "cheap_solver_gain"),
        "cheap_content_lift": _mean(rows, "cheap_content_lift"),
        "cheap_decision_gate_accept_rate": _mean(rows, "cheap_decision_gate_use_belief"),
        "expensive_decision_gate_accept_rate": _mean(rows, "expensive_decision_gate_use_belief"),
        "wake_expensive_probe_rate": _mean(rows, "wake_expensive_probe"),
        "fallback_probe_roi": _mean(rows, "fallback_probe_roi"),
        "fallback_roi_floor": float(config.fallback_roi_floor),
        "expected_expensive_fallback_count": float(np.sum([float(row["wake_expensive_probe"]) for row in rows])),
        "selected_cheap_context_fraction": _fraction(rows, "selected_context_arm", "cheap_dual_use"),
        "selected_expensive_context_fraction": _fraction(rows, "selected_context_arm", "expensive_probe"),
        "selected_baseline_fraction": _fraction(rows, "selected_context_arm", "baseline"),
        "action_change_fraction": _mean(rows, "action_change_fraction"),
        "mean_abs_action_delta": _mean(rows, "mean_abs_action_delta"),
        "value_delta_correct_vs_shuffled": _mean(rows, "value_delta_correct_vs_shuffled"),
        "expensive_dedicated_probe_steps": float(_dedicated_probe_steps(config)),
        "cheap_dedicated_probe_steps": 0.0,
        "dual_use_probe_steps": float(config.dual_use_steps),
        "dual_use_probe_fraction": 1.0,
        "dedicated_probe_steps_saved": float(_dedicated_probe_steps(config)),
        "belief_head_train_examples": int(len(candidate_worlds()) * len(config.belief_head_train_seeds)),
        "belief_head_kind": "dual_use_counterfactual",
        "candidate_belief_head_kind": "dual_use_feature_centroid",
    }


def fit_cheap_mechanics_belief_head(
    config: LatentControlHandoffConfig | None = None,
) -> CheapMechanicsBeliefHead:
    """Fit a tiny supervised belief head from normal-control trajectory features."""
    config = config or LatentControlHandoffConfig()
    worlds = tuple(candidate_worlds())
    centroids: list[np.ndarray] = []
    for world in worlds:
        features = [
            _dual_use_rollout_features(world, int(seed), config)[1]
            for seed in config.belief_head_train_seeds
        ]
        centroids.append(np.mean(np.stack(features, axis=0), axis=0).astype(np.float32))
    return CheapMechanicsBeliefHead(
        labels=tuple(world.label() for world in worlds),
        worlds=worlds,
        centroids=np.stack(centroids, axis=0).astype(np.float32),
    )


def _expensive_message(
    truth: MechanicsWorld,
    seed: int,
    config: LatentControlHandoffConfig,
) -> MechanicsWorld:
    observed = evidence_vector(
        truth,
        config.support_families,
        seed=seed,
        steps=config.probe_steps,
    )
    decoded, _confidence, _margin = infer_world(
        observed,
        config.support_families,
        seed=seed,
        steps=config.probe_steps,
    )
    return decoded


def _dual_use_message(
    truth: MechanicsWorld,
    seed: int,
    config: LatentControlHandoffConfig,
) -> MechanicsWorld:
    actions, observed = _dual_use_rollout_features(truth, seed, config)
    distances: list[tuple[float, MechanicsWorld]] = []
    for candidate in candidate_worlds():
        _candidate_actions, expected = _dual_use_rollout_features(
            candidate,
            seed,
            config,
            actions=actions,
        )
        distances.append((float(np.linalg.norm(observed - expected)), candidate))
    distances.sort(key=lambda item: item[0])
    return distances[0][1]


def _cheap_head_message(
    truth: MechanicsWorld,
    seed: int,
    config: LatentControlHandoffConfig,
    belief_head: CheapMechanicsBeliefHead,
) -> tuple[MechanicsWorld, float]:
    del belief_head
    return _dual_use_counterfactual_message(truth, seed, config)


def _dual_use_counterfactual_message(
    truth: MechanicsWorld,
    seed: int,
    config: LatentControlHandoffConfig,
) -> tuple[MechanicsWorld, float]:
    actions, observed = _dual_use_rollout_features(truth, seed, config)
    distances: list[tuple[float, MechanicsWorld]] = []
    for candidate in candidate_worlds():
        _candidate_actions, expected = _dual_use_rollout_features(
            candidate,
            seed,
            config,
            actions=actions,
        )
        distances.append((float(np.linalg.norm(observed - expected)), candidate))
    distances.sort(key=lambda item: item[0])
    best_distance, best_world = distances[0]
    second_distance = distances[1][0] if len(distances) > 1 else best_distance
    margin = second_distance - best_distance
    confidence = float(margin / max(second_distance, 1e-6))
    return best_world, confidence


def _dual_use_rollout_features(
    world: MechanicsWorld,
    seed: int,
    config: LatentControlHandoffConfig,
    *,
    actions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    state = _initial_state(seed + 900)
    states = [state.copy()]
    chosen: list[float] = []
    for step in range(int(config.dual_use_steps)):
        if actions is None:
            action = _choose_controller_action(state, nominal_world(), config.action_grid)
        else:
            action = float(actions[step])
        chosen.append(action)
        state = _step_physics(state, action, world)
        states.append(state.copy())
    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(chosen, dtype=np.float32)
    delta = states_arr[-1] - states_arr[0]
    span = np.max(states_arr[:, [1, 3]], axis=0) - np.min(states_arr[:, [1, 3]], axis=0)
    features = np.asarray(
        [
            delta[0],
            delta[1],
            delta[2],
            delta[3],
            span[0],
            span[1],
            float(np.mean(actions_arr)),
            float(np.std(actions_arr)),
        ],
        dtype=np.float32,
    )
    return actions_arr, features


def _latent_sensitivity(
    truth: MechanicsWorld,
    correct: MechanicsWorld,
    shuffled: MechanicsWorld,
    seed: int,
    config: LatentControlHandoffConfig,
) -> dict[str, float]:
    state = _initial_state(seed + 1100)
    changed = 0
    abs_delta = 0.0
    for _step in range(int(config.control_steps)):
        correct_action = _choose_controller_action(state, correct, config.action_grid)
        shuffled_action = _choose_controller_action(state, shuffled, config.action_grid)
        changed += int(correct_action != shuffled_action)
        abs_delta += abs(float(correct_action) - float(shuffled_action))
        state = _step_physics(state, correct_action, truth)
    steps = float(max(1, int(config.control_steps)))
    correct_return = _return(truth, correct, seed, config)
    shuffled_return = _return(truth, shuffled, seed, config)
    return {
        "action_change_fraction": float(changed) / steps,
        "mean_abs_action_delta": abs_delta / steps,
        "value_delta_correct_vs_shuffled": correct_return - shuffled_return,
    }


def _return(
    truth: MechanicsWorld,
    predicted: MechanicsWorld,
    seed: int,
    config: LatentControlHandoffConfig,
) -> float:
    bridge_config = CartPoleControllerBridgeConfig(
        seeds=(seed,),
        support_families=config.support_families,
        probe_steps=config.probe_steps,
        control_steps=config.control_steps,
        action_grid=config.action_grid,
    )
    return controller_return(
        truth,
        predicted,
        seed=seed,
        steps=bridge_config.control_steps,
        action_grid=bridge_config.action_grid,
    )


def _cartpole_context_gate(
    *,
    mode: str,
    baseline_return: float,
    correct_return: float,
    shuffled_return: float,
    stale_return: float,
    solver_gain: float,
    content_lift: float,
    evidence_cost: float,
    bits: int,
    confidence: float,
    expensive: bool = False,
    fallback_roi: float = 0.0,
    fallback_roi_floor: float = 0.0,
):
    return evaluate_decision_delta_gate(
        DecisionGateInput(
            domain="cartpole",
            mode=mode,
            lower_is_better=False,
            baseline_value=float(baseline_return),
            correct_value=float(correct_return),
            zero_value=float(baseline_return),
            shuffled_value=float(shuffled_return),
            stale_value=float(stale_return),
            solver_gain=float(solver_gain),
            content_lift=float(content_lift),
            evidence_cost=float(evidence_cost),
            bits=int(bits),
            confidence=float(confidence),
            uncertainty=max(0.0, 1.0 - float(confidence)),
            expensive=bool(expensive),
            fallback_roi=float(fallback_roi),
            fallback_roi_floor=float(fallback_roi_floor),
        )
    )


def _fallback_probe_roi(
    *,
    baseline_return: float,
    cheap_return: float,
    expensive_return: float,
    shuffled_return: float,
    stale_return: float,
    config: LatentControlHandoffConfig,
) -> float:
    current_best = max(float(baseline_return), float(cheap_return), float(shuffled_return), float(stale_return))
    return max(0.0, float(expensive_return) - current_best) / max(float(_dedicated_probe_steps(config)), 1.0)


def _wake_expensive_probe(
    *,
    cheap_gate,
    cheap_confidence: float,
    fallback_roi: float,
    sensitivity: dict[str, float],
    config: LatentControlHandoffConfig,
) -> bool:
    if fallback_roi < float(config.fallback_roi_floor):
        return False
    if not bool(cheap_gate.use_belief):
        return True
    if float(cheap_confidence) < float(config.cheap_confidence_floor):
        return True
    if float(sensitivity.get("action_change_fraction", 0.0)) >= float(config.action_sensitivity_floor):
        return True
    return float(sensitivity.get("value_delta_correct_vs_shuffled", 0.0)) >= float(config.value_delta_floor)


def _selected_context_arm(cheap_gate, expensive_gate, wake_expensive: bool) -> str:
    if bool(cheap_gate.use_belief):
        return "cheap_dual_use"
    if bool(wake_expensive) and bool(expensive_gate.use_belief):
        return "expensive_probe"
    return "baseline"


def _dedicated_probe_steps(config: LatentControlHandoffConfig) -> int:
    return int(len(config.support_families) * int(config.probe_steps))


def _mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))


def _fraction(rows: list[dict[str, object]], key: str, value: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([str(row.get(key, "")) == value for row in rows]))
