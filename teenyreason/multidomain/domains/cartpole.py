"""Controlled CartPole mechanics ladder for crawler-message validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


PARAM_NAMES = ("gravity", "masspole", "length", "force")
PROBE_FAMILIES = ("passive_decay", "impulse_left", "impulse_right", "chirp")
HELDOUT_FAMILY = "stabilize_counterfactual"


@dataclass(frozen=True)
class CartPoleMechanicsConfig:
    """Small deterministic mechanics-identification benchmark."""

    seeds: tuple[int, ...] = tuple(range(16))
    support_families: tuple[str, ...] = PROBE_FAMILIES
    rollout_steps: int = 18
    belief_dim: int = 4


@dataclass(frozen=True)
class CartPoleControllerBridgeConfig:
    """Short-horizon controller handoff check using inferred mechanics."""

    seeds: tuple[int, ...] = tuple(range(16))
    support_families: tuple[str, ...] = PROBE_FAMILIES
    probe_steps: int = 18
    control_steps: int = 80
    action_grid: tuple[float, ...] = (-1.0, 0.0, 1.0)


@dataclass(frozen=True)
class MechanicsWorld:
    """One hidden CartPole mechanics setting."""

    gravity: float
    masspole: float
    length: float
    force: float

    def vector(self) -> np.ndarray:
        return np.asarray([self.gravity, self.masspole, self.length, self.force], dtype=np.float32)

    def label(self) -> str:
        bits = ["hi" if value > 1.0 else "lo" for value in self.vector()]
        return "g{0}_m{1}_l{2}_f{3}".format(*bits)


def candidate_worlds() -> list[MechanicsWorld]:
    """Return the small discrete hidden-mechanics family for the ladder."""
    values = (0.82, 1.18)
    return [
        MechanicsWorld(gravity=g, masspole=m, length=l, force=f)
        for g, m, l, f in product(values, values, values, values)
    ]


def world_for_seed(seed: int) -> MechanicsWorld:
    """Choose a deterministic hidden world for one seed."""
    worlds = candidate_worlds()
    return worlds[int(seed) % len(worlds)]


def nominal_world() -> MechanicsWorld:
    return MechanicsWorld(gravity=1.0, masspole=1.0, length=1.0, force=1.0)


def _probe_actions(family: str, steps: int) -> np.ndarray:
    if family == "passive_decay":
        return np.zeros((steps,), dtype=np.float32)
    if family == "impulse_left":
        return np.asarray([-1.0 if idx < steps // 3 else 0.0 for idx in range(steps)], dtype=np.float32)
    if family == "impulse_right":
        return np.asarray([1.0 if idx < steps // 3 else 0.0 for idx in range(steps)], dtype=np.float32)
    if family == "chirp":
        return np.asarray([(-1.0 if idx % 4 < 2 else 1.0) * (0.3 + idx / max(steps, 1)) for idx in range(steps)], dtype=np.float32)
    if family == HELDOUT_FAMILY:
        return np.asarray([0.65, -0.65, 0.35, -0.35] * ((steps // 4) + 1), dtype=np.float32)[:steps]
    raise ValueError(f"Unknown probe family: {family}")


def _initial_state(seed: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + int(seed))
    return rng.uniform(low=-0.04, high=0.04, size=(4,)).astype(np.float32)


def _step_physics(state: np.ndarray, action: float, world: MechanicsWorld) -> np.ndarray:
    x, x_dot, theta, theta_dot = [float(value) for value in state]
    gravity = 9.8 * world.gravity
    masscart = 1.0
    masspole = 0.1 * world.masspole
    total_mass = masspole + masscart
    length = 0.5 * world.length
    polemass_length = masspole * length
    force = 30.0 * world.force * float(action)
    tau = 0.02

    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    return np.asarray(
        [
            x + tau * x_dot,
            x_dot + tau * xacc,
            theta + tau * theta_dot,
            theta_dot + tau * thetaacc,
        ],
        dtype=np.float32,
    )


def rollout(world: MechanicsWorld, family: str, *, seed: int, steps: int) -> np.ndarray:
    """Run one deterministic probe rollout."""
    state = _initial_state(seed)
    states = [state.copy()]
    for action in _probe_actions(family, steps):
        state = _step_physics(state, float(action), world)
        states.append(state.copy())
    return np.asarray(states, dtype=np.float32)


def rollout_features(world: MechanicsWorld, family: str, *, seed: int, steps: int) -> np.ndarray:
    """Convert one probe rollout into compact mechanics evidence."""
    states = rollout(world, family, seed=seed, steps=steps)
    delta = states[-1] - states[0]
    velocity_span = np.max(states[:, [1, 3]], axis=0) - np.min(states[:, [1, 3]], axis=0)
    energy_proxy = np.mean(np.square(states[:, 2])) + 0.1 * np.mean(np.square(states[:, 3]))
    return np.asarray(
        [
            delta[0],
            delta[1],
            delta[2],
            delta[3],
            velocity_span[0],
            velocity_span[1],
            energy_proxy,
        ],
        dtype=np.float32,
    )


def evidence_vector(
    world: MechanicsWorld,
    families: tuple[str, ...],
    *,
    seed: int,
    steps: int,
) -> np.ndarray:
    chunks = [
        rollout_features(world, family, seed=seed, steps=steps)
        for family in families
    ]
    return np.concatenate(chunks, axis=0).astype(np.float32)


def infer_world(
    observed: np.ndarray,
    families: tuple[str, ...],
    *,
    seed: int,
    steps: int,
) -> tuple[MechanicsWorld, float, float]:
    """Infer hidden mechanics by nearest counterfactual probe signature."""
    distances: list[tuple[float, MechanicsWorld]] = []
    for candidate in candidate_worlds():
        expected = evidence_vector(candidate, families, seed=seed, steps=steps)
        distances.append((float(np.linalg.norm(observed - expected)), candidate))
    distances.sort(key=lambda item: item[0])
    best_distance, best_world = distances[0]
    second_distance = distances[1][0] if len(distances) > 1 else best_distance
    margin = second_distance - best_distance
    confidence = float(margin / max(second_distance, 1e-6))
    return best_world, confidence, margin


def transition_error(
    predicted: MechanicsWorld,
    truth: MechanicsWorld,
    *,
    seed: int,
    steps: int,
) -> float:
    """Held-out counterfactual rollout MSE for a predicted mechanics message."""
    true_states = rollout(truth, HELDOUT_FAMILY, seed=seed + 50, steps=steps)
    predicted_states = rollout(predicted, HELDOUT_FAMILY, seed=seed + 50, steps=steps)
    return float(np.mean(np.square(true_states - predicted_states)))


def controller_return(
    truth: MechanicsWorld,
    predicted: MechanicsWorld,
    *,
    seed: int,
    steps: int,
    action_grid: tuple[float, ...],
) -> float:
    """Evaluate a one-step model-predictive controller under true mechanics."""
    state = _initial_state(seed + 700)
    total_cost = 0.0
    for _step in range(int(steps)):
        action = _choose_controller_action(state, predicted, action_grid)
        state = _step_physics(state, action, truth)
        total_cost += _state_cost(state)
    return float(-total_cost)


def _choose_controller_action(
    state: np.ndarray,
    predicted: MechanicsWorld,
    action_grid: tuple[float, ...],
) -> float:
    scored = [
        (_state_cost(_step_physics(state, float(action), predicted)), float(action))
        for action in action_grid
    ]
    scored.sort(key=lambda item: item[0])
    return float(scored[0][1])


def _state_cost(state: np.ndarray) -> float:
    x, x_dot, theta, theta_dot = [float(value) for value in state]
    return float(
        0.35 * x * x
        + 0.08 * x_dot * x_dot
        + 8.0 * theta * theta
        + 0.35 * theta_dot * theta_dot
    )


def _r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    total = float(np.sum(np.square(truth - np.mean(truth, axis=0, keepdims=True))))
    residual = float(np.sum(np.square(truth - prediction)))
    if total <= 1e-8:
        return 0.0
    return float(1.0 - residual / total)


def run_cartpole_mechanics_benchmark(
    config: CartPoleMechanicsConfig | None = None,
) -> dict[str, object]:
    """Run the controlled mechanics ladder without PPO."""
    config = config or CartPoleMechanicsConfig()
    worlds = candidate_worlds()
    rows: list[dict[str, object]] = []
    truth_vectors: list[np.ndarray] = []
    predicted_vectors: list[np.ndarray] = []
    uncertainties: list[float] = []
    learned_errors: list[float] = []
    baseline_errors: list[float] = []
    zero_errors: list[float] = []
    shuffled_errors: list[float] = []
    stale_errors: list[float] = []

    for seed in config.seeds:
        truth = world_for_seed(int(seed))
        observed = evidence_vector(
            truth,
            config.support_families,
            seed=int(seed),
            steps=config.rollout_steps,
        )
        decoded, confidence, margin = infer_world(
            observed,
            config.support_families,
            seed=int(seed),
            steps=config.rollout_steps,
        )
        first_families = tuple(config.support_families[::2])
        second_families = tuple(config.support_families[1::2])
        first_decoded, _first_conf, _first_margin = infer_world(
            evidence_vector(truth, first_families, seed=int(seed), steps=config.rollout_steps),
            first_families,
            seed=int(seed),
            steps=config.rollout_steps,
        )
        second_decoded, _second_conf, _second_margin = infer_world(
            evidence_vector(truth, second_families, seed=int(seed), steps=config.rollout_steps),
            second_families,
            seed=int(seed),
            steps=config.rollout_steps,
        )
        shuffled = worlds[(worlds.index(decoded) + 5) % len(worlds)]
        stale = world_for_seed(int(seed) - 1)
        baseline = nominal_world()
        learned_error = transition_error(decoded, truth, seed=int(seed), steps=config.rollout_steps)
        baseline_error = transition_error(baseline, truth, seed=int(seed), steps=config.rollout_steps)
        zero_error = transition_error(baseline, truth, seed=int(seed), steps=config.rollout_steps)
        shuffled_error = transition_error(shuffled, truth, seed=int(seed), steps=config.rollout_steps)
        stale_error = transition_error(stale, truth, seed=int(seed), steps=config.rollout_steps)
        best_ablation_error = min(zero_error, shuffled_error, stale_error)
        truth_vector = truth.vector()
        decoded_vector = decoded.vector()
        truth_vectors.append(truth_vector)
        predicted_vectors.append(decoded_vector)
        learned_errors.append(learned_error)
        baseline_errors.append(baseline_error)
        zero_errors.append(zero_error)
        shuffled_errors.append(shuffled_error)
        stale_errors.append(stale_error)
        uncertainties.append(1.0 - confidence)
        rows.append(
            {
                "seed": int(seed),
                "hidden_rule": truth.label(),
                "decoded_rule": decoded.label(),
                "decode_accuracy": float(decoded == truth),
                "mechanics_mae": float(np.mean(np.abs(truth_vector - decoded_vector))),
                "subset_agreement": float(first_decoded == second_decoded == decoded),
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "baseline_transition_mse": baseline_error,
                "belief_transition_mse": learned_error,
                "zero_transition_mse": zero_error,
                "shuffled_transition_mse": shuffled_error,
                "stale_transition_mse": stale_error,
                "content_lift": best_ablation_error - learned_error,
                "probe_count": int(len(config.support_families)),
                "support_families": list(config.support_families),
                "margin": margin,
            }
        )

    truth_arr = np.asarray(truth_vectors, dtype=np.float32)
    pred_arr = np.asarray(predicted_vectors, dtype=np.float32)
    errors = np.asarray(learned_errors, dtype=np.float32)
    uncertainty_arr = np.asarray(uncertainties, dtype=np.float32)
    corr = 0.0
    if len(errors) > 1 and float(np.std(errors)) > 0.0 and float(np.std(uncertainty_arr)) > 0.0:
        corr = float(np.corrcoef(errors, uncertainty_arr)[0, 1])
    mean_content_lift = float(np.mean([row["content_lift"] for row in rows])) if rows else 0.0
    decode_accuracy = float(np.mean([row["decode_accuracy"] for row in rows])) if rows else 0.0
    subset_agreement = float(np.mean([row["subset_agreement"] for row in rows])) if rows else 0.0
    mechanics_r2 = _r2_score(truth_arr, pred_arr)
    return {
        "domain": "cartpole",
        "dataset": "ControlledCartPoleMechanics",
        "model_family": "CounterfactualProbeNearestMechanics",
        "probe_objective": "discriminative_mechanics_questions",
        "rows": rows,
        "artifacts": [
            {
                "raw_evidence_windows": [
                    {
                        "modality": "rl_state",
                        "query_family": family,
                        "source_id": "controlled_cartpole",
                        "intervention_cost": float(config.rollout_steps),
                        "hidden_target": {"mechanics": list(PARAM_NAMES)},
                        "outcome": {"target": "heldout_transition_prediction"},
                    }
                    for family in config.support_families
                ],
                "query_families": list(config.support_families),
                "domain_belief": {
                    "mechanics": list(PARAM_NAMES),
                    "belief_dim": int(config.belief_dim),
                },
                "uncertainty_estimate": float(np.mean(uncertainty_arr)) if len(uncertainty_arr) else 0.0,
                "hidden_rule_targets": {"mechanics": list(PARAM_NAMES)},
                "subset_agreement": subset_agreement,
                "belief_bitrate": int(config.belief_dim * 32),
            }
        ],
        "mechanics_r2": mechanics_r2,
        "mechanics_decode_accuracy": decode_accuracy,
        "subset_agreement": subset_agreement,
        "uncertainty_error_corr": corr,
        "mean_content_lift": mean_content_lift,
        "mean_belief_transition_mse": float(np.mean(errors)) if len(errors) else 0.0,
        "mean_baseline_transition_mse": float(np.mean(baseline_errors)) if baseline_errors else 0.0,
        "mean_zero_transition_mse": float(np.mean(zero_errors)) if zero_errors else 0.0,
        "mean_shuffled_transition_mse": float(np.mean(shuffled_errors)) if shuffled_errors else 0.0,
        "mean_stale_transition_mse": float(np.mean(stale_errors)) if stale_errors else 0.0,
        "support_families": list(config.support_families),
        "rollout_steps": int(config.rollout_steps),
    }


def run_cartpole_controller_bridge(
    config: CartPoleControllerBridgeConfig | None = None,
) -> dict[str, object]:
    """Check whether decoded mechanics help a held-out controller."""
    config = config or CartPoleControllerBridgeConfig()
    worlds = candidate_worlds()
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        truth = world_for_seed(int(seed))
        observed = evidence_vector(
            truth,
            config.support_families,
            seed=int(seed),
            steps=config.probe_steps,
        )
        decoded, confidence, _margin = infer_world(
            observed,
            config.support_families,
            seed=int(seed),
            steps=config.probe_steps,
        )
        first_families = tuple(config.support_families[::2])
        second_families = tuple(config.support_families[1::2])
        first_decoded, _first_conf, _first_margin = infer_world(
            evidence_vector(truth, first_families, seed=int(seed), steps=config.probe_steps),
            first_families,
            seed=int(seed),
            steps=config.probe_steps,
        )
        second_decoded, _second_conf, _second_margin = infer_world(
            evidence_vector(truth, second_families, seed=int(seed), steps=config.probe_steps),
            second_families,
            seed=int(seed),
            steps=config.probe_steps,
        )
        shuffled = worlds[(worlds.index(decoded) + 5) % len(worlds)]
        stale = world_for_seed(int(seed) - 1)
        baseline = nominal_world()
        learned_return = controller_return(
            truth,
            decoded,
            seed=int(seed),
            steps=config.control_steps,
            action_grid=config.action_grid,
        )
        baseline_return = controller_return(
            truth,
            baseline,
            seed=int(seed),
            steps=config.control_steps,
            action_grid=config.action_grid,
        )
        zero_return = baseline_return
        shuffled_return = controller_return(
            truth,
            shuffled,
            seed=int(seed),
            steps=config.control_steps,
            action_grid=config.action_grid,
        )
        stale_return = controller_return(
            truth,
            stale,
            seed=int(seed),
            steps=config.control_steps,
            action_grid=config.action_grid,
        )
        best_ablation = max(zero_return, shuffled_return, stale_return)
        rows.append(
            {
                "seed": int(seed),
                "hidden_rule": truth.label(),
                "decoded_rule": decoded.label(),
                "decode_accuracy": float(decoded == truth),
                "subset_agreement": float(first_decoded == second_decoded == decoded),
                "confidence": float(confidence),
                "baseline_return": baseline_return,
                "belief_return": learned_return,
                "zero_return": zero_return,
                "shuffled_return": shuffled_return,
                "stale_return": stale_return,
                "solver_gain": learned_return - baseline_return,
                "content_lift": learned_return - best_ablation,
                "probe_count": int(len(config.support_families)),
                "control_steps": int(config.control_steps),
            }
        )
    return {
        "domain": "cartpole",
        "dataset": "ControlledCartPoleControllerBridge",
        "model_family": "MechanicsMessage+OneStepMPC",
        "hidden_target": "cartpole_mechanics_controller_handoff",
        "rows": rows,
        "decode_accuracy": _mean_row(rows, "decode_accuracy"),
        "subset_agreement": _mean_row(rows, "subset_agreement"),
        "baseline_return": _mean_row(rows, "baseline_return"),
        "belief_return": _mean_row(rows, "belief_return"),
        "zero_return": _mean_row(rows, "zero_return"),
        "shuffled_return": _mean_row(rows, "shuffled_return"),
        "stale_return": _mean_row(rows, "stale_return"),
        "solver_gain": _mean_row(rows, "solver_gain"),
        "content_lift": _mean_row(rows, "content_lift"),
    }


def _mean_row(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))
