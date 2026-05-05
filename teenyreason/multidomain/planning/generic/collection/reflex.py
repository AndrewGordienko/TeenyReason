"""Closed-loop reflex search collector for generic continuous Gym envs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .....envs import make_env
from ...gym_mpc import TransitionBatch, assert_box_spaces
from ..config import AdvancedGymMPCConfig
from .replay import max_return_to_go, mean_or_zero, sum_or_zero, trajectory_steps
from .trajectory import ReplayTrajectory, collect_probe_trajectories, make_trajectory, trajectories_to_batch


@dataclass(frozen=True)
class ReflexPolicy:
    """Tiny feedback controller: observation plus phase features to action."""

    weights: np.ndarray
    phase_weights: np.ndarray
    bias: np.ndarray
    smoothing: float
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray

    def action(self, observation: np.ndarray, *, step: int, previous_action: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        obs_z = np.clip((obs - self.obs_mean) / np.maximum(self.obs_std, 1e-4), -5.0, 5.0)
        phase = phase_features(step, count=int(self.phase_weights.shape[0] // 2))
        normalized = np.tanh(obs_z @ self.weights + phase @ self.phase_weights + self.bias)
        raw = denormalize_actions(normalized.reshape(1, -1), self.action_low, self.action_high)[0]
        blend = float(np.clip(self.smoothing, 0.0, 0.95))
        return np.clip(blend * previous_action + (1.0 - blend) * raw, self.action_low, self.action_high).astype(np.float32)


def collect_reflex_archive_transitions(
    config: AdvancedGymMPCConfig,
) -> tuple[TransitionBatch, np.ndarray, np.ndarray, dict[str, object]]:
    """Search reusable feedback reflexes and keep real high-return rollouts."""
    bootstrap_episodes = max(1, int(config.probe_episodes))
    bootstrap, action_low, action_high = collect_probe_trajectories(config, episodes=bootstrap_episodes)
    obs_mean, obs_std = observation_stats(bootstrap)
    episode_steps = max(int(config.probe_steps), int(config.control_steps))
    rng = np.random.default_rng(int(config.seed) + 510_000)
    state = make_search_state(config, obs_mean, obs_std, action_low, action_high)
    all_reflex: list[ReplayTrajectory] = []
    best_return = max_or_zero([item.episode_return for item in bootstrap])
    generations: list[float] = []
    for generation in range(max(1, int(config.reflex_generations))):
        candidates = sample_generation(config, state, rng, generation=generation)
        evaluated: list[ReplayTrajectory] = []
        for index, policy in enumerate(candidates):
            seed = int(config.seed + 520_000 + generation * 4099 + index)
            trajectory = collect_reflex_trajectory(config, policy, seed=seed, max_steps=episode_steps)
            evaluated.append(trajectory)
            all_reflex.append(trajectory)
        state = update_search_state(state, candidates, evaluated, config)
        generation_best = max_or_zero([item.episode_return for item in evaluated])
        best_return = max(best_return, generation_best)
        generations.append(generation_best)
    selected = select_training_trajectories(bootstrap, all_reflex, config)
    batch = trajectories_to_batch(selected)
    stats = reflex_diagnostics(config, bootstrap, all_reflex, selected, generations, batch, best_return)
    return batch, action_low, action_high, stats


@dataclass(frozen=True)
class ReflexSearchState:
    mean: np.ndarray
    std: np.ndarray
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    obs_dim: int
    action_dim: int
    phase_count: int


def make_search_state(
    config: AdvancedGymMPCConfig,
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> ReflexSearchState:
    obs_dim = int(obs_mean.shape[0])
    action_dim = int(action_low.shape[0])
    phase_count = max(0, int(config.reflex_phase_features))
    param_count = obs_dim * action_dim + (2 * phase_count) * action_dim + action_dim + 1
    return ReflexSearchState(
        mean=np.zeros((param_count,), dtype=np.float32),
        std=np.full((param_count,), float(config.reflex_weight_scale), dtype=np.float32),
        obs_mean=np.asarray(obs_mean, dtype=np.float32),
        obs_std=np.asarray(obs_std, dtype=np.float32),
        action_low=np.asarray(action_low, dtype=np.float32),
        action_high=np.asarray(action_high, dtype=np.float32),
        obs_dim=obs_dim,
        action_dim=action_dim,
        phase_count=phase_count,
    )


def sample_generation(
    config: AdvancedGymMPCConfig,
    state: ReflexSearchState,
    rng: np.random.Generator,
    *,
    generation: int,
) -> list[ReflexPolicy]:
    count = max(2, int(config.reflex_population))
    params = rng.normal(state.mean, state.std, size=(count, state.mean.shape[0])).astype(np.float32)
    if generation == 0:
        params[0] = 0.0
        for idx in range(1, min(count, state.action_dim + 1)):
            params[idx] = sparse_axis_params(state, action_index=idx - 1, scale=float(config.reflex_weight_scale))
    return [policy_from_params(row, state, config) for row in params]


def sparse_axis_params(state: ReflexSearchState, *, action_index: int, scale: float) -> np.ndarray:
    params = np.zeros_like(state.mean)
    weights = np.zeros((state.obs_dim, state.action_dim), dtype=np.float32)
    for obs_index in range(min(state.obs_dim, 4)):
        sign = -1.0 if obs_index % 2 else 1.0
        weights[obs_index, int(action_index)] = sign * float(scale)
    params[: state.obs_dim * state.action_dim] = weights.reshape(-1)
    return params


def update_search_state(
    state: ReflexSearchState,
    policies: list[ReflexPolicy],
    trajectories: list[ReplayTrajectory],
    config: AdvancedGymMPCConfig,
) -> ReflexSearchState:
    if not trajectories:
        return state
    order = np.argsort(np.asarray([item.episode_return for item in trajectories], dtype=np.float32))[::-1]
    elite_count = max(1, min(int(config.reflex_elite_count), len(order)))
    elite_params = np.stack([params_from_policy(policies[int(idx)]) for idx in order[:elite_count]], axis=0)
    std = np.maximum(np.std(elite_params, axis=0), float(config.reflex_mutation_floor)).astype(np.float32)
    return ReflexSearchState(
        mean=np.mean(elite_params, axis=0).astype(np.float32),
        std=std,
        obs_mean=state.obs_mean,
        obs_std=state.obs_std,
        action_low=state.action_low,
        action_high=state.action_high,
        obs_dim=state.obs_dim,
        action_dim=state.action_dim,
        phase_count=state.phase_count,
    )


def policy_from_params(params: np.ndarray, state: ReflexSearchState, config: AdvancedGymMPCConfig) -> ReflexPolicy:
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    obs_end = state.obs_dim * state.action_dim
    phase_end = obs_end + (2 * state.phase_count) * state.action_dim
    bias_end = phase_end + state.action_dim
    weights = params[:obs_end].reshape(state.obs_dim, state.action_dim)
    phase_weights = params[obs_end:phase_end].reshape(2 * state.phase_count, state.action_dim)
    bias = params[phase_end:bias_end].reshape(state.action_dim)
    smoothing = float(np.clip(float(config.reflex_action_smoothing) + 0.15 * np.tanh(float(params[bias_end])), 0.0, 0.95))
    return ReflexPolicy(weights, phase_weights, bias, smoothing, state.obs_mean, state.obs_std, state.action_low, state.action_high)


def params_from_policy(policy: ReflexPolicy) -> np.ndarray:
    smooth = np.asarray([(float(policy.smoothing) - 0.55) / 0.15], dtype=np.float32)
    return np.concatenate([policy.weights.reshape(-1), policy.phase_weights.reshape(-1), policy.bias.reshape(-1), smooth])


def collect_reflex_trajectory(
    config: AdvancedGymMPCConfig,
    policy: ReflexPolicy,
    *,
    seed: int,
    max_steps: int,
) -> ReplayTrajectory:
    env = make_env(config.env_name, max_episode_steps=max(1, int(max_steps)))
    try:
        assert_box_spaces(env)
        obs, _info = env.reset(seed=int(seed))
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        previous = np.clip(np.zeros_like(policy.action_low, dtype=np.float32), policy.action_low, policy.action_high)
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        next_observations: list[np.ndarray] = []
        dones: list[float] = []
        for step in range(max(1, int(max_steps))):
            action = policy.action(obs, step=step, previous_action=previous)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
            done = bool(terminated or truncated)
            observations.append(obs.copy())
            actions.append(action.copy())
            rewards.append(float(reward))
            next_observations.append(next_obs.copy())
            dones.append(float(done))
            previous = action
            obs = next_obs
            if done:
                break
        return make_trajectory(
            seed=seed,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            discount=float(config.discount),
        )
    finally:
        env.close()


def select_training_trajectories(
    bootstrap: list[ReplayTrajectory],
    reflex: list[ReplayTrajectory],
    config: AdvancedGymMPCConfig,
) -> list[ReplayTrajectory]:
    if not reflex:
        return list(bootstrap)
    keep = max(int(config.reflex_elite_count), int(round(len(reflex) * float(config.reflex_keep_fraction))))
    ranked = sorted(reflex, key=lambda item: item.episode_return, reverse=True)
    floor = reflex_frontier_floor(config, bootstrap)
    above_floor = [item for item in ranked if float(item.episode_return) >= floor]
    selected = above_floor[: max(1, min(keep, len(above_floor)))]
    if not selected:
        selected = ranked[: max(1, min(int(config.reflex_elite_count), len(ranked)))]
    return list(bootstrap) + selected


def reflex_diagnostics(
    config: AdvancedGymMPCConfig,
    bootstrap: list[ReplayTrajectory],
    reflex: list[ReplayTrajectory],
    selected: list[ReplayTrajectory],
    generations: list[float],
    batch: TransitionBatch,
    best_return: float,
) -> dict[str, object]:
    bootstrap_returns = [item.episode_return for item in bootstrap]
    reflex_returns = [item.episode_return for item in reflex]
    interaction_steps = trajectory_steps(bootstrap) + trajectory_steps(reflex)
    selected_steps = trajectory_steps(selected)
    return {
        "collector": "reflex_archive",
        "collector_samples": int(batch.observations.shape[0]),
        "collector_interaction_steps": int(interaction_steps),
        "collector_episode_count": int(len(bootstrap) + len(reflex)),
        "collector_best_return": float(best_return),
        "collector_return_mean": mean_or_zero(bootstrap_returns + reflex_returns),
        "collector_solve_gap": float(float(config.solve_return) - float(best_return)),
        "reflex_bootstrap_best_return": max_or_zero(bootstrap_returns),
        "reflex_best_return": max_or_zero(reflex_returns),
        "reflex_return_mean": mean_or_zero(reflex_returns),
        "reflex_return_std": std_or_zero(reflex_returns),
        "reflex_generation_best_max": max_or_zero(generations),
        "reflex_generation_best_last": float(generations[-1]) if generations else 0.0,
        "reflex_episode_count": int(len(reflex)),
        "reflex_selected_steps": int(selected_steps),
        "reflex_selected_fraction": float(selected_steps / interaction_steps) if interaction_steps > 0 else 0.0,
        "reflex_frontier_floor": reflex_frontier_floor(config, bootstrap),
        "reflex_value_target_max": max_return_to_go(batch, discount=float(config.discount)),
        "reflex_total_return_sum": sum_or_zero(reflex_returns),
    }


def reflex_frontier_floor(config: AdvancedGymMPCConfig, bootstrap: list[ReplayTrajectory]) -> float:
    best = max_or_zero([item.episode_return for item in bootstrap])
    gap = abs(float(config.solve_return) - best)
    window = max(float(config.success_archive_frontier_floor), float(config.success_archive_frontier_gap_fraction) * gap)
    return float(best - window)


def observation_stats(trajectories: list[ReplayTrajectory]) -> tuple[np.ndarray, np.ndarray]:
    observations = np.concatenate([item.observations for item in trajectories], axis=0)
    return np.mean(observations, axis=0).astype(np.float32), (np.std(observations, axis=0) + 1e-4).astype(np.float32)


def phase_features(step: int, *, count: int) -> np.ndarray:
    if int(count) <= 0:
        return np.zeros((0,), dtype=np.float32)
    rows: list[float] = []
    for idx in range(max(1, int(count))):
        period = float(8 * (idx + 1))
        phase = 2.0 * np.pi * float(step) / period
        rows.extend([float(np.sin(phase)), float(np.cos(phase))])
    return np.asarray(rows, dtype=np.float32)


def denormalize_actions(actions: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, -1)
    return (low + 0.5 * (np.clip(actions, -1.0, 1.0) + 1.0) * (high - low)).astype(np.float32)


def max_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.max(np.asarray(rows, dtype=np.float32))) if rows else 0.0


def std_or_zero(values: object) -> float:
    rows = list(values)
    return float(np.std(np.asarray(rows, dtype=np.float32))) if rows else 0.0


__all__ = ["collect_reflex_archive_transitions"]
