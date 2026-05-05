"""Generic training-target builders for the belief world model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.components import encode_probe_modes

FutureSummaryBuilder = Callable[..., np.ndarray]
DecisionTargetBuilder = Callable[..., np.ndarray]

_FUTURE_SUMMARY_BUILDERS: dict[str, FutureSummaryBuilder] = {}
_DECISION_TARGET_BUILDERS: dict[str, DecisionTargetBuilder] = {}
_DEFAULT_TARGET_BUILDERS_REGISTERED = False


def register_future_summary_builder(target_builder_key: str, builder: FutureSummaryBuilder) -> None:
    """Register one future-summary builder under a recipe or benchmark key."""
    _FUTURE_SUMMARY_BUILDERS[str(target_builder_key)] = builder


def register_decision_target_builder(target_builder_key: str, builder: DecisionTargetBuilder) -> None:
    """Register one decision-target builder under a recipe or benchmark key."""
    _DECISION_TARGET_BUILDERS[str(target_builder_key)] = builder


def _ensure_default_target_builders_registered() -> None:
    """Load built-in recipe registrations on demand."""
    global _DEFAULT_TARGET_BUILDERS_REGISTERED
    if _DEFAULT_TARGET_BUILDERS_REGISTERED:
        return
    from ....recipes import register_default_recipe_targets

    register_default_recipe_targets()
    _DEFAULT_TARGET_BUILDERS_REGISTERED = True


def resolve_target_builder_key(
    *,
    target_builder_key: str | None = None,
    env_name: str | None = None,
) -> str | None:
    """Resolve the target-builder key while keeping `env_name` as a compatibility alias."""
    if target_builder_key:
        return str(target_builder_key)
    if env_name:
        return str(env_name)
    return None


def build_generic_affordance_targets(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    action_vocab_size: int,
) -> np.ndarray:
    """Summarize short-horizon behavior directly from the local evidence slices."""
    initial_state = states[:, 0, :]
    current_state = states[:, -2, :]
    next_state = states[:, -1, :]

    from_start_delta = next_state - initial_state
    recent_delta = next_state - current_state
    state_span = np.max(states, axis=1) - np.min(states, axis=1)
    mean_abs_state = np.mean(np.abs(states), axis=1)
    reward_summary = np.stack(
        [
            np.sum(rewards, axis=1),
            np.mean(rewards, axis=1),
            np.min(rewards, axis=1),
            np.max(rewards, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    action_hist = np.zeros((actions.shape[0], action_vocab_size), dtype=np.float32)
    for row_idx in range(actions.shape[0]):
        counts = np.bincount(actions[row_idx], minlength=action_vocab_size).astype(np.float32)
        action_hist[row_idx] = counts / max(float(np.sum(counts)), 1.0)

    terminal_summary = np.stack(
        [
            terminated.astype(np.float32),
            truncated.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return np.concatenate(
        [
            from_start_delta.astype(np.float32),
            recent_delta.astype(np.float32),
            state_span.astype(np.float32),
            mean_abs_state.astype(np.float32),
            reward_summary,
            action_hist,
            terminal_summary,
        ],
        axis=1,
    ).astype(np.float32)


def build_generic_future_summary_targets(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    action_vocab_size: int,
    probe_mode: np.ndarray | None = None,
) -> np.ndarray:
    """Summarize the suffix of a trajectory for generic contrastive prediction."""
    del probe_mode
    start_state = states[:, 0, :]
    end_state = states[:, -1, :]
    end_delta = end_state - start_state
    step_delta = np.diff(states, axis=1)
    mean_step_delta = np.mean(step_delta, axis=1)
    state_span = np.max(states, axis=1) - np.min(states, axis=1)
    reward_sum = np.sum(rewards, axis=1, dtype=np.float32).reshape(-1, 1)
    reward_mean = np.mean(rewards, axis=1, dtype=np.float32).reshape(-1, 1)
    reward_max = np.max(rewards, axis=1).reshape(-1, 1)
    reward_min = np.min(rewards, axis=1).reshape(-1, 1)

    action_hist = np.zeros((actions.shape[0], action_vocab_size), dtype=np.float32)
    for row_idx in range(actions.shape[0]):
        counts = np.bincount(actions[row_idx], minlength=action_vocab_size).astype(np.float32)
        action_hist[row_idx] = counts / max(float(np.sum(counts)), 1.0)

    terminal_summary = np.stack(
        [
            terminated.astype(np.float32),
            truncated.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return np.concatenate(
        [
            end_delta.astype(np.float32),
            mean_step_delta.astype(np.float32),
            state_span.astype(np.float32),
            reward_sum.astype(np.float32),
            reward_mean.astype(np.float32),
            reward_max.astype(np.float32),
            reward_min.astype(np.float32),
            action_hist.astype(np.float32),
            terminal_summary,
        ],
        axis=1,
    ).astype(np.float32)


def build_future_summary_targets(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    action_vocab_size: int,
    probe_mode: np.ndarray | None = None,
    env_name: str | None = None,
    target_builder_key: str | None = None,
) -> np.ndarray:
    """Build future-summary targets with optional recipe-registered specialization."""
    _ensure_default_target_builders_registered()
    resolved_key = resolve_target_builder_key(
        target_builder_key=target_builder_key,
        env_name=env_name,
    )
    builder = _FUTURE_SUMMARY_BUILDERS.get(resolved_key or "")
    if builder is not None:
        return builder(
            states=states,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            action_vocab_size=action_vocab_size,
            probe_mode=probe_mode,
        )
    return build_generic_future_summary_targets(
        states=states,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        action_vocab_size=action_vocab_size,
        probe_mode=probe_mode,
    )


def build_bipedal_decision_targets(
    states: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> np.ndarray:
    """Compatibility wrapper for the recipe-registered locomotion target builder."""
    from ...envs import BIPEDAL_WALKER_NAME

    _ensure_default_target_builders_registered()
    builder = _DECISION_TARGET_BUILDERS.get(BIPEDAL_WALKER_NAME)
    if builder is None:
        return build_generic_decision_targets(states, rewards, terminated, truncated)
    return builder(
        states=states,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
    )


def build_generic_decision_targets(
    states: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> np.ndarray:
    """Generic decision targets when no recipe-specific heuristic is available."""
    current_state = states[:, -1, :]
    state_diff = np.diff(states, axis=1)
    delta_norm = np.linalg.norm(state_diff, axis=2)
    reward_sum = np.sum(rewards, axis=1)
    reward_mid = rewards.shape[1] // 2
    reward_trend = np.mean(rewards[:, reward_mid:], axis=1) - np.mean(rewards[:, :reward_mid], axis=1)
    state_energy = np.mean(np.abs(current_state), axis=1)
    motion_energy = np.mean(delta_norm, axis=1)
    state_span = np.mean(np.max(states, axis=1) - np.min(states, axis=1), axis=1)
    fall_risk = np.logical_or(terminated, truncated).astype(np.float32)
    return np.stack(
        [
            reward_sum.astype(np.float32),
            reward_trend.astype(np.float32),
            state_energy.astype(np.float32),
            motion_energy.astype(np.float32),
            state_span.astype(np.float32),
            fall_risk.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def build_decision_targets(
    states: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    env_name: str | None = None,
    target_builder_key: str | None = None,
) -> np.ndarray:
    """Build decision targets with optional recipe-registered specialization."""
    _ensure_default_target_builders_registered()
    resolved_key = resolve_target_builder_key(
        target_builder_key=target_builder_key,
        env_name=env_name,
    )
    builder = _DECISION_TARGET_BUILDERS.get(resolved_key or "")
    if builder is not None:
        return builder(
            states=states,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
        )
    return build_generic_decision_targets(states, rewards, terminated, truncated)


def normalize_targets(values: np.ndarray) -> np.ndarray:
    """Standardize a target matrix columnwise for easier training."""
    value_mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    value_std = values.std(axis=0, keepdims=True).astype(np.float32)
    value_std = np.where(value_std < 1e-6, 1.0, value_std)
    return ((values - value_mean) / value_std).astype(np.float32)


def build_training_tensors(
    windows: dict[str, np.ndarray],
    action_vocab_size: int,
    intervention_horizon: int,
    analytic_affordances: bool = True,
    env_name: str | None = None,
    target_builder_key: str | None = None,
) -> dict[str, np.ndarray]:
    """Convert recorded windows into the tensors consumed by encoder training."""
    del intervention_horizon
    del analytic_affordances
    states = windows["states"]
    actions = windows["actions"]
    rewards = windows["rewards"]
    env_params = windows["env_params"]
    env_instance_id = windows.get(
        "env_instance_id",
        np.arange(states.shape[0], dtype=np.int32),
    )
    probe_mode = np.asarray(windows["probe_mode"], dtype="U")
    probe_mode_idx = encode_probe_modes(probe_mode)
    terminated = windows["terminated"]
    truncated = windows["truncated"]
    resolved_key = resolve_target_builder_key(
        target_builder_key=target_builder_key,
        env_name=env_name,
    )

    current_state = states[:, -2, :]
    next_state = states[:, -1, :]
    delta_state = next_state - current_state
    current_action = actions[:, -1]
    split_idx = max(2, actions.shape[1] // 2)
    prefix_states = states[:, : split_idx + 1, :]
    prefix_actions = actions[:, :split_idx]
    prefix_rewards = rewards[:, :split_idx]
    future_states = states[:, split_idx:, :]
    future_actions = actions[:, split_idx:]
    future_rewards = rewards[:, split_idx:]
    return_target = np.sum(rewards, axis=1, dtype=np.float32).reshape(-1, 1)
    risk_target = np.logical_or(terminated, truncated).astype(np.float32).reshape(-1, 1)

    target_affordances = build_generic_affordance_targets(
        states=states,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        action_vocab_size=action_vocab_size,
    )
    decision_targets = build_decision_targets(
        states=states,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        env_name=env_name,
        target_builder_key=resolved_key,
    )
    future_summary_targets = build_future_summary_targets(
        states=future_states,
        actions=future_actions,
        rewards=future_rewards,
        terminated=terminated,
        truncated=truncated,
        action_vocab_size=action_vocab_size,
        probe_mode=probe_mode,
        env_name=env_name,
        target_builder_key=resolved_key,
    )
    env_param_mean = env_params.mean(axis=0, keepdims=True).astype(np.float32)
    env_param_std = env_params.std(axis=0, keepdims=True).astype(np.float32)
    env_param_std = np.where(env_param_std < 1e-6, 1.0, env_param_std)
    normalized_env_params = ((env_params - env_param_mean) / env_param_std).astype(np.float32)

    return {
        "window_states": states.astype(np.float32),
        "window_actions": actions.astype(np.int64),
        "window_rewards": rewards.astype(np.float32),
        "env_instance_id": np.asarray(env_instance_id, dtype=np.int64),
        "probe_mode_idx": np.asarray(probe_mode_idx, dtype=np.int64),
        "prefix_states": prefix_states.astype(np.float32),
        "prefix_actions": prefix_actions.astype(np.int64),
        "prefix_rewards": prefix_rewards.astype(np.float32),
        "current_state": current_state.astype(np.float32),
        "current_action": current_action.astype(np.int64),
        "target_delta": delta_state.astype(np.float32),
        "target_env_params": normalized_env_params,
        "target_env_params_mean": env_param_mean.reshape(-1).astype(np.float32),
        "target_env_params_std": env_param_std.reshape(-1).astype(np.float32),
        "target_affordances": normalize_targets(target_affordances).astype(np.float32),
        "target_decision": normalize_targets(decision_targets).astype(np.float32),
        "target_return": normalize_targets(return_target).astype(np.float32),
        "target_risk": risk_target.astype(np.float32),
        "target_future_summary": normalize_targets(future_summary_targets).astype(np.float32),
    }
