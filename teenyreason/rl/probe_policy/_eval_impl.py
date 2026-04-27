"""Probe-policy evaluation helpers.

This module keeps deterministic evaluation and probe-surprise estimation out of
the main PPO training file so training code can stay focused on rollout and
update logic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ...crawler import CrawlerModelBundle
from ...envs import make_env
from ...models.belief_world_model import build_future_summary_targets
from ...models.env_belief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ...representation import DeltaPredictorEnsemble, WorldEncoder
from ...probe.explorer import build_probe_planner
from ...probe.probe_data import apply_env_params
from ...probe.probe_latent import (
    aggregate_env_belief,
    belief_mean_z,
    belief_uncertainty,
    collect_adaptive_probe_window,
    encode_window_posterior,
    init_recurrent_belief_hidden,
    maybe_update_online_belief,
    nearest_probe_action_idx,
    probe_group_ids_from_families,
    select_episode_physics,
    update_recurrent_belief_from_window,
)
from ..core import (
    BeliefNativeActorCritic,
    PlainGaussianActorCritic,
    ProbeConditionedGaussianActorCritic,
    RunningNormalizer,
    mean_to_continuous_action,
    sanitize_numpy,
    sanitize_tensor,
)
from .messages import build_solver_episode_belief, build_solver_episode_expression


def compute_probe_surprise(
    *,
    env_future_predictor: nn.Module | None,
    belief: np.ndarray | None,
    window_states: np.ndarray,
    window_actions: np.ndarray,
    window_rewards: np.ndarray,
    action_vocab_size: int,
    device: torch.device,
    env_name: str | None = None,
    probe_family: str | None = None,
) -> float:
    """Measure how wrong the current belief was about what a fresh probe would reveal."""
    if env_future_predictor is None or belief is None:
        return 0.0

    split_idx = max(2, window_actions.shape[0] // 2)
    future_target = build_future_summary_targets(
        states=window_states[None, split_idx:, ...],
        actions=window_actions[None, split_idx:],
        rewards=window_rewards[None, split_idx:],
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        action_vocab_size=action_vocab_size,
        probe_mode=None if probe_family is None else np.asarray([probe_family], dtype="U"),
        env_name=env_name,
    )
    belief_t = torch.tensor(
        sanitize_numpy(belief_mean_z(belief)[None, :]),
        dtype=torch.float32,
        device=device,
    )
    future_target_t = torch.tensor(
        sanitize_numpy(future_target),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        future_pred = sanitize_tensor(env_future_predictor(belief_t))
    if future_pred.shape[-1] != future_target_t.shape[-1]:
        shared_dim = min(int(future_pred.shape[-1]), int(future_target_t.shape[-1]))
        if shared_dim <= 0:
            return 0.0
        future_pred = future_pred[..., :shared_dim]
        future_target_t = future_target_t[..., :shared_dim]
    return float(torch.mean(torch.abs(future_pred - future_target_t)).item())


def compute_control_surprise(
    *,
    predictor: DeltaPredictorEnsemble | None,
    belief: np.ndarray | None,
    prev_state: np.ndarray,
    action_idx: int,
    next_state: np.ndarray,
    device: torch.device,
) -> float:
    """Measure one-step action-conditioned surprise during downstream control."""
    if predictor is None or belief is None:
        return 0.0
    belief_t = torch.tensor(
        sanitize_numpy(belief_mean_z(belief)[None, :]),
        dtype=torch.float32,
        device=device,
    )
    prev_state_t = torch.tensor(
        sanitize_numpy(prev_state[None, :]),
        dtype=torch.float32,
        device=device,
    )
    action_t = torch.tensor([int(action_idx)], dtype=torch.long, device=device)
    target_delta_t = torch.tensor(
        sanitize_numpy((next_state - prev_state)[None, :]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        predicted_delta = sanitize_tensor(
            predictor.predict_all(prev_state_t, action_t, belief_t).mean(dim=0)
        )
    return float(torch.mean(torch.abs(predicted_delta - target_delta_t)).item())


def transform_controller_context_input(
    context_input: np.ndarray,
    *,
    disable_controller_context: bool = False,
    shuffle_controller_context: bool = False,
    rng: np.random.Generator | None = None,
    stale_context_input: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one evaluation ablation to the full-system controller context."""
    base_context = sanitize_numpy(np.asarray(context_input, dtype=np.float32).reshape(-1))
    if stale_context_input is not None:
        return sanitize_numpy(np.asarray(stale_context_input, dtype=np.float32).reshape(-1))
    if disable_controller_context:
        return np.zeros_like(base_context, dtype=np.float32)
    if not shuffle_controller_context or base_context.size <= 2:
        return base_context
    rng = np.random.default_rng(0) if rng is None else rng
    shuffled = base_context.copy()
    code = shuffled[:-2].copy()
    rng.shuffle(code)
    shuffled[:-2] = code
    return sanitize_numpy(shuffled)


def policy_action_value_step(
    *,
    policy: nn.Module,
    state_t: torch.Tensor,
    context_t: torch.Tensor,
    hidden_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Run one policy step for either the matched or belief-native controller."""
    if isinstance(policy, BeliefNativeActorCritic):
        mean, value, next_hidden, aux = policy.forward_with_hidden(
            state_t,
            context_t,
            hidden_state=hidden_state,
        )
        return mean, value, next_hidden, aux
    mean, value = policy(state_t, context_t)
    return mean, value, hidden_state, {}


def evaluate_plain_policy(
    policy: PlainGaussianActorCritic,
    state_normalizer: RunningNormalizer,
    env_name: str,
    action_low: np.ndarray,
    action_high: np.ndarray,
    randomize_physics: bool,
    base_physics,
    eval_episodes: int,
    seed: int,
) -> tuple[list[float], int]:
    """Run short deterministic eval episodes before declaring the baseline solved."""
    env = make_env(env_name)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    total_steps = 0
    device = next(policy.parameters()).device

    for eval_episode in range(eval_episodes):
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        apply_env_params(env, episode_physics)
        raw_state, _info = env.reset(seed=seed + eval_episode)
        raw_state = np.asarray(raw_state, dtype=np.float32)
        done = False
        episode_return = 0.0

        while not done:
            state = sanitize_numpy(state_normalizer.normalize(raw_state))
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, _value = policy(state_t)
            action = mean_to_continuous_action(mean, action_low, action_high)
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            total_steps += 1
            raw_state = np.asarray(next_raw_state, dtype=np.float32)
            episode_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(episode_return)

    env.close()
    return returns, total_steps


def evaluate_probe_policy(
    policy: nn.Module,
    crawler_bundle: CrawlerModelBundle | None,
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    env_future_predictor: nn.Module | None,
    predictor: DeltaPredictorEnsemble | None,
    state_normalizer: RunningNormalizer,
    env_name: str,
    action_values: np.ndarray,
    window_size: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    randomize_physics: bool,
    base_physics,
    base_probe_episodes: int,
    max_probe_episodes: int,
    probe_adaptive_budget: bool,
    uncertainty_probe_threshold: float,
    surprise_probe_threshold: float,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    num_training_episodes: int,
    disable_belief_message: bool,
    freeze_env_expression_per_episode: bool,
    eval_episodes: int,
    seed: int,
    device: torch.device,
    shadow_env_expression: bool = False,
    belief_native_controller: bool = False,
    disable_controller_context: bool = False,
    shuffle_controller_context: bool = False,
    use_stale_previous_context: bool = False,
    disable_online_refinement: bool = False,
    full_system_context_source: str = "learned",
    force_message_mode: str | None = None,
    forced_expression_scale: float | None = None,
) -> tuple[list[float], int, float]:
    """Run deterministic probe-conditioned eval episodes before declaring solved."""
    env = make_env(env_name)
    probe_env = make_env(env_name)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    total_steps = 0
    total_probe_windows = 0
    stale_context_input: np.ndarray | None = None

    for eval_episode in range(eval_episodes):
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        belief = None
        belief_hidden = init_recurrent_belief_hidden(encoder=encoder, device=device)
        belief_posteriors: list[tuple[np.ndarray, np.ndarray]] = []
        probe_families: list[str | None] = []
        probe_planner = build_probe_planner(
            action_space=env.action_space,
            action_values=action_values,
            rng=rng,
        )
        if probe_planner is not None:
            probe_planner.begin_env_instance()

        probe_target_count = max(1, int(base_probe_episodes))
        probe_idx = 0
        episode_belief = None
        episode_context_input = None
        while probe_idx < probe_target_count:
            prior_belief = None if belief is None else belief.copy()
            window_states, window_actions, window_rewards, probe_failed, probe_steps_used = collect_adaptive_probe_window(
                env=probe_env,
                encoder=encoder,
                predictor=predictor,
                device=device,
                rng=rng,
                window_size=window_size,
                episode_physics=episode_physics,
                action_values=action_values,
                env_name=env_name,
                prior_belief=belief,
                prior_hidden=belief_hidden,
                planner=probe_planner,
            )
            total_steps += probe_steps_used
            if probe_failed:
                belief = None
                break
            observed_family = (
                None
                if probe_planner is None
                else getattr(probe_planner, "current_goal", None)
            )
            probe_surprise = compute_probe_surprise(
                env_future_predictor=env_future_predictor,
                belief=prior_belief,
                window_states=window_states,
                window_actions=window_actions,
                window_rewards=window_rewards,
                action_vocab_size=len(action_values),
                device=device,
                env_name=env_name,
                probe_family=observed_family,
            )
            window_posterior = encode_window_posterior(
                encoder=encoder,
                device=device,
                window_states=window_states,
                window_actions=window_actions,
                window_rewards=window_rewards,
            )
            _window_belief, belief_hidden, _posterior = update_recurrent_belief_from_window(
                encoder=encoder,
                device=device,
                belief_hidden=belief_hidden,
                window_states=window_states,
                window_actions=window_actions,
                window_rewards=window_rewards,
                prior_belief=None,
                alpha=1.0,
            )
            belief_posteriors.append(window_posterior)
            probe_families.append(observed_family)
            probe_group_ids = probe_group_ids_from_families(
                probe_families,
                family_names=None if crawler_bundle is None else crawler_bundle.family_names,
            )
            belief, payload = aggregate_env_belief(
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                device=device,
                posterior_views=belief_posteriors,
                probe_group_ids=probe_group_ids,
            )
            if crawler_bundle is not None:
                step_result = crawler_bundle.build_step_result(
                    payload=payload,
                    expected_family_gain={},
                    realized_family_gain={},
                    stop_reason=None,
                    bits_per_dim=0,
                    use_residual_sketch=False,
                )
                if belief_native_controller:
                    selected_context = step_result.controller_context
                    if full_system_context_source == "oracle" and hasattr(episode_physics, "as_array"):
                        selected_context = crawler_bundle.build_oracle_controller_context(
                            episode_physics.as_array()
                        )
                    raw_context_input = selected_context.vector
                    episode_context_input = transform_controller_context_input(
                        raw_context_input,
                        disable_controller_context=disable_controller_context,
                        shuffle_controller_context=shuffle_controller_context,
                        rng=rng,
                        stale_context_input=stale_context_input if use_stale_previous_context else None,
                    )
                    episode_belief = episode_context_input
                else:
                    episode_belief, _message_scale = build_solver_episode_expression(
                        env_expression=step_result.env_expression,
                        current_episode=max(1, int(num_training_episodes)),
                        total_episodes=max(1, int(num_training_episodes)),
                        disable_env_expression=disable_belief_message,
                        strict_fair_mode=(
                            freeze_env_expression_per_episode and (not shadow_env_expression)
                        ),
                        shadow_expression_mode=shadow_env_expression,
                        force_message_mode=force_message_mode,
                        forced_expression_scale=forced_expression_scale,
                    )
            else:
                episode_belief, _message_scale = build_solver_episode_belief(
                    belief_message=belief_mean_z(belief),
                    uncertainty_scalar=belief_uncertainty(belief),
                    uncertainty_probe_threshold=uncertainty_probe_threshold,
                    current_episode=max(1, int(num_training_episodes)),
                    total_episodes=max(1, int(num_training_episodes)),
                    disable_belief_message=disable_belief_message,
                )
            total_probe_windows += 1
            if (
                probe_adaptive_budget
                and prior_belief is not None
                and probe_idx + 1 >= probe_target_count
                and probe_target_count < max_probe_episodes
                and (
                    belief_uncertainty(belief) >= uncertainty_probe_threshold
                    or probe_surprise >= surprise_probe_threshold
                )
            ):
                probe_target_count += 1
            probe_idx += 1

        if belief is None or episode_belief is None:
            returns.append(0.0)
            continue

        apply_env_params(env, episode_physics)
        raw_state, _info = env.reset(seed=seed + eval_episode)
        raw_state = np.asarray(raw_state, dtype=np.float32)
        done = False
        episode_return = 0.0
        episode_step = 0
        controller_hidden = None
        if belief_native_controller and isinstance(policy, BeliefNativeActorCritic):
            belief_t = torch.tensor(
                sanitize_numpy(episode_belief[None, :]),
                dtype=torch.float32,
                device=device,
            )
            controller_hidden = policy.init_recurrent_state(belief_t)

        while not done:
            state = sanitize_numpy(state_normalizer.normalize(raw_state))
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            belief_t = torch.tensor(episode_belief[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, _value, controller_hidden, _aux = policy_action_value_step(
                    policy=policy,
                    state_t=state_t,
                    context_t=belief_t,
                    hidden_state=controller_hidden,
                )
            action = mean_to_continuous_action(mean, action_low, action_high)

            prev_raw_state = raw_state.copy()
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            total_steps += 1
            next_raw_state = np.asarray(next_raw_state, dtype=np.float32)
            episode_step += 1
            control_surprise = compute_control_surprise(
                predictor=predictor,
                belief=belief,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                next_state=next_raw_state,
                device=device,
            )
            belief, belief_hidden, belief_posteriors = maybe_update_online_belief(
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                device=device,
                belief_hidden=belief_hidden,
                belief_posteriors=belief_posteriors,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                reward=float(reward),
                next_state=next_raw_state,
                belief=belief,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                episode_step=episode_step,
            )
            if (
                (not disable_online_refinement)
                and (
                    (
                        (not freeze_env_expression_per_episode)
                        and episode_step % online_z_update_freq == 0
                    )
                    or (belief_native_controller and control_surprise >= surprise_probe_threshold)
                )
                and belief_posteriors
            ):
                belief, payload = aggregate_env_belief(
                    belief_aggregator=belief_aggregator,
                    env_param_predictor=env_param_predictor,
                    device=device,
                    posterior_views=belief_posteriors,
                )
                if crawler_bundle is not None:
                    step_result = crawler_bundle.build_step_result(
                        payload=payload,
                        expected_family_gain={},
                        realized_family_gain={},
                        stop_reason=None,
                        bits_per_dim=0,
                        use_residual_sketch=False,
                    )
                    if belief_native_controller:
                        selected_context = step_result.controller_context
                        if full_system_context_source == "oracle" and hasattr(episode_physics, "as_array"):
                            selected_context = crawler_bundle.build_oracle_controller_context(
                                episode_physics.as_array()
                            )
                        raw_context_input = selected_context.vector
                        episode_context_input = transform_controller_context_input(
                            raw_context_input,
                            disable_controller_context=disable_controller_context,
                            shuffle_controller_context=shuffle_controller_context,
                            rng=rng,
                            stale_context_input=stale_context_input if use_stale_previous_context else None,
                        )
                        episode_belief = episode_context_input
                        if isinstance(policy, BeliefNativeActorCritic):
                            refresh_t = torch.tensor(
                                sanitize_numpy(episode_belief[None, :]),
                                dtype=torch.float32,
                                device=device,
                            )
                            controller_hidden = policy.refresh_recurrent_state(
                                refresh_t,
                                controller_hidden,
                            )
                    else:
                        episode_belief, _message_scale = build_solver_episode_expression(
                            env_expression=step_result.env_expression,
                            current_episode=max(1, int(num_training_episodes)),
                            total_episodes=max(1, int(num_training_episodes)),
                            disable_env_expression=disable_belief_message,
                            strict_fair_mode=(
                                freeze_env_expression_per_episode and (not shadow_env_expression)
                            ),
                            shadow_expression_mode=shadow_env_expression,
                            force_message_mode=force_message_mode,
                            forced_expression_scale=forced_expression_scale,
                        )
                else:
                    episode_belief, _message_scale = build_solver_episode_belief(
                        belief_message=belief_mean_z(belief),
                        uncertainty_scalar=belief_uncertainty(belief),
                        uncertainty_probe_threshold=uncertainty_probe_threshold,
                        current_episode=max(1, int(num_training_episodes)),
                        total_episodes=max(1, int(num_training_episodes)),
                        disable_belief_message=disable_belief_message,
                    )
            raw_state = next_raw_state
            episode_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(episode_return)
        if belief_native_controller and episode_context_input is not None:
            stale_context_input = sanitize_numpy(episode_context_input.copy())

    env.close()
    probe_env.close()
    avg_probe_windows = total_probe_windows / max(int(eval_episodes), 1)
    return returns, total_steps, float(avg_probe_windows)


def evaluate_probe_message_ablations(
    *,
    policy: nn.Module,
    crawler_bundle: CrawlerModelBundle | None,
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    env_future_predictor: nn.Module | None,
    predictor: DeltaPredictorEnsemble | None,
    state_normalizer: RunningNormalizer,
    env_name: str,
    action_values: np.ndarray,
    window_size: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    randomize_physics: bool,
    base_physics,
    base_probe_episodes: int,
    max_probe_episodes: int,
    probe_adaptive_budget: bool,
    uncertainty_probe_threshold: float,
    surprise_probe_threshold: float,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    num_training_episodes: int,
    freeze_env_expression_per_episode: bool,
    eval_episodes: int,
    seed: int,
    device: torch.device,
    forced_expression_scale: float,
) -> tuple[list[float], list[float]]:
    """Evaluate the same probe policy with env expression muted and force-admitted."""
    no_env_expression_eval_returns, _no_message_steps, _no_message_probe_windows = evaluate_probe_policy(
        policy=policy,
        crawler_bundle=crawler_bundle,
        encoder=encoder,
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        env_future_predictor=env_future_predictor,
        predictor=predictor,
        state_normalizer=state_normalizer,
        env_name=env_name,
        action_values=action_values,
        window_size=window_size,
        action_low=action_low,
        action_high=action_high,
        randomize_physics=randomize_physics,
        base_physics=base_physics,
        base_probe_episodes=base_probe_episodes,
        max_probe_episodes=max_probe_episodes,
        probe_adaptive_budget=probe_adaptive_budget,
        uncertainty_probe_threshold=uncertainty_probe_threshold,
        surprise_probe_threshold=surprise_probe_threshold,
        online_z_update_alpha=online_z_update_alpha,
        online_z_update_freq=online_z_update_freq,
        num_training_episodes=num_training_episodes,
        disable_belief_message=True,
        freeze_env_expression_per_episode=freeze_env_expression_per_episode,
        shadow_env_expression=False,
        eval_episodes=eval_episodes,
        seed=seed,
        device=device,
        belief_native_controller=False,
    )
    forced_env_expression_eval_returns, _forced_message_steps, _forced_probe_windows = evaluate_probe_policy(
        policy=policy,
        crawler_bundle=crawler_bundle,
        encoder=encoder,
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        env_future_predictor=env_future_predictor,
        predictor=predictor,
        state_normalizer=state_normalizer,
        env_name=env_name,
        action_values=action_values,
        window_size=window_size,
        action_low=action_low,
        action_high=action_high,
        randomize_physics=randomize_physics,
        base_physics=base_physics,
        base_probe_episodes=base_probe_episodes,
        max_probe_episodes=max_probe_episodes,
        probe_adaptive_budget=probe_adaptive_budget,
        uncertainty_probe_threshold=uncertainty_probe_threshold,
        surprise_probe_threshold=surprise_probe_threshold,
        online_z_update_alpha=online_z_update_alpha,
        online_z_update_freq=online_z_update_freq,
        num_training_episodes=num_training_episodes,
        disable_belief_message=False,
        freeze_env_expression_per_episode=freeze_env_expression_per_episode,
        shadow_env_expression=False,
        eval_episodes=eval_episodes,
        seed=seed,
        device=device,
        belief_native_controller=False,
        force_message_mode="diag",
        forced_expression_scale=forced_expression_scale,
    )
    return no_env_expression_eval_returns, forced_env_expression_eval_returns
