"""Policy training loops for plain PPO and probe-conditioned PPO."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ...crawler import CrawlerModelBundle
from ...envs import get_action_values, make_env
from ...models.env_belief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ..core import (
    BeliefNativeActorCritic,
    PlainGaussianActorCritic,
    ProbeConditionedGaussianActorCritic,
    RunningNormalizer,
    build_episode_batch,
    evaluate_continuous_actions,
    sample_continuous_action,
    sanitize_numpy,
    set_optimizer_lr,
    update_ppo_policy,
    validate_continuous_env,
)
from .budget import (
    choose_fair_probe_family,
    choose_quota_probe_family,
    choose_next_probe_family,
    choose_seed_probe_family,
    default_probe_stop_reasons,
    desired_family_coverage_budget,
    minimum_family_coverage_ratio,
    probe_family_selection_metrics,
    rank_probe_family_candidates,
    selectable_unseen_active_probe_families,
    should_continue_probing_adaptive,
    should_require_seed_probe_family,
    should_stop_probing_fair,
)
from .eval import (
    compute_control_surprise,
    compute_probe_surprise,
    evaluate_probe_message_ablations,
    evaluate_plain_policy,
    evaluate_probe_policy,
    policy_action_value_step,
)
from ..full_system.context_support import mix_controller_contexts as _shared_mix_controller_contexts
from ..full_system.curriculum import (
    DEFAULT_PROBE_CURRICULUM_SCHEDULE,
    full_system_oracle_weight_for_episode as _shared_full_system_oracle_weight_for_episode,
    normalize_oracle_curriculum_schedule,
)
from .handoff_diagnostics import compute_online_future_diagnostics
from .logging import print_plain_episode_status, print_probe_episode_status, print_probe_failure, print_solve_event
from .audit import build_solver_expression_audit, solver_message_content
from .messages import (
    DEFAULT_FORCED_EVAL_EXPRESSION_SCALE,
    apply_solver_expression_keep_scale,
    apply_solver_message_keep_scale,
    build_solver_episode_expression,
    build_solver_episode_belief,
    compute_solver_message_scale,
    fair_env_expression_enabled,
    sample_solver_training_message_keep_scale,
    shadow_env_expression_diagnostics,
    solver_belief_input_from_message,
)
from .types import MatchedEvalSummary, TrainingRunResult
from .reporting import (
    average_family_scalar_counter,
    average_family_score_counter,
    default_family_metric_counter,
    default_family_score_counter,
    format_peer_solve_status,
    restore_normalizer_state,
    snapshot_normalizer_state,
    snapshot_policy_state_dict,
    update_family_scalar_counter,
    update_family_score_counter,
)
from .rollouts import (
    RolloutChunk,
    append_rollout_step,
    build_rollout_batch,
    clear_rollout_chunk,
    init_rollout_chunk,
    rollout_chunk_step_count,
)
from ...probe.probe_data import apply_env_params, default_env_params
from ...probe.explorer import build_probe_planner
from ...probe.probe_latent import (
    EliteTrajectoryBuffer,
    LatentPerformanceMemory,
    adjust_entropy_coef,
    belief_mean_z,
    choose_policy_epochs,
    choose_probe_count,
    collect_adaptive_probe_window,
    encode_window_posterior,
    aggregate_env_belief,
    init_recurrent_belief_hidden,
    maybe_update_online_belief,
    nearest_probe_action_idx,
    probe_group_ids_from_families,
    sanitize_belief_vector,
    select_episode_physics,
    should_promote_episode_to_elite,
    update_recurrent_belief_from_window,
)
from ...representation import DeltaPredictorEnsemble, WorldEncoder


def normalize_full_system_curriculum_schedule(
    schedule: list[tuple[int, float]] | tuple[tuple[int, float], ...] | None,
) -> tuple[tuple[int, float], ...]:
    """Return one boring, explicit oracle-weight schedule for curriculum mode."""
    return normalize_oracle_curriculum_schedule(
        schedule,
        default_schedule=DEFAULT_PROBE_CURRICULUM_SCHEDULE,
    )


def full_system_oracle_weight_for_episode(
    *,
    context_source: str,
    current_episode: int,
    curriculum_schedule: tuple[tuple[int, float], ...],
) -> float:
    """Resolve how much oracle context should be mixed in this episode."""
    return _shared_full_system_oracle_weight_for_episode(
        context_source=context_source,
        current_episode=current_episode,
        curriculum_schedule=curriculum_schedule,
    )


def mix_controller_contexts(
    learned_context,
    oracle_context,
    *,
    oracle_weight: float,
):
    """Blend learned and oracle controller contexts for curriculum training."""
    return _shared_mix_controller_contexts(
        learned_context,
        oracle_context,
        oracle_weight=oracle_weight,
    )


def rolling_solve_average(returns: list[float], window: int) -> float | None:
    """Return the current solve average once the requested window is full."""
    active_window = max(1, int(window))
    if len(returns) < active_window:
        return None
    return float(np.mean(np.asarray(returns[-active_window:], dtype=np.float32)))


def maybe_extend_solve_episode_limit(
    *,
    episode_return: float,
    episode: int,
    episode_limit: int,
    max_episode_limit: int,
    solved_return: float,
    solve_avg_window: int,
) -> int:
    """Give a late threshold hit enough room to become a rolling-average solve."""
    if int(solve_avg_window) <= 1:
        return int(episode_limit)
    if float(episode_return) < float(solved_return):
        return int(episode_limit)

    needed_limit = int(episode) + max(1, int(solve_avg_window)) - 1
    return min(max(int(episode_limit), needed_limit), int(max_episode_limit))


def late_exploitation_entropy_coef(
    *,
    base_entropy_coef: float,
    returns: list[float],
    best_return_so_far: float,
    solved_return: float,
) -> float:
    """Back off exploration once a continuous-control policy is near solved."""
    base_entropy_coef = float(base_entropy_coef)
    if not returns or solved_return <= 0.0:
        return base_entropy_coef

    recent_window = returns[-20:]
    recent_return = float(np.mean(np.asarray(recent_window, dtype=np.float32)))
    best_ratio = float(best_return_so_far) / max(float(solved_return), 1e-6)
    recent_ratio = recent_return / max(float(solved_return), 1e-6)
    progress_ratio = max(best_ratio, recent_ratio)
    if progress_ratio >= 0.95 and recent_ratio >= 0.60:
        return max(1e-5, 0.15 * base_entropy_coef)
    if progress_ratio >= 0.90 and recent_ratio >= 0.50:
        return max(1e-5, 0.30 * base_entropy_coef)
    if progress_ratio >= 0.80 and recent_ratio >= 0.30:
        return max(1e-5, 0.60 * base_entropy_coef)
    return base_entropy_coef


def cap_late_exploitation_action_std(
    *,
    policy: nn.Module,
    returns: list[float],
    best_return_so_far: float,
    solved_return: float,
) -> None:
    """Trim sampling noise after the policy has shown a near-solve gait."""
    if not returns or solved_return <= 0.0 or not hasattr(policy, "log_std"):
        return

    recent_window = returns[-20:]
    recent_return = float(np.mean(np.asarray(recent_window, dtype=np.float32)))
    best_ratio = float(best_return_so_far) / max(float(solved_return), 1e-6)
    recent_ratio = recent_return / max(float(solved_return), 1e-6)
    max_log_std = None
    if best_ratio >= 0.95 and recent_ratio >= 0.60:
        max_log_std = -1.25
    elif best_ratio >= 0.90 and recent_ratio >= 0.50:
        max_log_std = -1.00
    elif best_ratio >= 0.80 and recent_ratio >= 0.30:
        max_log_std = -0.75
    if max_log_std is None:
        return

    with torch.no_grad():
        policy.log_std.clamp_(min=-5.0, max=float(max_log_std))


def compute_sil_loss(
    model: ProbeConditionedGaussianActorCritic,
    elite_buffer: EliteTrajectoryBuffer,
    batch_size: int,
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the self-imitation auxiliary loss from elite replay samples."""
    # Self-imitation learning replays especially good trajectories alongside PPO.
    states, beliefs, actions, returns_to_go, sample_weight = elite_buffer.sample(batch_size)
    state_t = torch.tensor(sanitize_numpy(states), dtype=torch.float32, device=device)
    belief_t = torch.tensor(sanitize_numpy(beliefs), dtype=torch.float32, device=device)
    action_t = torch.tensor(sanitize_numpy(actions), dtype=torch.float32, device=device)
    return_t = torch.tensor(sanitize_numpy(returns_to_go), dtype=torch.float32, device=device)
    weight_t = torch.tensor(sanitize_numpy(sample_weight), dtype=torch.float32, device=device)

    mean, value = model(state_t, belief_t)
    log_prob, _entropy = evaluate_continuous_actions(
        mean=mean,
        log_std=model.log_std,
        actions=action_t,
        action_low=action_low,
        action_high=action_high,
    )
    positive_advantage = torch.clamp(return_t - value.detach(), min=0.0)
    sil_policy_loss = -(log_prob * positive_advantage * weight_t).mean()
    sil_value_gap = torch.clamp(return_t - value, min=0.0)
    sil_value_loss = 0.5 * (sil_value_gap.pow(2) * weight_t).mean()
    return sil_policy_loss, sil_value_loss


def train_plain_ppo(
    env_name: str,
    num_episodes: int = 300,
    belief_dim: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    value_clip_ratio: float | None = 0.2,
    ppo_epochs: int = 10,
    minibatch_size: int = 64,
    value_loss_weight: float = 0.5,
    entropy_coef: float = 0.005,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.02,
    min_rollout_steps: int = 256,
    lr_anneal: bool = True,
    hidden_dim: int = 128,
    initial_log_std: float = -1.5,
    normalize_rewards: bool = False,
    elite_capacity: int = 20000,
    sil_batch_size: int = 64,
    sil_policy_weight: float = 0.10,
    sil_value_weight: float = 0.10,
    min_elite_return: float = 200.0,
    elite_warmup_episodes: int = 25,
    elite_threshold_std_scale: float = 1.5,
    seed: int = 0,
    randomize_physics: bool = False,
    solved_return: float = 500.0,
    solve_avg_window: int = 1,
    solve_grace_episodes: int = 0,
    solve_eval_episodes: int = 3,
    late_exploitation_enabled: bool = True,
    run_index: int = 1,
    total_runs: int = 1,
    variant_label: str = "baseline",
    peer_variant_label: str = "probe",
    peer_solved_episode: int | None = None,
    trace_writer=None,
) -> TrainingRunResult:
    """Train the plain PPO baseline with no probe conditioning."""
    env = make_env(env_name)
    action_low, action_high = validate_continuous_env(env)
    rng = np.random.default_rng(seed)
    base_physics = default_env_params(env_name, env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = PlainGaussianActorCritic(
        state_dim,
        action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
        initial_log_std=initial_log_std,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    state_normalizer = RunningNormalizer(state_dim)
    reward_normalizer = RunningNormalizer(1, clip=10.0) if normalize_rewards else None
    elite_buffer = EliteTrajectoryBuffer(elite_capacity)
    returns = []
    best_return_so_far = float("-inf")
    best_policy_state_dict = snapshot_policy_state_dict(policy)
    best_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
    best_episode = None
    best_env_steps = None
    solved_episode = None
    solved_env_steps = None
    solve_policy_state_dict = None
    solve_state_normalizer_state = None
    solve_eval_returns = None
    total_env_steps = 0
    rollout_chunk = init_rollout_chunk(track_beliefs=False)
    episode_limit = max(1, int(num_episodes))
    max_episode_limit = episode_limit + max(0, int(solve_grace_episodes))
    anneal_episode_count = max_episode_limit if solve_grace_episodes > 0 else episode_limit

    episode = 1
    while episode <= episode_limit:
        if trace_writer is not None and episode == 1:
            trace_writer.set_stage(
                "baseline_training",
                "Baseline Controller",
                "Training the plain controller with no belief message so we have a clean control reference.",
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant=variant_label,
            )
        if lr_anneal:
            progress_left = max(0.0, 1.0 - float(episode - 1) / max(float(anneal_episode_count), 1.0))
            set_optimizer_lr(optimizer, lr * progress_left)
        episode_entropy_coef = float(entropy_coef)
        if late_exploitation_enabled:
            episode_entropy_coef = late_exploitation_entropy_coef(
                base_entropy_coef=episode_entropy_coef,
                returns=returns,
                best_return_so_far=best_return_so_far,
                solved_return=solved_return,
            )
            cap_late_exploitation_action_std(
                policy=policy,
                returns=returns,
                best_return_so_far=best_return_so_far,
                solved_return=solved_return,
            )
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        apply_env_params(env, episode_physics)
        raw_state, _info = env.reset()
        raw_state = np.asarray(raw_state, dtype=np.float32)
        state_normalizer.update(raw_state)
        state = sanitize_numpy(state_normalizer.normalize(raw_state))

        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_values = []
        episode_terminals = []
        episode_return = 0.0
        done = False
        last_next_state = state.copy()
        last_terminated = False

        while not done:
            # Roll out one baseline policy step using only the normalized state.
            state_t = torch.tensor(sanitize_numpy(state[None, :]), dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, value = policy(state_t)
            action, log_prob = sample_continuous_action(
                mean=mean,
                log_std=policy.log_std,
                action_low=action_low,
                action_high=action_high,
            )
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            total_env_steps += 1
            next_raw_state = np.asarray(next_raw_state, dtype=np.float32)
            if trace_writer is not None:
                trace_writer.record_policy_step(
                    phase="baseline_control",
                    variant=variant_label,
                    state=raw_state,
                    action_value=float(np.asarray(action, dtype=np.float32).reshape(-1)[0]),
                    reward=float(reward),
                    episode=episode,
                    step_idx=len(episode_rewards),
                    episode_return=episode_return + float(reward),
                )
            state_normalizer.update(next_raw_state)
            next_state = sanitize_numpy(state_normalizer.normalize(next_raw_state))
            raw_reward = float(reward)
            train_reward = raw_reward
            if reward_normalizer is not None:
                reward_normalizer.update(np.asarray([[raw_reward]], dtype=np.float32))
                train_reward = float(
                    reward_normalizer.scale_only(np.asarray([raw_reward], dtype=np.float32))[0]
                )

            episode_states.append(sanitize_numpy(state.copy()))
            episode_actions.append(action.copy())
            episode_log_probs.append(log_prob)
            episode_rewards.append(train_reward)
            episode_values.append(float(value.item()))
            episode_terminals.append(float(terminated))
            append_rollout_step(
                rollout_chunk,
                state=state,
                action=action,
                log_prob=log_prob,
                reward=train_reward,
                value=float(value.item()),
                terminal=float(terminated),
            )

            episode_return += raw_reward
            state = next_state
            last_next_state = next_state.copy()
            last_terminated = bool(terminated)
            done = bool(terminated or truncated)

            if rollout_chunk_step_count(rollout_chunk) >= min_rollout_steps or (
                done and episode == episode_limit
            ):
                chunk_bootstrap_value = 0.0
                if not terminated:
                    next_state_t = torch.tensor(
                        sanitize_numpy(last_next_state[None, :]),
                        dtype=torch.float32,
                        device=device,
                    )
                    with torch.no_grad():
                        _mean, next_value = policy(next_state_t)
                    chunk_bootstrap_value = float(next_value.item())

                rollout_batch = build_rollout_batch(
                    rollout_chunk,
                    bootstrap_value=chunk_bootstrap_value,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                )
                auxiliary_loss_fn = None
                if len(elite_buffer) >= sil_batch_size:
                    def auxiliary_loss_fn():
                        states_np, _beliefs_np, actions_np, returns_np, weights_np = elite_buffer.sample(sil_batch_size)
                        state_t = torch.tensor(sanitize_numpy(states_np), dtype=torch.float32, device=device)
                        action_t = torch.tensor(sanitize_numpy(actions_np), dtype=torch.float32, device=device)
                        return_t = torch.tensor(sanitize_numpy(returns_np), dtype=torch.float32, device=device)
                        weight_t = torch.tensor(sanitize_numpy(weights_np), dtype=torch.float32, device=device)
                        mean, value = policy(state_t)
                        log_prob, _entropy = evaluate_continuous_actions(
                            mean=mean,
                            log_std=policy.log_std,
                            actions=action_t,
                            action_low=action_low,
                            action_high=action_high,
                        )
                        positive_advantage = torch.clamp(return_t - value.detach(), min=0.0)
                        sil_policy_loss = -(log_prob * positive_advantage * weight_t).mean()
                        sil_value_gap = torch.clamp(return_t - value, min=0.0)
                        sil_value_loss = 0.5 * (sil_value_gap.pow(2) * weight_t).mean()
                        return (
                            sil_policy_weight * sil_policy_loss
                            + sil_value_weight * sil_value_loss
                        )

                update_ppo_policy(
                    model=policy,
                    optimizer=optimizer,
                    batch=rollout_batch,
                    action_low=action_low,
                    action_high=action_high,
                    clip_ratio=clip_ratio,
                    value_loss_weight=value_loss_weight,
                    entropy_coef=episode_entropy_coef,
                    ppo_epochs=ppo_epochs,
                    minibatch_size=minibatch_size,
                    max_grad_norm=max_grad_norm,
                    target_kl=target_kl,
                    value_clip_ratio=value_clip_ratio,
                    auxiliary_loss_fn=auxiliary_loss_fn,
                    expression_consistency_weight=0.10,
                    expression_consistency_threshold=0.35,
                )
                clear_rollout_chunk(rollout_chunk)

        bootstrap_value = 0.0
        if not last_terminated:
            next_state_t = torch.tensor(sanitize_numpy(last_next_state[None, :]), dtype=torch.float32, device=device)
            with torch.no_grad():
                _mean, next_value = policy(next_state_t)
            bootstrap_value = float(next_value.item())

        batch = build_episode_batch(
            states=episode_states,
            actions=episode_actions,
            log_probs=episode_log_probs,
            rewards=episode_rewards,
            values=episode_values,
            terminals=episode_terminals,
            bootstrap_value=bootstrap_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        is_elite_episode, elite_threshold = should_promote_episode_to_elite(
            episode_return=episode_return,
            completed_returns=returns,
            best_return_so_far=best_return_so_far,
            min_elite_return=min_elite_return,
            current_episode=episode,
            warmup_episodes=elite_warmup_episodes,
            std_scale=elite_threshold_std_scale,
        )
        if is_elite_episode:
            episode_weight = min(3.0, max(1.0, episode_return / max(elite_threshold, 1.0)))
            elite_buffer.push_episode(
                states=batch.states,
                beliefs=np.zeros((len(batch.states), 1), dtype=np.float32),
                actions=batch.actions,
                returns_to_go=batch.returns,
                episode_weight=episode_weight,
            )

        returns.append(episode_return)
        new_best = episode_return >= best_return_so_far
        if new_best:
            best_policy_state_dict = snapshot_policy_state_dict(policy)
            best_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
            best_episode = episode
            best_env_steps = total_env_steps
        best_return_so_far = max(best_return_so_far, episode_return)
        avg_10 = np.mean(returns[-10:])
        current_solved_episode = solved_episode
        current_solved_env_steps = solved_env_steps
        del current_solved_env_steps
        print_plain_episode_status(
            run_index=run_index,
            total_runs=total_runs,
            seed=seed,
            variant_label=variant_label,
            episode=episode,
            episode_return=episode_return,
            avg10=float(avg_10),
            best_return=float(best_return_so_far),
            total_env_steps=total_env_steps,
            solved_episode=current_solved_episode,
            peer_status=format_peer_solve_status(peer_variant_label, peer_solved_episode),
            new_best=bool(new_best),
        )
        if trace_writer is not None:
            trace_writer.record_episode_summary(
                variant=variant_label,
                episode=episode,
                episode_return=episode_return,
                avg10=float(avg_10),
                avg50=float(np.mean(returns[-50:])),
                total_env_steps=total_env_steps,
            )

        episode_limit = maybe_extend_solve_episode_limit(
            episode_return=episode_return,
            episode=episode,
            episode_limit=episode_limit,
            max_episode_limit=max_episode_limit,
            solved_return=solved_return,
            solve_avg_window=solve_avg_window,
        )
        solve_average = rolling_solve_average(returns, solve_avg_window)
        if solve_average is not None and solve_average >= float(solved_return):
            solved_episode = episode
            solved_env_steps = total_env_steps
            solve_policy_state_dict = snapshot_policy_state_dict(policy)
            solve_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
            print_solve_event(
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant_label=variant_label,
                episode=episode,
                total_env_steps=total_env_steps,
                episode_return=float(solve_average),
            )
            break
        episode += 1

    if solve_avg_window <= 1 and solve_eval_episodes > 0 and best_episode is not None:
        eval_policy = PlainGaussianActorCritic(
            state_dim,
            action_dim,
            belief_dim=belief_dim,
            hidden_dim=hidden_dim,
            initial_log_std=initial_log_std,
        ).to(device)
        eval_policy.load_state_dict(best_policy_state_dict)
        eval_policy.eval()
        eval_normalizer = restore_normalizer_state(state_dim, best_state_normalizer_state)
        solve_eval_returns, _eval_steps = evaluate_plain_policy(
            policy=eval_policy,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_low=action_low,
            action_high=action_high,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            eval_episodes=solve_eval_episodes,
            seed=seed,
        )
        solve_eval_best_return = float(np.max(np.asarray(solve_eval_returns, dtype=np.float32)))
        if solve_eval_best_return >= float(solved_return):
            solved_episode = int(best_episode)
            solved_env_steps = int(best_env_steps or total_env_steps)
            solve_policy_state_dict = best_policy_state_dict
            solve_state_normalizer_state = best_state_normalizer_state
            print_solve_event(
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant_label=variant_label,
                episode=int(best_episode),
                total_env_steps=int(solved_env_steps),
                episode_return=solve_eval_best_return,
            )

    env.close()
    return TrainingRunResult(
        policy=policy,
        returns=returns,
        state_normalizer=state_normalizer,
        solved_episode=solved_episode,
        solved_env_steps=solved_env_steps,
        total_env_steps=total_env_steps,
        best_policy_state_dict=best_policy_state_dict,
        best_state_normalizer_state=best_state_normalizer_state,
        best_return=best_return_so_far,
        best_episode=best_episode,
        solve_policy_state_dict=solve_policy_state_dict,
        solve_state_normalizer_state=solve_state_normalizer_state,
        solve_eval_returns=solve_eval_returns,
        solve_probe_count=None,
        probe_env_steps_total=0,
        control_env_steps_total=total_env_steps,
        probe_windows_total=0,
        probe_stop_reasons=None,
        probe_family_expected_gain=None,
        probe_family_realized_gain=None,
        probe_family_future_error=None,
        probe_family_selection_count=None,
        last_probe_stop_reason=None,
        solve_probe_stop_reason=None,
        env_expression_eval_returns=solve_eval_returns,
        no_env_expression_eval_returns=None,
        env_expression_ablation_delta=None,
        forced_env_expression_eval_returns=None,
        forced_env_expression_ablation_delta=None,
        forced_env_expression_scale=None,
        post_expression_env_steps_total=total_env_steps,
        post_expression_episode_count=solved_episode,
    )


def train_probe_conditioned_ppo(
    env_name: str,
    crawler_bundle: CrawlerModelBundle | None = None,
    encoder: WorldEncoder | None = None,
    belief_aggregator: EnvBeliefAggregator | None = None,
    env_param_predictor: EnvParamPredictorEnsemble | None = None,
    env_future_predictor: nn.Module | None = None,
    predictor: DeltaPredictorEnsemble | None = None,
    device: torch.device | None = None,
    num_episodes: int = 300,
    window_size: int = 8,
    action_bins: int = 9,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    value_clip_ratio: float | None = 0.2,
    ppo_epochs: int = 10,
    minibatch_size: int = 64,
    value_loss_weight: float = 0.5,
    entropy_coef: float = 0.005,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.02,
    min_rollout_steps: int = 256,
    lr_anneal: bool = True,
    hidden_dim: int = 128,
    initial_log_std: float = -1.5,
    normalize_rewards: bool = False,
    seed: int = 0,
    randomize_physics: bool = False,
    latent_memory_capacity: int = 512,
    base_probe_episodes: int = 1,
    max_probe_episodes: int = 3,
    benchmark_mode: str = "fair",
    probe_budget_mode: str = "fair_two_probe_handoff",
    probe_adaptive_budget: bool = False,
    probe_adaptive_policy_schedule: bool = False,
    belief_bits_per_dim: int = 0,
    belief_use_residual_sketch: bool = False,
    novelty_probe_threshold: float = 0.12,
    low_return_probe_threshold: float = 180.0,
    exploit_return_threshold: float = 260.0,
    uncertainty_probe_threshold: float = 0.20,
    uncertainty_focus_threshold: float = 0.18,
    surprise_probe_threshold: float = 0.75,
    belief_message_dropout_prob: float = 0.08,
    online_z_update_alpha: float = 0.25,
    online_z_update_freq: int = 4,
    elite_capacity: int = 20000,
    sil_batch_size: int = 64,
    sil_policy_weight: float = 0.10,
    sil_value_weight: float = 0.10,
    min_elite_return: float = 200.0,
    elite_warmup_episodes: int = 25,
    elite_threshold_std_scale: float = 1.5,
    solved_return: float = 500.0,
    solve_avg_window: int = 1,
    solve_grace_episodes: int = 0,
    solve_eval_episodes: int = 3,
    run_index: int = 1,
    total_runs: int = 1,
    variant_label: str = "probe",
    peer_variant_label: str = "baseline",
    peer_solved_episode: int | None = None,
    disable_env_expression: bool = False,
    shadow_env_expression: bool = False,
    belief_native_controller: bool = False,
    full_system_online_refinement: bool = True,
    full_system_surprise_refresh_threshold: float | None = None,
    full_system_context_source: str = "learned",
    full_system_context_chunk_len: int = 32,
    full_system_context_zero_prob: float = 0.20,
    full_system_context_shuffle_prob: float = 0.10,
    full_system_context_stale_prob: float = 0.10,
    full_system_curriculum_schedule: list[tuple[int, float]] | tuple[tuple[int, float], ...] | None = None,
    trace_writer=None,
) -> TrainingRunResult:
    """Train PPO with one compact env expression built from active probe trajectories."""
    if crawler_bundle is not None:
        encoder = crawler_bundle.encoder
        belief_aggregator = crawler_bundle.belief_aggregator
        env_param_predictor = crawler_bundle.env_param_predictor
        env_future_predictor = crawler_bundle.env_future_predictor
        predictor = crawler_bundle.predictor
        device = crawler_bundle.device
    if encoder is None or belief_aggregator is None or device is None:
        raise ValueError("Probe-conditioned PPO needs either a crawler bundle or explicit crawler models.")

    train_env = make_env(env_name)
    probe_env = make_env(env_name)
    action_low, action_high = validate_continuous_env(train_env)
    action_values = get_action_values(train_env, action_bins, env_name=env_name)
    if action_values is None:
        raise ValueError("Probe-conditioned PPO expects a continuous control env")

    rng = np.random.default_rng(seed)
    base_physics = default_env_params(env_name, train_env)
    state_dim = train_env.observation_space.shape[0]

    with torch.no_grad():
        belief_dim = (
            int(crawler_bundle.full_system_controller_dim)
            if belief_native_controller and crawler_bundle is not None
            else (
                int(crawler_bundle.env_expression_dim) + 2
                if crawler_bundle is not None
                else int(encoder.z_dim) + 2
            )
        )

    if belief_native_controller:
        policy = BeliefNativeActorCritic(
            state_dim=state_dim,
            action_dim=train_env.action_space.shape[0],
            mechanics_dim=int(encoder.z_dim),
            affordance_dim=int(encoder.z_dim),
            hidden_dim=hidden_dim,
            initial_log_std=initial_log_std,
        ).to(device)
    else:
        policy = ProbeConditionedGaussianActorCritic(
            state_dim=state_dim,
            action_dim=train_env.action_space.shape[0],
            belief_dim=belief_dim,
            hidden_dim=hidden_dim,
            initial_log_std=initial_log_std,
        ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    state_normalizer = RunningNormalizer(state_dim)
    reward_normalizer = RunningNormalizer(1, clip=10.0) if normalize_rewards else None
    performance_memory = LatentPerformanceMemory(latent_memory_capacity)
    elite_buffer = EliteTrajectoryBuffer(elite_capacity)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    belief_aggregator.eval()
    if env_param_predictor is not None:
        env_param_predictor.eval()
    if env_future_predictor is not None:
        env_future_predictor.eval()
    fair_mode = benchmark_mode == "fair"
    if shadow_env_expression and disable_env_expression:
        raise ValueError("Shadow env expression mode cannot be combined with disable_env_expression.")
    if belief_native_controller and (shadow_env_expression or disable_env_expression):
        raise ValueError(
            "Belief-native controller mode is separate from the strict env-expression benchmark variants."
        )
    if belief_native_controller and full_system_context_source not in {"learned", "oracle", "curriculum"}:
        raise ValueError(
            f"Unsupported full-system context source: {full_system_context_source}"
        )
    full_system_surprise_refresh_threshold = (
        surprise_probe_threshold
        if full_system_surprise_refresh_threshold is None
        else float(full_system_surprise_refresh_threshold)
    )
    full_system_context_chunk_len = max(1, int(full_system_context_chunk_len))
    full_system_curriculum_schedule = normalize_full_system_curriculum_schedule(
        full_system_curriculum_schedule
    )
    fixed_cap_early_stop = fair_mode or probe_budget_mode in {
        "fixed_cap_early_stop",
        "fair_two_probe_handoff",
    }
    adaptive_expand = (not fair_mode) and (probe_budget_mode == "adaptive_expand" or probe_adaptive_budget)
    adaptive_policy_schedule = (not fair_mode) and probe_adaptive_policy_schedule
    effective_max_probe_episodes = (
        min(int(max_probe_episodes), 2)
        if fair_mode
        else int(max_probe_episodes)
    )

    returns = []
    best_return_so_far = float("-inf")
    best_policy_state_dict = snapshot_policy_state_dict(policy)
    best_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
    best_episode = None
    best_env_steps = None
    best_probe_count = None
    best_probe_stop_reason = None
    best_fair_handoff_probe_families = None
    solved_episode = None
    solved_env_steps = None
    solve_policy_state_dict = None
    solve_state_normalizer_state = None
    solve_eval_returns = None
    no_env_expression_eval_returns = None
    env_expression_ablation_delta = None
    forced_env_expression_eval_returns = None
    forced_env_expression_ablation_delta = None
    message_ablation_config_diff = None
    zero_context_eval_returns = None
    shuffled_context_eval_returns = None
    stale_context_eval_returns = None
    no_online_refinement_eval_returns = None
    zero_context_ablation_delta = None
    shuffled_context_ablation_delta = None
    stale_context_ablation_delta = None
    online_refinement_ablation_delta = None
    expression_scale_history: list[float] = []
    fair_ready_handoff_history: list[float] = []
    fair_expression_enabled_history: list[float] = []
    fair_expression_force_muted_history: list[float] = []
    fair_ready_confidence_history: list[float] = []
    fair_muted_confidence_history: list[float] = []
    expression_ready_but_muted_history: list[float] = []
    fair_stop_blocker_counts: dict[str, int] = {}
    readiness_reason_counts: dict[str, int] = {}
    readiness_component_keys = (
        "future_probe_quality",
        "subset_stability",
        "leaveout_stability",
        "support_diversity",
    )
    readiness_component_totals = {
        key: 0.0 for key in readiness_component_keys
    }
    readiness_component_count = 0
    shadow_expression_enabled_history: list[float] = []
    shadow_expression_scale_history: list[float] = []
    shadow_confidence_history: list[float] = []
    shadow_strict_miss_history: list[float] = []
    shadow_blocker_counts: dict[str, int] = {}
    second_probe_family_selection_count = {
        family: 0 for family in getattr(crawler_bundle, "family_names", ())
    }
    second_probe_raw_future_gain_history: list[float] = []
    second_probe_future_estimate_history: list[float] = []
    second_probe_choice_future_gain_history: list[float] = []
    fair_selection_event_count = 0
    fair_coverage_satisfied_selection_count = 0
    fair_value_driven_selection_count = 0
    fair_uniformity_pressure_selection_count = 0
    readiness_component_timeline: list[dict[str, float]] = []
    online_future_quality_trace: list[float] = []
    online_subset_stability_trace: list[float] = []
    online_offline_gap_trace: list[float] = []
    online_geometry_complete_history: list[float] = []
    online_split_latent_disagreement_history: list[float] = []
    online_split_retrieval_margin_deficit_history: list[float] = []
    online_leaveout_shift_history: list[float] = []
    message_input_delta_history: list[float] = []
    muted_message_input_delta_history: list[float] = []
    actor_message_norm_history: list[float] = []
    actor_message_nonzero_history: list[float] = []
    muted_actor_message_nonzero_history: list[float] = []
    matched_mute_parity_history: list[float] = []
    message_mode_history: list[str] = []
    last_fair_handoff_probe_families: list[str] | None = None
    solve_fair_handoff_probe_families: list[str] | None = None
    solve_probe_count = None
    last_probe_stop_reason = None
    solve_probe_stop_reason = None
    last_logged_message_mode: str | None = None
    last_logged_blocker: str | None = None
    total_env_steps = 0
    total_probe_env_steps = 0
    total_control_env_steps = 0
    total_probe_windows = 0
    probe_stop_reasons = default_probe_stop_reasons()
    probe_family_expected_gain_totals = default_family_score_counter(getattr(crawler_bundle, "family_names", ()))
    probe_family_expected_gain_counts = {family: 0 for family in getattr(crawler_bundle, "family_names", ())}
    probe_family_realized_gain_totals = default_family_metric_counter(getattr(crawler_bundle, "family_names", ()))
    probe_family_realized_gain_counts = {family: 0 for family in getattr(crawler_bundle, "family_names", ())}
    probe_family_future_error_totals = default_family_metric_counter(getattr(crawler_bundle, "family_names", ()))
    probe_family_future_error_counts = {family: 0 for family in getattr(crawler_bundle, "family_names", ())}
    probe_family_selection_count = {family: 0 for family in getattr(crawler_bundle, "family_names", ())}
    probe_family_bad_streaks = {family: 0 for family in getattr(crawler_bundle, "family_names", ())}
    rollout_chunk = init_rollout_chunk(
        track_beliefs=True,
        track_recurrent_hidden=belief_native_controller,
    )
    episode_limit = max(1, int(num_episodes))
    max_episode_limit = episode_limit + max(0, int(solve_grace_episodes))
    anneal_episode_count = max_episode_limit if solve_grace_episodes > 0 else episode_limit

    episode = 1
    while episode <= episode_limit:
        if trace_writer is not None and episode == 1:
            trace_writer.set_stage(
                "probe_training",
                (
                    "Belief-Native Controller"
                    if belief_native_controller
                    else "Probe-Conditioned Controller"
                ),
                (
                    "Alternating between probe support, structured belief-context updates, and downstream control."
                    if belief_native_controller
                    else "Alternating between short support probes and downstream control while the env expression stays frozen per episode."
                ),
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant=variant_label,
            )
        if lr_anneal:
            progress_left = max(0.0, 1.0 - float(episode - 1) / max(float(anneal_episode_count), 1.0))
            set_optimizer_lr(optimizer, lr * progress_left)
        episode_probe_steps = 0
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        belief_posteriors: list[tuple[np.ndarray, np.ndarray]] = []
        probe_planner = build_probe_planner(
            action_space=probe_env.action_space,
            action_values=action_values,
            rng=rng,
            env_name=env_name,
        )
        if probe_planner is not None:
            probe_planner.begin_env_instance()
        # Probe first, act second: the crawler gathers a small support set,
        # scores the next family by expected information gain, and emits the
        # solver-facing belief message the policy will consume.
        belief_hidden = init_recurrent_belief_hidden(encoder=encoder, device=device)
        min_seed_support = min(
            max(1, min(2, int(base_probe_episodes))),
            max(1, int(effective_max_probe_episodes)),
        )
        family_coverage_budget = desired_family_coverage_budget(
            getattr(crawler_bundle, "family_names", ()),
            int(effective_max_probe_episodes),
            min_seed_support,
        )
        min_family_coverage_ratio = minimum_family_coverage_ratio(
            family_coverage_budget=family_coverage_budget,
            min_seed_support=min_seed_support,
        )
        family_counts = {family: 0 for family in getattr(crawler_bundle, "family_names", ())}
        family_error_history: dict[str, float] = {}
        family_realized_gain_history: dict[str, float] = {}
        recent_realized_gains: list[float] = []
        recent_probe_families: list[str] = []
        probe_window_records: list[dict[str, object]] = []
        step_result = None
        belief = None
        probe_count = 0
        probe_stop_reason = None
        probe_surprise = 0.0

        while probe_count < max(1, int(effective_max_probe_episodes)):
            expected_family_gain = {} if step_result is None else step_result.expected_family_gain
            chosen_family = None
            chosen_family_metrics: dict[str, float] = {}
            if probe_planner is not None:
                if fair_mode and crawler_bundle is not None:
                    chosen_family = choose_fair_probe_family(
                        family_names=getattr(crawler_bundle, "family_names", ()),
                        expected_family_gain=expected_family_gain,
                        family_counts=family_counts,
                        probe_count=probe_count,
                        global_family_counts=probe_family_selection_count,
                        family_realized_gain_history=family_realized_gain_history,
                        recent_families=tuple(recent_probe_families),
                        probe_surprise=probe_surprise,
                    )
                    if chosen_family is not None:
                        fair_selection_event_count += 1
                        if probe_count > 0:
                            fair_coverage_satisfied_selection_count += 1
                            fair_value_driven_selection_count += 1
                        else:
                            fair_uniformity_pressure_selection_count += 1
                else:
                    allow_quota_family = (not fixed_cap_early_stop) or probe_count < max(int(min_seed_support), 2)
                    require_seed_family = should_require_seed_probe_family(
                        probe_count=probe_count,
                        family_coverage_budget=family_coverage_budget,
                        family_names=getattr(crawler_bundle, "family_names", ()),
                        expected_family_gain=expected_family_gain,
                        family_counts=family_counts,
                        global_family_counts=probe_family_selection_count,
                        family_realized_gain_history=family_realized_gain_history,
                        family_bad_streaks=probe_family_bad_streaks,
                    ) if crawler_bundle is not None else False
                    chosen_family = choose_next_probe_family(
                        crawler_bundle=crawler_bundle,
                        expected_family_gain=expected_family_gain,
                        family_counts=family_counts,
                        global_family_counts=probe_family_selection_count,
                        require_seed_family=require_seed_family,
                        allow_quota_family=allow_quota_family,
                        family_realized_gain_history=family_realized_gain_history,
                        family_bad_streaks=probe_family_bad_streaks,
                        recent_families=tuple(recent_probe_families),
                    ) if crawler_bundle is not None else None
                chosen_family_metrics = expected_family_gain.get(chosen_family, {}) if chosen_family is not None else {}
                probe_planner.begin_rollout(primary_goal=chosen_family)

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
                trace_writer=trace_writer,
                trace_context={"episode_id": episode, "env_instance_id": seed, "step_offset": episode_probe_steps},
            )
            total_env_steps += probe_steps_used
            total_probe_env_steps += probe_steps_used
            episode_probe_steps += probe_steps_used
            if probe_failed:
                probe_stop_reasons["probe_failure"] += 1
                if probe_count == 0:
                    print_probe_failure(
                        episode=episode,
                        probe_steps=episode_probe_steps,
                    )
                    returns.append(0.0)
                    probe_stop_reason = "probe_failure"
                    break
                probe_stop_reason = "probe_failure"
                break

            if prior_belief is not None:
                probe_surprise = compute_probe_surprise(
                    env_future_predictor=env_future_predictor,
                    belief=prior_belief,
                    window_states=window_states,
                    window_actions=window_actions,
                    window_rewards=window_rewards,
                    action_vocab_size=int(len(action_values)),
                    device=device,
                    env_name=env_name,
                    probe_family=chosen_family,
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
            probe_window_records.append(
                {
                    "states": sanitize_numpy(window_states),
                    "actions": np.asarray(window_actions, dtype=np.int64),
                    "rewards": sanitize_numpy(window_rewards),
                    "terminated": False,
                    "truncated": False,
                    "probe_family": (
                        chosen_family
                        if chosen_family is not None
                        else (
                            None
                            if probe_planner is None
                            else getattr(probe_planner, "current_goal", None)
                        )
                    ),
                }
            )
            total_probe_windows += 1
            probe_count += 1
            if chosen_family is not None:
                family_counts[chosen_family] = family_counts.get(chosen_family, 0) + 1
                probe_family_selection_count[chosen_family] = probe_family_selection_count.get(chosen_family, 0) + 1
                if fair_mode and probe_count == 2:
                    second_probe_family_selection_count[chosen_family] = (
                        second_probe_family_selection_count.get(chosen_family, 0) + 1
                    )
                    second_probe_raw_future_gain_history.append(
                        float(chosen_family_metrics.get("raw_predicted_future_error_reduction", 0.0))
                    )
                    second_probe_future_estimate_history.append(
                        float(chosen_family_metrics.get("future_error_estimate", 0.0))
                    )
                    second_probe_choice_future_gain_history.append(
                        float(chosen_family_metrics.get("future_gain_for_choice", 0.0))
                    )
                recent_probe_families.append(chosen_family)
                if len(recent_probe_families) > 3:
                    recent_probe_families = recent_probe_families[-3:]

            probe_group_ids = probe_group_ids_from_families(
                [record.get("probe_family") for record in probe_window_records],
                family_names=None if crawler_bundle is None else crawler_bundle.family_names,
            )
            particle_belief_mode = crawler_bundle is not None and str(getattr(crawler_bundle, "belief_mode", "latent_pool")) == "particle_sysid"
            if particle_belief_mode:
                belief, payload = crawler_bundle.build_particle_env_belief(
                    probe_window_records,
                    bits_per_dim=belief_bits_per_dim,
                    use_residual_sketch=belief_use_residual_sketch and (not fair_mode),
                )
            else:
                belief, payload = aggregate_env_belief(
                    belief_aggregator=belief_aggregator,
                    env_param_predictor=env_param_predictor,
                    device=device,
                    posterior_views=belief_posteriors,
                    probe_group_ids=probe_group_ids,
                )
            particle_payload_overrides = {}
            if particle_belief_mode:
                particle_payload_overrides = {
                    key: payload[key]
                    for key in (
                        "future_probe_error",
                        "full_future_prediction_error",
                        "observed_family_future_error",
                        "heldout_family_future_error",
                        "support_size_matched_future_error",
                        "online_subset_stability",
                        "online_geometry_complete",
                        "online_leaveout_shift",
                        "online_observed_family_count",
                        "online_offline_gap",
                        "leaveout_shift",
                        "leaveout_param_std",
                        "fair_handoff_probe_families",
                    )
                    if key in payload
                }

            if crawler_bundle is not None:
                future_diagnostics = compute_online_future_diagnostics(
                    crawler_bundle=crawler_bundle,
                    posterior_views=belief_posteriors,
                    probe_windows=probe_window_records,
                    env_name=env_name,
                )
                payload["future_probe_error"] = np.asarray(
                    [future_diagnostics["future_probe_error"]],
                    dtype=np.float32,
                )
                payload["full_future_prediction_error"] = np.asarray(
                    [future_diagnostics["full_future_prediction_error"]],
                    dtype=np.float32,
                )
                payload["observed_family_future_error"] = np.asarray(
                    [future_diagnostics["observed_family_future_error"]],
                    dtype=np.float32,
                )
                payload["heldout_family_future_error"] = np.asarray(
                    [future_diagnostics["heldout_family_future_error"]],
                    dtype=np.float32,
                )
                payload["support_size_matched_future_error"] = np.asarray(
                    [future_diagnostics["support_size_matched_future_error"]],
                    dtype=np.float32,
                )
                payload["online_offline_gap"] = np.asarray(
                    [future_diagnostics["online_offline_gap"]],
                    dtype=np.float32,
                )
                payload["online_subset_stability"] = np.asarray(
                    [future_diagnostics["online_subset_stability"]],
                    dtype=np.float32,
                )
                payload["online_geometry_complete"] = np.asarray(
                    [1.0 if future_diagnostics["online_geometry_complete"] else 0.0],
                    dtype=np.float32,
                )
                payload["online_split_latent_disagreement"] = np.asarray(
                    [future_diagnostics["online_split_latent_disagreement"]],
                    dtype=np.float32,
                )
                payload["online_split_retrieval_margin_deficit"] = np.asarray(
                    [future_diagnostics["online_split_retrieval_margin_deficit"]],
                    dtype=np.float32,
                )
                payload["online_leaveout_shift"] = np.asarray(
                    [future_diagnostics["online_leaveout_shift"]],
                    dtype=np.float32,
                )
                payload["online_observed_family_count"] = np.asarray(
                    [future_diagnostics["online_observed_family_count"]],
                    dtype=np.int32,
                )
                payload["fair_handoff_probe_families"] = np.asarray(
                    future_diagnostics["fair_handoff_probe_families"],
                    dtype="U",
                )
                if particle_payload_overrides:
                    payload.update(particle_payload_overrides)
                predictive_belief = crawler_bundle.build_predictive_belief(payload)
                uncertainty_estimate = crawler_bundle.build_uncertainty_estimate(payload)
                expected_family_gain = crawler_bundle.score_probe_families(
                    predictive_belief,
                    uncertainty_estimate,
                    family_counts=family_counts,
                    global_family_counts=probe_family_selection_count,
                    family_error_history=family_error_history,
                    family_realized_gain_history=family_realized_gain_history,
                    use_learned_family_value=not fair_mode,
                )
                realized_family_gain = {}
                if step_result is not None and chosen_family is not None:
                    previous_posterior_entropy = float(
                        step_result.predictive_belief.metadata.get("mechanics_posterior_entropy", 0.0)
                    )
                    next_posterior_entropy = float(
                        predictive_belief.metadata.get("mechanics_posterior_entropy", 0.0)
                    )
                    entropy_drop = max(0.0, previous_posterior_entropy - next_posterior_entropy)
                    future_error_drop = max(
                        0.0,
                        float(step_result.predictive_belief.future_probe_error)
                        - float(predictive_belief.future_probe_error),
                    )
                    scalar_drop = max(
                        0.0,
                        float(step_result.uncertainty.scalar) - float(uncertainty_estimate.scalar),
                    )
                    realized_family_gain[chosen_family] = max(
                        0.0,
                        0.50 * entropy_drop + 0.25 * future_error_drop + 0.25 * scalar_drop,
                    )
                step_result = crawler_bundle.build_step_result(
                    payload=payload,
                    expected_family_gain=expected_family_gain,
                    realized_family_gain=realized_family_gain,
                    stop_reason=None,
                    bits_per_dim=belief_bits_per_dim,
                    use_residual_sketch=belief_use_residual_sketch and (not fair_mode),
                )
                update_family_score_counter(
                    probe_family_expected_gain_totals,
                    probe_family_expected_gain_counts,
                    step_result.expected_family_gain,
                )
                update_family_scalar_counter(
                    probe_family_realized_gain_totals,
                    probe_family_realized_gain_counts,
                    step_result.realized_family_gain,
                )
                if chosen_family is not None:
                    family_error_history[chosen_family] = float(step_result.predictive_belief.future_probe_error)
                    if chosen_family in realized_family_gain:
                        recent_realized_gains.append(float(realized_family_gain[chosen_family]))
                        if len(recent_realized_gains) > 3:
                            recent_realized_gains = recent_realized_gains[-3:]
                        previous_realized_gain = float(family_realized_gain_history.get(chosen_family, realized_family_gain[chosen_family]))
                        family_realized_gain_history[chosen_family] = (
                            0.65 * previous_realized_gain
                            + 0.35 * float(realized_family_gain[chosen_family])
                        )
                    if chosen_family_metrics:
                        chosen_predicted_value, chosen_value_per_step, _chosen_selection_score, _chosen_score = (
                            probe_family_selection_metrics(chosen_family_metrics)
                        )
                        chosen_realized_gain = float(realized_family_gain.get(chosen_family, 0.0))
                        if (
                            chosen_predicted_value <= 0.0
                            or chosen_value_per_step <= 0.0
                            or chosen_realized_gain <= 0.02
                        ):
                            probe_family_bad_streaks[chosen_family] = probe_family_bad_streaks.get(chosen_family, 0) + 1
                        elif (
                            chosen_predicted_value > 0.08
                            and chosen_value_per_step > 0.08
                            and chosen_realized_gain > 0.05
                        ):
                            probe_family_bad_streaks[chosen_family] = 0
                        else:
                            probe_family_bad_streaks[chosen_family] = max(
                                0,
                                probe_family_bad_streaks.get(chosen_family, 0) - 1,
                            )
                    update_family_scalar_counter(
                        probe_family_future_error_totals,
                        probe_family_future_error_counts,
                        {chosen_family: float(step_result.predictive_belief.future_probe_error)},
                    )
                if trace_writer is not None:
                    probe_decision_shadow_diagnostics = shadow_env_expression_diagnostics(
                        env_expression=step_result.env_expression,
                    )
                    probe_decision_muted_by_policy = False
                    probe_decision_mute_reason = None
                    if shadow_env_expression and (not disable_env_expression):
                        probe_decision_muted_by_policy = not bool(
                            probe_decision_shadow_diagnostics["enabled"]
                        )
                        if probe_decision_muted_by_policy:
                            probe_decision_mute_reason = (
                                f"shadow_{probe_decision_shadow_diagnostics['blocker']}"
                            )
                    elif fair_mode and (not disable_env_expression):
                        probe_decision_muted_by_policy = not fair_env_expression_enabled(
                            env_expression=step_result.env_expression,
                        )
                        if probe_decision_muted_by_policy:
                            probe_decision_mute_reason = "fair_not_ready"
                    trace_writer.record_probe_decision(
                        episode=episode,
                        probe_count=probe_count,
                        max_probe_episodes=effective_max_probe_episodes,
                        chosen_family=chosen_family,
                        uncertainty=float(step_result.uncertainty.scalar),
                        surprise=float(probe_surprise),
                        message_scale=None,
                        expression_confidence=float(step_result.env_expression.confidence),
                        expression_ready=bool(step_result.env_expression.ready),
                        expression_muted_by_policy=probe_decision_muted_by_policy,
                        expression_mute_reason=probe_decision_mute_reason,
                        chosen_raw_future_estimate=chosen_family_metrics.get(
                            "raw_future_error_estimate"
                        ),
                        chosen_future_estimate=chosen_family_metrics.get(
                            "future_error_estimate"
                        ),
                        chosen_future_gain_for_choice=chosen_family_metrics.get(
                            "future_gain_for_choice"
                        ),
                        chosen_entropy_reduction=chosen_family_metrics.get(
                            "predicted_entropy_reduction"
                        ),
                        chosen_hypothesis_separation=chosen_family_metrics.get(
                            "predicted_hypothesis_separation"
                        ),
                        chosen_estimated_cost=chosen_family_metrics.get(
                            "estimated_probe_cost"
                        ),
                        expected_family_gain=step_result.expected_family_gain,
                        realized_family_gain=step_result.realized_family_gain,
                        stop_reason=None,
                    )
            else:
                step_result = None

            if step_result is None:
                continue

            best_expected_gain = max(
                (float(item.get("score", 0.0)) for item in step_result.expected_family_gain.values()),
                default=0.0,
            )
            best_entropy_reduction = max(
                (float(item.get("predicted_entropy_reduction", 0.0)) for item in step_result.expected_family_gain.values()),
                default=0.0,
            )
            best_hypothesis_separation = max(
                (float(item.get("predicted_hypothesis_separation", 0.0)) for item in step_result.expected_family_gain.values()),
                default=0.0,
            )
            best_value_per_probe_step = max(
                (float(item.get("value_per_probe_step", 0.0)) for item in step_result.expected_family_gain.values()),
                default=0.0,
            )
            best_marginal_value = max(
                (float(item.get("predicted_marginal_value", item.get("value_per_probe_step", 0.0))) for item in step_result.expected_family_gain.values()),
                default=0.0,
            )
            best_selection_score = max(
                (float(item.get("selection_score", item.get("score", 0.0))) for item in step_result.expected_family_gain.values()),
                default=0.0,
            )
            best_realized_gain = max(
                (float(family_realized_gain_history.get(family, 0.0)) for family in step_result.expected_family_gain.keys()),
                default=0.0,
            )
            recent_realized_gain = (
                float(np.mean(np.asarray(recent_realized_gains[-2:], dtype=np.float32)))
                if recent_realized_gains
                else None
            )
            observed_family_count = sum(1 for value in family_counts.values() if int(value) > 0)
            family_coverage_ratio = float(observed_family_count) / float(
                max(1, min(len(family_counts), family_coverage_budget))
            )
            if fair_mode:
                selectable_families = []
                next_selectable_family = choose_fair_probe_family(
                    family_names=getattr(crawler_bundle, "family_names", ()),
                    expected_family_gain=step_result.expected_family_gain,
                    family_counts=family_counts,
                    probe_count=probe_count,
                    global_family_counts=probe_family_selection_count,
                    family_realized_gain_history=family_realized_gain_history,
                    recent_families=tuple(recent_probe_families),
                    probe_surprise=probe_surprise,
                )
                if next_selectable_family is not None:
                    selectable_families = [next_selectable_family]
            else:
                selectable_families = rank_probe_family_candidates(
                    crawler_bundle,
                    step_result.expected_family_gain,
                    family_counts,
                    probe_family_selection_count,
                    family_realized_gain_history=family_realized_gain_history,
                    family_bad_streaks=probe_family_bad_streaks,
                    recent_families=tuple(recent_probe_families),
                )
                next_selectable_family = choose_next_probe_family(
                    crawler_bundle=crawler_bundle,
                    expected_family_gain=step_result.expected_family_gain,
                    family_counts=family_counts,
                    global_family_counts=probe_family_selection_count,
                    require_seed_family=should_require_seed_probe_family(
                        probe_count=probe_count,
                        family_coverage_budget=family_coverage_budget,
                        family_names=getattr(crawler_bundle, "family_names", ()),
                        expected_family_gain=step_result.expected_family_gain,
                        family_counts=family_counts,
                        global_family_counts=probe_family_selection_count,
                        family_realized_gain_history=family_realized_gain_history,
                        family_bad_streaks=probe_family_bad_streaks,
                    ),
                    allow_quota_family=(not fixed_cap_early_stop) or probe_count < max(int(min_seed_support), 2),
                    family_realized_gain_history=family_realized_gain_history,
                    family_bad_streaks=probe_family_bad_streaks,
                    recent_families=tuple(recent_probe_families),
                )
            next_selectable_metrics = (
                step_result.expected_family_gain.get(next_selectable_family, {})
                if next_selectable_family is not None
                else {}
            )
            (
                next_selectable_marginal_value,
                next_selectable_value_per_probe_step,
                next_selectable_selection_score,
                _next_selectable_score,
            ) = probe_family_selection_metrics(next_selectable_metrics)
            only_passive_family_viable = bool(selectable_families) and all(
                family == "passive_decay" for family in selectable_families
            )
            if fixed_cap_early_stop:
                fair_stop_ready = bool(
                    step_result.env_expression.metadata.get("fair_stop_ready", False)
                )
                should_stop, reason = should_stop_probing_fair(
                    probe_count=probe_count,
                    min_seed_support=min_seed_support,
                    max_probe_episodes=effective_max_probe_episodes,
                    uncertainty_scalar=float(step_result.uncertainty.scalar),
                    uncertainty_probe_threshold=uncertainty_probe_threshold,
                    posterior_entropy=float(
                        step_result.predictive_belief.metadata.get("mechanics_posterior_entropy", 0.0)
                    ),
                    best_expected_gain=best_expected_gain,
                    best_entropy_reduction=best_entropy_reduction,
                    best_hypothesis_separation=best_hypothesis_separation,
                    best_value_per_probe_step=best_value_per_probe_step,
                    best_marginal_value=best_marginal_value,
                    best_selection_score=best_selection_score,
                    best_realized_gain=best_realized_gain,
                    recent_realized_gain=recent_realized_gain,
                    future_probe_error=float(step_result.predictive_belief.future_probe_error),
                    support_diversity_ratio=float(step_result.predictive_belief.support_diversity_ratio),
                    family_coverage_ratio=family_coverage_ratio,
                    min_family_coverage_ratio=min_family_coverage_ratio,
                    has_selectable_family=next_selectable_family is not None,
                    best_selectable_value_per_probe_step=(
                        next_selectable_value_per_probe_step
                        if next_selectable_family is not None
                        else None
                    ),
                    best_selectable_marginal_value=(
                        next_selectable_marginal_value
                        if next_selectable_family is not None
                        else None
                    ),
                    best_selectable_selection_score=(
                        next_selectable_selection_score
                        if next_selectable_family is not None
                        else None
                    ),
                    best_selectable_realized_gain=(
                        float(family_realized_gain_history.get(next_selectable_family, 0.0))
                        if next_selectable_family is not None
                        else None
                    ),
                    best_selectable_family_count=(
                        int(family_counts.get(next_selectable_family, 0))
                        if next_selectable_family is not None
                        else 0
                    ),
                    best_selectable_bad_streak=(
                        int(probe_family_bad_streaks.get(next_selectable_family, 0))
                        if next_selectable_family is not None
                        else 0
                    ),
                    only_passive_family_viable=only_passive_family_viable,
                    fair_stop_ready=fair_stop_ready,
                    expression_ready=bool(step_result.env_expression.ready),
                )
                if should_stop:
                    probe_stop_reason = reason
                    if reason is not None:
                        probe_stop_reasons[reason] += 1
                    break

            if adaptive_expand and probe_count >= min_seed_support:
                if next_selectable_family is None:
                    probe_stop_reason = "nonpositive_value_per_step"
                    probe_stop_reasons[probe_stop_reason] += 1
                    break
                continue_probing, reason = should_continue_probing_adaptive(
                    probe_count=probe_count,
                    max_probe_episodes=effective_max_probe_episodes,
                    uncertainty_scalar=float(step_result.uncertainty.scalar),
                    uncertainty_probe_threshold=uncertainty_probe_threshold,
                    probe_surprise=probe_surprise,
                    surprise_probe_threshold=surprise_probe_threshold,
                    best_expected_gain=best_expected_gain,
                    best_marginal_value=next_selectable_marginal_value,
                    future_probe_error=float(step_result.predictive_belief.future_probe_error),
                    family_coverage_ratio=family_coverage_ratio,
                    min_family_coverage_ratio=min_family_coverage_ratio,
                )
                if continue_probing:
                    probe_stop_reasons["adaptive_continue"] += 1
                else:
                    probe_stop_reason = reason
                    probe_stop_reasons[reason] += 1
                    break

        if belief is None or step_result is None:
            continue
        if probe_stop_reason is None:
            if fair_mode:
                probe_stop_reason = "fair_two_probe_handoff"
            else:
                probe_stop_reason = (
                    "fixed_cap_reached" if fixed_cap_early_stop else "adaptive_expand_cap"
                )
            probe_stop_reasons[probe_stop_reason] += 1
        last_probe_stop_reason = probe_stop_reason

        expression_confidence = float(step_result.env_expression.confidence)
        expression_ready = bool(step_result.env_expression.ready)
        message_mode = str(
            step_result.env_expression.metadata.get(
                "message_mode",
                "on" if expression_ready else "off",
            )
        )
        message_blocker = str(
            step_result.env_expression.metadata.get("message_blocker", "unknown")
        )
        message_mode_history.append(message_mode)
        fair_stop_ready = bool(
            step_result.env_expression.metadata.get("fair_stop_ready", False)
        )
        fair_stop_blocker = str(
            step_result.env_expression.metadata.get("fair_stop_blocker", "enabled")
        )
        readiness_reason = str(
            step_result.env_expression.metadata.get("readiness_reason", "unknown")
        )
        readiness_reason_counts[readiness_reason] = (
            readiness_reason_counts.get(readiness_reason, 0) + 1
        )
        for readiness_key in readiness_component_keys:
            readiness_component_totals[readiness_key] += float(
                step_result.env_expression.metadata.get(readiness_key, 0.0)
            )
        readiness_component_count += 1
        readiness_component_timeline.append(
            {
                readiness_key: float(
                    step_result.env_expression.metadata.get(readiness_key, 0.0)
                )
                for readiness_key in readiness_component_keys
            }
        )
        online_future_quality_trace.append(
            float(
                step_result.env_expression.metadata.get(
                    "support_size_matched_future_quality",
                    0.0,
                )
            )
        )
        online_subset_stability_trace.append(
            float(
                step_result.env_expression.metadata.get(
                    "online_subset_stability",
                    0.0,
                )
            )
        )
        online_offline_gap_trace.append(
            float(step_result.env_expression.metadata.get("online_offline_gap", 0.0))
        )
        online_geometry_complete_history.append(
            float(
                bool(
                    step_result.env_expression.metadata.get(
                        "online_geometry_complete",
                        False,
                    )
                )
            )
        )
        online_split_latent_disagreement_history.append(
            float(
                step_result.env_expression.metadata.get(
                    "online_split_latent_disagreement",
                    0.0,
                )
            )
        )
        online_split_retrieval_margin_deficit_history.append(
            float(
                step_result.env_expression.metadata.get(
                    "online_split_retrieval_margin_deficit",
                    0.0,
                )
            )
        )
        online_leaveout_shift_history.append(
            float(
                step_result.env_expression.metadata.get(
                    "online_leaveout_shift",
                    0.0,
                )
            )
        )
        last_fair_handoff_probe_families = [
            str(family)
            for family in step_result.env_expression.metadata.get(
                "fair_handoff_probe_families",
                (),
            )
            if str(family)
        ]
        fair_expression_allowed = fair_env_expression_enabled(
            env_expression=step_result.env_expression,
        )
        shadow_expression_diagnostics = shadow_env_expression_diagnostics(
            env_expression=step_result.env_expression,
        )
        shadow_expression_allowed = bool(shadow_expression_diagnostics["enabled"])
        if shadow_env_expression and (not disable_env_expression):
            expression_force_muted_by_policy = not shadow_expression_allowed
            expression_mute_reason = (
                f"shadow_{shadow_expression_diagnostics['blocker']}"
                if expression_force_muted_by_policy
                else None
            )
            shadow_expression_enabled_history.append(float(shadow_expression_allowed))
            shadow_strict_miss_history.append(
                float((not fair_expression_allowed) and shadow_expression_allowed)
            )
            if shadow_expression_allowed:
                shadow_confidence_history.append(expression_confidence)
            else:
                shadow_blocker = str(shadow_expression_diagnostics["blocker"])
                shadow_blocker_counts[shadow_blocker] = (
                    shadow_blocker_counts.get(shadow_blocker, 0) + 1
                )
        else:
            expression_force_muted_by_policy = (
                fair_mode
                and (not disable_env_expression)
                and (
                    message_mode == "off"
                    or (message_mode == "on" and (not fair_expression_allowed))
                )
            )
            expression_mute_reason = (
                "fair_not_ready" if expression_force_muted_by_policy else None
            )
        fair_ready_handoff_history.append(float(expression_ready))
        fair_expression_enabled_history.append(
            float(fair_mode and (not disable_env_expression) and fair_expression_allowed)
        )
        fair_expression_force_muted_history.append(
            float(expression_force_muted_by_policy)
        )
        expression_ready_but_muted_history.append(
            float(expression_ready and expression_force_muted_by_policy)
        )
        if fair_mode and expression_ready:
            fair_ready_confidence_history.append(expression_confidence)
        if expression_force_muted_by_policy:
            fair_muted_confidence_history.append(expression_confidence)
        if fair_mode and (not fair_stop_ready) and fair_stop_blocker != "enabled":
            fair_stop_blocker_counts[fair_stop_blocker] = (
                fair_stop_blocker_counts.get(fair_stop_blocker, 0) + 1
            )

        if belief_native_controller:
            oracle_context_weight = full_system_oracle_weight_for_episode(
                context_source=full_system_context_source,
                current_episode=episode,
                curriculum_schedule=full_system_curriculum_schedule,
            )
            selected_context = step_result.controller_context
            if crawler_bundle is not None and hasattr(episode_physics, "as_array"):
                oracle_context = crawler_bundle.build_oracle_controller_context(
                    episode_physics.as_array()
                )
                selected_context = mix_controller_contexts(
                    step_result.controller_context,
                    oracle_context,
                    oracle_weight=oracle_context_weight,
                )
            controller_input = sanitize_numpy(selected_context.vector)
            controller_input_scale = float(selected_context.confidence)
            raw_controller_input_scale = float(controller_input_scale)
            episode_expression_keep_scale = 1.0
        else:
            expression_audit = build_solver_expression_audit(
                env_expression=step_result.env_expression,
                current_episode=episode,
                total_episodes=num_episodes,
                strict_fair_mode=(fair_mode and (not shadow_env_expression)),
                shadow_expression_mode=shadow_env_expression,
            )
            # Freeze the crawler's env expression for this episode in fair mode.
            controller_input, controller_input_scale = build_solver_episode_expression(
                env_expression=step_result.env_expression,
                current_episode=episode,
                total_episodes=num_episodes,
                disable_env_expression=disable_env_expression,
                strict_fair_mode=(fair_mode and (not shadow_env_expression)),
                shadow_expression_mode=shadow_env_expression,
            )
            raw_controller_input_scale = float(controller_input_scale)
            episode_expression_keep_scale = 1.0
            if controller_input_scale > 0.0:
                if message_mode not in {"diag", "on"}:
                    episode_expression_keep_scale = sample_solver_training_message_keep_scale(
                        rng=rng,
                        current_episode=episode,
                        total_episodes=num_episodes,
                        message_scale=controller_input_scale,
                        base_dropout_prob=belief_message_dropout_prob,
                    )
                controller_input = apply_solver_expression_keep_scale(
                    controller_input,
                    keep_scale=episode_expression_keep_scale,
                )
                controller_input_scale *= episode_expression_keep_scale
            actor_message = solver_message_content(controller_input)
            actor_message_norm = float(np.linalg.norm(actor_message))
            actor_message_nonzero = float(actor_message_norm > 1e-6)
            actor_message_norm_history.append(actor_message_norm)
            actor_message_nonzero_history.append(actor_message_nonzero)
            message_input_delta_history.append(float(expression_audit["input_delta"]))
            if expression_force_muted_by_policy:
                muted_message_input_delta_history.append(
                    float(expression_audit["input_delta"])
                )
                muted_actor_message_nonzero_history.append(actor_message_nonzero)
                matched_mute_parity_history.append(
                    float(
                        float(expression_audit["input_delta"]) <= 1e-6
                        and actor_message_nonzero <= 0.0
                    )
                )
        expression_scale_history.append(float(controller_input_scale))
        if shadow_env_expression:
            shadow_expression_scale_history.append(float(controller_input_scale))
        mean_z = sanitize_numpy(step_result.predictive_belief.mean_raw)
        uncertainty = float(step_result.uncertainty.scalar)
        expected_return = performance_memory.expected_return(mean_z)
        novelty = performance_memory.novelty(mean_z)
        if adaptive_policy_schedule:
            episode_entropy_coef = adjust_entropy_coef(
                base_entropy_coef=entropy_coef,
                novelty=novelty,
                expected_return=expected_return,
                uncertainty=uncertainty,
                novelty_probe_threshold=novelty_probe_threshold,
                low_return_probe_threshold=low_return_probe_threshold,
                exploit_return_threshold=exploit_return_threshold,
                uncertainty_focus_threshold=uncertainty_focus_threshold,
            )
            episode_ppo_epochs = choose_policy_epochs(
                base_ppo_epochs=ppo_epochs,
                expected_return=expected_return,
                uncertainty=uncertainty,
                exploit_return_threshold=exploit_return_threshold,
                uncertainty_focus_threshold=uncertainty_focus_threshold,
            )
        else:
            episode_entropy_coef = float(entropy_coef)
            episode_ppo_epochs = int(ppo_epochs)
        episode_entropy_coef = late_exploitation_entropy_coef(
            base_entropy_coef=episode_entropy_coef,
            returns=returns,
            best_return_so_far=best_return_so_far,
            solved_return=solved_return,
        )
        cap_late_exploitation_action_std(
            policy=policy,
            returns=returns,
            best_return_so_far=best_return_so_far,
            solved_return=solved_return,
        )

        apply_env_params(train_env, episode_physics)
        raw_state, _info = train_env.reset()
        raw_state = np.asarray(raw_state, dtype=np.float32)
        state_normalizer.update(raw_state)
        state = sanitize_numpy(state_normalizer.normalize(raw_state))

        episode_states = []
        episode_beliefs = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_values = []
        episode_terminals = []
        episode_return = 0.0
        episode_step = 0
        done = False
        last_next_state = state.copy()
        last_terminated = False
        controller_hidden = None
        if belief_native_controller and isinstance(policy, BeliefNativeActorCritic):
            controller_hidden = policy.init_recurrent_state(
                torch.tensor(
                    sanitize_numpy(controller_input[None, :]),
                    dtype=torch.float32,
                    device=device,
                )
            )

        while not done:
            # The actor sees normalized state plus either the strict env expression
            # or the richer controller belief context for the full-system path.
            controller_input = sanitize_numpy(controller_input)
            state_t = torch.tensor(sanitize_numpy(state[None, :]), dtype=torch.float32, device=device)
            controller_input_t = torch.tensor(
                sanitize_numpy(controller_input[None, :]),
                dtype=torch.float32,
                device=device,
            )
            controller_hidden_input = None
            if belief_native_controller and controller_hidden is not None:
                controller_hidden_input = sanitize_numpy(
                    controller_hidden.detach().cpu().numpy().squeeze(0)
                )
            with torch.no_grad():
                mean, value, controller_hidden, _aux = policy_action_value_step(
                    policy=policy,
                    state_t=state_t,
                    context_t=controller_input_t,
                    hidden_state=controller_hidden,
                )
            action, log_prob = sample_continuous_action(
                mean=mean,
                log_std=policy.log_std,
                action_low=action_low,
                action_high=action_high,
            )

            prev_raw_state = raw_state.copy()
            next_raw_state, reward, terminated, truncated, _info = train_env.step(action)
            total_env_steps += 1
            total_control_env_steps += 1
            next_raw_state = np.asarray(next_raw_state, dtype=np.float32)
            episode_step += 1
            raw_reward = float(reward)
            train_reward = raw_reward
            if reward_normalizer is not None:
                reward_normalizer.update(np.asarray([[raw_reward]], dtype=np.float32))
                train_reward = float(
                    reward_normalizer.scale_only(np.asarray([raw_reward], dtype=np.float32))[0]
                )
            if trace_writer is not None:
                trace_writer.record_policy_step(
                    phase="probe_control",
                    variant=variant_label,
                    state=prev_raw_state,
                    action_value=float(np.asarray(action, dtype=np.float32).reshape(-1)[0]),
                    reward=raw_reward,
                    episode=episode,
                    step_idx=episode_step,
                    episode_return=episode_return + raw_reward,
                    probe_count=probe_count,
                    uncertainty=uncertainty,
                    message_scale=controller_input_scale,
                    focus_label=probe_stop_reason if probe_stop_reason is not None else chosen_family,
                    expected_return=expected_return,
                    novelty=novelty,
                    expression_confidence=expression_confidence,
                    expression_ready=expression_ready,
                    expression_muted_by_policy=expression_force_muted_by_policy,
                    expression_mute_reason=expression_mute_reason,
                )

            # Convert the continuous action back to the probe vocabulary so the
            # encoder can digest the recent trajectory using the same action language.
            control_surprise = compute_control_surprise(
                predictor=predictor,
                belief=belief,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                next_state=next_raw_state,
                device=device,
            )
            next_belief, belief_hidden, belief_posteriors = maybe_update_online_belief(
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                device=device,
                belief_hidden=belief_hidden,
                belief_posteriors=belief_posteriors,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                reward=raw_reward,
                next_state=next_raw_state,
                belief=belief,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                episode_step=episode_step,
            )
            next_controller_input = controller_input
            if (
                crawler_bundle is not None
                and belief_posteriors
                and (
                    (
                        belief_native_controller
                        and full_system_online_refinement
                        and (
                            episode_step % online_z_update_freq == 0
                            or control_surprise >= full_system_surprise_refresh_threshold
                        )
                    )
                    or (
                        (not belief_native_controller)
                        and (not fair_mode)
                        and episode_step % online_z_update_freq == 0
                    )
                )
            ):
                next_belief, next_payload = aggregate_env_belief(
                    belief_aggregator=belief_aggregator,
                    env_param_predictor=env_param_predictor,
                    device=device,
                    posterior_views=belief_posteriors,
                )
                next_expected_family_gain = crawler_bundle.score_probe_families(
                    crawler_bundle.build_predictive_belief(next_payload),
                    crawler_bundle.build_uncertainty_estimate(next_payload),
                    family_counts=family_counts,
                    global_family_counts=probe_family_selection_count,
                    family_error_history=family_error_history,
                    family_realized_gain_history=family_realized_gain_history,
                    use_learned_family_value=not fair_mode,
                )
                next_step_result = crawler_bundle.build_step_result(
                    payload=next_payload,
                    expected_family_gain=next_expected_family_gain,
                    realized_family_gain={},
                    stop_reason=probe_stop_reason,
                    bits_per_dim=belief_bits_per_dim,
                    use_residual_sketch=belief_use_residual_sketch and (not fair_mode),
                )
                step_result = next_step_result
                if belief_native_controller:
                    oracle_context_weight = full_system_oracle_weight_for_episode(
                        context_source=full_system_context_source,
                        current_episode=episode,
                        curriculum_schedule=full_system_curriculum_schedule,
                    )
                    next_selected_context = next_step_result.controller_context
                    if crawler_bundle is not None and hasattr(episode_physics, "as_array"):
                        oracle_context = crawler_bundle.build_oracle_controller_context(
                            episode_physics.as_array()
                        )
                        next_selected_context = mix_controller_contexts(
                            next_step_result.controller_context,
                            oracle_context,
                            oracle_weight=oracle_context_weight,
                        )
                    next_controller_input = sanitize_numpy(
                        next_selected_context.vector
                    )
                    next_episode_expression_scale = float(
                        next_selected_context.confidence
                    )
                    if isinstance(policy, BeliefNativeActorCritic):
                        controller_hidden = policy.refresh_recurrent_state(
                            torch.tensor(
                                sanitize_numpy(next_controller_input[None, :]),
                                dtype=torch.float32,
                                device=device,
                            ),
                            controller_hidden,
                        )
                else:
                    next_controller_input, next_episode_expression_scale = build_solver_episode_expression(
                        env_expression=next_step_result.env_expression,
                        current_episode=episode,
                        total_episodes=num_episodes,
                        disable_env_expression=disable_env_expression,
                        strict_fair_mode=(fair_mode and (not shadow_env_expression)),
                        shadow_expression_mode=shadow_env_expression,
                    )
                    next_controller_input = apply_solver_expression_keep_scale(
                        next_controller_input,
                        keep_scale=episode_expression_keep_scale,
                    )
                    next_episode_expression_scale *= episode_expression_keep_scale
                expression_confidence = float(next_step_result.env_expression.confidence)
                expression_ready = bool(next_step_result.env_expression.ready)
            else:
                next_episode_expression_scale = controller_input_scale
            state_normalizer.update(next_raw_state)
            next_state = sanitize_numpy(state_normalizer.normalize(next_raw_state))

            episode_states.append(sanitize_numpy(state.copy()))
            episode_beliefs.append(sanitize_numpy(controller_input.copy()))
            episode_actions.append(action.copy())
            episode_log_probs.append(log_prob)
            episode_rewards.append(train_reward)
            episode_values.append(float(value.item()))
            episode_terminals.append(float(terminated))
            append_rollout_step(
                rollout_chunk,
                state=state,
                belief=controller_input,
                action=action,
                log_prob=log_prob,
                reward=train_reward,
                value=float(value.item()),
                terminal=float(terminated),
                recurrent_hidden_state=controller_hidden_input,
            )

            state = next_state
            belief = next_belief
            controller_input = next_controller_input
            controller_input_scale = next_episode_expression_scale
            episode_return += raw_reward
            last_next_state = next_state.copy()
            last_terminated = bool(terminated)
            done = bool(terminated or truncated)

            if rollout_chunk_step_count(rollout_chunk) >= min_rollout_steps or (
                done and episode == episode_limit
            ):
                chunk_bootstrap_value = 0.0
                if not terminated:
                    next_state_t = torch.tensor(
                        sanitize_numpy(last_next_state[None, :]),
                        dtype=torch.float32,
                        device=device,
                    )
                    next_belief_t = torch.tensor(
                        sanitize_numpy(controller_input[None, :]),
                        dtype=torch.float32,
                        device=device,
                    )
                    with torch.no_grad():
                        _mean, next_value, _next_hidden, _aux = policy_action_value_step(
                            policy=policy,
                            state_t=next_state_t,
                            context_t=next_belief_t,
                            hidden_state=controller_hidden,
                        )
                    chunk_bootstrap_value = float(next_value.item())

                rollout_batch = build_rollout_batch(
                    rollout_chunk,
                    bootstrap_value=chunk_bootstrap_value,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    sequence_length=(
                        full_system_context_chunk_len
                        if belief_native_controller
                        else None
                    ),
                )
                auxiliary_loss_fn = None
                if (not belief_native_controller) and len(elite_buffer) >= sil_batch_size:
                    def auxiliary_loss_fn():
                        sil_policy_loss, sil_value_loss = compute_sil_loss(
                            model=policy,
                            elite_buffer=elite_buffer,
                            batch_size=sil_batch_size,
                            device=device,
                            action_low=action_low,
                            action_high=action_high,
                        )
                        return (
                            sil_policy_weight * sil_policy_loss
                            + sil_value_weight * sil_value_loss
                        )

                update_ppo_policy(
                    model=policy,
                    optimizer=optimizer,
                    batch=rollout_batch,
                    action_low=action_low,
                    action_high=action_high,
                    clip_ratio=clip_ratio,
                    value_loss_weight=value_loss_weight,
                    entropy_coef=episode_entropy_coef,
                    ppo_epochs=episode_ppo_epochs,
                    minibatch_size=minibatch_size,
                    max_grad_norm=max_grad_norm,
                    target_kl=target_kl,
                    value_clip_ratio=value_clip_ratio,
                    auxiliary_loss_fn=auxiliary_loss_fn,
                    controller_context_zero_prob=(
                        full_system_context_zero_prob
                        if belief_native_controller
                        else 0.0
                    ),
                    controller_context_shuffle_prob=(
                        full_system_context_shuffle_prob
                        if belief_native_controller
                        else 0.0
                    ),
                    controller_context_stale_prob=(
                        full_system_context_stale_prob
                        if belief_native_controller
                        else 0.0
                    ),
                    controller_sequence_length=(
                        full_system_context_chunk_len
                        if belief_native_controller
                        else None
                    ),
                )
                clear_rollout_chunk(rollout_chunk)

        bootstrap_value = 0.0
        if not last_terminated:
            next_state_t = torch.tensor(sanitize_numpy(last_next_state[None, :]), dtype=torch.float32, device=device)
            next_belief_t = torch.tensor(
                sanitize_numpy(controller_input[None, :]),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                _mean, next_value, _next_hidden, _aux = policy_action_value_step(
                    policy=policy,
                    state_t=next_state_t,
                    context_t=next_belief_t,
                    hidden_state=controller_hidden,
                )
            bootstrap_value = float(next_value.item())

        batch = build_episode_batch(
            states=episode_states,
            actions=episode_actions,
            log_probs=episode_log_probs,
            rewards=episode_rewards,
            values=episode_values,
            terminals=episode_terminals,
            bootstrap_value=bootstrap_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
            beliefs=episode_beliefs,
        )

        is_elite_episode, elite_threshold = should_promote_episode_to_elite(
            episode_return=episode_return,
            completed_returns=returns,
            best_return_so_far=best_return_so_far,
            min_elite_return=min_elite_return,
            current_episode=episode,
            warmup_episodes=elite_warmup_episodes,
            std_scale=elite_threshold_std_scale,
        )
        if is_elite_episode:
            episode_weight = min(3.0, max(1.0, episode_return / max(elite_threshold, 1.0)))
            elite_buffer.push_episode(
                states=batch.states,
                beliefs=batch.beliefs,
                actions=batch.actions,
                returns_to_go=batch.returns,
                episode_weight=episode_weight,
            )

        returns.append(episode_return)
        new_best = episode_return >= best_return_so_far
        if new_best:
            best_policy_state_dict = snapshot_policy_state_dict(policy)
            best_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
            best_episode = episode
            best_env_steps = total_env_steps
            best_probe_count = int(probe_count)
            best_probe_stop_reason = probe_stop_reason
            best_fair_handoff_probe_families = (
                None
                if last_fair_handoff_probe_families is None
                else list(last_fair_handoff_probe_families)
            )
        best_return_so_far = max(best_return_so_far, episode_return)
        performance_memory.push(mean_z, episode_return)
        avg_10 = np.mean(returns[-10:])
        current_solved_episode = solved_episode
        if disable_env_expression:
            message_detail = "env expression disabled"
            current_blocker = "disabled"
        elif message_mode == "off":
            message_detail = f"blocked by {message_blocker}"
            current_blocker = str(message_blocker)
        elif message_mode == "diag":
            message_detail = (
                f"diagnostic handoff | scale {raw_controller_input_scale:.2f}->{controller_input_scale:.2f}"
            )
            current_blocker = "diag"
        else:
            message_detail = (
                f"live handoff | scale {raw_controller_input_scale:.2f}->{controller_input_scale:.2f}"
            )
            current_blocker = "live"
        show_detail_line = bool(
            new_best
            or message_mode != last_logged_message_mode
            or current_blocker != last_logged_blocker
        )
        print_probe_episode_status(
            run_index=run_index,
            total_runs=total_runs,
            seed=seed,
            variant_label=variant_label,
            episode=episode,
            episode_return=episode_return,
            avg10=float(avg_10),
            best_return=float(best_return_so_far),
            total_env_steps=total_env_steps,
            probe_count=int(probe_count),
            episode_probe_steps=int(episode_probe_steps),
            message_mode=message_mode,
            message_detail=message_detail,
            solved_episode=current_solved_episode,
            peer_status=format_peer_solve_status(peer_variant_label, peer_solved_episode),
            new_best=bool(new_best),
            show_detail_line=show_detail_line,
        )
        last_logged_message_mode = message_mode
        last_logged_blocker = current_blocker
        if trace_writer is not None:
            trace_writer.record_episode_summary(
                variant=variant_label,
                episode=episode,
                episode_return=episode_return,
                avg10=float(avg_10),
                avg50=float(np.mean(returns[-50:])),
                total_env_steps=total_env_steps,
                probe_steps=episode_probe_steps,
                probe_count=probe_count,
                uncertainty=uncertainty,
                message_scale=controller_input_scale,
                stop_reason=probe_stop_reason,
                expression_confidence=expression_confidence,
                expression_ready=expression_ready,
                expression_muted_by_policy=expression_force_muted_by_policy,
                expression_mute_reason=expression_mute_reason,
            )

        episode_limit = maybe_extend_solve_episode_limit(
            episode_return=episode_return,
            episode=episode,
            episode_limit=episode_limit,
            max_episode_limit=max_episode_limit,
            solved_return=solved_return,
            solve_avg_window=solve_avg_window,
        )
        solve_average = rolling_solve_average(returns, solve_avg_window)
        if solve_average is not None and solve_average >= float(solved_return):
            solved_episode = episode
            solved_env_steps = total_env_steps
            solve_policy_state_dict = snapshot_policy_state_dict(policy)
            solve_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
            solve_probe_count = int(probe_count)
            solve_probe_stop_reason = probe_stop_reason
            solve_fair_handoff_probe_families = (
                None
                if last_fair_handoff_probe_families is None
                else list(last_fair_handoff_probe_families)
            )
            print_solve_event(
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant_label=variant_label,
                episode=episode,
                total_env_steps=total_env_steps,
                episode_return=float(solve_average),
                probe_count=int(probe_count),
            )
            break
        episode += 1

    if solve_avg_window <= 1 and solve_eval_episodes > 0 and best_episode is not None:
        eval_full_system_context_source = (
            "learned"
            if full_system_context_source == "curriculum"
            else full_system_context_source
        )
        if belief_native_controller:
            eval_policy = BeliefNativeActorCritic(
                state_dim=state_dim,
                action_dim=train_env.action_space.shape[0],
                mechanics_dim=int(encoder.z_dim),
                affordance_dim=int(encoder.z_dim),
                hidden_dim=hidden_dim,
                initial_log_std=initial_log_std,
            ).to(device)
        else:
            eval_policy = ProbeConditionedGaussianActorCritic(
                state_dim=state_dim,
                action_dim=train_env.action_space.shape[0],
                belief_dim=belief_dim,
                hidden_dim=hidden_dim,
                initial_log_std=initial_log_std,
            ).to(device)
        eval_policy.load_state_dict(best_policy_state_dict)
        eval_policy.eval()
        eval_normalizer = restore_normalizer_state(state_dim, best_state_normalizer_state)
        solve_eval_returns, _eval_steps, avg_probe_windows = evaluate_probe_policy(
            policy=eval_policy,
            crawler_bundle=crawler_bundle,
            encoder=encoder,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            env_future_predictor=env_future_predictor,
            predictor=predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            window_size=window_size,
            action_low=action_low,
            action_high=action_high,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=effective_max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=surprise_probe_threshold,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            num_training_episodes=num_episodes,
            disable_belief_message=disable_env_expression,
            freeze_env_expression_per_episode=fair_mode,
            shadow_env_expression=shadow_env_expression,
            eval_episodes=solve_eval_episodes,
            seed=seed,
            device=device,
            belief_native_controller=belief_native_controller,
            disable_online_refinement=(belief_native_controller and (not full_system_online_refinement)),
            full_system_context_source=eval_full_system_context_source,
        )
        solve_eval_best_return = float(np.max(np.asarray(solve_eval_returns, dtype=np.float32)))
        if solve_eval_best_return >= float(solved_return):
            solved_episode = int(best_episode)
            solved_env_steps = int(best_env_steps or total_env_steps)
            solve_policy_state_dict = best_policy_state_dict
            solve_state_normalizer_state = best_state_normalizer_state
            solve_probe_count = None if best_probe_count is None else int(best_probe_count)
            solve_probe_stop_reason = best_probe_stop_reason
            solve_fair_handoff_probe_families = (
                None
                if best_fair_handoff_probe_families is None
                else list(best_fair_handoff_probe_families)
            )
            print_solve_event(
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant_label=variant_label,
                episode=int(best_episode),
                total_env_steps=int(solved_env_steps),
                episode_return=solve_eval_best_return,
                probe_count=solve_probe_count,
            )
        if belief_native_controller:
            zero_context_eval_returns, _zero_steps, _zero_probe_windows = evaluate_probe_policy(
                policy=eval_policy,
                crawler_bundle=crawler_bundle,
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                env_future_predictor=env_future_predictor,
                predictor=predictor,
                state_normalizer=eval_normalizer,
                env_name=env_name,
                action_values=action_values,
                window_size=window_size,
                action_low=action_low,
                action_high=action_high,
                randomize_physics=randomize_physics,
                base_physics=base_physics,
                base_probe_episodes=base_probe_episodes,
                max_probe_episodes=effective_max_probe_episodes,
                probe_adaptive_budget=probe_adaptive_budget,
                uncertainty_probe_threshold=uncertainty_probe_threshold,
                surprise_probe_threshold=surprise_probe_threshold,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                num_training_episodes=num_episodes,
                disable_belief_message=False,
                freeze_env_expression_per_episode=False,
                shadow_env_expression=False,
                eval_episodes=solve_eval_episodes,
                seed=seed,
                device=device,
                belief_native_controller=True,
                disable_controller_context=True,
                full_system_context_source=eval_full_system_context_source,
            )
            shuffled_context_eval_returns, _shuffle_steps, _shuffle_probe_windows = evaluate_probe_policy(
                policy=eval_policy,
                crawler_bundle=crawler_bundle,
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                env_future_predictor=env_future_predictor,
                predictor=predictor,
                state_normalizer=eval_normalizer,
                env_name=env_name,
                action_values=action_values,
                window_size=window_size,
                action_low=action_low,
                action_high=action_high,
                randomize_physics=randomize_physics,
                base_physics=base_physics,
                base_probe_episodes=base_probe_episodes,
                max_probe_episodes=effective_max_probe_episodes,
                probe_adaptive_budget=probe_adaptive_budget,
                uncertainty_probe_threshold=uncertainty_probe_threshold,
                surprise_probe_threshold=surprise_probe_threshold,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                num_training_episodes=num_episodes,
                disable_belief_message=False,
                freeze_env_expression_per_episode=False,
                shadow_env_expression=False,
                eval_episodes=solve_eval_episodes,
                seed=seed,
                device=device,
                belief_native_controller=True,
                shuffle_controller_context=True,
                full_system_context_source=eval_full_system_context_source,
            )
            stale_context_eval_returns, _stale_steps, _stale_probe_windows = evaluate_probe_policy(
                policy=eval_policy,
                crawler_bundle=crawler_bundle,
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                env_future_predictor=env_future_predictor,
                predictor=predictor,
                state_normalizer=eval_normalizer,
                env_name=env_name,
                action_values=action_values,
                window_size=window_size,
                action_low=action_low,
                action_high=action_high,
                randomize_physics=randomize_physics,
                base_physics=base_physics,
                base_probe_episodes=base_probe_episodes,
                max_probe_episodes=effective_max_probe_episodes,
                probe_adaptive_budget=probe_adaptive_budget,
                uncertainty_probe_threshold=uncertainty_probe_threshold,
                surprise_probe_threshold=surprise_probe_threshold,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                num_training_episodes=num_episodes,
                disable_belief_message=False,
                freeze_env_expression_per_episode=False,
                shadow_env_expression=False,
                eval_episodes=solve_eval_episodes,
                seed=seed,
                device=device,
                belief_native_controller=True,
                use_stale_previous_context=True,
                full_system_context_source=eval_full_system_context_source,
            )
            no_online_refinement_eval_returns, _frozen_steps, _frozen_probe_windows = evaluate_probe_policy(
                policy=eval_policy,
                crawler_bundle=crawler_bundle,
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                env_future_predictor=env_future_predictor,
                predictor=predictor,
                state_normalizer=eval_normalizer,
                env_name=env_name,
                action_values=action_values,
                window_size=window_size,
                action_low=action_low,
                action_high=action_high,
                randomize_physics=randomize_physics,
                base_physics=base_physics,
                base_probe_episodes=base_probe_episodes,
                max_probe_episodes=effective_max_probe_episodes,
                probe_adaptive_budget=probe_adaptive_budget,
                uncertainty_probe_threshold=uncertainty_probe_threshold,
                surprise_probe_threshold=surprise_probe_threshold,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                num_training_episodes=num_episodes,
                disable_belief_message=False,
                freeze_env_expression_per_episode=False,
                shadow_env_expression=False,
                eval_episodes=solve_eval_episodes,
                seed=seed,
                device=device,
                belief_native_controller=True,
                disable_online_refinement=True,
                full_system_context_source=eval_full_system_context_source,
            )
            solve_eval_mean = float(np.mean(np.asarray(solve_eval_returns, dtype=np.float32)))
            zero_context_ablation_delta = solve_eval_mean - float(
                np.mean(np.asarray(zero_context_eval_returns, dtype=np.float32))
            )
            shuffled_context_ablation_delta = solve_eval_mean - float(
                np.mean(np.asarray(shuffled_context_eval_returns, dtype=np.float32))
            )
            stale_context_ablation_delta = solve_eval_mean - float(
                np.mean(np.asarray(stale_context_eval_returns, dtype=np.float32))
            )
            online_refinement_ablation_delta = solve_eval_mean - float(
                np.mean(np.asarray(no_online_refinement_eval_returns, dtype=np.float32))
            )
            print(
                f"[run {run_index}/{total_runs} | seed {seed} | {variant_label}] "
                f"belief-context eval | full={solve_eval_mean:7.2f} | "
                f"zero={np.mean(zero_context_eval_returns):7.2f} | "
                f"stale={np.mean(stale_context_eval_returns):7.2f} | "
                f"shuffle={np.mean(shuffled_context_eval_returns):7.2f} | "
                f"frozen={np.mean(no_online_refinement_eval_returns):7.2f} | "
                f"avg-probes={avg_probe_windows:4.2f}"
            )
        elif not disable_env_expression:
            message_ablation_config_diff = {
                key: {"on": on_value, "off": off_value}
                for key, on_value, off_value in (
                    ("disable_belief_message", False, True),
                    (
                        "freeze_env_expression_per_episode",
                        bool(fair_mode),
                        bool(fair_mode),
                    ),
                    (
                        "shadow_env_expression",
                        bool(shadow_env_expression),
                        False,
                    ),
                )
                if on_value != off_value
            }
            (
                no_env_expression_eval_returns,
                forced_env_expression_eval_returns,
            ) = evaluate_probe_message_ablations(
                policy=eval_policy,
                crawler_bundle=crawler_bundle,
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                env_future_predictor=env_future_predictor,
                predictor=predictor,
                state_normalizer=eval_normalizer,
                env_name=env_name,
                action_values=action_values,
                window_size=window_size,
                action_low=action_low,
                action_high=action_high,
                randomize_physics=randomize_physics,
                base_physics=base_physics,
                base_probe_episodes=base_probe_episodes,
                max_probe_episodes=effective_max_probe_episodes,
                probe_adaptive_budget=probe_adaptive_budget,
                uncertainty_probe_threshold=uncertainty_probe_threshold,
                surprise_probe_threshold=surprise_probe_threshold,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                num_training_episodes=num_episodes,
                freeze_env_expression_per_episode=fair_mode,
                eval_episodes=solve_eval_episodes,
                seed=seed,
                device=device,
                forced_expression_scale=DEFAULT_FORCED_EVAL_EXPRESSION_SCALE,
            )
            env_expression_ablation_delta = float(
                np.mean(np.asarray(solve_eval_returns, dtype=np.float32))
                - np.mean(np.asarray(no_env_expression_eval_returns, dtype=np.float32))
            )
            forced_env_expression_ablation_delta = float(
                np.mean(np.asarray(forced_env_expression_eval_returns, dtype=np.float32))
                - np.mean(np.asarray(no_env_expression_eval_returns, dtype=np.float32))
            )
            print(
                f"[run {run_index}/{total_runs} | seed {seed} | {variant_label}] "
                f"env-expression eval | on={np.mean(solve_eval_returns):7.2f} | "
                f"off={np.mean(no_env_expression_eval_returns):7.2f} | "
                f"forced@{DEFAULT_FORCED_EVAL_EXPRESSION_SCALE:.2f}="
                f"{np.mean(forced_env_expression_eval_returns):7.2f} | "
                f"delta={env_expression_ablation_delta:7.2f} | "
                f"forced-delta={forced_env_expression_ablation_delta:7.2f} | "
                f"avg-probes={avg_probe_windows:4.2f}"
            )

    train_env.close()
    probe_env.close()
    expression_scale_median = None
    expression_scale_active_fraction = None
    fair_ready_handoff_fraction = None
    fair_expression_enabled_fraction = None
    fair_expression_force_muted_fraction = None
    fair_ready_confidence_median = None
    fair_muted_confidence_median = None
    expression_ready_but_muted_fraction = None
    readiness_component_means = None
    shadow_expression_enabled_fraction = None
    shadow_expression_scale_median = None
    shadow_confidence_median = None
    shadow_strict_miss_fraction = None
    second_probe_raw_future_gain_mean = None
    second_probe_future_estimate_mean = None
    second_probe_choice_future_gain_mean = None
    family_coverage_satisfied_fraction = None
    second_probe_value_driven_fraction = None
    uniformity_pressure_active_fraction = None
    if expression_scale_history:
        expression_scale_array = np.asarray(expression_scale_history, dtype=np.float32)
        expression_scale_median = float(np.median(expression_scale_array))
        expression_scale_active_fraction = float(np.mean(expression_scale_array > 0.10))
    if fair_ready_handoff_history:
        fair_ready_handoff_fraction = float(
            np.mean(np.asarray(fair_ready_handoff_history, dtype=np.float32))
        )
        fair_expression_enabled_fraction = float(
            np.mean(np.asarray(fair_expression_enabled_history, dtype=np.float32))
        )
        fair_expression_force_muted_fraction = float(
            np.mean(np.asarray(fair_expression_force_muted_history, dtype=np.float32))
        )
    if fair_ready_confidence_history:
        fair_ready_confidence_median = float(
            np.median(np.asarray(fair_ready_confidence_history, dtype=np.float32))
        )
    if fair_muted_confidence_history:
        fair_muted_confidence_median = float(
            np.median(np.asarray(fair_muted_confidence_history, dtype=np.float32))
        )
    if readiness_component_count > 0:
        readiness_component_means = {
            key: float(readiness_component_totals[key]) / float(readiness_component_count)
            for key in readiness_component_keys
        }
    if shadow_expression_enabled_history:
        shadow_expression_enabled_fraction = float(
            np.mean(np.asarray(shadow_expression_enabled_history, dtype=np.float32))
        )
    if shadow_expression_scale_history:
        shadow_expression_scale_median = float(
            np.median(np.asarray(shadow_expression_scale_history, dtype=np.float32))
        )
    if shadow_confidence_history:
        shadow_confidence_median = float(
            np.median(np.asarray(shadow_confidence_history, dtype=np.float32))
        )
    if shadow_strict_miss_history:
        shadow_strict_miss_fraction = float(
            np.mean(np.asarray(shadow_strict_miss_history, dtype=np.float32))
        )
    if second_probe_raw_future_gain_history:
        second_probe_raw_future_gain_mean = float(
            np.mean(np.asarray(second_probe_raw_future_gain_history, dtype=np.float32))
        )
    if second_probe_future_estimate_history:
        second_probe_future_estimate_mean = float(
            np.mean(np.asarray(second_probe_future_estimate_history, dtype=np.float32))
        )
    if second_probe_choice_future_gain_history:
        second_probe_choice_future_gain_mean = float(
            np.mean(np.asarray(second_probe_choice_future_gain_history, dtype=np.float32))
        )
    if fair_selection_event_count > 0:
        family_coverage_satisfied_fraction = float(
            fair_coverage_satisfied_selection_count / float(fair_selection_event_count)
        )
        second_probe_value_driven_fraction = float(
            fair_value_driven_selection_count / float(fair_selection_event_count)
        )
        uniformity_pressure_active_fraction = float(
            fair_uniformity_pressure_selection_count / float(fair_selection_event_count)
        )
    online_offline_gap_mean = None
    online_geometry_complete_fraction = None
    online_split_latent_disagreement_mean = None
    online_split_retrieval_margin_deficit_mean = None
    online_leaveout_shift_mean = None
    message_input_delta_mean = None
    message_input_delta_max = None
    muted_message_input_delta_mean = None
    muted_message_input_delta_max = None
    actor_message_norm_mean = None
    actor_message_nonzero_fraction = None
    muted_actor_message_nonzero_fraction = None
    matched_mute_parity_fraction = None
    message_off_fraction = None
    message_diag_fraction = None
    message_on_fraction = None
    if online_offline_gap_trace:
        online_offline_gap_mean = float(
            np.mean(np.asarray(online_offline_gap_trace, dtype=np.float32))
        )
    if expression_ready_but_muted_history:
        expression_ready_but_muted_fraction = float(
            np.mean(np.asarray(expression_ready_but_muted_history, dtype=np.float32))
        )
    if online_geometry_complete_history:
        online_geometry_complete_fraction = float(
            np.mean(np.asarray(online_geometry_complete_history, dtype=np.float32))
        )
    if online_split_latent_disagreement_history:
        online_split_latent_disagreement_mean = float(
            np.mean(np.asarray(online_split_latent_disagreement_history, dtype=np.float32))
        )
    if online_split_retrieval_margin_deficit_history:
        online_split_retrieval_margin_deficit_mean = float(
            np.mean(
                np.asarray(
                    online_split_retrieval_margin_deficit_history,
                    dtype=np.float32,
                )
            )
        )
    if online_leaveout_shift_history:
        online_leaveout_shift_mean = float(
            np.mean(np.asarray(online_leaveout_shift_history, dtype=np.float32))
        )
    if message_input_delta_history:
        message_input_delta_array = np.asarray(message_input_delta_history, dtype=np.float32)
        message_input_delta_mean = float(np.mean(message_input_delta_array))
        message_input_delta_max = float(np.max(message_input_delta_array))
    if muted_message_input_delta_history:
        muted_message_input_delta_array = np.asarray(
            muted_message_input_delta_history,
            dtype=np.float32,
        )
        muted_message_input_delta_mean = float(
            np.mean(muted_message_input_delta_array)
        )
        muted_message_input_delta_max = float(
            np.max(muted_message_input_delta_array)
        )
    if actor_message_norm_history:
        actor_message_norm_mean = float(
            np.mean(np.asarray(actor_message_norm_history, dtype=np.float32))
        )
    if actor_message_nonzero_history:
        actor_message_nonzero_fraction = float(
            np.mean(np.asarray(actor_message_nonzero_history, dtype=np.float32))
        )
    if muted_actor_message_nonzero_history:
        muted_actor_message_nonzero_fraction = float(
            np.mean(np.asarray(muted_actor_message_nonzero_history, dtype=np.float32))
        )
    if matched_mute_parity_history:
        matched_mute_parity_fraction = float(
            np.mean(np.asarray(matched_mute_parity_history, dtype=np.float32))
        )
    if message_mode_history:
        message_mode_array = np.asarray(message_mode_history, dtype=object)
        message_off_fraction = float(np.mean(message_mode_array == "off"))
        message_diag_fraction = float(np.mean(message_mode_array == "diag"))
        message_on_fraction = float(np.mean(message_mode_array == "on"))
    return TrainingRunResult(
        policy=policy,
        returns=returns,
        state_normalizer=state_normalizer,
        solved_episode=solved_episode,
        solved_env_steps=solved_env_steps,
        total_env_steps=total_env_steps,
        best_policy_state_dict=best_policy_state_dict,
        best_state_normalizer_state=best_state_normalizer_state,
        best_return=best_return_so_far,
        best_episode=best_episode,
        solve_policy_state_dict=solve_policy_state_dict,
        solve_state_normalizer_state=solve_state_normalizer_state,
        solve_eval_returns=solve_eval_returns,
        solve_probe_count=solve_probe_count,
        probe_env_steps_total=total_probe_env_steps,
        control_env_steps_total=total_control_env_steps,
        probe_windows_total=total_probe_windows,
        probe_stop_reasons=probe_stop_reasons,
        probe_family_expected_gain=average_family_score_counter(
            probe_family_expected_gain_totals,
            probe_family_expected_gain_counts,
        ),
        probe_family_realized_gain=average_family_scalar_counter(
            probe_family_realized_gain_totals,
            probe_family_realized_gain_counts,
        ),
        probe_family_future_error=average_family_scalar_counter(
            probe_family_future_error_totals,
            probe_family_future_error_counts,
        ),
        probe_family_selection_count={
            family: int(value)
            for family, value in probe_family_selection_count.items()
        },
        last_probe_stop_reason=last_probe_stop_reason,
        solve_probe_stop_reason=solve_probe_stop_reason,
        env_expression_eval_returns=solve_eval_returns,
        no_env_expression_eval_returns=no_env_expression_eval_returns,
        env_expression_ablation_delta=env_expression_ablation_delta,
        forced_env_expression_eval_returns=forced_env_expression_eval_returns,
        forced_env_expression_ablation_delta=forced_env_expression_ablation_delta,
        forced_env_expression_scale=(
            None
            if forced_env_expression_eval_returns is None
            else DEFAULT_FORCED_EVAL_EXPRESSION_SCALE
        ),
        post_expression_env_steps_total=total_control_env_steps,
        post_expression_episode_count=solved_episode,
        expression_scale_median=expression_scale_median,
        expression_scale_active_fraction=expression_scale_active_fraction,
        fair_ready_handoff_fraction=fair_ready_handoff_fraction,
        fair_expression_enabled_fraction=fair_expression_enabled_fraction,
        fair_expression_force_muted_fraction=fair_expression_force_muted_fraction,
        fair_ready_confidence_median=fair_ready_confidence_median,
        fair_muted_confidence_median=fair_muted_confidence_median,
        fair_stop_blocker_counts=dict(fair_stop_blocker_counts),
        expression_ready_but_muted_fraction=expression_ready_but_muted_fraction,
        readiness_reason_counts=dict(readiness_reason_counts),
        readiness_component_means=readiness_component_means,
        shadow_expression_enabled_fraction=shadow_expression_enabled_fraction,
        shadow_expression_scale_median=shadow_expression_scale_median,
        shadow_confidence_median=shadow_confidence_median,
        shadow_blocker_counts=dict(shadow_blocker_counts),
        shadow_strict_miss_fraction=shadow_strict_miss_fraction,
        second_probe_family_selection_count={
            family: int(value)
            for family, value in second_probe_family_selection_count.items()
            if int(value) > 0
        },
        second_probe_raw_future_gain_mean=second_probe_raw_future_gain_mean,
        second_probe_future_estimate_mean=second_probe_future_estimate_mean,
        second_probe_choice_future_gain_mean=second_probe_choice_future_gain_mean,
        family_coverage_satisfied_fraction=family_coverage_satisfied_fraction,
        second_probe_value_driven_fraction=second_probe_value_driven_fraction,
        uniformity_pressure_active_fraction=uniformity_pressure_active_fraction,
        fair_handoff_probe_families=(
            solve_fair_handoff_probe_families
            if solve_fair_handoff_probe_families is not None
            else last_fair_handoff_probe_families
        ),
        readiness_component_timeline=readiness_component_timeline,
        online_future_quality_trace=online_future_quality_trace,
        online_subset_stability_trace=online_subset_stability_trace,
        online_offline_gap_trace=online_offline_gap_trace,
        online_offline_gap_mean=online_offline_gap_mean,
        online_geometry_complete_fraction=online_geometry_complete_fraction,
        online_split_latent_disagreement_mean=online_split_latent_disagreement_mean,
        online_split_retrieval_margin_deficit_mean=online_split_retrieval_margin_deficit_mean,
        online_leaveout_shift_mean=online_leaveout_shift_mean,
        message_input_delta_mean=message_input_delta_mean,
        message_input_delta_max=message_input_delta_max,
        muted_message_input_delta_mean=muted_message_input_delta_mean,
        muted_message_input_delta_max=muted_message_input_delta_max,
        actor_message_norm_mean=actor_message_norm_mean,
        actor_message_nonzero_fraction=actor_message_nonzero_fraction,
        muted_actor_message_nonzero_fraction=muted_actor_message_nonzero_fraction,
        matched_mute_parity_fraction=matched_mute_parity_fraction,
        message_off_fraction=message_off_fraction,
        message_diag_fraction=message_diag_fraction,
        message_on_fraction=message_on_fraction,
        message_ablation_config_diff=message_ablation_config_diff,
        teacher_action_agreement=None,
        controller_style=(
            (
                f"belief_native_context_{full_system_context_source}"
                if belief_native_controller
                else "matched_env_expression"
            )
        ),
        zero_context_eval_returns=zero_context_eval_returns,
        shuffled_context_eval_returns=shuffled_context_eval_returns,
        stale_context_eval_returns=stale_context_eval_returns,
        no_online_refinement_eval_returns=no_online_refinement_eval_returns,
        zero_context_ablation_delta=zero_context_ablation_delta,
        shuffled_context_ablation_delta=shuffled_context_ablation_delta,
        stale_context_ablation_delta=stale_context_ablation_delta,
        online_refinement_ablation_delta=online_refinement_ablation_delta,
    )
