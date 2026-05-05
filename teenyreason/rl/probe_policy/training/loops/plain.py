"""Plain PPO baseline training loop."""

import numpy as np
import torch
import torch.optim as optim

from .....crawler.probes.data import apply_env_params, default_env_params
from .....crawler.probes.latent import EliteTrajectoryBuffer, select_episode_physics, should_promote_episode_to_elite
from .....envs import make_env
from ....core import (
    PlainGaussianActorCritic,
    RunningNormalizer,
    build_episode_batch,
    evaluate_continuous_actions,
    sample_continuous_action,
    sanitize_numpy,
    set_optimizer_lr,
    update_ppo_policy,
    validate_continuous_env,
)
from ...types import TrainingRunResult
from ..evaluation.plain import evaluate_plain_policy
from ..logging import print_plain_episode_status, print_solve_event
from ..support.progress import (
    cap_late_exploitation_action_std,
    late_exploitation_entropy_coef,
    maybe_extend_solve_episode_limit,
    rolling_solve_average,
)
from ..reporting import (
    format_peer_solve_status,
    restore_normalizer_state,
    snapshot_normalizer_state,
    snapshot_policy_state_dict,
)
from ..rollouts import (
    append_rollout_step,
    build_rollout_batch,
    clear_rollout_chunk,
    init_rollout_chunk,
    rollout_chunk_step_count,
)


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
        new_best = episode_return > best_return_so_far
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
        best_env_steps=best_env_steps,
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
