"""Policy training loops.

This module contains the direct comparison the project cares about:

- `train_plain_ppo`: standard PPO that only sees environment state
- `train_probe_conditioned_ppo`: PPO that first runs scripted probes, builds a
  latent belief, and conditions the policy/value function on that belief
"""

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs import BIPEDAL_WALKER_NAME, get_action_values, make_env
from ppo_core import (
    PlainGaussianActorCritic,
    ProbeConditionedGaussianActorCritic,
    RunningNormalizer,
    build_episode_batch,
    concat_episode_batches,
    evaluate_continuous_actions,
    sample_continuous_action,
    update_ppo_policy,
    validate_continuous_env,
)
from probe_data import ProbePolicy, apply_env_params, default_env_params
from probe_latent import (
    EliteTrajectoryBuffer,
    LatentPerformanceMemory,
    adjust_entropy_coef,
    belief_mean_z,
    belief_uncertainty,
    build_belief_vector,
    choose_policy_epochs,
    choose_probe_count,
    collect_probe_window,
    encode_window,
    maybe_update_online_belief,
    nearest_probe_action_idx,
    select_episode_physics,
    should_promote_episode_to_elite,
)
from world_model import WorldEncoder


def format_solve_status(solved_episode: int | None) -> str:
    """Render a compact solved/not-solved status for the episode logs."""
    if solved_episode is None:
        return "no"
    return f"yes@{solved_episode:04d}"


def format_peer_solve_status(peer_label: str, peer_solved_episode: int | None) -> str:
    """Render the sibling run's solve status in the same log-friendly format."""
    if peer_solved_episode is None:
        return f"{peer_label}=pending"
    return f"{peer_label}={peer_solved_episode:04d}"


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
    state_t = torch.tensor(states, dtype=torch.float32, device=device)
    belief_t = torch.tensor(beliefs, dtype=torch.float32, device=device)
    action_t = torch.tensor(actions, dtype=torch.float32, device=device)
    return_t = torch.tensor(returns_to_go, dtype=torch.float32, device=device)
    weight_t = torch.tensor(sample_weight, dtype=torch.float32, device=device)

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
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    ppo_epochs: int = 10,
    minibatch_size: int = 64,
    value_loss_weight: float = 0.5,
    entropy_coef: float = 0.005,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.02,
    min_rollout_steps: int = 256,
    hidden_dim: int = 128,
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
    run_index: int = 1,
    total_runs: int = 1,
    variant_label: str = "baseline",
    peer_variant_label: str = "probe",
    peer_solved_episode: int | None = None,
):
    """Train the plain PPO baseline with no probe conditioning."""
    env = make_env(env_name)
    action_low, action_high = validate_continuous_env(env)
    rng = np.random.default_rng(seed)
    base_physics = default_env_params(env_name, env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = PlainGaussianActorCritic(state_dim, action_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    state_normalizer = RunningNormalizer(state_dim)
    reward_normalizer = RunningNormalizer(1, clip=10.0) if normalize_rewards else None
    elite_buffer = EliteTrajectoryBuffer(elite_capacity)
    returns = []
    best_return_so_far = 0.0
    pending_batches = []
    pending_steps = 0
    solved_episode = None

    for episode in range(1, num_episodes + 1):
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        apply_env_params(env, episode_physics)
        raw_state, _info = env.reset()
        raw_state = np.asarray(raw_state, dtype=np.float32)
        state_normalizer.update(raw_state)
        state = state_normalizer.normalize(raw_state)

        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        terminals = []
        episode_return = 0.0
        done = False
        last_next_state = state.copy()
        last_terminated = False

        while not done:
            # Roll out one baseline policy step using only the normalized state.
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, value = policy(state_t)
            action, log_prob = sample_continuous_action(
                mean=mean,
                log_std=policy.log_std,
                action_low=action_low,
                action_high=action_high,
            )
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            next_raw_state = np.asarray(next_raw_state, dtype=np.float32)
            state_normalizer.update(next_raw_state)
            next_state = state_normalizer.normalize(next_raw_state)
            raw_reward = float(reward)
            train_reward = raw_reward
            if reward_normalizer is not None:
                reward_normalizer.update(np.asarray([[raw_reward]], dtype=np.float32))
                train_reward = float(
                    reward_normalizer.scale_only(np.asarray([raw_reward], dtype=np.float32))[0]
                )

            states.append(state.copy())
            actions.append(action.copy())
            log_probs.append(log_prob)
            rewards.append(train_reward)
            values.append(float(value.item()))
            terminals.append(float(terminated))

            episode_return += raw_reward
            state = next_state
            last_next_state = next_state.copy()
            last_terminated = bool(terminated)
            done = bool(terminated or truncated)

        bootstrap_value = 0.0
        if not last_terminated:
            next_state_t = torch.tensor(last_next_state[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                _mean, next_value = policy(next_state_t)
            bootstrap_value = float(next_value.item())

        batch = build_episode_batch(
            states=states,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            values=values,
            terminals=terminals,
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
        best_return_so_far = max(best_return_so_far, episode_return)
        pending_batches.append(batch)
        pending_steps += len(batch.states)
        avg_10 = np.mean(returns[-10:])
        avg_50 = np.mean(returns[-50:])
        current_solved_episode = episode if episode_return >= solved_return else solved_episode
        elite_tag = " | elite=yes" if is_elite_episode else ""
        print(
            f"[run {run_index}/{total_runs} | seed {seed} | {variant_label}] "
            f"episode {episode:04d} | "
            f"return={episode_return:7.2f} | "
            f"avg10={avg_10:7.2f} | "
            f"avg50={avg_50:7.2f} | "
            f"elite={len(elite_buffer):5d}"
            f"{elite_tag}"
        )
        print(
            "  summary | "
            f"solved={format_solve_status(current_solved_episode)} | "
            f"target={solved_return:7.2f} | "
            f"best={best_return_so_far:7.2f} | "
            f"{format_peer_solve_status(peer_variant_label, peer_solved_episode)}"
        )

        should_update = (
            pending_steps >= min_rollout_steps
            or episode == num_episodes
            or episode_return >= solved_return
        )
        if should_update:
            # Batch together several recent episodes before each PPO update.
            merged_batch = concat_episode_batches(pending_batches)
            auxiliary_loss_fn = None
            if len(elite_buffer) >= sil_batch_size:
                def auxiliary_loss_fn():
                    states_np, _beliefs_np, actions_np, returns_np, weights_np = elite_buffer.sample(sil_batch_size)
                    state_t = torch.tensor(states_np, dtype=torch.float32, device=device)
                    action_t = torch.tensor(actions_np, dtype=torch.float32, device=device)
                    return_t = torch.tensor(returns_np, dtype=torch.float32, device=device)
                    weight_t = torch.tensor(weights_np, dtype=torch.float32, device=device)
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
                batch=merged_batch,
                action_low=action_low,
                action_high=action_high,
                clip_ratio=clip_ratio,
                value_loss_weight=value_loss_weight,
                entropy_coef=entropy_coef,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                auxiliary_loss_fn=auxiliary_loss_fn,
            )
            pending_batches.clear()
            pending_steps = 0

        if episode_return >= solved_return:
            solved_episode = episode
            print(
                f"[run {run_index}/{total_runs} | seed {seed} | {variant_label}] "
                f"Solved environment at episode {episode:04d} with return {episode_return:.2f}"
            )
            break

    env.close()
    return policy, returns


def train_probe_conditioned_ppo(
    env_name: str,
    encoder: WorldEncoder,
    device: torch.device,
    num_episodes: int = 300,
    window_size: int = 8,
    action_bins: int = 9,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 3e-4,
    clip_ratio: float = 0.2,
    ppo_epochs: int = 10,
    minibatch_size: int = 64,
    value_loss_weight: float = 0.5,
    entropy_coef: float = 0.005,
    max_grad_norm: float = 0.5,
    target_kl: float = 0.02,
    min_rollout_steps: int = 256,
    hidden_dim: int = 128,
    normalize_rewards: bool = False,
    seed: int = 0,
    randomize_physics: bool = False,
    latent_memory_capacity: int = 512,
    base_probe_episodes: int = 1,
    max_probe_episodes: int = 3,
    novelty_probe_threshold: float = 0.12,
    low_return_probe_threshold: float = 180.0,
    exploit_return_threshold: float = 260.0,
    uncertainty_probe_threshold: float = 0.20,
    uncertainty_focus_threshold: float = 0.18,
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
    run_index: int = 1,
    total_runs: int = 1,
    variant_label: str = "probe",
    peer_variant_label: str = "baseline",
    peer_solved_episode: int | None = None,
):
    """Train PPO with an extra belief vector built from scripted environment probes."""
    train_env = make_env(env_name)
    probe_env = make_env(env_name)
    action_low, action_high = validate_continuous_env(train_env)
    action_values = get_action_values(train_env, action_bins, env_name=env_name)
    if action_values is None:
        raise ValueError("Probe-conditioned PPO expects a continuous control env")

    rng = np.random.default_rng(seed)
    base_physics = default_env_params(env_name, train_env)
    probe_profile = "scalar"
    if env_name == "LunarLanderContinuous-v3":
        probe_profile = "lunar_lander"
    if env_name == BIPEDAL_WALKER_NAME:
        probe_profile = "bipedal"
    probe_policy = ProbePolicy(len(action_values), profile=probe_profile)
    state_dim = train_env.observation_space.shape[0]

    with torch.no_grad():
        # Ask the encoder what latent size it produces so the policy can size its belief input.
        dummy_states = torch.zeros((1, window_size + 1, state_dim), dtype=torch.float32, device=device)
        dummy_actions = torch.zeros((1, window_size), dtype=torch.long, device=device)
        latent_dim = int(encoder(dummy_states, dummy_actions).shape[-1])
        belief_dim = latent_dim * 2

    policy = ProbeConditionedGaussianActorCritic(
        state_dim=state_dim,
        action_dim=train_env.action_space.shape[0],
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    state_normalizer = RunningNormalizer(state_dim)
    reward_normalizer = RunningNormalizer(1, clip=10.0) if normalize_rewards else None
    performance_memory = LatentPerformanceMemory(latent_memory_capacity)
    elite_buffer = EliteTrajectoryBuffer(elite_capacity)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    returns = []
    best_return_so_far = 0.0
    pending_batches = []
    pending_entropy = []
    pending_epochs = []
    pending_steps = 0
    solved_episode = None

    for episode in range(1, num_episodes + 1):
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        # Probe first, act second: each episode starts by asking the environment a few fixed questions.
        probe_modes = (
            "center_hold",
            "pulse_left",
            "pulse_right",
            "reverse",
            "sweep",
            "sticky_random",
        )

        first_states, first_actions, probe_failed = collect_probe_window(
            env=probe_env,
            probe_policy=probe_policy,
            rng=rng,
            window_size=window_size,
            episode_physics=episode_physics,
            action_values=action_values,
            probe_mode=probe_modes[0],
        )
        if probe_failed:
            # Skip this episode entirely if we cannot get even the initial probe window.
            print(f"episode {episode:04d} | probe phase failed repeatedly, skipping")
            returns.append(0.0)
            continue

        # Start with one probe latent, then optionally ask follow-up probes if the
        # memory/uncertainty heuristics suggest the environment is still unclear.
        probe_latents = [encode_window(encoder, device, first_states, first_actions)]
        probe_target_count, novelty, expected_return = choose_probe_count(
            z=probe_latents[0],
            performance_memory=performance_memory,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=max_probe_episodes,
            novelty_probe_threshold=novelty_probe_threshold,
            low_return_probe_threshold=low_return_probe_threshold,
        )

        probe_idx = 1
        while probe_idx < probe_target_count:
            extra_states, extra_actions, extra_failed = collect_probe_window(
                env=probe_env,
                probe_policy=probe_policy,
                rng=rng,
                window_size=window_size,
                episode_physics=episode_physics,
                action_values=action_values,
                probe_mode=probe_modes[probe_idx % len(probe_modes)],
            )
            if extra_failed:
                probe_idx += 1
                continue

            probe_latents.append(encode_window(encoder, device, extra_states, extra_actions))
            belief = build_belief_vector(probe_latents)
            if (
                belief_uncertainty(belief) >= uncertainty_probe_threshold
                and probe_target_count < max_probe_episodes
                and (
                    novelty >= novelty_probe_threshold
                    or expected_return < low_return_probe_threshold
                )
            ):
                probe_target_count += 1
            probe_idx += 1

        # Freeze the initial probe belief for logging/memory, then optionally refine it online.
        belief = build_belief_vector(probe_latents)
        episode_belief = belief.copy()
        mean_z = belief_mean_z(belief)
        uncertainty = belief_uncertainty(belief)
        expected_return = performance_memory.expected_return(mean_z)
        novelty = performance_memory.novelty(mean_z)
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

        apply_env_params(train_env, episode_physics)
        raw_state, _info = train_env.reset()
        raw_state = np.asarray(raw_state, dtype=np.float32)
        state_normalizer.update(raw_state)
        state = state_normalizer.normalize(raw_state)
        recent_states = deque([raw_state.copy()], maxlen=window_size + 1)
        recent_action_idx = deque(maxlen=window_size)

        states = []
        beliefs = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        terminals = []
        episode_return = 0.0
        episode_step = 0
        done = False
        last_next_state = state.copy()
        last_terminated = False

        while not done:
            # The actor sees the current environment state plus the current belief.
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            belief_t = torch.tensor(belief[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, value = policy(state_t, belief_t)
            action, log_prob = sample_continuous_action(
                mean=mean,
                log_std=policy.log_std,
                action_low=action_low,
                action_high=action_high,
            )

            next_raw_state, reward, terminated, truncated, _info = train_env.step(action)
            next_raw_state = np.asarray(next_raw_state, dtype=np.float32)
            recent_states.append(next_raw_state.copy())
            recent_action_idx.append(nearest_probe_action_idx(action, action_values))
            episode_step += 1
            raw_reward = float(reward)
            train_reward = raw_reward
            if reward_normalizer is not None:
                reward_normalizer.update(np.asarray([[raw_reward]], dtype=np.float32))
                train_reward = float(
                    reward_normalizer.scale_only(np.asarray([raw_reward], dtype=np.float32))[0]
                )

            # Convert the continuous action back to the probe vocabulary so the
            # encoder can digest the recent trajectory using the same action language.
            next_belief = maybe_update_online_belief(
                encoder=encoder,
                device=device,
                recent_states=recent_states,
                recent_action_idx=recent_action_idx,
                belief=belief,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                episode_step=episode_step,
            )
            state_normalizer.update(next_raw_state)
            next_state = state_normalizer.normalize(next_raw_state)

            states.append(state.copy())
            beliefs.append(belief.copy())
            actions.append(action.copy())
            log_probs.append(log_prob)
            rewards.append(train_reward)
            values.append(float(value.item()))
            terminals.append(float(terminated))

            state = next_state
            belief = next_belief
            episode_return += raw_reward
            last_next_state = next_state.copy()
            last_terminated = bool(terminated)
            done = bool(terminated or truncated)

        bootstrap_value = 0.0
        if not last_terminated:
            next_state_t = torch.tensor(last_next_state[None, :], dtype=torch.float32, device=device)
            next_belief_t = torch.tensor(belief[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                _mean, next_value = policy(next_state_t, next_belief_t)
            bootstrap_value = float(next_value.item())

        batch = build_episode_batch(
            states=states,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            values=values,
            terminals=terminals,
            bootstrap_value=bootstrap_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
            beliefs=beliefs,
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
        best_return_so_far = max(best_return_so_far, episode_return)
        performance_memory.push(belief_mean_z(episode_belief), episode_return)
        pending_batches.append(batch)
        pending_steps += len(batch.states)
        pending_entropy.append(episode_entropy_coef)
        pending_epochs.append(episode_ppo_epochs)
        avg_10 = np.mean(returns[-10:])
        avg_50 = np.mean(returns[-50:])
        current_solved_episode = episode if episode_return >= solved_return else solved_episode
        elite_tag = " | elite=yes" if is_elite_episode else ""
        print(
            f"[run {run_index}/{total_runs} | seed {seed} | {variant_label}] "
            f"episode {episode:04d} | "
            f"return={episode_return:7.2f} | "
            f"avg10={avg_10:7.2f} | "
            f"avg50={avg_50:7.2f} | "
            f"probes={probe_target_count:2d} | "
            f"uncert={belief_uncertainty(episode_belief):5.3f} | "
            f"epochs={episode_ppo_epochs:2d} | "
            f"elite={len(elite_buffer):5d}"
            f"{elite_tag}"
        )
        print(
            "  summary | "
            f"solved={format_solve_status(current_solved_episode)} | "
            f"target={solved_return:7.2f} | "
            f"best={best_return_so_far:7.2f} | "
            f"{format_peer_solve_status(peer_variant_label, peer_solved_episode)}"
        )

        should_update = (
            pending_steps >= min_rollout_steps
            or episode == num_episodes
            or episode_return >= solved_return
        )
        if should_update:
            # Probe-conditioned PPO can also average its entropy/epoch settings across pending episodes.
            merged_batch = concat_episode_batches(pending_batches)
            merged_entropy_coef = float(np.mean(pending_entropy))
            merged_ppo_epochs = int(round(np.mean(pending_epochs)))
            auxiliary_loss_fn = None
            if len(elite_buffer) >= sil_batch_size:
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
                batch=merged_batch,
                action_low=action_low,
                action_high=action_high,
                clip_ratio=clip_ratio,
                value_loss_weight=value_loss_weight,
                entropy_coef=merged_entropy_coef,
                ppo_epochs=merged_ppo_epochs,
                minibatch_size=minibatch_size,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                auxiliary_loss_fn=auxiliary_loss_fn,
            )
            pending_batches.clear()
            pending_entropy.clear()
            pending_epochs.clear()
            pending_steps = 0

        if episode_return >= solved_return:
            solved_episode = episode
            print(
                f"[run {run_index}/{total_runs} | seed {seed} | {variant_label}] "
                f"Solved environment at episode {episode:04d} with return {episode_return:.2f}"
            )
            break

    train_env.close()
    probe_env.close()
    return policy, returns
