"""Render a saved probe-conditioned PPO policy."""

import argparse
import pickle

import numpy as np
import torch

from teenyreason.crawler import load_crawler_bundle_from_checkpoint
from teenyreason.envs import get_action_values, make_env
from teenyreason.probe.explorer import build_probe_planner
from teenyreason.rl.core import (
    ProbeConditionedGaussianActorCritic,
    RunningNormalizer,
    mean_to_continuous_action,
    sample_continuous_action,
    validate_continuous_env,
)
from teenyreason.probe.probe_data import apply_env_params, default_env_params
from teenyreason.probe.probe_latent import (
    collect_adaptive_probe_window,
    init_recurrent_belief_hidden,
    maybe_update_online_belief,
    nearest_probe_action_idx,
    probe_group_ids_from_families,
    select_episode_physics,
    update_recurrent_belief_from_window,
)


def load_normalizer(normalizer_state: dict) -> RunningNormalizer:
    """Rebuild a RunningNormalizer from a saved checkpoint dict."""
    mean = np.asarray(normalizer_state["mean"], dtype=np.float64)
    var = np.asarray(normalizer_state["var"], dtype=np.float64)
    normalizer = RunningNormalizer(
        shape=mean.shape,
        clip=float(normalizer_state["clip"]),
    )
    normalizer.mean = mean
    normalizer.var = var
    normalizer.count = float(normalizer_state["count"])
    return normalizer


def load_checkpoint(path: str) -> dict:
    """Load both new safe checkpoints and older local checkpoints."""
    try:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # Older checkpoints in this repo stored NumPy arrays in the normalizer
        # state, which requires opting out of weights_only loading.
        return torch.load(path, map_location="cpu", weights_only=False)


def build_solver_belief_input(payload: dict[str, np.ndarray]) -> np.ndarray:
    """Build the solver-facing env-expression vector stored in the crawler payload."""
    message = np.asarray(
        payload.get("env_expression", payload.get("belief_message", payload["belief"])),
        dtype=np.float32,
    ).reshape(-1)
    confidence = np.asarray(
        payload.get(
            "env_expression_confidence",
            payload.get("belief_message_confidence", np.asarray([1.0], dtype=np.float32)),
        ),
        dtype=np.float32,
    ).reshape(-1)
    uncertainty = np.asarray(
        payload.get(
            "env_expression_uncertainty",
            payload.get("belief_message_uncertainty", np.asarray([0.0], dtype=np.float32)),
        ),
        dtype=np.float32,
    ).reshape(-1)
    return np.concatenate([message, confidence[:1], uncertainty[:1]], axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Play a saved probe-conditioned PPO policy.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/bipedal_walker_ppo_seed_0_probe_ppo_checkpoint.pt",
        help="Path to the saved probe PPO checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to render.",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=None,
        help="Override how many probe windows to run before each rendered episode.",
    )
    parser.add_argument(
        "--use-final-policy",
        action="store_true",
        help="Render the final saved policy instead of the best-return snapshot when available.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of using the deterministic mean action.",
    )
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    env_name = checkpoint["env_name"]
    window_size = int(checkpoint["window_size"])
    z_dim = int(checkpoint["z_dim"])
    belief_dim = int(checkpoint.get("belief_dim", z_dim + 2))
    action_bins = int(checkpoint["action_bins"])
    hidden_dim = int(checkpoint["hidden_dim"])
    online_z_update_alpha = float(checkpoint["online_z_update_alpha"])
    online_z_update_freq = int(checkpoint["online_z_update_freq"])
    base_probe_episodes = int(checkpoint["base_probe_episodes"])
    max_probe_episodes = int(checkpoint.get("max_probe_episodes", base_probe_episodes))
    randomize_physics = bool(checkpoint.get("randomize_physics", False))
    solve_probe_count = checkpoint.get("solve_probe_count")

    device = torch.device("cpu")
    env = make_env(env_name, render_mode="human")
    probe_env = make_env(env_name)
    action_low, action_high = validate_continuous_env(env)
    action_values = get_action_values(env, action_bins, env_name=env_name)
    if action_values is None:
        raise ValueError("Probe-conditioned playback expects a continuous control environment.")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    crawler_bundle = load_crawler_bundle_from_checkpoint(
        checkpoint=checkpoint,
        state_dim=state_dim,
        action_vocab_size=len(action_values),
        device=device,
    )
    encoder = crawler_bundle.encoder
    predictor = crawler_bundle.predictor
    belief_aggregator = crawler_bundle.belief_aggregator
    env_param_predictor = crawler_bundle.env_param_predictor

    policy = ProbeConditionedGaussianActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        belief_dim=belief_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    policy_state_dict = checkpoint["policy_state_dict"]
    normalizer_state = checkpoint["state_normalizer"]
    selected_policy_label = "final"
    if not args.use_final_policy and checkpoint.get("solve_policy_state_dict") is not None:
        policy_state_dict = checkpoint["solve_policy_state_dict"]
        normalizer_state = checkpoint.get("solve_state_normalizer", normalizer_state)
        selected_policy_label = "solve"
    elif not args.use_final_policy and "best_policy_state_dict" in checkpoint:
        policy_state_dict = checkpoint["best_policy_state_dict"]
        normalizer_state = checkpoint.get("best_state_normalizer", normalizer_state)
        selected_policy_label = "best"
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    state_normalizer = load_normalizer(normalizer_state)
    if selected_policy_label == "solve":
        if checkpoint.get("solve_eval_returns") is not None:
            solve_eval_returns = np.asarray(checkpoint["solve_eval_returns"], dtype=np.float32)
            print(
                "Loaded solve policy snapshot | "
                f"episode={checkpoint.get('solved_episode')} | "
                f"eval_returns={np.round(solve_eval_returns, 2).tolist()} | "
                f"probe_count={solve_probe_count}"
            )
        else:
            print(
                "Loaded solve policy snapshot | "
                f"episode={checkpoint.get('solved_episode')} | "
                f"probe_count={solve_probe_count}"
            )
    elif selected_policy_label == "best" and checkpoint.get("best_episode") is not None:
        print(
            "Loaded best policy snapshot | "
            f"episode={checkpoint.get('best_episode')} | "
            f"return={float(checkpoint.get('best_return', 0.0)):.2f}"
        )
    else:
        print("Loaded final policy snapshot")
    probe_count = args.probe_count
    if probe_count is None:
        if selected_policy_label == "solve" and solve_probe_count is not None:
            probe_count = int(solve_probe_count)
        elif selected_policy_label in {"solve", "best"}:
            probe_count = max_probe_episodes
        else:
            probe_count = base_probe_episodes

    base_physics = default_env_params(env_name, env)

    for episode in range(1, args.episodes + 1):
        rng = np.random.default_rng(episode)
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        belief = None
        belief_hidden = init_recurrent_belief_hidden(encoder=encoder, device=device)
        belief_posteriors: list[tuple[np.ndarray, np.ndarray]] = []
        probe_families: list[str | None] = []
        probe_planner = build_probe_planner(
            action_space=env.action_space,
            action_values=action_values,
            rng=rng,
            env_name=env_name,
        )
        if probe_planner is not None:
            probe_planner.begin_env_instance()
        for _ in range(probe_count):
            window_states, window_actions, window_rewards, probe_failed, _probe_steps_used = collect_adaptive_probe_window(
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
            if probe_failed:
                raise RuntimeError("Could not collect a full probe window for playback.")
            window_posterior = crawler_bundle.encode_probe_window(
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
            observed_family = (
                None
                if probe_planner is None
                else getattr(probe_planner, "current_goal", None)
            )
            probe_families.append(observed_family)
            probe_group_ids = probe_group_ids_from_families(
                probe_families,
                family_names=crawler_bundle.family_names,
            )
            belief, payload = crawler_bundle.build_env_belief(
                posterior_views=belief_posteriors,
                probe_group_ids=probe_group_ids,
                bits_per_dim=int(checkpoint.get("belief_bits_per_dim", 0)),
                use_residual_sketch=bool(checkpoint.get("belief_use_residual_sketch", False)),
            )
        episode_belief = build_solver_belief_input(payload)

        apply_env_params(env, episode_physics)
        raw_state, _info = env.reset()
        raw_state = np.asarray(raw_state, dtype=np.float32)
        episode_return = 0.0
        episode_step = 0
        done = False

        while not done:
            env.render()
            state = state_normalizer.normalize(raw_state)
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            belief_t = torch.tensor(episode_belief[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, _value = policy(state_t, belief_t)
            if args.stochastic:
                action, _log_prob = sample_continuous_action(
                    mean=mean,
                    log_std=policy.log_std,
                    action_low=action_low,
                    action_high=action_high,
                )
            else:
                action = mean_to_continuous_action(mean, action_low, action_high)

            prev_raw_state = raw_state.copy()
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            next_raw_state = np.asarray(next_raw_state, dtype=np.float32)
            episode_step += 1
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
            if episode_step % online_z_update_freq == 0:
                belief, payload = crawler_bundle.build_env_belief(
                    posterior_views=belief_posteriors,
                    bits_per_dim=int(checkpoint.get("belief_bits_per_dim", 0)),
                    use_residual_sketch=bool(checkpoint.get("belief_use_residual_sketch", False)),
                )
                episode_belief = build_solver_belief_input(payload)
            raw_state = next_raw_state
            episode_return += float(reward)
            done = bool(terminated or truncated)

        print(
            f"rendered episode {episode:02d} | return={episode_return:.2f} | "
            f"probes={probe_count} | stochastic={'yes' if args.stochastic else 'no'} | "
            f"randomized_physics={'yes' if randomize_physics else 'no'}"
        )

    env.close()
    probe_env.close()


if __name__ == "__main__":
    main()
