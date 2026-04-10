import random
import statistics
from pathlib import Path

import numpy as np
import torch

from envs import BIPEDAL_WALKER_NAME
from probe_data import ProbeCrawler
from probe_ppo import train_plain_ppo, train_probe_conditioned_ppo
from world_model import train_encoder_predictor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_array_shapes(title: str, arrays: dict[str, np.ndarray]):
    print(title)
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")


def save_training_artifacts(encoder, baseline_returns, probe_returns, benchmark_tag: str):
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    encoder_path = output_dir / f"{benchmark_tag}_encoder_state_dict.pt"
    baseline_returns_path = output_dir / f"{benchmark_tag}_baseline_ppo_returns.npy"
    probe_returns_path = output_dir / f"{benchmark_tag}_probe_ppo_returns.npy"

    torch.save(encoder.state_dict(), encoder_path)
    np.save(baseline_returns_path, np.asarray(baseline_returns, dtype=np.float32))
    np.save(probe_returns_path, np.asarray(probe_returns, dtype=np.float32))

    print(f"Saved encoder to {encoder_path}")
    print(f"Saved baseline returns to {baseline_returns_path}")
    print(f"Saved probe-conditioned returns to {probe_returns_path}")


def save_benchmark_results(benchmark_tag: str, seeds, baseline_solves, probe_solves, solve_cap: int):
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    benchmark_path = output_dir / f"{benchmark_tag}_solve_benchmark.npz"
    np.savez(
        benchmark_path,
        seeds=np.asarray(seeds, dtype=np.int64),
        baseline_solves=np.asarray(baseline_solves, dtype=np.int64),
        probe_solves=np.asarray(probe_solves, dtype=np.int64),
        solve_cap=np.asarray([solve_cap], dtype=np.int64),
    )
    print(f"Saved benchmark results to {benchmark_path}")


def print_return_summary(name: str, returns):
    print(
        f"{name}: "
        f"avg10={np.mean(returns[-10:]):.2f} | "
        f"avg50={np.mean(returns[-50:]):.2f}"
    )


def solve_episode(returns, solved_return: float):
    for idx, value in enumerate(returns, start=1):
        if value >= solved_return:
            return idx
    return None


def print_solve_summary(name: str, solves, solve_cap: int):
    success_count = sum(1 for value in solves if value > 0)
    capped_solves = [value if value > 0 else solve_cap for value in solves]
    print(
        f"{name}: "
        f"success_rate={success_count}/{len(solves)} | "
        f"capped_median={statistics.median(capped_solves):.2f} | "
        f"capped_mean={sum(capped_solves) / len(capped_solves):.2f} | "
        f"solves={solves}"
    )


def run_single_seed(seed: int, run_index: int = 1, total_runs: int = 1):
    set_seed(seed)

    env_name = BIPEDAL_WALKER_NAME
    window_size = 8
    z_dim = 16
    action_bins = 9
    randomize_physics = False
    physics_loss_weight = 0.0
    affordance_loss_weight = 1.25
    intervention_horizon = 12
    probe_episodes_per_mode = 60
    probe_max_steps = 400
    encoder_epochs = 60
    encoder_batch_size = 128
    benchmark_tag = "bipedal_walker_ppo"
    latent_memory_capacity = 512
    base_probe_episodes = 1
    max_probe_episodes = 2
    novelty_probe_threshold = 0.15
    low_return_probe_threshold = 75.0
    exploit_return_threshold = 160.0
    uncertainty_probe_threshold = 0.24
    uncertainty_focus_threshold = 0.20
    online_z_update_alpha = 0.20
    online_z_update_freq = 6
    sil_batch_size = 64
    sil_policy_weight = 0.10
    sil_value_weight = 0.10
    min_elite_return = 100.0
    elite_warmup_episodes = 25
    elite_threshold_std_scale = 1.5
    solved_return = 300.0
    gamma = 0.99
    gae_lambda = 0.95
    lr = 2.5e-4
    clip_ratio = 0.2
    ppo_epochs = 10
    minibatch_size = 256
    value_loss_weight = 0.5
    entropy_coef = 2e-3
    max_grad_norm = 0.5
    target_kl = 0.02
    min_rollout_steps = 1024
    hidden_dim = 256
    normalize_rewards = True
    num_episodes = 2000

    print(f"\n=== Seed {seed} ===")
    print("Collecting probe data for BipedalWalker...")
    crawler = ProbeCrawler(
        env_name=env_name,
        window_size=window_size,
        seed=seed,
        randomize_physics=randomize_physics,
        action_bins=action_bins,
    )
    crawler.collect(episodes_per_mode=probe_episodes_per_mode, max_steps=probe_max_steps)

    transitions = crawler.get_transition_arrays()
    windows = crawler.get_window_arrays()

    print_array_shapes("Transitions:", transitions)
    print()
    print_array_shapes("Windows:", windows)

    print("\nTraining encoder + delta predictor...")
    encoder, _predictor, device = train_encoder_predictor(
        windows=windows,
        z_dim=z_dim,
        epochs=encoder_epochs,
        batch_size=encoder_batch_size,
        lr=1e-3,
        physics_loss_weight=physics_loss_weight,
        affordance_loss_weight=affordance_loss_weight,
        action_vocab_size=crawler.action_dim,
        intervention_horizon=intervention_horizon,
        analytic_affordances=False,
    )

    crawler.close()

    print("\nTraining baseline PPO...")
    _baseline_policy, baseline_returns = train_plain_ppo(
        env_name=env_name,
        num_episodes=num_episodes,
        gamma=gamma,
        gae_lambda=gae_lambda,
        lr=lr,
        clip_ratio=clip_ratio,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        value_loss_weight=value_loss_weight,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        min_rollout_steps=min_rollout_steps,
        hidden_dim=hidden_dim,
        normalize_rewards=normalize_rewards,
        seed=seed,
        randomize_physics=randomize_physics,
        solved_return=solved_return,
        run_index=run_index,
        total_runs=total_runs,
        variant_label="baseline",
        peer_variant_label="probe",
        peer_solved_episode=None,
    )

    baseline_solve = solve_episode(baseline_returns, solved_return=solved_return)

    print("\nTraining probe-conditioned PPO...")
    _probe_policy, probe_returns = train_probe_conditioned_ppo(
        env_name=env_name,
        encoder=encoder,
        device=device,
        num_episodes=num_episodes,
        window_size=window_size,
        action_bins=action_bins,
        gamma=gamma,
        gae_lambda=gae_lambda,
        lr=lr,
        clip_ratio=clip_ratio,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        value_loss_weight=value_loss_weight,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        min_rollout_steps=min_rollout_steps,
        hidden_dim=hidden_dim,
        normalize_rewards=normalize_rewards,
        seed=seed,
        randomize_physics=randomize_physics,
        latent_memory_capacity=latent_memory_capacity,
        base_probe_episodes=base_probe_episodes,
        max_probe_episodes=max_probe_episodes,
        novelty_probe_threshold=novelty_probe_threshold,
        low_return_probe_threshold=low_return_probe_threshold,
        exploit_return_threshold=exploit_return_threshold,
        uncertainty_probe_threshold=uncertainty_probe_threshold,
        uncertainty_focus_threshold=uncertainty_focus_threshold,
        online_z_update_alpha=online_z_update_alpha,
        online_z_update_freq=online_z_update_freq,
        sil_batch_size=sil_batch_size,
        sil_policy_weight=sil_policy_weight,
        sil_value_weight=sil_value_weight,
        min_elite_return=min_elite_return,
        elite_warmup_episodes=elite_warmup_episodes,
        elite_threshold_std_scale=elite_threshold_std_scale,
        solved_return=solved_return,
        run_index=run_index,
        total_runs=total_runs,
        variant_label="probe",
        peer_variant_label="baseline",
        peer_solved_episode=baseline_solve,
    )

    save_training_artifacts(
        encoder,
        baseline_returns,
        probe_returns,
        benchmark_tag=benchmark_tag,
    )

    probe_solve = solve_episode(probe_returns, solved_return=solved_return)

    print_return_summary("Baseline PPO", baseline_returns)
    print_return_summary("Probe-conditioned PPO", probe_returns)
    print(
        "Solve episodes: "
        f"baseline={baseline_solve} | "
        f"probe-conditioned={probe_solve}"
    )

    return {
        "seed": seed,
        "baseline_solve": baseline_solve,
        "probe_solve": probe_solve,
    }


def run_training_pipeline():
    seeds = [0, 1, 2, 3, 4]
    benchmark_tag = "bipedal_walker_ppo"
    num_episodes = 2000
    solve_cap = num_episodes + 1
    results = []

    total_runs = len(seeds)
    for run_index, seed in enumerate(seeds, start=1):
        results.append(run_single_seed(seed, run_index=run_index, total_runs=total_runs))

    baseline_solves = [
        item["baseline_solve"] if item["baseline_solve"] is not None else -1
        for item in results
    ]
    probe_solves = [
        item["probe_solve"] if item["probe_solve"] is not None else -1
        for item in results
    ]

    print("\n=== Benchmark Summary ===")
    for item in results:
        print(
            f"seed={item['seed']} | "
            f"baseline_solve={item['baseline_solve']} | "
            f"probe_solve={item['probe_solve']}"
        )

    print_solve_summary("Baseline PPO solve episode", baseline_solves, solve_cap)
    print_solve_summary("Probe-conditioned PPO solve episode", probe_solves, solve_cap)
    save_benchmark_results(benchmark_tag, seeds, baseline_solves, probe_solves, solve_cap)


if __name__ == "__main__":
    run_training_pipeline()
