"""Top-level experiment driver.

This file wires the whole benchmark together:

1. collect scripted probe data from the environment
2. train a latent encoder on those probe windows
3. train a plain PPO baseline
4. train a PPO agent conditioned on the learned probe belief
5. compare how quickly each variant solves the task across seeds
"""

import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..envs import (
    BIPEDAL_WALKER_NAME,
    CONTINUOUS_CARTPOLE_NAME,
    CONTINUOUS_LUNAR_LANDER_NAME,
)
from ..probe.probe_data import ProbeCrawler
from ..representation import build_latent_snapshot, save_latent_snapshot, train_encoder_predictor
from ..rl.probe_ppo import train_plain_ppo, train_probe_conditioned_ppo


@dataclass(frozen=True)
class ExperimentConfig:
    env_name: str
    benchmark_tag: str
    window_size: int
    z_dim: int
    action_bins: int
    randomize_physics: bool
    physics_loss_weight: float
    affordance_loss_weight: float
    intervention_horizon: int
    probe_episodes_per_mode: int
    probe_max_steps: int
    encoder_epochs: int
    encoder_batch_size: int
    latent_memory_capacity: int
    base_probe_episodes: int
    max_probe_episodes: int
    novelty_probe_threshold: float
    low_return_probe_threshold: float
    exploit_return_threshold: float
    uncertainty_probe_threshold: float
    uncertainty_focus_threshold: float
    online_z_update_alpha: float
    online_z_update_freq: int
    sil_batch_size: int
    sil_policy_weight: float
    sil_value_weight: float
    min_elite_return: float
    elite_warmup_episodes: int
    elite_threshold_std_scale: float
    solved_return: float
    solve_eval_episodes: int
    gamma: float
    gae_lambda: float
    lr: float
    clip_ratio: float
    ppo_epochs: int
    minibatch_size: int
    value_loss_weight: float
    entropy_coef: float
    max_grad_norm: float
    target_kl: float
    min_rollout_steps: int
    hidden_dim: int
    normalize_rewards: bool
    num_episodes: int


def build_experiment_config(env_name: str) -> ExperimentConfig:
    """Return a sensible per-environment benchmark config."""
    if env_name == BIPEDAL_WALKER_NAME:
        return ExperimentConfig(
            env_name=env_name,
            benchmark_tag="bipedal_walker_ppo",
            window_size=24,
            z_dim=24,
            action_bins=9,
            randomize_physics=False,
            physics_loss_weight=0.0,
            affordance_loss_weight=1.25,
            intervention_horizon=12,
            probe_episodes_per_mode=80,
            probe_max_steps=400,
            encoder_epochs=80,
            encoder_batch_size=128,
            latent_memory_capacity=512,
            base_probe_episodes=2,
            max_probe_episodes=3,
            novelty_probe_threshold=0.15,
            low_return_probe_threshold=75.0,
            exploit_return_threshold=160.0,
            uncertainty_probe_threshold=0.24,
            uncertainty_focus_threshold=0.20,
            online_z_update_alpha=0.30,
            online_z_update_freq=4,
            sil_batch_size=64,
            sil_policy_weight=0.10,
            sil_value_weight=0.10,
            min_elite_return=100.0,
            elite_warmup_episodes=25,
            elite_threshold_std_scale=1.5,
            solved_return=300.0,
            solve_eval_episodes=3,
            gamma=0.99,
            gae_lambda=0.95,
            lr=2.5e-4,
            clip_ratio=0.2,
            ppo_epochs=10,
            minibatch_size=256,
            value_loss_weight=0.5,
            entropy_coef=2e-3,
            max_grad_norm=0.5,
            target_kl=0.02,
            min_rollout_steps=1024,
            hidden_dim=256,
            normalize_rewards=True,
            num_episodes=2000,
        )

    if env_name == CONTINUOUS_LUNAR_LANDER_NAME:
        return ExperimentConfig(
            env_name=env_name,
            benchmark_tag="continuous_lunar_lander_ppo",
            window_size=24,
            z_dim=24,
            action_bins=9,
            randomize_physics=False,
            physics_loss_weight=0.05,
            affordance_loss_weight=1.0,
            intervention_horizon=12,
            probe_episodes_per_mode=60,
            probe_max_steps=300,
            encoder_epochs=60,
            encoder_batch_size=128,
            latent_memory_capacity=512,
            base_probe_episodes=2,
            max_probe_episodes=3,
            novelty_probe_threshold=0.12,
            low_return_probe_threshold=80.0,
            exploit_return_threshold=180.0,
            uncertainty_probe_threshold=0.20,
            uncertainty_focus_threshold=0.16,
            online_z_update_alpha=0.25,
            online_z_update_freq=4,
            sil_batch_size=64,
            sil_policy_weight=0.10,
            sil_value_weight=0.10,
            min_elite_return=120.0,
            elite_warmup_episodes=25,
            elite_threshold_std_scale=1.5,
            solved_return=200.0,
            solve_eval_episodes=3,
            gamma=0.99,
            gae_lambda=0.95,
            lr=3e-4,
            clip_ratio=0.2,
            ppo_epochs=10,
            minibatch_size=256,
            value_loss_weight=0.5,
            entropy_coef=3e-3,
            max_grad_norm=0.5,
            target_kl=0.02,
            min_rollout_steps=1024,
            hidden_dim=256,
            normalize_rewards=True,
            num_episodes=1500,
        )

    if env_name == CONTINUOUS_CARTPOLE_NAME:
        return ExperimentConfig(
            env_name=env_name,
            benchmark_tag="continuous_cartpole_ppo",
            window_size=16,
            z_dim=16,
            action_bins=9,
            randomize_physics=True,
            physics_loss_weight=0.10,
            affordance_loss_weight=1.0,
            intervention_horizon=12,
            probe_episodes_per_mode=50,
            probe_max_steps=250,
            encoder_epochs=40,
            encoder_batch_size=128,
            latent_memory_capacity=256,
            base_probe_episodes=2,
            max_probe_episodes=3,
            novelty_probe_threshold=0.10,
            low_return_probe_threshold=150.0,
            exploit_return_threshold=300.0,
            uncertainty_probe_threshold=0.18,
            uncertainty_focus_threshold=0.12,
            online_z_update_alpha=0.25,
            online_z_update_freq=3,
            sil_batch_size=64,
            sil_policy_weight=0.10,
            sil_value_weight=0.10,
            min_elite_return=150.0,
            elite_warmup_episodes=20,
            elite_threshold_std_scale=1.5,
            solved_return=500.0,
            solve_eval_episodes=3,
            gamma=0.99,
            gae_lambda=0.95,
            lr=3e-4,
            clip_ratio=0.2,
            ppo_epochs=10,
            minibatch_size=256,
            value_loss_weight=0.5,
            entropy_coef=2e-3,
            max_grad_norm=0.5,
            target_kl=0.02,
            min_rollout_steps=512,
            hidden_dim=128,
            normalize_rewards=False,
            num_episodes=1000,
        )

    raise ValueError(f"Unsupported benchmark environment: {env_name}")


def set_seed(seed: int):
    """Keep Python, NumPy, and Torch aligned for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_array_shapes(title: str, arrays: dict[str, np.ndarray]):
    """Small helper for inspecting the collected dataset before training."""
    print(title)
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")


def serialize_normalizer(normalizer) -> dict[str, float | torch.Tensor]:
    """Save enough running-normalizer state to reproduce evaluation preprocessing."""
    return {
        # Store tensors rather than NumPy arrays so checkpoints remain compatible
        # with PyTorch's safer weights-only loading path.
        "mean": torch.tensor(normalizer.mean, dtype=torch.float32),
        "var": torch.tensor(normalizer.var, dtype=torch.float32),
        "count": float(normalizer.count),
        "clip": float(normalizer.clip),
    }


def serialize_normalizer_state(normalizer_state: dict[str, np.ndarray | float]) -> dict[str, float | torch.Tensor]:
    """Serialize a cloned normalizer snapshot into a checkpoint-safe payload."""
    return {
        "mean": torch.tensor(normalizer_state["mean"], dtype=torch.float32),
        "var": torch.tensor(normalizer_state["var"], dtype=torch.float32),
        "count": float(normalizer_state["count"]),
        "clip": float(normalizer_state["clip"]),
    }


def save_training_artifacts(
    encoder,
    predictor,
    baseline_result,
    probe_result,
    artifact_tag: str,
    env_name: str,
    window_size: int,
    z_dim: int,
    action_bins: int,
    hidden_dim: int,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    base_probe_episodes: int,
    max_probe_episodes: int,
    randomize_physics: bool,
    solve_eval_episodes: int,
    solved_return: float,
):
    """Persist the main outputs from one benchmark configuration."""
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    encoder_path = output_dir / f"{artifact_tag}_encoder_state_dict.pt"
    baseline_returns_path = output_dir / f"{artifact_tag}_baseline_ppo_returns.npy"
    probe_returns_path = output_dir / f"{artifact_tag}_probe_ppo_returns.npy"
    baseline_checkpoint_path = output_dir / f"{artifact_tag}_baseline_ppo_checkpoint.pt"
    probe_checkpoint_path = output_dir / f"{artifact_tag}_probe_ppo_checkpoint.pt"

    torch.save(encoder.state_dict(), encoder_path)
    np.save(baseline_returns_path, np.asarray(baseline_result.returns, dtype=np.float32))
    np.save(probe_returns_path, np.asarray(probe_result.returns, dtype=np.float32))
    torch.save(
        {
            "env_name": env_name,
            "hidden_dim": hidden_dim,
            "action_bins": action_bins,
            "policy_state_dict": baseline_result.policy.state_dict(),
            "state_normalizer": serialize_normalizer(baseline_result.state_normalizer),
            "best_policy_state_dict": baseline_result.best_policy_state_dict,
            "best_state_normalizer": serialize_normalizer_state(
                baseline_result.best_state_normalizer_state
            ),
            "best_return": float(baseline_result.best_return),
            "best_episode": baseline_result.best_episode,
            "solve_policy_state_dict": baseline_result.solve_policy_state_dict,
            "solve_state_normalizer": None
            if baseline_result.solve_state_normalizer_state is None
            else serialize_normalizer_state(baseline_result.solve_state_normalizer_state),
            "solve_eval_returns": None
            if baseline_result.solve_eval_returns is None
            else torch.tensor(baseline_result.solve_eval_returns, dtype=torch.float32),
            "solved_episode": baseline_result.solved_episode,
            "solved_env_steps": baseline_result.solved_env_steps,
            "total_env_steps": baseline_result.total_env_steps,
            "randomize_physics": randomize_physics,
            "solve_eval_episodes": solve_eval_episodes,
            "solved_return": solved_return,
        },
        baseline_checkpoint_path,
    )
    torch.save(
        {
            "env_name": env_name,
            "window_size": window_size,
            "z_dim": z_dim,
            "action_bins": action_bins,
            "hidden_dim": hidden_dim,
            "online_z_update_alpha": online_z_update_alpha,
            "online_z_update_freq": online_z_update_freq,
            "base_probe_episodes": base_probe_episodes,
            "max_probe_episodes": max_probe_episodes,
            "encoder_state_dict": encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "predictor_ensemble_size": int(predictor.ensemble_size),
            "policy_state_dict": probe_result.policy.state_dict(),
            "state_normalizer": serialize_normalizer(probe_result.state_normalizer),
            "best_policy_state_dict": probe_result.best_policy_state_dict,
            "best_state_normalizer": serialize_normalizer_state(
                probe_result.best_state_normalizer_state
            ),
            "best_return": float(probe_result.best_return),
            "best_episode": probe_result.best_episode,
            "solve_policy_state_dict": probe_result.solve_policy_state_dict,
            "solve_state_normalizer": None
            if probe_result.solve_state_normalizer_state is None
            else serialize_normalizer_state(probe_result.solve_state_normalizer_state),
            "solve_eval_returns": None
            if probe_result.solve_eval_returns is None
            else torch.tensor(probe_result.solve_eval_returns, dtype=torch.float32),
            "solve_probe_count": probe_result.solve_probe_count,
            "solved_episode": probe_result.solved_episode,
            "solved_env_steps": probe_result.solved_env_steps,
            "total_env_steps": probe_result.total_env_steps,
            "randomize_physics": randomize_physics,
            "solve_eval_episodes": solve_eval_episodes,
            "solved_return": solved_return,
        },
        probe_checkpoint_path,
    )

    print(f"Saved encoder to {encoder_path}")
    print(f"Saved baseline returns to {baseline_returns_path}")
    print(f"Saved probe-conditioned returns to {probe_returns_path}")
    print(f"Saved baseline PPO checkpoint to {baseline_checkpoint_path}")
    print(f"Saved probe-conditioned PPO checkpoint to {probe_checkpoint_path}")


def save_benchmark_results(
    benchmark_tag: str,
    seeds,
    baseline_episode_solves,
    probe_episode_solves,
    baseline_step_solves,
    probe_step_solves,
    baseline_total_env_steps,
    probe_total_env_steps,
    baseline_completed_episodes,
    probe_completed_episodes,
    probe_encoder_steps,
):
    """Save the cross-seed solve summary in one compact file."""
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)

    benchmark_path = output_dir / f"{benchmark_tag}_solve_benchmark.npz"
    np.savez(
        benchmark_path,
        seeds=np.asarray(seeds, dtype=np.int64),
        baseline_solves=np.asarray(baseline_episode_solves, dtype=np.int64),
        probe_solves=np.asarray(probe_episode_solves, dtype=np.int64),
        baseline_episode_solves=np.asarray(baseline_episode_solves, dtype=np.int64),
        probe_episode_solves=np.asarray(probe_episode_solves, dtype=np.int64),
        baseline_step_solves=np.asarray(baseline_step_solves, dtype=np.int64),
        probe_step_solves=np.asarray(probe_step_solves, dtype=np.int64),
        baseline_total_env_steps=np.asarray(baseline_total_env_steps, dtype=np.int64),
        probe_total_env_steps=np.asarray(probe_total_env_steps, dtype=np.int64),
        baseline_completed_episodes=np.asarray(baseline_completed_episodes, dtype=np.int64),
        probe_completed_episodes=np.asarray(probe_completed_episodes, dtype=np.int64),
        probe_encoder_steps=np.asarray(probe_encoder_steps, dtype=np.int64),
    )
    print(f"Saved benchmark results to {benchmark_path}")


def print_return_summary(name: str, returns):
    """Report short-horizon and medium-horizon return averages."""
    print(
        f"{name}: "
        f"avg10={np.mean(returns[-10:]):.2f} | "
        f"avg50={np.mean(returns[-50:]):.2f}"
    )


def print_solve_summary(name: str, solves, unsolved_caps):
    """Summarize solve speed with an explicit per-run cap for unsolved seeds."""
    success_count = sum(1 for value in solves if value is not None)
    capped_solves = [
        value if value is not None else cap
        for value, cap in zip(solves, unsolved_caps)
    ]
    display_solves = [value if value is not None else -1 for value in solves]
    print(
        f"{name}: "
        f"success_rate={success_count}/{len(solves)} | "
        f"capped_median={statistics.median(capped_solves):.2f} | "
        f"capped_mean={sum(capped_solves) / len(capped_solves):.2f} | "
        f"solves={display_solves}"
    )


def run_single_seed(
    seed: int,
    config: ExperimentConfig,
    run_index: int = 1,
    total_runs: int = 1,
):
    """Run the full benchmark pipeline for one seed."""
    set_seed(seed)

    artifact_tag = f"{config.benchmark_tag}_seed_{seed}"

    print(f"\n=== Seed {seed} | env={config.env_name} ===")
    print(f"Collecting probe data for {config.env_name}...")
    # First collect short scripted probe rollouts that the encoder will learn from.
    crawler = ProbeCrawler(
        env_name=config.env_name,
        window_size=config.window_size,
        seed=seed,
        randomize_physics=config.randomize_physics,
        action_bins=config.action_bins,
    )
    crawler.collect(episodes_per_mode=config.probe_episodes_per_mode, max_steps=config.probe_max_steps)

    transitions = crawler.get_transition_arrays()
    windows = crawler.get_window_arrays()
    encoder_probe_steps = int(transitions["state"].shape[0])

    print_array_shapes("Transitions:", transitions)
    print()
    print_array_shapes("Windows:", windows)
    print(f"\nProbe encoder data collection steps: {encoder_probe_steps}")

    print("\nTraining encoder + delta predictor...")
    # Train the latent encoder before either PPO variant so both runs see the same probe model.
    encoder, predictor, device = train_encoder_predictor(
        windows=windows,
        z_dim=config.z_dim,
        epochs=config.encoder_epochs,
        batch_size=config.encoder_batch_size,
        lr=1e-3,
        physics_loss_weight=config.physics_loss_weight,
        affordance_loss_weight=config.affordance_loss_weight,
        decision_loss_weight=1.0,
        return_loss_weight=0.5,
        risk_loss_weight=0.25,
        kl_loss_weight=2e-3,
        ensemble_size=4,
        action_vocab_size=crawler.action_dim,
        intervention_horizon=config.intervention_horizon,
        analytic_affordances=False,
        env_name=config.env_name,
    )

    latent_snapshot = build_latent_snapshot(
        encoder=encoder,
        device=device,
        windows=windows,
    )
    latent_snapshot_path = Path("artifacts") / f"{artifact_tag}_latent_snapshot.npz"
    save_latent_snapshot(latent_snapshot_path, latent_snapshot)
    print(f"Saved latent snapshot to {latent_snapshot_path}")

    crawler.close()

    print("\nTraining baseline PPO...")
    # Baseline PPO gets only environment state.
    baseline_result = train_plain_ppo(
        env_name=config.env_name,
        num_episodes=config.num_episodes,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        lr=config.lr,
        clip_ratio=config.clip_ratio,
        ppo_epochs=config.ppo_epochs,
        minibatch_size=config.minibatch_size,
        value_loss_weight=config.value_loss_weight,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        min_rollout_steps=config.min_rollout_steps,
        hidden_dim=config.hidden_dim,
        normalize_rewards=config.normalize_rewards,
        seed=seed,
        randomize_physics=config.randomize_physics,
        solved_return=config.solved_return,
        solve_eval_episodes=config.solve_eval_episodes,
        run_index=run_index,
        total_runs=total_runs,
        variant_label="baseline",
        peer_variant_label="probe",
        peer_solved_episode=None,
    )

    baseline_returns = baseline_result.returns
    baseline_solve = baseline_result.solved_episode

    print("\nTraining probe-conditioned PPO...")
    # Probe-conditioned PPO gets both state and the probe-derived belief vector.
    probe_result = train_probe_conditioned_ppo(
        env_name=config.env_name,
        encoder=encoder,
        predictor=predictor,
        device=device,
        num_episodes=config.num_episodes,
        window_size=config.window_size,
        action_bins=config.action_bins,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        lr=config.lr,
        clip_ratio=config.clip_ratio,
        ppo_epochs=config.ppo_epochs,
        minibatch_size=config.minibatch_size,
        value_loss_weight=config.value_loss_weight,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
        min_rollout_steps=config.min_rollout_steps,
        hidden_dim=config.hidden_dim,
        normalize_rewards=config.normalize_rewards,
        seed=seed,
        randomize_physics=config.randomize_physics,
        latent_memory_capacity=config.latent_memory_capacity,
        base_probe_episodes=config.base_probe_episodes,
        max_probe_episodes=config.max_probe_episodes,
        novelty_probe_threshold=config.novelty_probe_threshold,
        low_return_probe_threshold=config.low_return_probe_threshold,
        exploit_return_threshold=config.exploit_return_threshold,
        uncertainty_probe_threshold=config.uncertainty_probe_threshold,
        uncertainty_focus_threshold=config.uncertainty_focus_threshold,
        online_z_update_alpha=config.online_z_update_alpha,
        online_z_update_freq=config.online_z_update_freq,
        sil_batch_size=config.sil_batch_size,
        sil_policy_weight=config.sil_policy_weight,
        sil_value_weight=config.sil_value_weight,
        min_elite_return=config.min_elite_return,
        elite_warmup_episodes=config.elite_warmup_episodes,
        elite_threshold_std_scale=config.elite_threshold_std_scale,
        solved_return=config.solved_return,
        solve_eval_episodes=config.solve_eval_episodes,
        run_index=run_index,
        total_runs=total_runs,
        variant_label="probe",
        peer_variant_label="baseline",
        peer_solved_episode=baseline_solve,
    )

    probe_returns = probe_result.returns

    save_training_artifacts(
        encoder,
        predictor,
        baseline_result,
        probe_result,
        artifact_tag=artifact_tag,
        env_name=config.env_name,
        window_size=config.window_size,
        z_dim=config.z_dim,
        action_bins=config.action_bins,
        hidden_dim=config.hidden_dim,
        online_z_update_alpha=config.online_z_update_alpha,
        online_z_update_freq=config.online_z_update_freq,
        base_probe_episodes=config.base_probe_episodes,
        max_probe_episodes=config.max_probe_episodes,
        randomize_physics=config.randomize_physics,
        solve_eval_episodes=config.solve_eval_episodes,
        solved_return=config.solved_return,
    )

    probe_solve = probe_result.solved_episode
    probe_solve_env_steps = (
        None
        if probe_result.solved_env_steps is None
        else probe_result.solved_env_steps + encoder_probe_steps
    )
    probe_total_env_steps = probe_result.total_env_steps + encoder_probe_steps

    print_return_summary("Baseline PPO", baseline_returns)
    print_return_summary("Probe-conditioned PPO", probe_returns)
    print(
        "Solve episodes: "
        f"baseline={baseline_solve} | "
        f"probe-conditioned={probe_solve}"
    )
    print(
        "Solve env steps (end-to-end): "
        f"baseline={baseline_result.solved_env_steps} | "
        f"probe-conditioned={probe_solve_env_steps}"
    )

    return {
        "seed": seed,
        "baseline_solve_episode": baseline_solve,
        "probe_solve_episode": probe_solve,
        "baseline_solve_env_steps": baseline_result.solved_env_steps,
        "probe_solve_env_steps": probe_solve_env_steps,
        "baseline_total_env_steps": baseline_result.total_env_steps,
        "probe_total_env_steps": probe_total_env_steps,
        "baseline_completed_episodes": len(baseline_returns),
        "probe_completed_episodes": len(probe_returns),
        "probe_encoder_steps": encoder_probe_steps,
    }


def run_training_pipeline(
    env_name: str = BIPEDAL_WALKER_NAME,
    seeds: list[int] | None = None,
):
    """Benchmark the current setup across a small fixed seed set."""
    config = build_experiment_config(env_name)
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    results = []

    total_runs = len(seeds)
    # Treat each seed as one benchmark run so the logs can report progress cleanly.
    for run_index, seed in enumerate(seeds, start=1):
        results.append(
            run_single_seed(
                seed,
                config=config,
                run_index=run_index,
                total_runs=total_runs,
            )
        )

    baseline_episode_solves = [item["baseline_solve_episode"] for item in results]
    probe_episode_solves = [item["probe_solve_episode"] for item in results]
    baseline_step_solves = [item["baseline_solve_env_steps"] for item in results]
    probe_step_solves = [item["probe_solve_env_steps"] for item in results]
    baseline_total_env_steps = [item["baseline_total_env_steps"] for item in results]
    probe_total_env_steps = [item["probe_total_env_steps"] for item in results]
    baseline_completed_episodes = [item["baseline_completed_episodes"] for item in results]
    probe_completed_episodes = [item["probe_completed_episodes"] for item in results]
    probe_encoder_steps = [item["probe_encoder_steps"] for item in results]

    print("\n=== Benchmark Summary ===")
    for item in results:
        print(
            f"seed={item['seed']} | "
            f"baseline_ep={item['baseline_solve_episode']} | "
            f"baseline_steps={item['baseline_solve_env_steps']} | "
            f"probe_ep={item['probe_solve_episode']} | "
            f"probe_steps={item['probe_solve_env_steps']} | "
            f"probe_encoder_steps={item['probe_encoder_steps']}"
        )

    print_solve_summary(
        "Baseline PPO solve episode",
        baseline_episode_solves,
        baseline_completed_episodes,
    )
    print_solve_summary(
        "Probe-conditioned PPO solve episode",
        probe_episode_solves,
        probe_completed_episodes,
    )
    print_solve_summary(
        "Baseline PPO solve env steps",
        baseline_step_solves,
        baseline_total_env_steps,
    )
    print_solve_summary(
        "Probe-conditioned PPO solve env steps",
        probe_step_solves,
        probe_total_env_steps,
    )
    save_benchmark_results(
        config.benchmark_tag,
        seeds,
        [value if value is not None else -1 for value in baseline_episode_solves],
        [value if value is not None else -1 for value in probe_episode_solves],
        [value if value is not None else -1 for value in baseline_step_solves],
        [value if value is not None else -1 for value in probe_step_solves],
        baseline_total_env_steps,
        probe_total_env_steps,
        baseline_completed_episodes,
        probe_completed_episodes,
        probe_encoder_steps,
    )


if __name__ == "__main__":
    run_training_pipeline()
