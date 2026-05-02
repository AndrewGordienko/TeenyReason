"""Benchmark configuration presets."""

from __future__ import annotations

from dataclasses import dataclass

from ..envs import (
    BIPEDAL_WALKER_NAME,
    CONTINUOUS_CARTPOLE_NAME,
    CONTINUOUS_LUNAR_LANDER_NAME,
)


@dataclass(frozen=True)
class ExperimentConfig:
    env_name: str
    benchmark_tag: str
    benchmark_profile: str
    benchmark_mode: str
    probe_budget_mode: str
    belief_mode: str
    sysid_epochs: int
    sysid_batch_size: int
    sysid_lr: float
    sysid_negative_count: int
    sysid_particle_count: int
    sysid_likelihood_scale: float
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
    encoder_lr: float
    encoder_kl_loss_weight: float
    encoder_contrastive_loss_weight: float
    encoder_contrastive_dim: int
    encoder_env_consistency_loss_weight: float
    encoder_env_geometry_loss_weight: float
    encoder_mode_adversary_loss_weight: float
    encoder_latent_rollout_loss_weight: float
    encoder_env_within_between_loss_weight: float
    encoder_belief_subset_count: int
    encoder_belief_subset_size: int
    belief_bits_per_dim: int
    belief_use_residual_sketch: bool
    latent_memory_capacity: int
    base_probe_episodes: int
    max_probe_episodes: int
    probe_adaptive_budget: bool
    probe_adaptive_policy_schedule: bool
    novelty_probe_threshold: float
    low_return_probe_threshold: float
    exploit_return_threshold: float
    uncertainty_probe_threshold: float
    uncertainty_focus_threshold: float
    surprise_probe_threshold: float
    online_z_update_alpha: float
    online_z_update_freq: int
    full_system_enabled: bool
    full_system_online_refinement: bool
    full_system_surprise_refresh_threshold: float
    full_system_context_source: str
    full_system_context_chunk_len: int
    full_system_context_zero_prob: float
    full_system_context_shuffle_prob: float
    full_system_context_stale_prob: float
    full_system_curriculum_schedule: tuple[tuple[int, float], ...]
    full_system_plateau_warmup_episodes: int
    full_system_plateau_patience: int
    full_system_plateau_best_return_delta: float
    full_system_plateau_avg50_delta: float
    sil_batch_size: int
    sil_policy_weight: float
    sil_value_weight: float
    min_elite_return: float
    elite_warmup_episodes: int
    elite_threshold_std_scale: float
    solved_return: float
    solve_eval_episodes: int
    full_system_ablation_eval_episodes: int
    belief_controller_eval_interval: int
    gamma: float
    gae_lambda: float
    lr: float
    clip_ratio: float
    value_clip_ratio: float | None
    ppo_epochs: int
    minibatch_size: int
    value_loss_weight: float
    entropy_coef: float
    max_grad_norm: float
    target_kl: float
    min_rollout_steps: int
    lr_anneal: bool
    hidden_dim: int
    initial_log_std: float
    normalize_rewards: bool
    num_episodes: int


def build_experiment_config(env_name: str) -> ExperimentConfig:
    """Return a sensible per-environment benchmark config."""
    if env_name == BIPEDAL_WALKER_NAME:
        return ExperimentConfig(
            env_name=env_name,
            benchmark_tag="bipedal_walker_ppo",
            benchmark_profile="full",
            benchmark_mode="fair",
            probe_budget_mode="fair_two_probe_handoff",
            belief_mode="latent_pool",
            sysid_epochs=0,
            sysid_batch_size=256,
            sysid_lr=3e-4,
            sysid_negative_count=15,
            sysid_particle_count=128,
            sysid_likelihood_scale=0.35,
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
            encoder_lr=1e-3,
            encoder_kl_loss_weight=2e-3,
            encoder_contrastive_loss_weight=0.30,
            encoder_contrastive_dim=64,
            encoder_env_consistency_loss_weight=0.45,
            encoder_env_geometry_loss_weight=0.25,
            encoder_mode_adversary_loss_weight=0.15,
            encoder_latent_rollout_loss_weight=0.20,
            encoder_env_within_between_loss_weight=0.30,
            encoder_belief_subset_count=4,
            encoder_belief_subset_size=6,
            belief_bits_per_dim=0,
            belief_use_residual_sketch=False,
            latent_memory_capacity=512,
            base_probe_episodes=2,
            max_probe_episodes=3,
            probe_adaptive_budget=False,
            probe_adaptive_policy_schedule=False,
            novelty_probe_threshold=0.15,
            low_return_probe_threshold=75.0,
            exploit_return_threshold=160.0,
            uncertainty_probe_threshold=0.24,
            uncertainty_focus_threshold=0.20,
            surprise_probe_threshold=0.75,
            online_z_update_alpha=0.30,
            online_z_update_freq=4,
            full_system_enabled=True,
            full_system_online_refinement=True,
            full_system_surprise_refresh_threshold=0.35,
            full_system_context_source="curriculum",
            full_system_context_chunk_len=32,
            full_system_context_zero_prob=0.20,
            full_system_context_shuffle_prob=0.10,
            full_system_context_stale_prob=0.10,
            full_system_curriculum_schedule=((250, 1.0), (500, 0.75), (750, 0.50), (900, 0.25), (10**9, 0.0)),
            full_system_plateau_warmup_episodes=400,
            full_system_plateau_patience=250,
            full_system_plateau_best_return_delta=10.0,
            full_system_plateau_avg50_delta=5.0,
            sil_batch_size=64,
            sil_policy_weight=0.10,
            sil_value_weight=0.10,
            min_elite_return=100.0,
            elite_warmup_episodes=25,
            elite_threshold_std_scale=1.5,
            solved_return=300.0,
            solve_eval_episodes=3,
            full_system_ablation_eval_episodes=5,
            belief_controller_eval_interval=50,
            gamma=0.99,
            gae_lambda=0.95,
            lr=2.5e-4,
            clip_ratio=0.2,
            value_clip_ratio=0.2,
            ppo_epochs=10,
            minibatch_size=256,
            value_loss_weight=0.5,
            entropy_coef=2e-3,
            max_grad_norm=0.5,
            target_kl=0.02,
            min_rollout_steps=1024,
            lr_anneal=True,
            hidden_dim=256,
            initial_log_std=-1.5,
            normalize_rewards=True,
            num_episodes=2000,
        )

    if env_name == CONTINUOUS_LUNAR_LANDER_NAME:
        return ExperimentConfig(
            env_name=env_name,
            benchmark_tag="continuous_lunar_lander_ppo",
            benchmark_profile="full",
            benchmark_mode="fair",
            probe_budget_mode="fair_two_probe_handoff",
            belief_mode="latent_pool",
            sysid_epochs=0,
            sysid_batch_size=256,
            sysid_lr=3e-4,
            sysid_negative_count=15,
            sysid_particle_count=128,
            sysid_likelihood_scale=0.35,
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
            encoder_lr=1e-3,
            encoder_kl_loss_weight=2e-3,
            encoder_contrastive_loss_weight=0.25,
            encoder_contrastive_dim=64,
            encoder_env_consistency_loss_weight=0.40,
            encoder_env_geometry_loss_weight=0.20,
            encoder_mode_adversary_loss_weight=0.10,
            encoder_latent_rollout_loss_weight=0.15,
            encoder_env_within_between_loss_weight=0.25,
            encoder_belief_subset_count=4,
            encoder_belief_subset_size=6,
            belief_bits_per_dim=0,
            belief_use_residual_sketch=False,
            latent_memory_capacity=512,
            base_probe_episodes=2,
            max_probe_episodes=3,
            probe_adaptive_budget=False,
            probe_adaptive_policy_schedule=False,
            novelty_probe_threshold=0.12,
            low_return_probe_threshold=80.0,
            exploit_return_threshold=180.0,
            uncertainty_probe_threshold=0.20,
            uncertainty_focus_threshold=0.16,
            surprise_probe_threshold=0.75,
            online_z_update_alpha=0.25,
            online_z_update_freq=4,
            full_system_enabled=True,
            full_system_online_refinement=True,
            full_system_surprise_refresh_threshold=0.35,
            full_system_context_source="curriculum",
            full_system_context_chunk_len=32,
            full_system_context_zero_prob=0.20,
            full_system_context_shuffle_prob=0.10,
            full_system_context_stale_prob=0.10,
            full_system_curriculum_schedule=((250, 1.0), (500, 0.75), (750, 0.50), (900, 0.25), (10**9, 0.0)),
            full_system_plateau_warmup_episodes=300,
            full_system_plateau_patience=175,
            full_system_plateau_best_return_delta=8.0,
            full_system_plateau_avg50_delta=4.0,
            sil_batch_size=64,
            sil_policy_weight=0.10,
            sil_value_weight=0.10,
            min_elite_return=120.0,
            elite_warmup_episodes=25,
            elite_threshold_std_scale=1.5,
            solved_return=200.0,
            solve_eval_episodes=3,
            full_system_ablation_eval_episodes=5,
            belief_controller_eval_interval=50,
            gamma=0.99,
            gae_lambda=0.95,
            lr=3e-4,
            clip_ratio=0.2,
            value_clip_ratio=0.2,
            ppo_epochs=10,
            minibatch_size=256,
            value_loss_weight=0.5,
            entropy_coef=3e-3,
            max_grad_norm=0.5,
            target_kl=0.02,
            min_rollout_steps=1024,
            lr_anneal=True,
            hidden_dim=256,
            initial_log_std=-0.5,
            normalize_rewards=True,
            num_episodes=1500,
        )

    if env_name == CONTINUOUS_CARTPOLE_NAME:
        return ExperimentConfig(
            env_name=env_name,
            benchmark_tag="continuous_cartpole_ppo",
            benchmark_profile="fast",
            benchmark_mode="fair",
            probe_budget_mode="fair_two_probe_handoff",
            belief_mode="particle_sysid",
            sysid_epochs=80,
            sysid_batch_size=256,
            sysid_lr=3e-4,
            sysid_negative_count=15,
            sysid_particle_count=128,
            sysid_likelihood_scale=0.35,
            window_size=16,
            z_dim=16,
            action_bins=9,
            randomize_physics=True,
            physics_loss_weight=0.10,
            affordance_loss_weight=1.0,
            intervention_horizon=12,
            probe_episodes_per_mode=50,
            probe_max_steps=320,
            encoder_epochs=40,
            encoder_batch_size=128,
            encoder_lr=3e-4,
            encoder_kl_loss_weight=2e-3,
            encoder_contrastive_loss_weight=0.20,
            encoder_contrastive_dim=64,
            encoder_env_consistency_loss_weight=0.55,
            encoder_env_geometry_loss_weight=0.35,
            encoder_mode_adversary_loss_weight=0.20,
            encoder_latent_rollout_loss_weight=0.20,
            encoder_env_within_between_loss_weight=0.35,
            encoder_belief_subset_count=1,
            encoder_belief_subset_size=4,
            belief_bits_per_dim=0,
            belief_use_residual_sketch=False,
            latent_memory_capacity=256,
            base_probe_episodes=2,
            max_probe_episodes=3,
            probe_adaptive_budget=False,
            probe_adaptive_policy_schedule=False,
            novelty_probe_threshold=0.10,
            low_return_probe_threshold=150.0,
            exploit_return_threshold=300.0,
            uncertainty_probe_threshold=0.18,
            uncertainty_focus_threshold=0.12,
            surprise_probe_threshold=0.75,
            online_z_update_alpha=0.25,
            online_z_update_freq=3,
            full_system_enabled=True,
            full_system_online_refinement=True,
            full_system_surprise_refresh_threshold=0.30,
            full_system_context_source="curriculum",
            full_system_context_chunk_len=32,
            full_system_context_zero_prob=0.20,
            full_system_context_shuffle_prob=0.10,
            full_system_context_stale_prob=0.10,
            full_system_curriculum_schedule=((250, 1.0), (500, 0.75), (750, 0.50), (900, 0.25), (10**9, 0.0)),
            full_system_plateau_warmup_episodes=200,
            full_system_plateau_patience=125,
            full_system_plateau_best_return_delta=10.0,
            full_system_plateau_avg50_delta=5.0,
            sil_batch_size=64,
            sil_policy_weight=0.10,
            sil_value_weight=0.10,
            min_elite_return=150.0,
            elite_warmup_episodes=20,
            elite_threshold_std_scale=1.5,
            solved_return=500.0,
            solve_eval_episodes=3,
            full_system_ablation_eval_episodes=3,
            belief_controller_eval_interval=25,
            gamma=0.99,
            gae_lambda=0.95,
            lr=3e-4,
            clip_ratio=0.2,
            value_clip_ratio=0.2,
            ppo_epochs=10,
            minibatch_size=256,
            value_loss_weight=0.5,
            entropy_coef=2e-3,
            max_grad_norm=0.5,
            target_kl=0.02,
            min_rollout_steps=512,
            lr_anneal=True,
            hidden_dim=128,
            initial_log_std=-1.5,
            normalize_rewards=False,
            num_episodes=1000,
        )

    raise ValueError(f"Unsupported benchmark environment: {env_name}")
