"""Scenario-memory policy improvement for generic continuous control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...gym_mpc import TransitionBatch, normalize_actions
from ..config import AdvancedGymMPCConfig
from .actor_critic import ActorCriticAgent, ReplayBuffer, soft_update


@dataclass
class ScenarioPolicyState:
    """Persistent actor-critic state for scenario-based practice."""

    agent: ActorCriticAgent
    replay: ReplayBuffer
    behavior_pretrain_loss: float


def create_policy_state(
    config: AdvancedGymMPCConfig,
    batch: TransitionBatch,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> ScenarioPolicyState:
    replay = ReplayBuffer.from_batch(batch, config)
    agent = ActorCriticAgent.create(batch, action_low, action_high, config)
    pretrain_loss = pretrain_real_behavior(config, replay, agent)
    return ScenarioPolicyState(agent=agent, replay=replay, behavior_pretrain_loss=float(pretrain_loss))


def append_real_batch(state: ScenarioPolicyState, batch: TransitionBatch | None, config: AdvancedGymMPCConfig) -> None:
    if batch is not None and int(batch.observations.shape[0]) > 0:
        state.replay.append_batch(batch, config)


def append_imagined_variants(
    state: ScenarioPolicyState,
    variants: list[object],
    config: AdvancedGymMPCConfig,
) -> dict[str, float]:
    rows: list[dict[str, np.ndarray | float]] = []
    priorities: list[float] = []
    for variant in variants:
        row_priority = variant_priority(variant, config)
        for row in variant.rows:
            rows.append(row)
            priorities.append(row_priority)
    if not rows:
        return {"scenario_imagined_replay_added": 0.0, "scenario_imagined_priority_mean": 0.0}
    obs = np.asarray([row["observation"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    rewards = np.asarray([float(row["reward"]) for row in rows], dtype=np.float32)
    next_obs = np.asarray([row["next_observation"] for row in rows], dtype=np.float32)
    dones = np.asarray([float(row["done"]) for row in rows], dtype=np.float32)
    priority = np.maximum(np.asarray(priorities, dtype=np.float32), 1e-4)
    priority = priority / max(float(np.mean(priority)), 1e-6)
    state.replay.append(obs, actions, rewards, next_obs, dones, priority, imagined=True)
    return {
        "scenario_imagined_replay_added": float(len(rows)),
        "scenario_imagined_priority_mean": float(np.mean(priority)),
        "scenario_imagined_priority_max": float(np.max(priority)),
    }


def train_scenario_policy(
    config: AdvancedGymMPCConfig,
    state: ScenarioPolicyState,
    *,
    round_idx: int,
) -> dict[str, float]:
    updates = max(1, int(config.scenario_actor_critic_updates))
    rng = np.random.default_rng(int(config.seed) + 241_000 + int(round_idx) * 997)
    stats = {
        "scenario_critic_td_loss": [],
        "scenario_critic_q_mean": [],
        "scenario_actor_q_mean": [],
        "scenario_real_behavior_loss": [],
        "scenario_actor_smoothness_loss": [],
    }
    for update in range(updates):
        train_critic_step(config, state, rng, stats)
        if update % max(1, int(config.actor_critic_policy_delay)) == 0:
            train_actor_step(config, state, rng, stats)
            soft_update(state.agent.actor_target, state.agent.actor, tau=float(config.actor_critic_tau))
            soft_update(state.agent.critic1_target, state.agent.critic1, tau=float(config.actor_critic_tau))
            soft_update(state.agent.critic2_target, state.agent.critic2, tau=float(config.actor_critic_tau))
    return {
        "scenario_policy_update_mode": 1.0,
        "scenario_policy_updates": float(updates),
        "scenario_replay_real_count": float(state.replay.real_count),
        "scenario_replay_imagined_count": float(state.replay.imagined_count),
        "scenario_behavior_pretrain_loss": float(state.behavior_pretrain_loss),
        **{key: mean(values) for key, values in stats.items()},
    }


def snapshot_agent(agent: ActorCriticAgent) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "actor": clone_state_dict(agent.actor.state_dict()),
        "actor_target": clone_state_dict(agent.actor_target.state_dict()),
        "critic1": clone_state_dict(agent.critic1.state_dict()),
        "critic2": clone_state_dict(agent.critic2.state_dict()),
        "critic1_target": clone_state_dict(agent.critic1_target.state_dict()),
        "critic2_target": clone_state_dict(agent.critic2_target.state_dict()),
    }


def restore_agent(agent: ActorCriticAgent, snapshot: dict[str, dict[str, torch.Tensor]]) -> None:
    agent.actor.load_state_dict(snapshot["actor"])
    agent.actor_target.load_state_dict(snapshot["actor_target"])
    agent.critic1.load_state_dict(snapshot["critic1"])
    agent.critic2.load_state_dict(snapshot["critic2"])
    agent.critic1_target.load_state_dict(snapshot["critic1_target"])
    agent.critic2_target.load_state_dict(snapshot["critic2_target"])


def clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def train_critic_step(
    config: AdvancedGymMPCConfig,
    state: ScenarioPolicyState,
    rng: np.random.Generator,
    stats: dict[str, list[float]],
) -> None:
    batch = state.replay.sample(rng, int(config.actor_critic_batch_size))
    agent = state.agent
    obs = agent.obs_tensor(batch["observations"])
    next_obs = agent.obs_tensor(batch["next_observations"])
    action_z = torch.as_tensor(normalize_actions(batch["actions"], agent.action_low, agent.action_high), dtype=torch.float32)
    reward = torch.as_tensor(batch["rewards"].reshape(-1, 1) / state.replay.reward_scale, dtype=torch.float32)
    done = torch.as_tensor(batch["dones"].reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        noise = torch.clamp(
            torch.randn_like(action_z) * float(config.actor_critic_policy_noise),
            -float(config.actor_critic_noise_clip),
            float(config.actor_critic_noise_clip),
        )
        next_action = torch.clamp(agent.actor_target(next_obs) + noise, -1.0, 1.0)
        target_q = torch.min(agent.critic1_target(next_obs, next_action), agent.critic2_target(next_obs, next_action))
        target = reward + float(config.discount) * (1.0 - done) * target_q
    q1 = agent.critic1(obs, action_z)
    q2 = agent.critic2(obs, action_z)
    loss = torch.mean(torch.square(q1 - target) + torch.square(q2 - target))
    agent.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(agent.critic1.parameters()) + list(agent.critic2.parameters()), 5.0)
    agent.critic_optimizer.step()
    stats["scenario_critic_td_loss"].append(float(loss.detach().item()))
    stats["scenario_critic_q_mean"].append(float(torch.mean(0.5 * (q1 + q2)).detach().item()))


def train_actor_step(
    config: AdvancedGymMPCConfig,
    state: ScenarioPolicyState,
    rng: np.random.Generator,
    stats: dict[str, list[float]],
) -> None:
    agent = state.agent
    policy_batch = state.replay.sample(rng, int(config.actor_critic_batch_size))
    policy_obs = agent.obs_tensor(policy_batch["observations"])
    actor_action = agent.actor(policy_obs)
    q_loss = -torch.mean(agent.critic1(policy_obs, actor_action))

    real_batch = state.replay.sample(rng, int(config.actor_critic_batch_size), real_only=True)
    real_obs = agent.obs_tensor(real_batch["observations"])
    real_action = torch.as_tensor(normalize_actions(real_batch["actions"], agent.action_low, agent.action_high), dtype=torch.float32)
    behavior_loss = torch.mean(torch.square(agent.actor(real_obs) - real_action))
    smooth_loss = torch.mean(torch.square(actor_action))
    loss = q_loss + float(config.actor_critic_behavior_weight) * behavior_loss + float(config.actor_critic_smoothness_weight) * smooth_loss
    agent.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 5.0)
    agent.actor_optimizer.step()
    stats["scenario_actor_q_mean"].append(float((-q_loss).detach().item()))
    stats["scenario_real_behavior_loss"].append(float(behavior_loss.detach().item()))
    stats["scenario_actor_smoothness_loss"].append(float(smooth_loss.detach().item()))


def pretrain_real_behavior(config: AdvancedGymMPCConfig, replay: ReplayBuffer, agent: ActorCriticAgent) -> float:
    rng = np.random.default_rng(int(config.seed) + 240_000)
    losses: list[float] = []
    epochs = max(1, int(config.scenario_behavior_pretrain_epochs))
    for _epoch in range(epochs):
        batch = replay.sample(rng, int(config.actor_critic_batch_size), real_only=True)
        obs = agent.obs_tensor(batch["observations"])
        target = torch.as_tensor(normalize_actions(batch["actions"], agent.action_low, agent.action_high), dtype=torch.float32)
        loss = torch.mean(torch.square(agent.actor(obs) - target))
        agent.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 5.0)
        agent.actor_optimizer.step()
        losses.append(float(loss.detach().item()))
    state = agent.actor.state_dict()
    agent.actor_target.load_state_dict(state)
    return mean(losses)


def variant_priority(variant: object, config: AdvancedGymMPCConfig) -> float:
    base = float(variant.weights.combined)
    lift = float(getattr(variant, "predicted_lift", 0.0))
    scale = max(1e-4, float(config.scenario_advantage_temperature))
    lift_bonus = 1.0 + float(np.clip(max(0.0, lift) / scale, 0.0, 3.0))
    uncertainty = float(getattr(variant, "uncertainty", 0.0))
    uncertainty_cost = 1.0 / (1.0 + max(0.0, uncertainty))
    return float(max(1e-4, base * lift_bonus * uncertainty_cost))


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


__all__ = [
    "ScenarioPolicyState",
    "append_imagined_variants",
    "append_real_batch",
    "create_policy_state",
    "restore_agent",
    "snapshot_agent",
    "train_scenario_policy",
]
