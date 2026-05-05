"""Bipedal benchmark recipe helpers and target-builder registration."""

from __future__ import annotations

from functools import partial

import numpy as np

from ...crawler.core import (
    LinearMessageProjector,
    RoundRobinQueryPolicy,
    ScriptedWorldAdapter,
    SupportLimitStopPolicy,
    VectorBeliefBackend,
)
from ...envs import BIPEDAL_WALKER_NAME
from ...models.belief.objectives.targets import register_decision_target_builder
from ..base import BenchmarkSpec, CrawlerRecipe


def build_bipedal_decision_targets(
    *,
    states: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> np.ndarray:
    """Decision-focused supervision for locomotion-style control."""
    current_state = states[:, -1, :]
    state_diff = np.diff(states, axis=1)
    delta_norm = np.linalg.norm(state_diff, axis=2)
    reward_sum = np.sum(rewards, axis=1)
    reward_mid = rewards.shape[1] // 2
    reward_trend = np.mean(rewards[:, reward_mid:], axis=1) - np.mean(rewards[:, :reward_mid], axis=1)
    hull_angle = current_state[:, 0]
    hull_angular_velocity = current_state[:, 1]
    forward_speed = current_state[:, 2]
    vertical_speed = current_state[:, 3]
    left_contact = np.clip(current_state[:, 8], 0.0, 1.0)
    right_contact = np.clip(current_state[:, 13], 0.0, 1.0)
    both_contact = left_contact * right_contact
    contact_balance = 1.0 - np.abs(left_contact - right_contact)
    upright_margin = 1.0 - np.clip(np.abs(hull_angle), 0.0, 1.5) / 1.5
    angular_stability = 1.0 / (1.0 + np.abs(hull_angular_velocity))
    motion_energy = np.mean(delta_norm, axis=1)
    recoverability = upright_margin + 0.25 * reward_trend - 0.10 * np.abs(vertical_speed)
    fall_risk = np.logical_or(terminated, truncated).astype(np.float32)
    return np.stack(
        [
            reward_sum.astype(np.float32),
            reward_trend.astype(np.float32),
            forward_speed.astype(np.float32),
            vertical_speed.astype(np.float32),
            upright_margin.astype(np.float32),
            angular_stability.astype(np.float32),
            motion_energy.astype(np.float32),
            left_contact.astype(np.float32),
            right_contact.astype(np.float32),
            both_contact.astype(np.float32),
            contact_balance.astype(np.float32),
            recoverability.astype(np.float32),
            fall_risk.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def register_bipedal_recipe_targets() -> None:
    """Register the locomotion-specific decision builder."""
    register_decision_target_builder(
        BIPEDAL_WALKER_NAME,
        build_bipedal_decision_targets,
    )


def build_bipedal_recipe() -> CrawlerRecipe:
    """Build the BipedalWalker benchmark recipe."""
    register_bipedal_recipe_targets()
    query_payloads = {
        "stability_probe": {"vector": np.asarray([1.0, 0.2, 0.1], dtype=np.float32)},
        "stride_probe": {"vector": np.asarray([0.4, 1.0, 0.2], dtype=np.float32)},
        "recovery_probe": {"vector": np.asarray([0.3, 0.5, 1.0], dtype=np.float32)},
    }
    return CrawlerRecipe(
        name="bipedal",
        description="Generic crawler composition for the BipedalWalker benchmark.",
        world_adapter_factory=partial(
            ScriptedWorldAdapter,
            query_payloads=query_payloads,
            source_prefix="bipedal",
        ),
        belief_backend_factory=partial(VectorBeliefBackend, vector_key="vector"),
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(SupportLimitStopPolicy, min_support=2),
        message_projector_factory=LinearMessageProjector,
        max_steps=2,
        metadata={
            "modality": "rl",
            "recipe_family": "bipedal",
        },
        benchmark=BenchmarkSpec(kind="ppo", env_name=BIPEDAL_WALKER_NAME),
    )
