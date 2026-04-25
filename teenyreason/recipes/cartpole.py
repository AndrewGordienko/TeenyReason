"""CartPole recipe composition and target-builder registration."""

from __future__ import annotations

from functools import partial

import numpy as np

from ..crawler.core import (
    LinearMessageProjector,
    RoundRobinQueryPolicy,
    ScriptedWorldAdapter,
    SupportLimitStopPolicy,
    VectorBeliefBackend,
)
from ..envs import CONTINUOUS_CARTPOLE_NAME
from ..models.belief.belief_targets import register_future_summary_builder
from .base import BenchmarkSpec, CrawlerRecipe


def _cartpole_query_payloads() -> dict[str, dict[str, np.ndarray]]:
    return {
        "passive_decay": {"vector": np.asarray([1.0, 0.1, 0.8, 0.2], dtype=np.float32)},
        "impulse_left": {"vector": np.asarray([0.3, 1.0, 0.8, 0.2], dtype=np.float32)},
        "impulse_right": {"vector": np.asarray([0.3, 1.0, 0.8, 0.3], dtype=np.float32)},
        "chirp": {"vector": np.asarray([0.4, 0.9, 0.7, 0.4], dtype=np.float32)},
        "boundary_push": {"vector": np.asarray([0.2, 0.5, 0.6, 1.0], dtype=np.float32)},
        "cart_brake": {"vector": np.asarray([0.2, 0.8, 1.0, 0.8], dtype=np.float32)},
    }


def _cartpole_probe_family_features(probe_mode: np.ndarray | None) -> np.ndarray:
    """Encode CartPole intervention identity into a small semantics vector."""
    if probe_mode is None:
        return np.zeros((0, 4), dtype=np.float32)
    probe_mode_np = np.asarray(probe_mode, dtype="U").reshape(-1)
    feature_rows: list[np.ndarray] = []
    lookup = _cartpole_query_payloads()
    for family in probe_mode_np.tolist():
        feature_rows.append(
            np.asarray(
                lookup.get(str(family), {}).get("vector", np.zeros((4,), dtype=np.float32)),
                dtype=np.float32,
            )
        )
    return np.stack(feature_rows, axis=0).astype(np.float32)


def build_cartpole_future_summary_targets(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    action_vocab_size: int,
    probe_mode: np.ndarray | None,
) -> np.ndarray:
    """Summarize CartPole response with mechanics-relevant, family-aware features."""
    start_state = states[:, 0, :]
    end_state = states[:, -1, :]
    end_delta = end_state - start_state
    step_delta = np.diff(states, axis=1)
    mean_step_delta = np.mean(step_delta, axis=1)
    state_span = np.max(states, axis=1) - np.min(states, axis=1)
    mean_abs_state = np.mean(np.abs(states), axis=1)
    peak_abs_state = np.max(np.abs(states), axis=1)
    reward_summary = np.stack(
        [
            np.sum(rewards, axis=1),
            np.mean(rewards, axis=1),
            np.min(rewards, axis=1),
            np.max(rewards, axis=1),
        ],
        axis=1,
    ).astype(np.float32)

    centered_actions = np.zeros(actions.shape, dtype=np.float32)
    if action_vocab_size > 1:
        centered_actions = (
            2.0 * actions.astype(np.float32) / float(action_vocab_size - 1) - 1.0
        )
    action_delta = (
        np.diff(centered_actions, axis=1)
        if centered_actions.shape[1] > 1
        else np.zeros((centered_actions.shape[0], 1), dtype=np.float32)
    )
    action_switch_rate = np.zeros((centered_actions.shape[0],), dtype=np.float32)
    if centered_actions.shape[1] > 1:
        action_switch_rate = np.mean(
            np.sign(centered_actions[:, 1:]) != np.sign(centered_actions[:, :-1]),
            axis=1,
        ).astype(np.float32)
    action_summary = np.stack(
        [
            np.mean(centered_actions, axis=1),
            np.mean(np.abs(centered_actions), axis=1),
            np.mean(np.abs(action_delta), axis=1),
            action_switch_rate,
        ],
        axis=1,
    ).astype(np.float32)

    response_summary = np.stack(
        [
            np.mean(np.abs(states[:, :, 0]), axis=1),
            np.mean(np.abs(states[:, :, 2]), axis=1),
            np.max(np.abs(states[:, :, 0]), axis=1),
            np.max(np.abs(states[:, :, 2]), axis=1),
            np.mean(np.abs(states[:, :, 1]), axis=1),
            np.mean(np.abs(states[:, :, 3]), axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    terminal_summary = np.stack(
        [
            terminated.astype(np.float32),
            truncated.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    family_features = _cartpole_probe_family_features(probe_mode)
    if family_features.shape[0] == 0:
        family_features = np.zeros((states.shape[0], 4), dtype=np.float32)

    return np.concatenate(
        [
            end_delta.astype(np.float32),
            mean_step_delta.astype(np.float32),
            state_span.astype(np.float32),
            mean_abs_state.astype(np.float32),
            peak_abs_state.astype(np.float32),
            reward_summary,
            action_summary,
            response_summary,
            family_features.astype(np.float32),
            terminal_summary,
        ],
        axis=1,
    ).astype(np.float32)


def register_cartpole_recipe_targets() -> None:
    """Register the CartPole-specific target builders with the generic registry."""
    register_future_summary_builder(
        CONTINUOUS_CARTPOLE_NAME,
        build_cartpole_future_summary_targets,
    )


def build_cartpole_recipe() -> CrawlerRecipe:
    """Build the user-facing CartPole recipe on top of the generic crawler API."""
    register_cartpole_recipe_targets()
    query_payloads = _cartpole_query_payloads()
    return CrawlerRecipe(
        name="cartpole",
        description="Generic crawler composition for the Continuous CartPole benchmark.",
        world_adapter_factory=partial(
            ScriptedWorldAdapter,
            query_payloads=query_payloads,
            source_prefix="cartpole",
        ),
        belief_backend_factory=partial(VectorBeliefBackend, vector_key="vector"),
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(SupportLimitStopPolicy, min_support=2),
        message_projector_factory=LinearMessageProjector,
        max_steps=2,
        metadata={
            "modality": "rl",
            "recipe_family": "cartpole",
        },
        benchmark=BenchmarkSpec(kind="ppo", env_name=CONTINUOUS_CARTPOLE_NAME),
    )
