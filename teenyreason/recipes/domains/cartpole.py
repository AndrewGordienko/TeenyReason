"""CartPole recipe composition and target-builder registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import numpy as np

from ...crawler.core import (
    LinearMessageProjector,
    RoundRobinQueryPolicy,
    SupportLimitStopPolicy,
    VectorBeliefBackend,
)
from ...crawler.types import BeliefState, EvidenceSlice
from ...envs import CONTINUOUS_CARTPOLE_NAME
from ...envs.continuous_cartpole import ContinuousCartPoleEnv
from ...models.belief.objectives.targets import register_future_summary_builder
from ...crawler.probes.data.probe_env import (
    apply_cartpole_physics,
    sample_cartpole_physics,
)
from ..base import BenchmarkSpec, CrawlerRecipe
from ..evidence import evidence_metadata, evidence_payload


def _cartpole_query_payloads() -> dict[str, dict[str, np.ndarray]]:
    return {
        "passive_decay": {"vector": np.asarray([1.0, 0.1, 0.8, 0.2], dtype=np.float32)},
        "impulse_left": {"vector": np.asarray([0.3, 1.0, 0.8, 0.2], dtype=np.float32)},
        "impulse_right": {"vector": np.asarray([0.3, 1.0, 0.8, 0.3], dtype=np.float32)},
        "chirp": {"vector": np.asarray([0.4, 0.9, 0.7, 0.4], dtype=np.float32)},
        "boundary_push": {"vector": np.asarray([0.2, 0.5, 0.6, 1.0], dtype=np.float32)},
        "cart_brake": {"vector": np.asarray([0.2, 0.8, 1.0, 0.8], dtype=np.float32)},
    }


def _cartpole_query_cost(query_name: str) -> float:
    if query_name == "passive_decay":
        return 0.70
    if query_name == "chirp":
        return 0.90
    if query_name in {"boundary_push", "cart_brake"}:
        return 1.15
    return 1.0


def _cartpole_action_schedule(query_name: str, window_size: int) -> np.ndarray:
    steps = max(1, int(window_size))
    if query_name == "passive_decay":
        return np.zeros((steps,), dtype=np.float32)
    if query_name == "impulse_left":
        actions = np.zeros((steps,), dtype=np.float32)
        actions[: max(1, steps // 3)] = -0.85
        return actions
    if query_name == "impulse_right":
        actions = np.zeros((steps,), dtype=np.float32)
        actions[: max(1, steps // 3)] = 0.85
        return actions
    if query_name == "chirp":
        phase = np.linspace(0.0, 2.0 * np.pi, num=steps, endpoint=False)
        return (0.65 * np.sin(phase * 1.5)).astype(np.float32)
    if query_name == "boundary_push":
        return np.asarray(
            [0.95 if idx % 2 == 0 else -0.95 for idx in range(steps)],
            dtype=np.float32,
        )
    if query_name == "cart_brake":
        actions = np.ones((steps,), dtype=np.float32) * 0.75
        actions[steps // 2 :] = -0.75
        return actions
    return np.zeros((steps,), dtype=np.float32)


def _cartpole_observation_vector(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: bool,
) -> np.ndarray:
    start_state = states[0]
    end_state = states[-1]
    delta = end_state - start_state
    span = np.max(states, axis=0) - np.min(states, axis=0)
    mean_abs = np.mean(np.abs(states), axis=0)
    action_stats = np.asarray(
        [
            float(np.mean(actions)) if actions.size else 0.0,
            float(np.mean(np.abs(actions))) if actions.size else 0.0,
            float(np.std(actions)) if actions.size else 0.0,
        ],
        dtype=np.float32,
    )
    reward_stats = np.asarray(
        [
            float(np.sum(rewards)) if rewards.size else 0.0,
            float(np.mean(rewards)) if rewards.size else 0.0,
            1.0 if terminated else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [delta, span, mean_abs, action_stats, reward_stats],
        axis=0,
    ).astype(np.float32)


@dataclass
class CartPoleProbeWorldAdapter:
    """Execute real named CartPole probe families for the generic crawler."""

    window_size: int = 10
    source_prefix: str = "cartpole"
    _env: ContinuousCartPoleEnv = field(init=False, default_factory=ContinuousCartPoleEnv)
    _rng: np.random.Generator = field(init=False, default_factory=np.random.default_rng)
    _source_id: str = field(init=False, default="cartpole:0")
    _hidden_target: dict[str, float] = field(init=False, default_factory=dict)
    _query_counter: int = field(init=False, default=0)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        sampled = sample_cartpole_physics(self._rng)
        apply_cartpole_physics(self._env, sampled)
        self._source_id = f"{self.source_prefix}:{0 if seed is None else int(seed)}"
        self._hidden_target = {
            "gravity": float(sampled.gravity),
            "masscart": float(sampled.masscart),
            "masspole": float(sampled.masspole),
            "length": float(sampled.length),
            "force_mag": float(sampled.force_mag),
        }
        self._query_counter = 0

    def available_queries(
        self,
        *,
        belief_state: BeliefState,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
    ) -> tuple[str, ...]:
        del belief_state
        names = tuple(_cartpole_query_payloads().keys())
        seen = {item.query_name for item in history}
        unseen = tuple(name for name in names if name not in seen)
        return unseen if unseen else names

    def execute_query(
        self,
        query_name: str,
        *,
        belief_state: BeliefState,
        history: tuple[EvidenceSlice, ...] | list[EvidenceSlice],
    ) -> EvidenceSlice:
        del belief_state
        del history
        self._query_counter += 1
        actions = _cartpole_action_schedule(query_name, self.window_size)
        state, _info = self._env.reset(seed=int(self._rng.integers(0, 2**31 - 1)))
        states: list[np.ndarray] = [np.asarray(state, dtype=np.float32)]
        rewards: list[float] = []
        terminated = False
        truncated = False
        for action in actions:
            next_state, reward, terminated, truncated, _info = self._env.step(
                np.asarray([float(action)], dtype=np.float32)
            )
            states.append(np.asarray(next_state, dtype=np.float32))
            rewards.append(float(reward))
            if bool(terminated or truncated):
                break
        while len(states) < actions.shape[0] + 1:
            states.append(states[-1].copy())
            rewards.append(0.0)
        states_np = np.stack(states[: actions.shape[0] + 1], axis=0).astype(np.float32)
        rewards_np = np.asarray(rewards[: actions.shape[0]], dtype=np.float32)
        vector = _cartpole_observation_vector(
            states=states_np,
            actions=actions,
            rewards=rewards_np,
            terminated=bool(terminated or truncated),
        )
        payload = evidence_payload(
            modality="control",
            query_family=str(query_name),
            source_id=self._source_id,
            intervention_cost=_cartpole_query_cost(str(query_name)),
            hidden_target=self._hidden_target,
            local_state={
                "initial_state": states_np[0],
                "states": states_np,
            },
            outcome={
                "actions": actions.astype(np.float32),
                "rewards": rewards_np,
                "final_state": states_np[-1],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
            vector=vector,
            belief_source="sysid",
            extra={
                "states": states_np,
                "actions": actions.astype(np.float32),
                "rewards": rewards_np,
            },
        )
        return EvidenceSlice(
            query_name=str(query_name),
            source_id=self._source_id,
            payload=payload,
            metadata=evidence_metadata(payload=payload, query_index=self._query_counter),
        )


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
    return CrawlerRecipe(
        name="cartpole",
        description="Generic crawler composition for the Continuous CartPole benchmark.",
        world_adapter_factory=CartPoleProbeWorldAdapter,
        belief_backend_factory=partial(VectorBeliefBackend, vector_key="vector"),
        query_policy_factory=RoundRobinQueryPolicy,
        stop_policy_factory=partial(SupportLimitStopPolicy, min_support=2),
        message_projector_factory=LinearMessageProjector,
        max_steps=2,
        metadata={
            "modality": "control",
            "recipe_family": "cartpole",
            "belief_source": "sysid",
        },
        benchmark=BenchmarkSpec(kind="ppo", env_name=CONTINUOUS_CARTPOLE_NAME),
    )
