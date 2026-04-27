"""Probe dataset collection and crawler orchestration."""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Optional

import numpy as np

from ...envs import (
    action_index_to_env_action,
    get_action_values,
    make_env,
)
from ..explorer import build_probe_planner
from .probe_env import apply_env_params, default_env_params, sample_env_params
from .probe_policy import PROBE_MODES, ProbePolicy
from .probe_records import Transition, WindowRecord


class ProbeCrawler:
    """Collect probe trajectories and convert them into encoder-ready windows."""

    def __init__(
        self,
        env_name: str = "CartPole-v1",
        window_size: int = 8,
        seed: int = 0,
        randomize_physics: bool = True,
        action_bins: int = 9,
        trace_writer=None,
    ):
        self.env = make_env(env_name)
        self.env_name = env_name
        self.window_size = window_size
        self.rng = np.random.default_rng(seed)
        self.randomize_physics = randomize_physics
        self.base_physics = default_env_params(env_name, self.env)
        self.action_values = get_action_values(self.env, action_bins, env_name=env_name)
        self.action_dim = int(self.env.action_space.n) if self.action_values is None else int(len(self.action_values))
        self.probe_policy = ProbePolicy(self.action_dim, profile="scalar")
        self.trace_writer = trace_writer
        self.default_active_window_stride = max(2, window_size // 2)
        self.default_active_step_cap = max(window_size + 8, 24)
        self.transitions: list[Transition] = []
        self.windows: list[WindowRecord] = []

    def _append_transition(
        self,
        env_instance_id: int,
        episode_id: int,
        step_idx: int,
        probe_mode: str,
        env_params: np.ndarray,
        state: np.ndarray,
        action_idx: int,
        next_state: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.transitions.append(
            Transition(
                env_instance_id=env_instance_id,
                episode_id=episode_id,
                step_idx=step_idx,
                probe_mode=probe_mode,
                env_params=env_params.copy(),
                state=np.asarray(state, dtype=np.float32).copy(),
                action=int(action_idx),
                next_state=np.asarray(next_state, dtype=np.float32).copy(),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    def _append_window(
        self,
        env_instance_id: int,
        episode_id: int,
        step_idx: int,
        probe_mode: str,
        env_params: np.ndarray,
        state_window: deque[np.ndarray],
        action_window: deque[int],
        reward_window: deque[float],
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.windows.append(
            WindowRecord(
                env_instance_id=env_instance_id,
                episode_id=episode_id,
                end_step_idx=step_idx,
                probe_mode=probe_mode,
                env_params=env_params.copy(),
                states=np.stack(state_window, axis=0),
                actions=np.asarray(action_window, dtype=np.int64),
                rewards=np.asarray(reward_window, dtype=np.float32),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    def run_episode(
        self,
        env_instance_id: int,
        episode_id: int,
        probe_mode: str,
        max_steps: int = 200,
        episode_physics=None,
        reset_options: Optional[dict[str, Any]] = None,
    ):
        """Run one scripted probe episode and record both steps and windows."""
        if episode_physics is None:
            episode_physics = sample_env_params(self.rng, self.base_physics) if self.randomize_physics else self.base_physics

        apply_env_params(self.env, episode_physics)
        env_params = episode_physics.as_array()
        self.probe_policy.reset_episode()
        state, _info = self.env.reset(options=reset_options)

        state_window = deque(maxlen=self.window_size + 1)
        action_window = deque(maxlen=self.window_size)
        reward_window = deque(maxlen=self.window_size)
        state_window.append(np.asarray(state, dtype=np.float32))

        for step_idx in range(max_steps):
            action_idx = self.probe_policy.sample_action(probe_mode, step_idx, self.rng)
            env_action = action_index_to_env_action(action_idx, self.action_values)
            next_state, reward, terminated, truncated, _info = self.env.step(env_action)

            state_np = np.asarray(state, dtype=np.float32)
            next_state_np = np.asarray(next_state, dtype=np.float32)
            self._append_transition(
                env_instance_id=env_instance_id,
                episode_id=episode_id,
                step_idx=step_idx,
                probe_mode=probe_mode,
                env_params=env_params,
                state=state_np,
                action_idx=int(action_idx),
                next_state=next_state_np,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            if self.trace_writer is not None:
                action_scalar = float(np.asarray(env_action, dtype=np.float32).reshape(-1)[0])
                self.trace_writer.record_probe_collection_step(
                    phase="dataset_scripted_probe",
                    state=state_np,
                    action_value=action_scalar,
                    action_index=int(action_idx),
                    probe_mode=probe_mode,
                    env_params=env_params,
                    env_instance_id=env_instance_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    reward=float(reward),
                )

            action_window.append(int(action_idx))
            reward_window.append(float(reward))
            state_window.append(next_state_np.copy())

            if len(action_window) == self.window_size and len(state_window) == self.window_size + 1:
                self._append_window(
                    env_instance_id=env_instance_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    probe_mode=probe_mode,
                    env_params=env_params,
                    state_window=state_window,
                    action_window=action_window,
                    reward_window=reward_window,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                if self.trace_writer is not None:
                    self.trace_writer.record_probe_window(
                        probe_mode=probe_mode,
                        episode_id=episode_id,
                        env_instance_id=env_instance_id,
                        reward_sum=float(np.sum(np.asarray(reward_window, dtype=np.float32))),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )

            state = next_state
            if terminated or truncated:
                break

    def run_active_probe_episode(
        self,
        env_instance_id: int,
        episode_id: int,
        episode_physics,
        max_steps: int,
        planner,
        active_step_cap: int,
        active_window_stride: int,
        max_windows_per_rollout: int | None = None,
        primary_goal: str | None = None,
        reset_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Run one short active scientist rollout for a single hidden world."""
        apply_env_params(self.env, episode_physics)
        env_params = episode_physics.as_array()
        state, _info = self.env.reset(options=reset_options)
        state_np = np.asarray(state, dtype=np.float32)
        rollout_goal = primary_goal or planner.choose_rollout_goal(state_np)
        planner.begin_rollout(primary_goal=rollout_goal)

        state_window = deque([state_np.copy()], maxlen=self.window_size + 1)
        action_window: deque[int] = deque(maxlen=self.window_size)
        reward_window: deque[float] = deque(maxlen=self.window_size)
        mode_window: deque[str] = deque(maxlen=self.window_size)
        next_window_step = self.window_size - 1
        emitted_windows = 0
        last_step_idx = -1
        last_probe_mode = rollout_goal
        last_terminated = False
        last_truncated = False
        window_limit = (
            None
            if max_windows_per_rollout is None
            else max(1, int(max_windows_per_rollout))
        )

        step_limit = min(max_steps, max(self.window_size, int(active_step_cap)))
        for step_idx in range(step_limit):
            probe_mode = planner.ensure_goal(state_np)
            action_prior = planner.action_prior_scores(state_np, list(action_window))
            logits = action_prior - float(np.max(action_prior))
            weights = np.exp(logits / 0.35).astype(np.float32)
            weights /= np.clip(np.sum(weights), 1e-6, None)
            action_idx = int(self.rng.choice(np.arange(self.action_dim), p=weights))

            env_action = action_index_to_env_action(action_idx, self.action_values)
            next_state, reward, terminated, truncated, _info = self.env.step(env_action)
            next_state_np = np.asarray(next_state, dtype=np.float32)
            last_step_idx = int(step_idx)
            last_probe_mode = probe_mode
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)

            self._append_transition(
                env_instance_id=env_instance_id,
                episode_id=episode_id,
                step_idx=step_idx,
                probe_mode=probe_mode,
                env_params=env_params,
                state=state_np,
                action_idx=action_idx,
                next_state=next_state_np,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            if self.trace_writer is not None:
                action_scalar = float(np.asarray(env_action, dtype=np.float32).reshape(-1)[0])
                self.trace_writer.record_probe_collection_step(
                    phase="dataset_active_probe",
                    state=state_np,
                    action_value=action_scalar,
                    action_index=int(action_idx),
                    probe_mode=probe_mode,
                    env_params=env_params,
                    env_instance_id=env_instance_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    reward=float(reward),
                )

            planner.observe_transition(
                prev_state=state_np,
                action_idx=action_idx,
                next_state=next_state_np,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )

            action_window.append(action_idx)
            reward_window.append(float(reward))
            state_window.append(next_state_np.copy())
            mode_window.append(probe_mode)

            if len(action_window) == self.window_size and len(state_window) == self.window_size + 1 and step_idx >= next_window_step:
                dominant_mode = Counter(mode_window).most_common(1)[0][0]
                window_mode = planner.rollout_goal if planner.rollout_goal is not None else dominant_mode
                self._append_window(
                    env_instance_id=env_instance_id,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    probe_mode=window_mode,
                    env_params=env_params,
                    state_window=state_window,
                    action_window=action_window,
                    reward_window=reward_window,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                if self.trace_writer is not None:
                    self.trace_writer.record_probe_window(
                        probe_mode=window_mode,
                        episode_id=episode_id,
                        env_instance_id=env_instance_id,
                        reward_sum=float(np.sum(np.asarray(reward_window, dtype=np.float32))),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                next_window_step += max(1, int(active_window_stride))
                emitted_windows += 1
                if window_limit is not None and emitted_windows >= window_limit:
                    break

            state_np = next_state_np
            if terminated or truncated:
                break

        emit_partial = bool(getattr(planner, "emit_partial_terminal_window", False))
        min_partial_steps = int(getattr(planner, "partial_window_min_steps", self.window_size))
        if (
            emitted_windows == 0
            and emit_partial
            and (last_terminated or last_truncated)
            and len(action_window) >= max(1, min_partial_steps)
        ):
            padded_states = deque(list(state_window), maxlen=self.window_size + 1)
            padded_actions = deque(list(action_window), maxlen=self.window_size)
            padded_rewards = deque(list(reward_window), maxlen=self.window_size)
            while len(padded_actions) < self.window_size:
                padded_actions.append(int(padded_actions[-1]))
                padded_rewards.append(0.0)
                padded_states.append(np.asarray(padded_states[-1], dtype=np.float32).copy())
            window_mode = planner.rollout_goal if planner.rollout_goal is not None else str(last_probe_mode)
            self._append_window(
                env_instance_id=env_instance_id,
                episode_id=episode_id,
                step_idx=max(0, last_step_idx),
                probe_mode=window_mode,
                env_params=env_params,
                state_window=padded_states,
                action_window=padded_actions,
                reward_window=padded_rewards,
                terminated=last_terminated,
                truncated=last_truncated,
            )
            if self.trace_writer is not None:
                self.trace_writer.record_probe_window(
                    probe_mode=window_mode,
                    episode_id=episode_id,
                    env_instance_id=env_instance_id,
                    reward_sum=float(np.sum(np.asarray(padded_rewards, dtype=np.float32))),
                    terminated=last_terminated,
                    truncated=last_truncated,
                )

    def _collect_active_dataset(self, env_instances: int, max_steps: int) -> None:
        """Collect probe data via active short-horizon experiments per world."""
        episode_id = 0
        planner = build_probe_planner(
            action_space=self.env.action_space,
            action_values=self.action_values,
            rng=self.rng,
            env_name=self.env_name,
        )
        if planner is None:
            raise ValueError("Active dataset requested without a scientist planner")

        support_goal_sequence = tuple(getattr(planner, "support_goal_sequence", ()))
        if not support_goal_sequence:
            support_goal_sequence = (None,)
        active_step_cap = int(getattr(planner, "active_step_cap", self.default_active_step_cap))
        active_window_stride = int(getattr(planner, "active_window_stride", self.default_active_window_stride))
        raw_max_windows_per_rollout = getattr(planner, "max_windows_per_rollout", None)
        max_windows_per_rollout = (
            None
            if raw_max_windows_per_rollout is None
            else max(1, int(raw_max_windows_per_rollout))
        )
        min_windows_per_env = int(getattr(planner, "min_windows_per_env", len(support_goal_sequence)))
        min_windows_per_family = max(1, int(getattr(planner, "min_windows_per_family", 1)))
        max_support_retry_rollouts = int(
            getattr(planner, "max_support_retry_rollouts", len(support_goal_sequence))
        )
        required_window_counts = {
            goal: min_windows_per_family
            for goal in support_goal_sequence
            if goal is not None
        }

        def count_goal_windows(start_idx: int) -> Counter[str]:
            return Counter(str(window.probe_mode) for window in self.windows[start_idx:])

        for env_instance_id in range(env_instances):
            episode_physics = sample_env_params(self.rng, self.base_physics) if self.randomize_physics else self.base_physics
            planner.begin_env_instance()
            env_window_start = len(self.windows)
            rollout_budget = (
                len(support_goal_sequence) * min_windows_per_family
                + max(0, max_support_retry_rollouts)
            )
            for rollout_idx in range(rollout_budget):
                goal_counts = count_goal_windows(env_window_start)
                undercovered_goals = [
                    goal
                    for goal in support_goal_sequence
                    if goal is not None and goal_counts[str(goal)] < required_window_counts.get(goal, 1)
                ]
                primary_goal = (
                    undercovered_goals[0]
                    if undercovered_goals
                    else support_goal_sequence[rollout_idx % len(support_goal_sequence)]
                )
                self.run_active_probe_episode(
                    env_instance_id=env_instance_id,
                    episode_id=episode_id,
                    episode_physics=episode_physics,
                    max_steps=max_steps,
                    planner=planner,
                    active_step_cap=active_step_cap,
                    active_window_stride=active_window_stride,
                    max_windows_per_rollout=max_windows_per_rollout,
                    primary_goal=primary_goal,
                )
                episode_id += 1
                env_window_count = len(self.windows) - env_window_start
                goal_counts = count_goal_windows(env_window_start)
                completed_family_targets = all(
                    goal_counts[str(goal)] >= required_count
                    for goal, required_count in required_window_counts.items()
                )
                completed_support_cycle = rollout_idx + 1 >= len(support_goal_sequence)
                if (
                    completed_support_cycle
                    and completed_family_targets
                    and env_window_count >= max(1, min_windows_per_env)
                ):
                    break

    def collect(self, episodes_per_mode: int = 20, max_steps: int = 200):
        """Collect an encoder dataset for the current environment family."""
        if self.trace_writer is not None:
            self.trace_writer.set_stage(
                "probe_dataset_collection",
                "Crawler Support Search",
                "Collecting short probe windows to build the first mechanics belief.",
            )
        if build_probe_planner(
            action_space=self.env.action_space,
            action_values=self.action_values,
            rng=self.rng,
            env_name=self.env_name,
        ) is not None:
            self._collect_active_dataset(env_instances=episodes_per_mode, max_steps=max_steps)
            return

        episode_id = 0
        for env_instance_id in range(episodes_per_mode):
            episode_physics = sample_env_params(self.rng, self.base_physics) if self.randomize_physics else self.base_physics
            for mode in PROBE_MODES:
                self.run_episode(
                    env_instance_id=env_instance_id,
                    episode_id=episode_id,
                    probe_mode=mode,
                    max_steps=max_steps,
                    episode_physics=episode_physics,
                )
                episode_id += 1

    def get_transition_arrays(self) -> dict[str, np.ndarray]:
        """Convert the recorded transition list into stacked NumPy arrays."""
        if not self.transitions:
            raise ValueError("No probe transitions collected; call collect() before reading arrays.")
        return {
            "env_instance_id": np.asarray([t.env_instance_id for t in self.transitions], dtype=np.int32),
            "episode_id": np.asarray([t.episode_id for t in self.transitions], dtype=np.int32),
            "step_idx": np.asarray([t.step_idx for t in self.transitions], dtype=np.int32),
            "probe_mode": np.asarray([t.probe_mode for t in self.transitions], dtype=object),
            "env_params": np.stack([t.env_params for t in self.transitions], axis=0),
            "state": np.stack([t.state for t in self.transitions], axis=0),
            "action": np.asarray([t.action for t in self.transitions], dtype=np.int64),
            "next_state": np.stack([t.next_state for t in self.transitions], axis=0),
            "reward": np.asarray([t.reward for t in self.transitions], dtype=np.float32),
            "terminated": np.asarray([t.terminated for t in self.transitions], dtype=np.bool_),
            "truncated": np.asarray([t.truncated for t in self.transitions], dtype=np.bool_),
        }

    def get_window_arrays(self) -> dict[str, np.ndarray]:
        """Convert the recorded window list into stacked NumPy arrays."""
        if not self.windows:
            raise ValueError("No probe windows collected; check probe settings before training.")
        return {
            "env_instance_id": np.asarray([w.env_instance_id for w in self.windows], dtype=np.int32),
            "episode_id": np.asarray([w.episode_id for w in self.windows], dtype=np.int32),
            "end_step_idx": np.asarray([w.end_step_idx for w in self.windows], dtype=np.int32),
            "probe_mode": np.asarray([w.probe_mode for w in self.windows], dtype=object),
            "env_params": np.stack([w.env_params for w in self.windows], axis=0),
            "states": np.stack([w.states for w in self.windows], axis=0),
            "actions": np.stack([w.actions for w in self.windows], axis=0),
            "rewards": np.stack([w.rewards for w in self.windows], axis=0),
            "terminated": np.asarray([w.terminated for w in self.windows], dtype=np.bool_),
            "truncated": np.asarray([w.truncated for w in self.windows], dtype=np.bool_),
        }

    def close(self):
        """Close the owned environment."""
        self.env.close()


CartPoleCrawler = ProbeCrawler
