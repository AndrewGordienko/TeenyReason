"""Generic active probe explorer for gym-like action spaces."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


MECHANICS_SCALAR_PROBE_FAMILIES = (
    "passive_decay",
    "impulse_left",
    "impulse_right",
    "chirp",
    "boundary_push",
    "cart_brake",
)


def _center_action(action_space: gym.spaces.Space) -> np.ndarray | None:
    if not isinstance(action_space, gym.spaces.Box):
        return None
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    return np.clip(np.zeros_like(low, dtype=np.float32), low, high)


def _dedupe_labels(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped: list[str] = []
    for label in labels:
        count = counts.get(label, 0)
        counts[label] = count + 1
        deduped.append(label if count == 0 else f"{label}_{count + 1}")
    return deduped


def _infer_discrete_family_labels(action_count: int) -> list[str]:
    return [f"action_{idx}" for idx in range(max(0, int(action_count)))]


def _infer_scalar_box_family_labels(
    action_space: gym.spaces.Box,
    action_values: np.ndarray,
) -> list[str]:
    values = np.asarray(action_values, dtype=np.float32).reshape(-1)
    center_value = float(_center_action(action_space)[0])
    center_idx = int(np.argmin(np.abs(values - center_value)))
    labels = [""] * values.shape[0]
    labels[center_idx] = "center"

    negative_indices = [
        idx for idx in range(values.shape[0]) if idx != center_idx and float(values[idx]) < center_value
    ]
    positive_indices = [
        idx for idx in range(values.shape[0]) if idx != center_idx and float(values[idx]) > center_value
    ]
    negative_indices.sort(key=lambda idx: abs(float(values[idx]) - center_value))
    positive_indices.sort(key=lambda idx: abs(float(values[idx]) - center_value))

    for rank, idx in enumerate(negative_indices, start=1):
        labels[idx] = f"neg_{rank}"
    for rank, idx in enumerate(positive_indices, start=1):
        labels[idx] = f"pos_{rank}"
    return labels


def _looks_like_continuous_cartpole(env_name: str | None, action_values: np.ndarray | None) -> bool:
    """Detect the one-dimensional CartPole action grid that needs named probes."""
    del action_values
    return env_name is not None and "cartpole" in str(env_name).lower()


def _infer_box_family_labels(
    action_space: gym.spaces.Box,
    action_values: np.ndarray,
) -> list[str]:
    values = np.asarray(action_values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected 2-D Box action values")
    if values.shape[1] == 1:
        return _infer_scalar_box_family_labels(action_space, values)

    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    center = _center_action(action_space)
    if center is None:
        return [f"prototype_{idx}" for idx in range(values.shape[0])]

    labels: list[str] = []
    tol = 1e-4
    for idx, value in enumerate(values):
        if np.allclose(value, center, atol=tol):
            labels.append("center")
            continue
        if np.allclose(value, low, atol=tol):
            labels.append("all_min")
            continue
        if np.allclose(value, high, atol=tol):
            labels.append("all_max")
            continue

        changed = np.flatnonzero(np.abs(value - center) > tol)
        if changed.shape[0] == 1:
            axis = int(changed[0])
            low_distance = abs(float(value[axis]) - float(low[axis]))
            high_distance = abs(float(value[axis]) - float(high[axis]))
            edge = "min" if low_distance <= high_distance else "max"
            labels.append(f"axis{axis}_{edge}")
            continue

        signs = []
        for axis in changed.tolist():
            direction = "max" if abs(float(value[axis]) - float(high[axis])) <= abs(float(value[axis]) - float(low[axis])) else "min"
            signs.append(f"a{axis}{direction[0]}")
        labels.append(f"mix_{'_'.join(signs) or idx}")
    return _dedupe_labels(labels)


def infer_probe_family_labels(
    *,
    action_space: gym.spaces.Space,
    action_values: np.ndarray | None,
) -> list[str]:
    """Build one stable probe-family label for each action prototype."""
    if isinstance(action_space, gym.spaces.Discrete):
        return _infer_discrete_family_labels(int(action_space.n))
    if isinstance(action_space, gym.spaces.Box):
        if action_values is None:
            raise ValueError("Box probe explorer needs prototype actions")
        return _infer_box_family_labels(action_space, action_values)
    raise ValueError(f"Unsupported action space for probing: {type(action_space).__name__}")


@dataclass
class ProbeCoverage:
    """Small generic summary of what the explorer has seen in this world."""

    total_steps: int = 0
    boundary_events: int = 0
    surprise_events: int = 0
    termination_events: int = 0
    controllability_changes: int = 0


class GenericProbeExplorer:
    """Choose short probe families directly from the action space and interaction."""

    active_step_cap = 48
    active_window_stride = 8
    max_windows_per_rollout: int | None = None
    emit_partial_terminal_window = False
    partial_window_min_steps = 0
    min_windows_per_family = 1

    def __init__(
        self,
        *,
        action_space: gym.spaces.Space,
        action_values: np.ndarray | None,
        rng: np.random.Generator,
        env_name: str | None = None,
        goal_horizon: int = 6,
    ):
        self.action_space = action_space
        self.action_values = None if action_values is None else np.asarray(action_values, dtype=np.float32)
        self.rng = rng
        self.goal_horizon = max(3, int(goal_horizon))
        self.rollout_goal_locked_steps = max(3, self.goal_horizon - 1)
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_vocab_size = int(action_space.n)
        elif self.action_values is not None:
            self.action_vocab_size = int(self.action_values.shape[0])
        else:
            raise ValueError("Generic probe explorer needs a discrete action count or action prototypes")

        self.action_labels = infer_probe_family_labels(
            action_space=action_space,
            action_values=self.action_values,
        )
        family_names: list[str] = []
        for label in self.action_labels:
            if label not in family_names:
                family_names.append(label)
        scalar_box_families = bool(family_names) and all(
            name == "center" or name.startswith("neg_") or name.startswith("pos_")
            for name in family_names
        )
        self.mechanics_scalar_probe = bool(
            scalar_box_families
            and _looks_like_continuous_cartpole(env_name, self.action_values)
        )
        if self.mechanics_scalar_probe:
            self.support_goal_sequence = MECHANICS_SCALAR_PROBE_FAMILIES
        elif scalar_box_families:
            directional_names = [
                name for name in family_names if name.startswith("neg_") or name.startswith("pos_")
            ]
            center_names = [name for name in family_names if name == "center"]
            self.support_goal_sequence = tuple(directional_names + center_names)
        else:
            self.support_goal_sequence = tuple(family_names)
        self.active_step_cap = type(self).active_step_cap
        self.active_window_stride = type(self).active_window_stride
        self.max_windows_per_rollout = type(self).max_windows_per_rollout
        self.emit_partial_terminal_window = bool(type(self).emit_partial_terminal_window)
        self.partial_window_min_steps = int(type(self).partial_window_min_steps)
        self.min_windows_per_family = max(1, int(type(self).min_windows_per_family))
        self.center_recovery_bonus = 0.45
        self.min_windows_per_env = len(self.support_goal_sequence)
        self.max_support_retry_rollouts = max(2, len(self.support_goal_sequence) // 2)
        if scalar_box_families:
            # Scalar continuous controls are easy to oversample. Emit one
            # compact window per named experiment so the encoder learns from a
            # small, auditable support budget plus genuine held-out windows.
            self.active_step_cap = max(self.active_step_cap, 64)
            self.active_window_stride = min(self.active_window_stride, 6)
            self.max_windows_per_rollout = 1
            self.emit_partial_terminal_window = True
            self.partial_window_min_steps = 3
            self.center_recovery_bonus = 0.20
            self.min_windows_per_env = len(self.support_goal_sequence)
            self.max_support_retry_rollouts = max(
                self.max_support_retry_rollouts,
                len(self.support_goal_sequence),
            )
            if self.mechanics_scalar_probe:
                # Two independent windows per named experiment let split
                # diagnostics separate same-family stability from cross-family
                # transfer instead of treating every split as purely cross-family.
                self.min_windows_per_family = 2
                self.min_windows_per_env = len(self.support_goal_sequence) * self.min_windows_per_family
                self.max_support_retry_rollouts = max(
                    self.max_support_retry_rollouts,
                    self.min_windows_per_env,
                )
        self.family_indices = {
            family: np.asarray(
                [idx for idx, label in enumerate(self.action_labels) if label == family],
                dtype=np.int64,
            )
            for family in self.support_goal_sequence
        }
        if self.mechanics_scalar_probe and self.action_values is not None:
            scalar_actions = np.asarray(self.action_values, dtype=np.float32).reshape(-1)
            center_idx = int(np.argmin(np.abs(scalar_actions)))
            left_idx = int(np.argmin(scalar_actions))
            right_idx = int(np.argmax(scalar_actions))
            self.scalar_center_idx = center_idx
            self.scalar_left_idx = left_idx
            self.scalar_right_idx = right_idx
            self.family_indices = {
                "passive_decay": np.asarray([center_idx], dtype=np.int64),
                "impulse_left": np.asarray([left_idx], dtype=np.int64),
                "impulse_right": np.asarray([right_idx], dtype=np.int64),
                "chirp": np.asarray([left_idx, right_idx], dtype=np.int64),
                "boundary_push": np.asarray([left_idx, right_idx], dtype=np.int64),
                "cart_brake": np.asarray([left_idx, right_idx], dtype=np.int64),
            }
        self.goal_counts = {goal: 0 for goal in self.support_goal_sequence}
        self.goal_transition_sum = {goal: 0.0 for goal in self.support_goal_sequence}
        self.goal_transition_count = {goal: 0 for goal in self.support_goal_sequence}
        self.goal_boundary_events = {goal: 0 for goal in self.support_goal_sequence}
        self.goal_termination_events = {goal: 0 for goal in self.support_goal_sequence}
        self.recent_goals: deque[str] = deque(maxlen=4)
        self.recent_transition_scales: deque[float] = deque(maxlen=16)
        self.coverage = ProbeCoverage()
        self.current_goal: str | None = None
        self.rollout_goal: str | None = None
        self.steps_in_goal = 0
        self.center_family = "center" if "center" in self.family_indices else None

    def begin_env_instance(self):
        self.goal_counts = {goal: 0 for goal in self.support_goal_sequence}
        self.goal_transition_sum = {goal: 0.0 for goal in self.support_goal_sequence}
        self.goal_transition_count = {goal: 0 for goal in self.support_goal_sequence}
        self.goal_boundary_events = {goal: 0 for goal in self.support_goal_sequence}
        self.goal_termination_events = {goal: 0 for goal in self.support_goal_sequence}
        self.recent_goals.clear()
        self.recent_transition_scales.clear()
        self.coverage = ProbeCoverage()
        self.current_goal = None
        self.rollout_goal = None
        self.steps_in_goal = 0

    def begin_rollout(self, primary_goal: str | None = None):
        self.current_goal = None
        self.rollout_goal = primary_goal if primary_goal in self.family_indices else None
        self.steps_in_goal = 0

    def choose_rollout_goal(self, state: np.ndarray) -> str:
        del state
        scores: dict[str, float] = {}
        mean_effect = float(np.mean(np.asarray(tuple(self.recent_transition_scales) or (0.0,), dtype=np.float32)))
        for goal in self.support_goal_sequence:
            count = float(self.goal_counts.get(goal, 0))
            transition_count = max(1, int(self.goal_transition_count.get(goal, 0)))
            goal_effect = float(self.goal_transition_sum.get(goal, 0.0)) / float(transition_count)
            boundary_bonus = 0.20 if int(self.goal_boundary_events.get(goal, 0)) <= 0 else 0.0
            termination_penalty = 0.15 * float(self.goal_termination_events.get(goal, 0)) / float(transition_count)
            effect_bonus = max(0.0, goal_effect - mean_effect)
            novelty_bonus = 0.0 if goal in self.recent_goals else 0.20
            score = 1.20 / (1.0 + count) + boundary_bonus + 0.15 * effect_bonus + novelty_bonus - termination_penalty
            score += float(self.rng.normal(0.0, 0.02))
            scores[goal] = score
        return max(scores.items(), key=lambda item: item[1])[0]

    def ensure_goal(self, state: np.ndarray) -> str:
        del state
        if self.current_goal is None:
            self.current_goal = self.rollout_goal or self.choose_rollout_goal(np.zeros((1,), dtype=np.float32))
            self.steps_in_goal = 0
            self.goal_counts[self.current_goal] = self.goal_counts.get(self.current_goal, 0) + 1
            self.recent_goals.append(self.current_goal)
            return self.current_goal

        if self.steps_in_goal < self.rollout_goal_locked_steps:
            return self.current_goal

        self.current_goal = self.rollout_goal or self.choose_rollout_goal(np.zeros((1,), dtype=np.float32))
        self.steps_in_goal = 0
        self.goal_counts[self.current_goal] = self.goal_counts.get(self.current_goal, 0) + 1
        self.recent_goals.append(self.current_goal)
        return self.current_goal

    def action_prior_scores(self, state: np.ndarray, recent_actions: list[int]) -> np.ndarray:
        goal = self.ensure_goal(np.zeros((1,), dtype=np.float32))
        counts = np.bincount(recent_actions, minlength=self.action_vocab_size).astype(np.float32)
        scores = 0.05 / (1.0 + counts)
        if self.mechanics_scalar_probe:
            state_np = np.asarray(state, dtype=np.float32).reshape(-1)
            left_idx = int(getattr(self, "scalar_left_idx", 0))
            right_idx = int(getattr(self, "scalar_right_idx", max(self.action_vocab_size - 1, 0)))
            center_idx = int(getattr(self, "scalar_center_idx", self.action_vocab_size // 2))
            if goal == "passive_decay":
                scores[center_idx] += 1.0
            elif goal == "impulse_left":
                scores[left_idx] += 1.0
            elif goal == "impulse_right":
                scores[right_idx] += 1.0
            elif goal == "chirp":
                scores[left_idx if (self.steps_in_goal // 2) % 2 == 0 else right_idx] += 1.0
            elif goal == "boundary_push":
                # Push in the direction the cart/pole is already leaning so
                # the window exposes edge dynamics instead of another center roll.
                position = float(state_np[0]) if state_np.size > 0 else 0.0
                angle = float(state_np[2]) if state_np.size > 2 else position
                scores[right_idx if position + 0.5 * angle >= 0.0 else left_idx] += 1.0
            elif goal == "cart_brake":
                velocity = float(state_np[1]) if state_np.size > 1 else 0.0
                scores[left_idx if velocity >= 0.0 else right_idx] += 1.0
            return scores.astype(np.float32)

        target_indices = self.family_indices.get(goal, np.asarray([], dtype=np.int64))
        if target_indices.size > 0:
            scores[target_indices] += 1.0
        if self.center_family is not None and goal != self.center_family and self.steps_in_goal >= 2:
            center_indices = self.family_indices[self.center_family]
            scores[center_indices] += float(self.center_recovery_bonus)
        return scores.astype(np.float32)

    def observe_transition(
        self,
        prev_state: np.ndarray,
        action_idx: int,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        del action_idx
        del truncated
        goal = self.current_goal or self.rollout_goal or (self.support_goal_sequence[0] if self.support_goal_sequence else "probe")
        prev_state_np = np.asarray(prev_state, dtype=np.float32).reshape(-1)
        next_state_np = np.asarray(next_state, dtype=np.float32).reshape(-1)
        transition_scale = float(np.linalg.norm(next_state_np - prev_state_np) / np.sqrt(max(float(prev_state_np.size), 1.0)))
        state_scale = float(np.mean(np.abs(next_state_np))) if next_state_np.size > 0 else 0.0
        baseline = float(np.mean(np.asarray(tuple(self.recent_transition_scales) or (transition_scale,), dtype=np.float32)))
        surprise_event = transition_scale > max(0.10, 1.75 * max(baseline, 1e-3))
        boundary_event = bool(terminated) or state_scale > max(0.30, 1.75 * max(baseline, 1e-3))

        self.coverage.total_steps += 1
        self.coverage.surprise_events += int(surprise_event)
        self.coverage.boundary_events += int(boundary_event)
        self.coverage.termination_events += int(bool(terminated))
        self.goal_transition_sum[goal] = self.goal_transition_sum.get(goal, 0.0) + transition_scale
        self.goal_transition_count[goal] = self.goal_transition_count.get(goal, 0) + 1
        self.goal_boundary_events[goal] = self.goal_boundary_events.get(goal, 0) + int(boundary_event)
        self.goal_termination_events[goal] = self.goal_termination_events.get(goal, 0) + int(bool(terminated))
        if self.recent_transition_scales:
            previous_scale = float(self.recent_transition_scales[-1])
            change = abs(transition_scale - previous_scale)
            if change > max(0.05, 0.75 * max(previous_scale, baseline, 1e-3)):
                self.coverage.controllability_changes += 1
        self.recent_transition_scales.append(transition_scale)
        self.steps_in_goal += 1
        if terminated:
            self.current_goal = None
            self.steps_in_goal = 0


def build_probe_planner(
    *,
    action_space: gym.spaces.Space,
    action_values: np.ndarray | None,
    rng: np.random.Generator,
    env_name: str | None = None,
):
    """Return the default generic probe explorer for supported gym action spaces."""
    if isinstance(action_space, (gym.spaces.Discrete, gym.spaces.Box)):
        return GenericProbeExplorer(
            action_space=action_space,
            action_values=action_values,
            rng=rng,
            env_name=env_name,
        )
    return None
