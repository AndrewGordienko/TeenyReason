"""CartPole-specific active experiment planner.

This module owns the "scientist" behavior for Continuous CartPole. The goal is
not to solve the task directly, but to run short informative interventions that
reveal hidden mechanics such as pole length, mass, or force scale.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..envs import CONTINUOUS_CARTPOLE_NAME


CARTPOLE_ACTIVE_GOALS = (
    "passive_decay",
    "impulse_left",
    "impulse_right",
    "chirp",
    "counter_balance",
    "boundary_push",
    "cart_brake",
)

CARTPOLE_SUPPORT_GOAL_SEQUENCE = (
    "passive_decay",
    "impulse_left",
    "impulse_right",
    "chirp",
    "boundary_push",
    "cart_brake",
)


@dataclass
class CartPoleCoverage:
    """Lightweight coverage summary for one sampled env instance."""

    high_angle_steps: int = 0
    near_boundary_steps: int = 0
    sign_flip_steps: int = 0
    strong_action_steps: int = 0
    centered_decay_steps: int = 0


def build_probe_planner(env_name: str, action_values: np.ndarray, rng: np.random.Generator):
    """Return an env-specific scientist planner when the env supports one."""
    if env_name == CONTINUOUS_CARTPOLE_NAME:
        return CartPoleScientistPlanner(action_values=action_values, rng=rng)
    return None


class CartPoleScientistPlanner:
    """Choose informative CartPole experiments instead of replaying fixed scripts."""

    def __init__(
        self,
        action_values: np.ndarray,
        rng: np.random.Generator,
        goal_horizon: int = 6,
        theta_limit: float = 12.0 * 2.0 * np.pi / 360.0,
        x_threshold: float = 2.4,
    ):
        action_values_np = np.asarray(action_values, dtype=np.float32).reshape(-1)
        if action_values_np.ndim != 1:
            raise ValueError("CartPole scientist expects scalar continuous probe actions")

        self.action_values = action_values_np
        self.rng = rng
        self.goal_horizon = max(2, int(goal_horizon))
        self.theta_limit = float(theta_limit)
        self.x_threshold = float(x_threshold)
        self.rollout_goal_locked_steps = max(4, self.goal_horizon - 1)
        self.goal_target_counts = {
            "passive_decay": 1,
            "impulse_left": 1,
            "impulse_right": 1,
            "chirp": 1,
            "counter_balance": 1,
            "boundary_push": 1,
            "cart_brake": 1,
        }
        self.goal_counts = {goal: 0 for goal in CARTPOLE_ACTIVE_GOALS}
        self.coverage = CartPoleCoverage()
        self.recent_goals: deque[str] = deque(maxlen=4)
        self.current_goal: str | None = None
        self.rollout_goal: str | None = None
        self.steps_in_goal = 0
        self.last_impulse_sign = 1.0

    def begin_env_instance(self):
        """Reset persistent coverage when the crawler meets a new hidden world."""
        self.goal_counts = {goal: 0 for goal in CARTPOLE_ACTIVE_GOALS}
        self.coverage = CartPoleCoverage()
        self.recent_goals.clear()
        self.current_goal = None
        self.rollout_goal = None
        self.steps_in_goal = 0
        self.last_impulse_sign = 1.0

    def begin_rollout(self, primary_goal: str | None = None):
        """Reset the short-horizon experiment state while keeping env coverage."""
        self.current_goal = None
        self.rollout_goal = primary_goal
        self.steps_in_goal = 0

    def choose_rollout_goal(self, state: np.ndarray) -> str:
        """Pick the next experiment family for this rollout before acting."""
        state_np = np.asarray(state, dtype=np.float32)
        scores: dict[str, float] = {}
        has_unmet_goal = any(
            self.goal_counts[goal] < self.goal_target_counts[goal]
            for goal in CARTPOLE_ACTIVE_GOALS
        )
        for goal in CARTPOLE_ACTIVE_GOALS:
            score = self._rollout_goal_suitability(goal, state_np)
            if has_unmet_goal:
                score += 1.25 * max(0, self.goal_target_counts[goal] - self.goal_counts[goal])
            else:
                score += 0.30 / (1.0 + self.goal_counts[goal])
            if goal in self.recent_goals:
                score -= 0.25
            score += float(self.rng.normal(0.0, 0.02))
            scores[goal] = score
        return max(scores.items(), key=lambda item: item[1])[0]

    def ensure_goal(self, state: np.ndarray) -> str:
        """Return the current goal, switching when the state suggests a new test."""
        state_np = np.asarray(state, dtype=np.float32)
        if self.current_goal is None:
            if self.rollout_goal is None:
                self.rollout_goal = self.choose_rollout_goal(state_np)
            self.current_goal = self.rollout_goal
            self.steps_in_goal = 0
            self.goal_counts[self.current_goal] += 1
            self.recent_goals.append(self.current_goal)
            return self.current_goal

        if (
            self.rollout_goal is not None
            and self.current_goal == self.rollout_goal
            and self.steps_in_goal < self.rollout_goal_locked_steps
            and not self._needs_rescue_goal(state_np)
        ):
            return self.current_goal

        if self._should_switch_goal(state_np):
            self.current_goal = self._choose_goal(state_np)
            self.steps_in_goal = 0
            self.goal_counts[self.current_goal] += 1
            self.recent_goals.append(self.current_goal)
        return self.current_goal

    def action_prior_scores(self, state: np.ndarray, recent_actions: list[int]) -> np.ndarray:
        """Return goal-aware action priors before model-based probe scoring."""
        state_np = np.asarray(state, dtype=np.float32)
        goal = self.ensure_goal(state_np)
        scores = self._goal_action_scores(goal=goal, state=state_np)
        if recent_actions:
            counts = np.bincount(recent_actions, minlength=self.action_values.shape[0]).astype(np.float32)
            scores += 0.08 / (1.0 + counts)
        return scores.astype(np.float32)

    def observe_transition(
        self,
        prev_state: np.ndarray,
        action_idx: int,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Update coverage summary from one real interaction step."""
        del truncated
        prev_state_np = np.asarray(prev_state, dtype=np.float32)
        next_state_np = np.asarray(next_state, dtype=np.float32)
        prev_theta = float(prev_state_np[2])
        next_theta = float(next_state_np[2])
        next_x = float(next_state_np[0])
        action_value = float(self.action_values[int(action_idx)])

        if abs(next_theta) >= 0.45 * self.theta_limit:
            self.coverage.high_angle_steps += 1
        if abs(next_theta) >= 0.75 * self.theta_limit or abs(next_x) >= 0.65 * self.x_threshold:
            self.coverage.near_boundary_steps += 1
        if np.sign(prev_theta) != 0.0 and np.sign(prev_theta) != np.sign(next_theta):
            self.coverage.sign_flip_steps += 1
        if abs(action_value) >= 0.6:
            self.coverage.strong_action_steps += 1
        if abs(prev_theta) <= 0.04 and abs(next_theta) <= 0.04 and abs(action_value) <= 0.15:
            self.coverage.centered_decay_steps += 1

        self.steps_in_goal += 1
        if terminated:
            self.current_goal = None
            self.steps_in_goal = 0

    def _should_switch_goal(self, state: np.ndarray) -> bool:
        if self.current_goal is None:
            return True
        if self.steps_in_goal >= self.goal_horizon:
            return True

        theta = float(state[2])
        theta_dot = float(state[3])
        x = float(state[0])
        x_dot = float(state[1])
        angle_pressure = abs(theta) / max(self.theta_limit, 1e-6)
        cart_pressure = abs(x) / max(self.x_threshold, 1e-6)

        if self.current_goal in {"impulse_left", "impulse_right", "chirp"} and angle_pressure >= 0.55:
            return True
        if self.current_goal == "passive_decay" and (abs(theta_dot) > 1.0 or cart_pressure > 0.35):
            return True
        if self.current_goal == "boundary_push" and angle_pressure >= 0.85:
            return True
        if self.current_goal == "cart_brake" and abs(x_dot) < 0.25 and abs(x) < 0.35:
            return True
        return False

    def _needs_rescue_goal(self, state: np.ndarray) -> bool:
        theta = float(state[2])
        x = float(state[0])
        theta_dot = float(state[3])
        x_dot = float(state[1])
        return (
            abs(theta) >= 0.70 * self.theta_limit
            or abs(x) >= 0.70 * self.x_threshold
            or abs(theta_dot) >= 2.2
            or abs(x_dot) >= 1.8
        )

    def _choose_goal(self, state: np.ndarray, prefer_coverage: bool = False) -> str:
        x, x_dot, theta, theta_dot = [float(value) for value in state]
        abs_theta = abs(theta)
        abs_theta_dot = abs(theta_dot)
        abs_x = abs(x)
        abs_x_dot = abs(x_dot)
        neutral_state = abs_theta < 0.05 and abs_theta_dot < 0.7 and abs_x < 0.6
        scores: dict[str, float] = {}

        left_impulses = self.goal_counts["impulse_left"]
        right_impulses = self.goal_counts["impulse_right"]
        impulse_bias = float(np.clip(right_impulses - left_impulses, -3, 3))

        for goal in CARTPOLE_ACTIVE_GOALS:
            score = 0.35 / (1.0 + self.goal_counts[goal])
            target_count = self.goal_target_counts[goal]
            score += 0.70 * max(0, target_count - self.goal_counts[goal])
            score -= 0.55 * max(0, self.goal_counts[goal] - target_count)
            if goal in self.recent_goals:
                score -= 0.35
            if prefer_coverage and self.goal_counts[goal] == 0:
                score += 0.45
            scores[goal] = score

        scores["passive_decay"] += 1.2 if neutral_state else -0.2
        if self.coverage.centered_decay_steps < 24:
            scores["passive_decay"] += 0.15

        scores["impulse_left"] += 1.25 if neutral_state else 0.0
        scores["impulse_left"] += 0.20 * impulse_bias

        scores["impulse_right"] += 1.25 if neutral_state else 0.0
        scores["impulse_right"] -= 0.20 * impulse_bias

        scores["chirp"] += 1.15 if neutral_state else 0.20
        if self.coverage.sign_flip_steps < 12:
            scores["chirp"] += 0.60

        scores["counter_balance"] += 1.6 if abs_theta > 0.05 or abs_theta_dot > 0.8 else 0.2
        if abs_theta >= 0.75 * self.theta_limit:
            scores["counter_balance"] += 1.0
        if sum(self.goal_counts[goal] for goal in CARTPOLE_ACTIVE_GOALS if goal != "counter_balance") < 5:
            scores["counter_balance"] -= 1.15

        scores["boundary_push"] += 1.15 if 0.04 < abs_theta < 0.85 * self.theta_limit else -0.15
        if self.coverage.near_boundary_steps < 20:
            scores["boundary_push"] += 0.55
        if prefer_coverage and neutral_state:
            scores["boundary_push"] += 0.35

        scores["cart_brake"] += 1.1 if abs_x > 0.55 or abs_x_dot > 0.85 else 0.0
        if abs_x >= 0.75 * self.x_threshold:
            scores["cart_brake"] += 0.6
        if prefer_coverage and abs_x > 0.30:
            scores["cart_brake"] += 0.25

        for goal in scores:
            scores[goal] += float(self.rng.normal(0.0, 0.03))

        return max(scores.items(), key=lambda item: item[1])[0]

    def _goal_action_scores(self, goal: str, state: np.ndarray) -> np.ndarray:
        x, x_dot, theta, theta_dot = [float(value) for value in state]
        abs_theta = abs(theta)
        lean = self._signed_direction(theta + 0.25 * theta_dot, fallback=self.last_impulse_sign)
        destabilize = -lean
        brake = self._signed_direction(-(x_dot + 0.45 * x), fallback=-lean)

        if goal == "passive_decay":
            if abs_theta < 0.05 and abs(x_dot) < 0.5:
                return self._target_score(0.0, weight=1.0, sigma=0.22)
            recovery_target = lean * min(0.55, 0.25 + 2.5 * abs_theta)
            return self._target_score(recovery_target, weight=1.0, sigma=0.25)

        if goal == "impulse_left":
            if self.steps_in_goal < 2:
                self.last_impulse_sign = -1.0
                return self._target_score(-0.65, weight=1.35, sigma=0.20)
            coast_target = 0.0 if abs_theta < 0.06 else lean * 0.25
            return self._target_score(coast_target, weight=1.0, sigma=0.25)

        if goal == "impulse_right":
            if self.steps_in_goal < 2:
                self.last_impulse_sign = 1.0
                return self._target_score(0.65, weight=1.35, sigma=0.20)
            coast_target = 0.0 if abs_theta < 0.06 else lean * 0.25
            return self._target_score(coast_target, weight=1.0, sigma=0.25)

        if goal == "chirp":
            pattern = (-0.65, 0.0, 0.65, 0.0, -0.9, 0.0, 0.9, 0.0)
            target = pattern[self.steps_in_goal % len(pattern)]
            return self._target_score(target, weight=1.1, sigma=0.24)

        if goal == "counter_balance":
            magnitude = float(np.clip(0.35 + 3.0 * abs_theta + 0.35 * abs(theta_dot), 0.3, 1.0))
            return self._target_score(lean * magnitude, weight=1.2, sigma=0.24)

        if goal == "boundary_push":
            magnitude = float(np.clip(0.18 + 1.8 * abs_theta + 0.20 * abs(theta_dot), 0.18, 0.75))
            if abs_theta < 0.02:
                magnitude = 0.45
            return self._target_score(destabilize * magnitude, weight=1.1, sigma=0.24)

        if goal == "cart_brake":
            primary = self._target_score(brake * 0.8, weight=1.0, sigma=0.26)
            if abs_theta > 0.04:
                primary += self._target_score(lean * 0.35, weight=0.45, sigma=0.30)
            return primary

        raise ValueError(f"Unknown CartPole scientist goal: {goal}")

    def _rollout_goal_suitability(self, goal: str, state: np.ndarray) -> float:
        x, x_dot, theta, theta_dot = [float(value) for value in state]
        abs_theta = abs(theta)
        abs_theta_dot = abs(theta_dot)
        abs_x = abs(x)
        abs_x_dot = abs(x_dot)
        neutral_state = abs_theta < 0.05 and abs_theta_dot < 0.7 and abs_x < 0.6

        if goal == "passive_decay":
            return 0.75 if neutral_state else 0.15
        if goal in {"impulse_left", "impulse_right"}:
            return 1.15 if neutral_state else 0.35
        if goal == "chirp":
            return 1.10 if neutral_state else 0.45
        if goal == "boundary_push":
            return 0.95 if neutral_state else 0.70
        if goal == "counter_balance":
            return 0.45 if neutral_state else (1.05 if abs_theta > 0.06 or abs_theta_dot > 0.9 else 0.55)
        if goal == "cart_brake":
            return 0.40 if neutral_state else (1.0 if abs_x > 0.45 or abs_x_dot > 0.7 else 0.55)
        return 0.5

    def _target_score(self, target: float, weight: float, sigma: float) -> np.ndarray:
        sigma = max(float(sigma), 1e-3)
        squared_distance = np.square(self.action_values - float(np.clip(target, -1.0, 1.0)))
        return (weight * np.exp(-0.5 * squared_distance / (sigma * sigma))).astype(np.float32)

    def _signed_direction(self, value: float, fallback: float) -> float:
        if value > 1e-4:
            return 1.0
        if value < -1e-4:
            return -1.0
        if abs(fallback) > 1e-4:
            return float(np.sign(fallback))
        return -1.0 if float(self.rng.random()) < 0.5 else 1.0
