"""Scripted probe-policy library."""

from __future__ import annotations

import numpy as np


PROBE_MODES = (
    "random",
    "hold_left",
    "hold_right",
    "center_hold",
    "pulse_left",
    "pulse_right",
    "reverse",
    "sweep",
    "sticky_random",
    "burst_random",
    "anti_repeat",
    "phase_random",
)


class ProbePolicy:
    """Small library of repeatable probe behaviors."""

    def __init__(self, action_space_n: int, profile: str = "scalar"):
        if int(action_space_n) <= 0:
            raise ValueError("ProbePolicy needs at least one action")
        self.n = action_space_n
        self.profile = profile
        self._sticky = 0
        self._burst_len = 2

    def reset_episode(self):
        """Clear mode-local state so each probe episode starts from a clean slate."""
        self._sticky = 0
        self._burst_len = 2

    def _center_action(self) -> int:
        if self.profile == "lunar_lander":
            return 1
        if self.profile == "bipedal":
            return 0
        return (self.n - 1) // 2

    def _small_left_action(self) -> int:
        if self.profile == "lunar_lander":
            return 5
        if self.profile == "bipedal":
            return 1
        center = self._center_action()
        offset = max(1, (self.n - 1) // 4)
        return max(0, center - offset)

    def _small_right_action(self) -> int:
        if self.profile == "lunar_lander":
            return 6
        if self.profile == "bipedal":
            return 2
        center = self._center_action()
        offset = max(1, (self.n - 1) // 4)
        return min(self.n - 1, center + offset)

    def _hard_left_action(self) -> int:
        if self.profile == "lunar_lander":
            return 3
        if self.profile == "bipedal":
            return 3
        return 0

    def _hard_right_action(self) -> int:
        if self.profile == "lunar_lander":
            return 4
        if self.profile == "bipedal":
            return 4
        return self.n - 1

    def sample_action(self, mode: str, step_idx: int, rng: np.random.Generator) -> int:
        """Return the next discrete probe action for a named scripted mode."""
        if mode == "random":
            return int(rng.integers(0, self.n))
        if mode == "hold_left":
            return self._hard_left_action()
        if mode == "hold_right":
            return self._hard_right_action()
        if mode == "center_hold":
            return self._center_action()
        if mode == "alternate":
            return step_idx % self.n
        if mode == "pulse_left":
            return self._small_left_action() if step_idx < 2 else self._center_action()
        if mode == "pulse_right":
            return self._small_right_action() if step_idx < 2 else self._center_action()
        if mode == "reverse":
            if step_idx < 2:
                return self._small_left_action()
            if step_idx < 4:
                return self._small_right_action()
            return self._center_action()
        if mode == "sweep":
            pattern = (self._small_left_action(), self._center_action(), self._small_right_action(), self._center_action())
            return int(pattern[step_idx % len(pattern)])
        if mode == "sticky_random":
            burst_len = 4
            if step_idx % burst_len == 0:
                self._sticky = int(rng.integers(0, self.n))
            return self._sticky
        if mode == "burst_random":
            if step_idx == 0 or step_idx % self._burst_len == 0:
                self._burst_len = int(rng.integers(1, 5))
                self._sticky = int(rng.integers(0, self.n))
            return self._sticky
        if mode == "anti_repeat":
            if self.n == 1:
                return 0
            if step_idx == 0:
                self._sticky = int(rng.integers(0, self.n))
                return self._sticky
            choices = np.delete(np.arange(self.n), self._sticky)
            self._sticky = int(rng.choice(choices))
            return self._sticky
        if mode == "phase_random":
            center = self._center_action()
            if step_idx < 2:
                candidates = np.asarray([center, self._small_left_action(), self._small_right_action()])
            else:
                candidates = np.arange(self.n)
            return int(rng.choice(candidates))
        raise ValueError(f"Unknown probe mode: {mode}")
