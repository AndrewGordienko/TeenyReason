"""Recorded probe-transition datatypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """Single recorded interaction step from a probe episode."""

    env_instance_id: int
    episode_id: int
    step_idx: int
    probe_mode: str
    env_params: np.ndarray
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    terminated: bool
    truncated: bool


@dataclass
class WindowRecord:
    """Fixed-length temporal slice used to train the latent encoder."""

    env_instance_id: int
    episode_id: int
    end_step_idx: int
    probe_mode: str
    env_params: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool
    truncated: bool
