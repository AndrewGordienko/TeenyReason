"""Small data containers shared by PPO rollout collection and optimization.

The rest of the RL package passes `EpisodeBatch` around instead of many
parallel arrays. Keeping it isolated makes it easy to inspect what an
optimizer step can see, which is usually the first place to look when PPO
metrics drift or a rollout looks misaligned.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeBatch:
    """Everything PPO needs from one or more collected episodes."""
    states: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    beliefs: np.ndarray | None
    recurrent_hidden_states: np.ndarray | None = None
    sequence_length: int | None = None
