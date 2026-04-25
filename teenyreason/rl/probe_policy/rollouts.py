"""Probe-policy rollout chunk helpers."""

from dataclasses import dataclass

import numpy as np

from ..core.ppo_core import build_episode_batch, sanitize_numpy


@dataclass
class RolloutChunk:
    """Single-env fixed-horizon rollout chunk used for PPO updates."""

    states: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[float]
    rewards: list[float]
    values: list[float]
    terminals: list[float]
    beliefs: list[np.ndarray] | None
    recurrent_hidden_states: list[np.ndarray] | None


def init_rollout_chunk(
    track_beliefs: bool,
    *,
    track_recurrent_hidden: bool = False,
) -> RolloutChunk:
    """Create one empty rollout chunk, optionally tracking belief vectors."""
    return RolloutChunk(
        states=[],
        actions=[],
        log_probs=[],
        rewards=[],
        values=[],
        terminals=[],
        beliefs=[] if track_beliefs else None,
        recurrent_hidden_states=[] if track_recurrent_hidden else None,
    )


def rollout_chunk_step_count(chunk: RolloutChunk) -> int:
    """Return how many environment steps are currently buffered."""
    return len(chunk.states)


def append_rollout_step(
    chunk: RolloutChunk,
    *,
    state: np.ndarray,
    action: np.ndarray,
    log_prob: float,
    reward: float,
    value: float,
    terminal: float,
    belief: np.ndarray | None = None,
    recurrent_hidden_state: np.ndarray | None = None,
):
    """Append one transition to the current fixed-horizon rollout chunk."""
    chunk.states.append(sanitize_numpy(state.copy()))
    chunk.actions.append(sanitize_numpy(action.copy()))
    chunk.log_probs.append(float(log_prob))
    chunk.rewards.append(float(reward))
    chunk.values.append(float(value))
    chunk.terminals.append(float(terminal))
    if chunk.beliefs is not None and belief is not None:
        chunk.beliefs.append(sanitize_numpy(belief.copy()))
    if chunk.recurrent_hidden_states is not None and recurrent_hidden_state is not None:
        chunk.recurrent_hidden_states.append(sanitize_numpy(recurrent_hidden_state.copy()))


def clear_rollout_chunk(chunk: RolloutChunk):
    """Reset the rollout chunk in place after one PPO update."""
    chunk.states.clear()
    chunk.actions.clear()
    chunk.log_probs.clear()
    chunk.rewards.clear()
    chunk.values.clear()
    chunk.terminals.clear()
    if chunk.beliefs is not None:
        chunk.beliefs.clear()
    if chunk.recurrent_hidden_states is not None:
        chunk.recurrent_hidden_states.clear()


def build_rollout_batch(
    chunk: RolloutChunk,
    *,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
    sequence_length: int | None = None,
):
    """Pack the current rollout chunk into one PPO batch."""
    return build_episode_batch(
        states=chunk.states,
        actions=chunk.actions,
        log_probs=chunk.log_probs,
        rewards=chunk.rewards,
        values=chunk.values,
        terminals=chunk.terminals,
        bootstrap_value=bootstrap_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
        beliefs=chunk.beliefs,
        recurrent_hidden_states=chunk.recurrent_hidden_states,
        sequence_length=sequence_length,
    )
