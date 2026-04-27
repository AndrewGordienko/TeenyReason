"""Rollout packing helpers for feed-forward and recurrent PPO updates.

Collectors append Python lists because that is simple while stepping envs.
Before optimization, this module converts those lists into finite, shaped
arrays/tensors and builds padded recurrent sequences when the controller
needs hidden-state continuity.
"""

import numpy as np
import torch

from .numerics import compute_gae, sanitize_tensor
from .types import EpisodeBatch


def build_episode_batch(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    log_probs: list[float],
    rewards: list[float],
    values: list[float],
    terminals: list[float],
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
    beliefs: list[np.ndarray] | None = None,
    recurrent_hidden_states: list[np.ndarray] | None = None,
    sequence_length: int | None = None,
) -> EpisodeBatch:
    """Pack one collected episode into the normalized PPO batch structure."""
    rewards_np = np.asarray(rewards, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)
    terminals_np = np.asarray(terminals, dtype=np.float32)
    advantages, returns = compute_gae(
        rewards=rewards_np,
        values=values_np,
        terminals=terminals_np,
        bootstrap_value=bootstrap_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return EpisodeBatch(
        states=np.stack(states, axis=0).astype(np.float32),
        actions=np.stack(actions, axis=0).astype(np.float32),
        old_log_probs=np.asarray(log_probs, dtype=np.float32),
        old_values=values_np.astype(np.float32),
        returns=returns,
        advantages=advantages,
        beliefs=None if beliefs is None else np.stack(beliefs, axis=0).astype(np.float32),
        recurrent_hidden_states=(
            None
            if recurrent_hidden_states is None
            else np.stack(recurrent_hidden_states, axis=0).astype(np.float32)
        ),
        sequence_length=None if sequence_length is None else int(sequence_length),
    )


def concat_episode_batches(batches: list[EpisodeBatch]) -> EpisodeBatch:
    """Concatenate several episode batches before an optimizer step."""
    beliefs = None
    if batches[0].beliefs is not None:
        beliefs = np.concatenate([batch.beliefs for batch in batches], axis=0).astype(np.float32)
    recurrent_hidden_states = None
    if batches[0].recurrent_hidden_states is not None:
        recurrent_hidden_states = np.concatenate(
            [batch.recurrent_hidden_states for batch in batches],
            axis=0,
        ).astype(np.float32)

    return EpisodeBatch(
        states=np.concatenate([batch.states for batch in batches], axis=0).astype(np.float32),
        actions=np.concatenate([batch.actions for batch in batches], axis=0).astype(np.float32),
        old_log_probs=np.concatenate([batch.old_log_probs for batch in batches], axis=0).astype(np.float32),
        old_values=np.concatenate([batch.old_values for batch in batches], axis=0).astype(np.float32),
        returns=np.concatenate([batch.returns for batch in batches], axis=0).astype(np.float32),
        advantages=np.concatenate([batch.advantages for batch in batches], axis=0).astype(np.float32),
        beliefs=beliefs,
        recurrent_hidden_states=recurrent_hidden_states,
        sequence_length=batches[0].sequence_length,
    )


def _shuffle_context_codes_batch(sequences: torch.Tensor) -> torch.Tensor:
    """Shuffle only the code portion of each controller-context sequence."""
    if sequences.shape[-1] <= 2 or sequences.shape[0] == 0:
        return sequences
    shuffled = sequences.clone()
    code_dim = sequences.shape[-1] - 2
    permutations = torch.rand((sequences.shape[0], code_dim), device=sequences.device).argsort(dim=-1)
    gather_index = permutations[:, None, :].expand(-1, sequences.shape[1], -1)
    shuffled[..., :-2] = torch.gather(shuffled[..., :-2], dim=-1, index=gather_index)
    return shuffled


def corrupt_controller_context_sequences(
    beliefs: torch.Tensor,
    *,
    zero_prob: float,
    shuffle_prob: float,
    stale_prob: float,
) -> torch.Tensor:
    """Apply belief-native training corruption so the policy learns a safe fallback."""
    if beliefs.ndim != 3:
        return beliefs
    zero_prob = max(0.0, float(zero_prob))
    shuffle_prob = max(0.0, float(shuffle_prob))
    stale_prob = max(0.0, float(stale_prob))
    if zero_prob <= 0.0 and shuffle_prob <= 0.0 and stale_prob <= 0.0:
        return beliefs
    corrupted = beliefs.clone()
    thresholds = torch.tensor(
        np.cumsum([zero_prob, shuffle_prob, stale_prob]).tolist(),
        dtype=beliefs.dtype,
        device=beliefs.device,
    )
    samples = torch.rand((corrupted.shape[0],), device=beliefs.device)
    zero_mask = samples < thresholds[0]
    shuffle_mask = (samples >= thresholds[0]) & (samples < thresholds[1])
    stale_mask = (samples >= thresholds[1]) & (samples < thresholds[2])

    # Keep this vectorized: this function sits inside the recurrent PPO minibatch
    # loop, and per-sequence `.item()` calls can accidentally synchronize GPU work.
    if torch.any(zero_mask):
        corrupted[zero_mask] = 0.0
    if torch.any(shuffle_mask):
        corrupted[shuffle_mask] = _shuffle_context_codes_batch(corrupted[shuffle_mask])
    if torch.any(stale_mask):
        corrupted[stale_mask] = corrupted[stale_mask, :1, :].expand_as(corrupted[stale_mask])
    return sanitize_tensor(corrupted)


def prepare_recurrent_minibatch(
    *,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    beliefs: torch.Tensor,
    recurrent_hidden_states: torch.Tensor,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Pack flat rollout tensors into fixed-length padded sequences."""
    total_steps = int(states.shape[0])
    sequence_length = max(1, int(sequence_length))
    num_sequences = int(np.ceil(float(total_steps) / float(sequence_length)))
    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    belief_dim = beliefs.shape[-1]
    hidden_dim = recurrent_hidden_states.shape[-1]
    states_seq = torch.zeros(
        (num_sequences, sequence_length, state_dim),
        dtype=states.dtype,
        device=states.device,
    )
    actions_seq = torch.zeros(
        (num_sequences, sequence_length, action_dim),
        dtype=actions.dtype,
        device=actions.device,
    )
    old_log_probs_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=old_log_probs.dtype,
        device=old_log_probs.device,
    )
    old_values_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=old_values.dtype,
        device=old_values.device,
    )
    returns_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=returns.dtype,
        device=returns.device,
    )
    advantages_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=advantages.dtype,
        device=advantages.device,
    )
    beliefs_seq = torch.zeros(
        (num_sequences, sequence_length, belief_dim),
        dtype=beliefs.dtype,
        device=beliefs.device,
    )
    hidden_seq = torch.zeros(
        (num_sequences, hidden_dim),
        dtype=recurrent_hidden_states.dtype,
        device=recurrent_hidden_states.device,
    )
    mask_seq = torch.zeros(
        (num_sequences, sequence_length),
        dtype=states.dtype,
        device=states.device,
    )
    for seq_idx, start in enumerate(range(0, total_steps, sequence_length)):
        end = min(start + sequence_length, total_steps)
        valid = end - start
        states_seq[seq_idx, :valid] = states[start:end]
        actions_seq[seq_idx, :valid] = actions[start:end]
        old_log_probs_seq[seq_idx, :valid] = old_log_probs[start:end]
        old_values_seq[seq_idx, :valid] = old_values[start:end]
        returns_seq[seq_idx, :valid] = returns[start:end]
        advantages_seq[seq_idx, :valid] = advantages[start:end]
        beliefs_seq[seq_idx, :valid] = beliefs[start:end]
        hidden_seq[seq_idx] = recurrent_hidden_states[start]
        mask_seq[seq_idx, :valid] = 1.0
    return {
        "states": states_seq,
        "actions": actions_seq,
        "old_log_probs": old_log_probs_seq,
        "old_values": old_values_seq,
        "returns": returns_seq,
        "advantages": advantages_seq,
        "beliefs": beliefs_seq,
        "hidden": hidden_seq,
        "mask": mask_seq,
    }
