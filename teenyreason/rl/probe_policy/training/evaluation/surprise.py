"""Prediction-surprise diagnostics used during training and evaluation."""

import numpy as np
import torch
import torch.nn as nn

from .....models.belief import build_future_summary_targets
from .....cognition.representation import DeltaPredictorEnsemble
from ....core import sanitize_numpy, sanitize_tensor
from .....crawler.probes.latent import belief_mean_z


def compute_probe_surprise(
    *,
    env_future_predictor: nn.Module | None,
    belief: np.ndarray | None,
    window_states: np.ndarray,
    window_actions: np.ndarray,
    window_rewards: np.ndarray,
    action_vocab_size: int,
    device: torch.device,
    env_name: str | None = None,
    probe_family: str | None = None,
) -> float:
    """Measure how wrong the current belief was about what a fresh probe would reveal."""
    if env_future_predictor is None or belief is None:
        return 0.0

    split_idx = max(2, window_actions.shape[0] // 2)
    future_target = build_future_summary_targets(
        states=window_states[None, split_idx:, ...],
        actions=window_actions[None, split_idx:],
        rewards=window_rewards[None, split_idx:],
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        action_vocab_size=action_vocab_size,
        probe_mode=None if probe_family is None else np.asarray([probe_family], dtype="U"),
        env_name=env_name,
    )
    belief_t = torch.tensor(
        sanitize_numpy(belief_mean_z(belief)[None, :]),
        dtype=torch.float32,
        device=device,
    )
    future_target_t = torch.tensor(
        sanitize_numpy(future_target),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        future_pred = sanitize_tensor(env_future_predictor(belief_t))
    if future_pred.shape[-1] != future_target_t.shape[-1]:
        shared_dim = min(int(future_pred.shape[-1]), int(future_target_t.shape[-1]))
        if shared_dim <= 0:
            return 0.0
        future_pred = future_pred[..., :shared_dim]
        future_target_t = future_target_t[..., :shared_dim]
    return float(torch.mean(torch.abs(future_pred - future_target_t)).item())


def compute_control_surprise(
    *,
    predictor: DeltaPredictorEnsemble | None,
    belief: np.ndarray | None,
    prev_state: np.ndarray,
    action_idx: int,
    next_state: np.ndarray,
    device: torch.device,
) -> float:
    """Measure one-step action-conditioned surprise during downstream control."""
    if predictor is None or belief is None:
        return 0.0
    belief_t = torch.tensor(
        sanitize_numpy(belief_mean_z(belief)[None, :]),
        dtype=torch.float32,
        device=device,
    )
    prev_state_t = torch.tensor(
        sanitize_numpy(prev_state[None, :]),
        dtype=torch.float32,
        device=device,
    )
    action_t = torch.tensor([int(action_idx)], dtype=torch.long, device=device)
    target_delta_t = torch.tensor(
        sanitize_numpy((next_state - prev_state)[None, :]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        predicted_delta = sanitize_tensor(
            predictor.predict_all(prev_state_t, action_t, belief_t).mean(dim=0)
        )
    return float(torch.mean(torch.abs(predicted_delta - target_delta_t)).item())
