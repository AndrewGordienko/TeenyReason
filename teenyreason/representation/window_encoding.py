"""Window-level encoding helpers for latent snapshot analysis."""

from __future__ import annotations

import numpy as np
import torch

from ..models.belief_world_model import WorldEncoder


def encode_window_dataset(
    encoder: WorldEncoder,
    device: torch.device,
    windows: dict[str, np.ndarray],
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode every saved probe window into posterior mean and log-variance."""
    encoder.eval()
    states = windows["states"].astype(np.float32)
    actions = windows["actions"].astype(np.int64)
    rewards = windows["rewards"].astype(np.float32)

    means = []
    logvars = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            stop = start + batch_size
            state_t = torch.tensor(states[start:stop], dtype=torch.float32, device=device)
            action_t = torch.tensor(actions[start:stop], dtype=torch.long, device=device)
            reward_t = torch.tensor(rewards[start:stop], dtype=torch.float32, device=device)
            mean_t, logvar_t = encoder.encode_posterior(state_t, action_t, rewards=reward_t)
            means.append(mean_t.cpu().numpy().astype(np.float32))
            logvars.append(logvar_t.cpu().numpy().astype(np.float32))

    return (
        np.concatenate(means, axis=0).astype(np.float32),
        np.concatenate(logvars, axis=0).astype(np.float32),
    )
