"""Self-imitation helpers for probe-conditioned policy training."""

import torch

from .....crawler.probes.latent import EliteTrajectoryBuffer
from ....core import ProbeConditionedGaussianActorCritic, evaluate_continuous_actions, sanitize_numpy


def compute_sil_loss(
    model: ProbeConditionedGaussianActorCritic,
    elite_buffer: EliteTrajectoryBuffer,
    batch_size: int,
    device: torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the self-imitation auxiliary loss from elite replay samples."""
    # Self-imitation learning replays especially good trajectories alongside PPO.
    states, beliefs, actions, returns_to_go, sample_weight = elite_buffer.sample(batch_size)
    state_t = torch.tensor(sanitize_numpy(states), dtype=torch.float32, device=device)
    belief_t = torch.tensor(sanitize_numpy(beliefs), dtype=torch.float32, device=device)
    action_t = torch.tensor(sanitize_numpy(actions), dtype=torch.float32, device=device)
    return_t = torch.tensor(sanitize_numpy(returns_to_go), dtype=torch.float32, device=device)
    weight_t = torch.tensor(sanitize_numpy(sample_weight), dtype=torch.float32, device=device)

    mean, value = model(state_t, belief_t)
    log_prob, _entropy = evaluate_continuous_actions(
        mean=mean,
        log_std=model.log_std,
        actions=action_t,
        action_low=action_low,
        action_high=action_high,
    )
    positive_advantage = torch.clamp(return_t - value.detach(), min=0.0)
    sil_policy_loss = -(log_prob * positive_advantage * weight_t).mean()
    sil_value_gap = torch.clamp(return_t - value, min=0.0)
    sil_value_loss = 0.5 * (sil_value_gap.pow(2) * weight_t).mean()
    return sil_policy_loss, sil_value_loss
