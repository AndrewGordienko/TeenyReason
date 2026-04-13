"""Recurrent world-model and structured latent supervision for probe-conditioned RL."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..envs import BIPEDAL_WALKER_NAME
from .env_belief import (
    EnvBeliefAggregator,
    EnvParamPredictorEnsemble,
    build_env_group_tensors,
    build_env_subset_masks,
    group_window_latents_torch,
    sample_env_belief_subsets,
)


def split_latent_dims(z_dim: int) -> tuple[int, int, int]:
    """Split the latent into dynamics, control, and contact-ish subspaces."""
    dyn_dim = max(4, z_dim // 2)
    ctrl_dim = max(4, z_dim // 3)
    contact_dim = max(2, z_dim - dyn_dim - ctrl_dim)
    total = dyn_dim + ctrl_dim + contact_dim
    if total != z_dim:
        dyn_dim += z_dim - total
    return dyn_dim, ctrl_dim, contact_dim


class WorldEncoder(nn.Module):
    """Encode a probe trajectory into a posterior over a structured latent."""

    def __init__(
        self,
        state_dim: int = 4,
        window_size: int = 8,
        action_vocab_size: int = 2,
        z_dim: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.window_size = window_size
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.max_context = max(window_size * 4, 128)
        self.action_emb = nn.Embedding(action_vocab_size, 8)
        self.position_emb = nn.Embedding(self.max_context, hidden_dim)
        token_dim = state_dim * 2 + 8 + 1
        self.token_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.context_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.context_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.context_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        self.z_dyn_dim, self.z_ctrl_dim, self.z_contact_dim = split_latent_dims(z_dim)

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return an empty recurrent belief state for one or more trajectories."""
        if device is None:
            device = self.mean_head.weight.device
        return torch.zeros(1, batch_size, self.hidden_dim, dtype=torch.float32, device=device)

    def build_tokens(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prev_state = states[:, :-1, :]
        next_state = states[:, 1:, :]
        delta_state = next_state - prev_state
        action_emb = self.action_emb(actions)
        if rewards is None:
            rewards = torch.zeros(actions.shape, dtype=torch.float32, device=actions.device)
        reward_feat = rewards.unsqueeze(-1)
        token = torch.cat([prev_state, delta_state, action_emb, reward_feat], dim=-1)
        token_features = self.token_net(token)
        step_positions = torch.arange(actions.shape[1], device=actions.device)
        step_positions = torch.clamp(step_positions, max=self.max_context - 1)
        pos_emb = self.position_emb(step_positions).unsqueeze(0)
        return token_features + pos_emb

    def build_step_token(
        self,
        prev_state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build one recurrent token from a single `(s, a, r, s')` transition."""
        if reward is None:
            reward = torch.zeros(action.shape[0], dtype=torch.float32, device=action.device)
        delta_state = next_state - prev_state
        action_emb = self.action_emb(action)
        token = torch.cat([prev_state, delta_state, action_emb, reward.unsqueeze(-1)], dim=-1)
        return self.token_net(token)

    def posterior_from_hidden(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode posterior parameters from the current recurrent hidden state."""
        final_hidden = hidden[-1]
        mean = self.mean_head(final_hidden)
        logvar = torch.clamp(self.logvar_head(final_hidden), -5.0, 2.0)
        return mean, logvar

    def summarize_outputs(
        self,
        outputs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Combine final recurrent state with attention over the whole trajectory."""
        query = self.context_query.expand(outputs.shape[0], -1, -1)
        attended, _attn_weights = self.context_attn(query, outputs, outputs)
        final_hidden = hidden[-1].unsqueeze(1)
        return self.context_fuse(torch.cat([final_hidden, attended], dim=-1)).squeeze(1)

    def posterior_from_context(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode posterior parameters from an attention-pooled trajectory summary."""
        mean = self.mean_head(context)
        logvar = torch.clamp(self.logvar_head(context), -5.0, 2.0)
        return mean, logvar

    def update_belief(
        self,
        prev_state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the recurrent belief with one transition and return the new posterior."""
        if hidden is None:
            hidden = self.init_hidden(prev_state.shape[0], device=prev_state.device)
        token = self.build_step_token(prev_state, next_state, action, reward=reward).unsqueeze(1)
        _outputs, hidden = self.gru(token, hidden)
        mean, logvar = self.posterior_from_hidden(hidden)
        return hidden, mean, logvar

    def encode_posterior(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.build_tokens(states, actions, rewards=rewards)
        if hidden is None:
            hidden = self.init_hidden(states.shape[0], device=states.device)
        outputs, hidden = self.gru(tokens, hidden)
        context = self.summarize_outputs(outputs, hidden)
        mean, logvar = self.posterior_from_context(context)
        return mean, logvar

    def encode_step_posteriors(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior parameters after every observed transition in the window."""
        tokens = self.build_tokens(states, actions, rewards=rewards)
        if hidden is None:
            hidden = self.init_hidden(states.shape[0], device=states.device)
        outputs, _hidden = self.gru(tokens, hidden)
        mean = self.mean_head(outputs)
        logvar = torch.clamp(self.logvar_head(outputs), -5.0, 2.0)
        return mean, logvar

    def sample_latent(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mean + noise * std

    def split_latent(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        dyn_end = self.z_dyn_dim
        ctrl_end = dyn_end + self.z_ctrl_dim
        return {
            "dyn": z[:, :dyn_end],
            "ctrl": z[:, dyn_end:ctrl_end],
            "contact": z[:, ctrl_end:],
        }

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
        sample: bool = False,
    ) -> torch.Tensor:
        mean, logvar = self.encode_posterior(states, actions, rewards=rewards)
        if sample:
            return self.sample_latent(mean, logvar)
        return mean


class DeltaPredictor(nn.Module):
    """One ensemble member that predicts next-state delta."""

    def __init__(
        self,
        state_dim: int = 4,
        action_vocab_size: int = 2,
        z_dim: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(action_vocab_size, 8)
        self.net = nn.Sequential(
            nn.Linear(state_dim + 8 + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        action_emb = self.action_emb(action)
        return self.net(torch.cat([state, action_emb, z], dim=1))


class DeltaPredictorEnsemble(nn.Module):
    """Ensemble for transition prediction and disagreement-based active probing."""

    def __init__(
        self,
        ensemble_size: int,
        state_dim: int,
        action_vocab_size: int,
        z_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                DeltaPredictor(
                    state_dim=state_dim,
                    action_vocab_size=action_vocab_size,
                    z_dim=z_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.ensemble_size = ensemble_size

    def predict_all(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack([head(state, action, z) for head in self.heads], dim=0)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_all(state, action, z).mean(dim=0)


class LatentTransitionModel(nn.Module):
    """Predict the next belief posterior from the current belief and new evidence."""

    def __init__(
        self,
        state_dim: int,
        action_vocab_size: int,
        z_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(action_vocab_size, 8)
        self.net = nn.Sequential(
            nn.Linear(z_dim + state_dim + 8 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)

    def forward(
        self,
        latent_mean: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_emb = self.action_emb(action)
        features = self.net(torch.cat([latent_mean, state, action_emb, reward.unsqueeze(-1)], dim=-1))
        next_mean = self.mean_head(features)
        next_logvar = torch.clamp(self.logvar_head(features), -5.0, 2.0)
        return next_mean, next_logvar


class OutcomePredictor(nn.Module):
    """Generic MLP decoder from a latent subspace to a target vector."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ContrastiveProjector(nn.Module):
    """Small normalized projection head for contrastive latent supervision."""

    def __init__(self, input_dim: int, output_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class GradientReversal(torch.autograd.Function):
    """Reverse gradients so the encoder can be trained against a nuisance classifier."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply a simple gradient-reversal layer."""
    return GradientReversal.apply(x, scale)


class ProbeModeAdversary(nn.Module):
    """Predict probe mode from the latent so the encoder can learn to hide it."""

    def __init__(self, input_dim: int, num_modes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modes),
        )

    def forward(self, x: torch.Tensor, reverse_scale: float = 1.0) -> torch.Tensor:
        return self.net(gradient_reverse(x, scale=reverse_scale))


def encode_probe_modes(probe_modes: np.ndarray) -> np.ndarray:
    """Map probe-mode strings to stable integer ids."""
    unique_modes, probe_mode_idx = np.unique(probe_modes.astype("U"), return_inverse=True)
    del unique_modes
    return probe_mode_idx.astype(np.int64)


def supervised_same_env_contrastive_loss(
    embeddings: torch.Tensor,
    env_instance_id: torch.Tensor,
    probe_mode_idx: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pull together embeddings from the same env instance across probe modes."""
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0

    logits = embeddings @ embeddings.T / max(temperature, 1e-4)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    same_env = env_instance_id[:, None] == env_instance_id[None, :]
    different_mode = probe_mode_idx[:, None] != probe_mode_idx[None, :]
    positive_mask = same_env & different_mode & (~self_mask)
    fallback_mask = same_env & (~self_mask)
    has_cross_mode_positive = positive_mask.any(dim=1, keepdim=True)
    positive_mask = torch.where(has_cross_mode_positive, positive_mask, fallback_mask)
    valid_rows = positive_mask.any(dim=1)
    if not torch.any(valid_rows):
        return embeddings.sum() * 0.0

    exp_logits = torch.exp(logits) * (~self_mask)
    positive_mass = (exp_logits * positive_mask).sum(dim=1).clamp_min(1e-6)
    total_mass = exp_logits.sum(dim=1).clamp_min(1e-6)
    loss = -torch.log(positive_mass / total_mass)
    return loss[valid_rows].mean()


def pairwise_env_geometry_loss(
    latent_mean: torch.Tensor,
    normalized_env_params: torch.Tensor,
) -> torch.Tensor:
    """Encourage latent distances to reflect true env-parameter distances."""
    if latent_mean.shape[0] < 2:
        return latent_mean.sum() * 0.0

    normalized_latent = F.normalize(latent_mean, dim=-1)
    latent_distance = 1.0 - normalized_latent @ normalized_latent.T
    env_distance = torch.cdist(normalized_env_params, normalized_env_params, p=2)
    mask = ~torch.eye(latent_mean.shape[0], dtype=torch.bool, device=latent_mean.device)
    if not torch.any(mask):
        return latent_mean.sum() * 0.0

    latent_values = latent_distance[mask]
    env_values = env_distance[mask]
    latent_values = latent_values / latent_values.mean().clamp_min(1e-6)
    env_values = env_values / env_values.mean().clamp_min(1e-6)
    return F.mse_loss(latent_values, env_values)


def within_between_env_loss(
    env_mean: torch.Tensor,
    subset_env_mean: torch.Tensor,
    env_params: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Keep same-env subset beliefs tighter than neighboring different-env beliefs."""
    if env_mean.shape[0] < 2:
        return env_mean.sum() * 0.0

    within = torch.norm(subset_env_mean - env_mean.unsqueeze(1), dim=-1).mean(dim=1)
    between = torch.cdist(env_mean, env_mean, p=2)
    self_mask = torch.eye(env_mean.shape[0], dtype=torch.bool, device=env_mean.device)
    between = between.masked_fill(self_mask, float("inf"))

    param_distance = torch.cdist(env_params, env_params, p=2)
    non_self = ~self_mask
    valid_param = param_distance[non_self]
    if valid_param.numel() == 0:
        nearest_between = between.min(dim=1).values
    else:
        hard_negative_cutoff = torch.quantile(valid_param, 0.35)
        hard_negative_mask = (param_distance >= hard_negative_cutoff) & non_self
        masked_between = between.masked_fill(~hard_negative_mask, float("inf"))
        nearest_between = masked_between.min(dim=1).values
        fallback_between = between.min(dim=1).values
        nearest_between = torch.where(torch.isfinite(nearest_between), nearest_between, fallback_between)

    margin_loss = F.relu(within + margin - nearest_between).mean()
    ratio_loss = (within / nearest_between.clamp_min(1e-6)).mean()
    return margin_loss + 0.10 * ratio_loss


def build_generic_affordance_targets(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    action_vocab_size: int,
) -> np.ndarray:
    """Summarize short-horizon behavior directly from the probe windows."""
    initial_state = states[:, 0, :]
    current_state = states[:, -2, :]
    next_state = states[:, -1, :]

    from_start_delta = next_state - initial_state
    recent_delta = next_state - current_state
    state_span = np.max(states, axis=1) - np.min(states, axis=1)
    mean_abs_state = np.mean(np.abs(states), axis=1)
    reward_summary = np.stack(
        [
            np.sum(rewards, axis=1),
            np.mean(rewards, axis=1),
            np.min(rewards, axis=1),
            np.max(rewards, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    action_hist = np.zeros((actions.shape[0], action_vocab_size), dtype=np.float32)
    for row_idx in range(actions.shape[0]):
        counts = np.bincount(actions[row_idx], minlength=action_vocab_size).astype(np.float32)
        action_hist[row_idx] = counts / max(float(np.sum(counts)), 1.0)

    terminal_summary = np.stack(
        [
            terminated.astype(np.float32),
            truncated.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return np.concatenate(
        [
            from_start_delta.astype(np.float32),
            recent_delta.astype(np.float32),
            state_span.astype(np.float32),
            mean_abs_state.astype(np.float32),
            reward_summary,
            action_hist,
            terminal_summary,
        ],
        axis=1,
    ).astype(np.float32)


def build_future_summary_targets(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    action_vocab_size: int,
) -> np.ndarray:
    """Summarize the suffix of a trajectory for contrastive future prediction."""
    start_state = states[:, 0, :]
    end_state = states[:, -1, :]
    end_delta = end_state - start_state
    step_delta = np.diff(states, axis=1)
    mean_step_delta = np.mean(step_delta, axis=1)
    state_span = np.max(states, axis=1) - np.min(states, axis=1)
    reward_sum = np.sum(rewards, axis=1, dtype=np.float32).reshape(-1, 1)
    reward_mean = np.mean(rewards, axis=1, dtype=np.float32).reshape(-1, 1)
    reward_max = np.max(rewards, axis=1).reshape(-1, 1)
    reward_min = np.min(rewards, axis=1).reshape(-1, 1)

    action_hist = np.zeros((actions.shape[0], action_vocab_size), dtype=np.float32)
    for row_idx in range(actions.shape[0]):
        counts = np.bincount(actions[row_idx], minlength=action_vocab_size).astype(np.float32)
        action_hist[row_idx] = counts / max(float(np.sum(counts)), 1.0)

    terminal_summary = np.stack(
        [
            terminated.astype(np.float32),
            truncated.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    return np.concatenate(
        [
            end_delta.astype(np.float32),
            mean_step_delta.astype(np.float32),
            state_span.astype(np.float32),
            reward_sum.astype(np.float32),
            reward_mean.astype(np.float32),
            reward_max.astype(np.float32),
            reward_min.astype(np.float32),
            action_hist.astype(np.float32),
            terminal_summary,
        ],
        axis=1,
    ).astype(np.float32)


def build_bipedal_decision_targets(
    states: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> np.ndarray:
    """Decision-relevant supervision for BipedalWalker-style locomotion."""
    current_state = states[:, -1, :]
    state_diff = np.diff(states, axis=1)
    delta_norm = np.linalg.norm(state_diff, axis=2)
    reward_sum = np.sum(rewards, axis=1)
    reward_mid = rewards.shape[1] // 2
    reward_trend = np.mean(rewards[:, reward_mid:], axis=1) - np.mean(rewards[:, :reward_mid], axis=1)
    hull_angle = current_state[:, 0]
    hull_angular_velocity = current_state[:, 1]
    forward_speed = current_state[:, 2]
    vertical_speed = current_state[:, 3]
    left_contact = np.clip(current_state[:, 8], 0.0, 1.0)
    right_contact = np.clip(current_state[:, 13], 0.0, 1.0)
    both_contact = left_contact * right_contact
    contact_balance = 1.0 - np.abs(left_contact - right_contact)
    upright_margin = 1.0 - np.clip(np.abs(hull_angle), 0.0, 1.5) / 1.5
    angular_stability = 1.0 / (1.0 + np.abs(hull_angular_velocity))
    motion_energy = np.mean(delta_norm, axis=1)
    recoverability = upright_margin + 0.25 * reward_trend - 0.10 * np.abs(vertical_speed)
    fall_risk = np.logical_or(terminated, truncated).astype(np.float32)
    return np.stack(
        [
            reward_sum.astype(np.float32),
            reward_trend.astype(np.float32),
            forward_speed.astype(np.float32),
            vertical_speed.astype(np.float32),
            upright_margin.astype(np.float32),
            angular_stability.astype(np.float32),
            motion_energy.astype(np.float32),
            left_contact.astype(np.float32),
            right_contact.astype(np.float32),
            both_contact.astype(np.float32),
            contact_balance.astype(np.float32),
            recoverability.astype(np.float32),
            fall_risk.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def build_generic_decision_targets(
    states: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> np.ndarray:
    """Generic decision targets when no environment-specific heuristic is available."""
    current_state = states[:, -1, :]
    state_diff = np.diff(states, axis=1)
    delta_norm = np.linalg.norm(state_diff, axis=2)
    reward_sum = np.sum(rewards, axis=1)
    reward_mid = rewards.shape[1] // 2
    reward_trend = np.mean(rewards[:, reward_mid:], axis=1) - np.mean(rewards[:, :reward_mid], axis=1)
    state_energy = np.mean(np.abs(current_state), axis=1)
    motion_energy = np.mean(delta_norm, axis=1)
    state_span = np.mean(np.max(states, axis=1) - np.min(states, axis=1), axis=1)
    fall_risk = np.logical_or(terminated, truncated).astype(np.float32)
    return np.stack(
        [
            reward_sum.astype(np.float32),
            reward_trend.astype(np.float32),
            state_energy.astype(np.float32),
            motion_energy.astype(np.float32),
            state_span.astype(np.float32),
            fall_risk.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def build_decision_targets(
    states: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    env_name: str | None,
) -> np.ndarray:
    """Dispatch to environment-specific or generic decision-focused targets."""
    if env_name == BIPEDAL_WALKER_NAME and states.shape[-1] >= 14:
        return build_bipedal_decision_targets(states, rewards, terminated, truncated)
    return build_generic_decision_targets(states, rewards, terminated, truncated)


def normalize_targets(values: np.ndarray) -> np.ndarray:
    """Standardize a target matrix columnwise for easier training."""
    value_mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    value_std = values.std(axis=0, keepdims=True).astype(np.float32)
    value_std = np.where(value_std < 1e-6, 1.0, value_std)
    return ((values - value_mean) / value_std).astype(np.float32)


def build_training_tensors(
    windows: dict[str, np.ndarray],
    action_vocab_size: int,
    intervention_horizon: int,
    analytic_affordances: bool = True,
    env_name: str | None = None,
) -> dict[str, np.ndarray]:
    """Convert recorded windows into the tensors consumed by encoder training."""
    del intervention_horizon
    del analytic_affordances
    states = windows["states"]
    actions = windows["actions"]
    rewards = windows["rewards"]
    env_params = windows["env_params"]
    env_instance_id = windows.get(
        "env_instance_id",
        np.arange(states.shape[0], dtype=np.int32),
    )
    probe_mode_idx = encode_probe_modes(np.asarray(windows["probe_mode"], dtype="U"))
    terminated = windows["terminated"]
    truncated = windows["truncated"]

    current_state = states[:, -2, :]
    next_state = states[:, -1, :]
    delta_state = next_state - current_state
    current_action = actions[:, -1]
    split_idx = max(2, actions.shape[1] // 2)
    prefix_states = states[:, : split_idx + 1, :]
    prefix_actions = actions[:, :split_idx]
    prefix_rewards = rewards[:, :split_idx]
    future_states = states[:, split_idx:, :]
    future_actions = actions[:, split_idx:]
    future_rewards = rewards[:, split_idx:]
    return_target = np.sum(rewards, axis=1, dtype=np.float32).reshape(-1, 1)
    risk_target = np.logical_or(terminated, truncated).astype(np.float32).reshape(-1, 1)

    target_affordances = build_generic_affordance_targets(
        states=states,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        action_vocab_size=action_vocab_size,
    )
    decision_targets = build_decision_targets(
        states=states,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        env_name=env_name,
    )
    future_summary_targets = build_future_summary_targets(
        states=future_states,
        actions=future_actions,
        rewards=future_rewards,
        terminated=terminated,
        truncated=truncated,
        action_vocab_size=action_vocab_size,
    )

    return {
        "window_states": states.astype(np.float32),
        "window_actions": actions.astype(np.int64),
        "window_rewards": rewards.astype(np.float32),
        "env_instance_id": np.asarray(env_instance_id, dtype=np.int64),
        "probe_mode_idx": np.asarray(probe_mode_idx, dtype=np.int64),
        "prefix_states": prefix_states.astype(np.float32),
        "prefix_actions": prefix_actions.astype(np.int64),
        "prefix_rewards": prefix_rewards.astype(np.float32),
        "current_state": current_state.astype(np.float32),
        "current_action": current_action.astype(np.int64),
        "target_delta": delta_state.astype(np.float32),
        "target_env_params": normalize_targets(env_params).astype(np.float32),
        "target_affordances": normalize_targets(target_affordances).astype(np.float32),
        "target_decision": normalize_targets(decision_targets).astype(np.float32),
        "target_return": normalize_targets(return_target).astype(np.float32),
        "target_risk": risk_target.astype(np.float32),
        "target_future_summary": normalize_targets(future_summary_targets).astype(np.float32),
    }


def info_nce_loss(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between latent-prefix queries and future-summary keys."""
    logits = query_embeddings @ key_embeddings.T / max(temperature, 1e-4)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def train_encoder_predictor(
    windows: dict[str, np.ndarray],
    z_dim: int = 8,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    physics_loss_weight: float = 0.1,
    affordance_loss_weight: float = 1.0,
    decision_loss_weight: float = 1.0,
    return_loss_weight: float = 0.5,
    risk_loss_weight: float = 0.25,
    kl_loss_weight: float = 1e-3,
    contrastive_loss_weight: float = 0.25,
    env_consistency_loss_weight: float = 0.35,
    env_geometry_loss_weight: float = 0.20,
    mode_adversary_loss_weight: float = 0.10,
    latent_rollout_loss_weight: float = 0.15,
    env_within_between_loss_weight: float = 0.30,
    belief_subset_count: int = 4,
    belief_subset_size: int = 6,
    contrastive_dim: int = 64,
    ensemble_size: int = 3,
    action_vocab_size: int | None = None,
    intervention_horizon: int = 12,
    analytic_affordances: bool = True,
    env_name: str | None = None,
) -> tuple[
    WorldEncoder,
    DeltaPredictorEnsemble,
    EnvBeliefAggregator,
    EnvParamPredictorEnsemble,
    torch.device,
]:
    """Train the recurrent posterior encoder and its structured decoders jointly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if action_vocab_size is None:
        action_vocab_size = int(np.max(windows["actions"])) + 1

    tensors = build_training_tensors(
        windows,
        action_vocab_size=action_vocab_size,
        intervention_horizon=intervention_horizon,
        analytic_affordances=analytic_affordances,
        env_name=env_name,
    )

    window_states = torch.tensor(tensors["window_states"], dtype=torch.float32, device=device)
    window_actions = torch.tensor(tensors["window_actions"], dtype=torch.long, device=device)
    window_rewards = torch.tensor(tensors["window_rewards"], dtype=torch.float32, device=device)
    prefix_states = torch.tensor(tensors["prefix_states"], dtype=torch.float32, device=device)
    prefix_actions = torch.tensor(tensors["prefix_actions"], dtype=torch.long, device=device)
    prefix_rewards = torch.tensor(tensors["prefix_rewards"], dtype=torch.float32, device=device)
    env_instance_id = torch.tensor(tensors["env_instance_id"], dtype=torch.long, device=device)
    probe_mode_idx = torch.tensor(tensors["probe_mode_idx"], dtype=torch.long, device=device)
    current_state = torch.tensor(tensors["current_state"], dtype=torch.float32, device=device)
    current_action = torch.tensor(tensors["current_action"], dtype=torch.long, device=device)
    target_delta = torch.tensor(tensors["target_delta"], dtype=torch.float32, device=device)
    target_env_params = torch.tensor(tensors["target_env_params"], dtype=torch.float32, device=device)
    target_affordances = torch.tensor(tensors["target_affordances"], dtype=torch.float32, device=device)
    target_decision = torch.tensor(tensors["target_decision"], dtype=torch.float32, device=device)
    target_return = torch.tensor(tensors["target_return"], dtype=torch.float32, device=device)
    target_risk = torch.tensor(tensors["target_risk"], dtype=torch.float32, device=device)
    target_future_summary = torch.tensor(
        tensors["target_future_summary"],
        dtype=torch.float32,
        device=device,
    )

    encoder = WorldEncoder(
        state_dim=window_states.shape[-1],
        window_size=window_actions.shape[1],
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    predictor = DeltaPredictorEnsemble(
        ensemble_size=ensemble_size,
        state_dim=window_states.shape[-1],
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    belief_aggregator = EnvBeliefAggregator(window_z_dim=z_dim).to(device)
    env_group_tensors = build_env_group_tensors(
        window_mean=np.zeros((window_states.shape[0], z_dim), dtype=np.float32),
        window_logvar=np.zeros((window_states.shape[0], z_dim), dtype=np.float32),
        env_instance_id=tensors["env_instance_id"],
        env_params=tensors["target_env_params"],
    )
    env_target_env_params = torch.tensor(
        env_group_tensors["env_params"],
        dtype=torch.float32,
        device=device,
    )
    env_param_predictor = EnvParamPredictorEnsemble(
        ensemble_size=ensemble_size,
        input_dim=z_dim,
        output_dim=env_target_env_params.shape[-1],
    ).to(device)
    latent_transition_model = LatentTransitionModel(
        state_dim=window_states.shape[-1],
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    affordance_predictor = OutcomePredictor(z_dim, target_affordances.shape[-1]).to(device)
    decision_predictor = OutcomePredictor(
        encoder.z_ctrl_dim + encoder.z_contact_dim,
        target_decision.shape[-1],
    ).to(device)
    return_predictor = OutcomePredictor(
        encoder.z_dyn_dim + encoder.z_ctrl_dim,
        target_return.shape[-1],
    ).to(device)
    risk_predictor = OutcomePredictor(
        encoder.z_ctrl_dim + encoder.z_contact_dim,
        target_risk.shape[-1],
    ).to(device)
    contrastive_query = ContrastiveProjector(z_dim, output_dim=contrastive_dim).to(device)
    contrastive_key = ContrastiveProjector(
        target_future_summary.shape[-1],
        output_dim=contrastive_dim,
    ).to(device)
    env_projector = ContrastiveProjector(z_dim, output_dim=contrastive_dim).to(device)
    env_belief_projector = ContrastiveProjector(z_dim, output_dim=contrastive_dim).to(device)
    mode_adversary = ProbeModeAdversary(
        input_dim=z_dim,
        num_modes=int(torch.max(probe_mode_idx).item()) + 1,
    ).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(belief_aggregator.parameters())
        + list(env_param_predictor.parameters())
        + list(latent_transition_model.parameters())
        + list(affordance_predictor.parameters())
        + list(decision_predictor.parameters())
        + list(return_predictor.parameters())
        + list(risk_predictor.parameters())
        + list(contrastive_query.parameters())
        + list(contrastive_key.parameters())
        + list(env_projector.parameters())
        + list(env_belief_projector.parameters())
        + list(mode_adversary.parameters()),
        lr=lr,
    )
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    num_windows = window_states.shape[0]

    for epoch in range(epochs):
        permutation = torch.randperm(num_windows, device=device)
        total_loss = 0.0
        total_contrastive_loss = 0.0
        total_env_consistency_loss = 0.0
        total_env_geometry_loss = 0.0
        total_mode_adversary_loss = 0.0
        total_latent_rollout_loss = 0.0
        total_env_param_loss = 0.0
        total_env_split_loss = 0.0
        total_env_split_contrastive_loss = 0.0
        total_uncertainty_calibration_loss = 0.0
        total_env_within_between_loss = 0.0

        for start in range(0, num_windows, batch_size):
            idx = permutation[start:start + batch_size]
            batch_states = window_states[idx]
            batch_actions = window_actions[idx]
            batch_rewards = window_rewards[idx]
            batch_prefix_states = prefix_states[idx]
            batch_prefix_actions = prefix_actions[idx]
            batch_prefix_rewards = prefix_rewards[idx]
            batch_env_instance_id = env_instance_id[idx]
            batch_probe_mode_idx = probe_mode_idx[idx]
            batch_current_state = current_state[idx]
            batch_current_action = current_action[idx]
            batch_target_delta = target_delta[idx]
            batch_target_env_params = target_env_params[idx]
            batch_target_affordances = target_affordances[idx]
            batch_target_decision = target_decision[idx]
            batch_target_return = target_return[idx]
            batch_target_risk = target_risk[idx]
            batch_target_future_summary = target_future_summary[idx]

            mean, logvar = encoder.encode_posterior(
                batch_states,
                batch_actions,
                rewards=batch_rewards,
            )
            step_mean, step_logvar = encoder.encode_step_posteriors(
                batch_states,
                batch_actions,
                rewards=batch_rewards,
            )
            prefix_mean, _prefix_logvar = encoder.encode_posterior(
                batch_prefix_states,
                batch_prefix_actions,
                rewards=batch_prefix_rewards,
            )
            z = encoder.sample_latent(mean, logvar)
            parts = encoder.split_latent(z)
            delta_preds = predictor.predict_all(batch_current_state, batch_current_action, z)
            affordance_pred = affordance_predictor(z)
            decision_pred = decision_predictor(torch.cat([parts["ctrl"], parts["contact"]], dim=1))
            return_pred = return_predictor(torch.cat([parts["dyn"], parts["ctrl"]], dim=1))
            risk_pred = risk_predictor(torch.cat([parts["ctrl"], parts["contact"]], dim=1))

            delta_loss = torch.stack(
                [mse_loss(delta_preds[member_idx], batch_target_delta) for member_idx in range(predictor.ensemble_size)],
                dim=0,
            ).mean()
            affordance_loss = mse_loss(affordance_pred, batch_target_affordances)
            decision_loss = mse_loss(decision_pred, batch_target_decision)
            return_loss = mse_loss(return_pred, batch_target_return)
            risk_loss = bce_loss(risk_pred, batch_target_risk)
            kl_loss = 0.5 * torch.mean(torch.exp(logvar) + mean.pow(2) - 1.0 - logvar)
            contrastive_loss = info_nce_loss(
                contrastive_query(prefix_mean),
                contrastive_key(batch_target_future_summary),
            )
            env_consistency_loss = supervised_same_env_contrastive_loss(
                embeddings=env_projector(mean),
                env_instance_id=batch_env_instance_id,
                probe_mode_idx=batch_probe_mode_idx,
            )
            env_geometry_loss = pairwise_env_geometry_loss(
                latent_mean=mean,
                normalized_env_params=batch_target_env_params,
            )
            mode_adversary_loss = ce_loss(
                mode_adversary(mean, reverse_scale=1.0),
                batch_probe_mode_idx,
            )
            latent_rollout_loss = mean.sum() * 0.0
            if step_mean.shape[1] > 1:
                rollout_input_mean = step_mean[:, :-1, :].reshape(-1, step_mean.shape[-1])
                rollout_state = batch_states[:, 1:-1, :].reshape(-1, batch_states.shape[-1])
                rollout_action = batch_actions[:, 1:].reshape(-1)
                rollout_reward = batch_rewards[:, 1:].reshape(-1)
                target_rollout_mean = step_mean[:, 1:, :].reshape(-1, step_mean.shape[-1])
                target_rollout_logvar = step_logvar[:, 1:, :].reshape(-1, step_logvar.shape[-1])
                pred_rollout_mean, pred_rollout_logvar = latent_transition_model(
                    latent_mean=rollout_input_mean,
                    state=rollout_state,
                    action=rollout_action,
                    reward=rollout_reward,
                )
                latent_rollout_loss = (
                    mse_loss(pred_rollout_mean, target_rollout_mean)
                    + 0.5 * mse_loss(pred_rollout_logvar, target_rollout_logvar)
                )

            loss = (
                delta_loss
                + affordance_loss_weight * affordance_loss
                + decision_loss_weight * decision_loss
                + return_loss_weight * return_loss
                + risk_loss_weight * risk_loss
                + kl_loss_weight * kl_loss
                + contrastive_loss_weight * contrastive_loss
                + env_consistency_loss_weight * env_consistency_loss
                + env_geometry_loss_weight * env_geometry_loss
                + mode_adversary_loss_weight * mode_adversary_loss
                + latent_rollout_loss_weight * latent_rollout_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            total_contrastive_loss += contrastive_loss.item() * len(idx)
            total_env_consistency_loss += env_consistency_loss.item() * len(idx)
            total_env_geometry_loss += env_geometry_loss.item() * len(idx)
            total_mode_adversary_loss += mode_adversary_loss.item() * len(idx)
            total_latent_rollout_loss += latent_rollout_loss.item() * len(idx)

        env_mean_full, env_logvar_full = encoder.encode_posterior(
            window_states,
            window_actions,
            rewards=window_rewards,
        )
        env_group_tensors_torch = group_window_latents_torch(
            window_mean=env_mean_full,
            window_logvar=env_logvar_full,
            env_instance_id=env_instance_id,
            env_params=target_env_params,
        )
        grouped_mean = env_group_tensors_torch["window_mean"]
        grouped_logvar = env_group_tensors_torch["window_logvar"]
        grouped_mask = env_group_tensors_torch["mask"]
        env_target_env_params = env_group_tensors_torch["env_params"]

        env_mean, env_logvar, env_view_spread = belief_aggregator(
            grouped_mean,
            grouped_logvar,
            grouped_mask,
        )
        env_param_preds = env_param_predictor.predict_all(env_mean)
        env_param_mean = env_param_preds.mean(dim=0)
        env_param_std = env_param_preds.std(dim=0)
        env_param_loss = torch.stack(
            [
                mse_loss(env_param_preds[member_idx], env_target_env_params)
                for member_idx in range(env_param_predictor.ensemble_size)
            ],
            dim=0,
        ).mean()

        split_mask_a, split_mask_b = build_env_subset_masks(grouped_mask)
        env_mean_a, env_logvar_a, _env_view_spread_a = belief_aggregator(
            grouped_mean,
            grouped_logvar,
            split_mask_a,
        )
        env_mean_b, env_logvar_b, _env_view_spread_b = belief_aggregator(
            grouped_mean,
            grouped_logvar,
            split_mask_b,
        )
        env_split_loss = (
            mse_loss(env_mean_a, env_mean_b)
            + 0.25 * mse_loss(torch.exp(0.5 * env_logvar_a), torch.exp(0.5 * env_logvar_b))
        )
        env_split_contrastive_loss = info_nce_loss(
            env_belief_projector(env_mean_a),
            env_belief_projector(env_mean_b),
        )
        subset_payload = sample_env_belief_subsets(
            aggregator=belief_aggregator,
            grouped_mean=grouped_mean,
            grouped_logvar=grouped_logvar,
            grouped_mask=grouped_mask,
            env_param_predictor=env_param_predictor,
            subset_count=belief_subset_count,
            subset_size=belief_subset_size,
        )
        subset_env_mean = subset_payload["env_mean"]
        subset_env_param_mean = subset_payload["env_param_mean"]
        subset_env_param_std = subset_env_param_mean.std(dim=1)
        subset_env_latent_std = subset_env_mean.std(dim=1)
        env_subset_consistency_loss = mse_loss(
            subset_env_mean,
            env_mean.unsqueeze(1).expand_as(subset_env_mean),
        )
        env_within_between_loss = within_between_env_loss(
            env_mean=env_mean,
            subset_env_mean=subset_env_mean,
            env_params=env_target_env_params,
        )
        env_geometry_belief_loss = pairwise_env_geometry_loss(
            latent_mean=env_mean,
            normalized_env_params=env_target_env_params,
        )
        env_prediction_error = torch.mean(torch.abs(env_param_mean - env_target_env_params), dim=1)
        uncertainty_calibration_loss = F.l1_loss(
            subset_env_param_std.mean(dim=1) + 0.25 * subset_env_latent_std.mean(dim=1),
            env_prediction_error.detach(),
        )
        env_loss = (
            physics_loss_weight * env_param_loss
            + env_consistency_loss_weight * (env_split_loss + env_subset_consistency_loss)
            + contrastive_loss_weight * env_split_contrastive_loss
            + env_geometry_loss_weight * env_geometry_belief_loss
            + env_within_between_loss_weight * env_within_between_loss
            + 0.10 * uncertainty_calibration_loss
        )
        optimizer.zero_grad()
        env_loss.backward()
        optimizer.step()
        total_env_param_loss = float(env_param_loss.item())
        total_env_split_loss = float(env_split_loss.item())
        total_env_split_contrastive_loss = float(env_split_contrastive_loss.item())
        total_uncertainty_calibration_loss = float(uncertainty_calibration_loss.item())
        total_env_within_between_loss = float(env_within_between_loss.item())

        avg_loss = total_loss / num_windows
        avg_contrastive_loss = total_contrastive_loss / num_windows
        avg_env_consistency_loss = total_env_consistency_loss / num_windows
        avg_env_geometry_loss = total_env_geometry_loss / num_windows
        avg_mode_adversary_loss = total_mode_adversary_loss / num_windows
        avg_latent_rollout_loss = total_latent_rollout_loss / num_windows
        print(
            f"encoder epoch {epoch + 1:02d} | "
            f"total loss = {avg_loss:.6f} | "
            f"contrastive = {avg_contrastive_loss:.6f} | "
            f"same-env = {avg_env_consistency_loss:.6f} | "
            f"geometry = {avg_env_geometry_loss:.6f} | "
            f"mode-adv = {avg_mode_adversary_loss:.6f} | "
            f"rollout = {avg_latent_rollout_loss:.6f} | "
            f"env-param = {total_env_param_loss:.6f} | "
            f"env-split = {total_env_split_loss:.6f} | "
            f"env-contrast = {total_env_split_contrastive_loss:.6f} | "
            f"env-metric = {total_env_within_between_loss:.6f} | "
            f"uncert-cal = {total_uncertainty_calibration_loss:.6f}"
        )

    return encoder, predictor, belief_aggregator, env_param_predictor, device
