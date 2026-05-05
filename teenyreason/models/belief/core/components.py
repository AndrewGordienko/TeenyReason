"""World-model components used by the belief encoder stack."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .common import safe_normalize, sanitize_tensor


class WorldEncoder(nn.Module):
    """Encode a probe trajectory into a posterior over one shared latent."""

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
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FamilyConditionedOutcomePredictor(nn.Module):
    """Predict a future probe summary conditioned on the chosen probe family."""

    def __init__(
        self,
        input_dim: int,
        num_families: int,
        output_dim: int,
        hidden_dim: int = 128,
        family_emb_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_families = int(num_families)
        self.output_dim = int(output_dim)
        self.family_emb = nn.Embedding(self.num_families, family_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + family_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, z: torch.Tensor, family_idx: torch.Tensor) -> torch.Tensor:
        family_idx = torch.clamp(family_idx.long(), min=0, max=self.num_families - 1)
        family_emb = self.family_emb(family_idx)
        return self.net(torch.cat([z, family_emb], dim=-1))


class FamilyConditionedValuePredictor(nn.Module):
    """Predict family-specific probe value terms from the current env belief."""

    def __init__(
        self,
        input_dim: int,
        num_families: int,
        output_dim: int = 3,
        hidden_dim: int = 128,
        family_emb_dim: int = 16,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_families = int(num_families)
        self.output_dim = int(output_dim)
        self.family_emb = nn.Embedding(self.num_families, family_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + family_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor, family_idx: torch.Tensor) -> torch.Tensor:
        family_idx = torch.clamp(family_idx.long(), min=0, max=self.num_families - 1)
        family_emb = self.family_emb(family_idx)
        return sanitize_tensor(torch.nn.functional.softplus(self.net(torch.cat([x, family_emb], dim=-1))))


class BeliefMessageProjector(nn.Module):
    """Project the predictive belief plus uncertainty into a solver-facing message."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, z: torch.Tensor, uncertainty_scalar: torch.Tensor) -> torch.Tensor:
        if uncertainty_scalar.ndim == 1:
            uncertainty_scalar = uncertainty_scalar.unsqueeze(-1)
        return sanitize_tensor(self.net(torch.cat([z, uncertainty_scalar], dim=-1)))


class ControllerBeliefContextProjector(nn.Module):
    """Project one predictive belief into structured mechanics and affordance codes."""

    def __init__(
        self,
        input_dim: int,
        mechanics_dim: int,
        affordance_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.mechanics_dim = int(mechanics_dim)
        self.affordance_dim = int(affordance_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mechanics_head = nn.Linear(hidden_dim, self.mechanics_dim)
        self.affordance_head = nn.Linear(hidden_dim, self.affordance_dim)

    @property
    def output_dim(self) -> int:
        return int(self.mechanics_dim + self.affordance_dim)

    def forward(
        self,
        z: torch.Tensor,
        uncertainty_scalar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if uncertainty_scalar.ndim == 1:
            uncertainty_scalar = uncertainty_scalar.unsqueeze(-1)
        features = sanitize_tensor(self.net(torch.cat([z, uncertainty_scalar], dim=-1)))
        mechanics_code = sanitize_tensor(self.mechanics_head(features))
        affordance_code = sanitize_tensor(self.affordance_head(features))
        return mechanics_code, affordance_code


class OracleControllerContextProjector(nn.Module):
    """Project normalized true env parameters into controller-context codes."""

    def __init__(
        self,
        input_dim: int,
        mechanics_dim: int,
        affordance_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.mechanics_dim = int(mechanics_dim)
        self.affordance_dim = int(affordance_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mechanics_head = nn.Linear(hidden_dim, self.mechanics_dim)
        self.affordance_head = nn.Linear(hidden_dim, self.affordance_dim)

    @property
    def output_dim(self) -> int:
        return int(self.mechanics_dim + self.affordance_dim)

    def forward(self, env_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = sanitize_tensor(self.net(env_params))
        mechanics_code = sanitize_tensor(self.mechanics_head(features))
        affordance_code = sanitize_tensor(self.affordance_head(features))
        return mechanics_code, affordance_code


class ControllerTrustPredictor(nn.Module):
    """Predict how much the full-system controller should trust the context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        uncertainty_scalar: torch.Tensor,
    ) -> torch.Tensor:
        if uncertainty_scalar.ndim == 1:
            uncertainty_scalar = uncertainty_scalar.unsqueeze(-1)
        trust_logit = sanitize_tensor(self.net(torch.cat([z, uncertainty_scalar], dim=-1)))
        return sanitize_tensor(torch.sigmoid(trust_logit))


class ContrastiveProjector(nn.Module):
    """Small normalized projection head for contrastive latent supervision."""

    def __init__(self, input_dim: int, output_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def project_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unnormalized projection before cosine-style comparisons."""
        return sanitize_tensor(torch.tanh(self.net(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return safe_normalize(self.project_raw(x), dim=-1)


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
