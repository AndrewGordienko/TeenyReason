"""Recurrent world-model and structured latent supervision for probe-conditioned RL."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..envs import BIPEDAL_WALKER_NAME


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
        self.action_emb = nn.Embedding(action_vocab_size, 8)
        token_dim = state_dim * 2 + 8 + 1
        self.token_net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.mean_head = nn.Linear(hidden_dim, z_dim)
        self.logvar_head = nn.Linear(hidden_dim, z_dim)
        self.z_dyn_dim, self.z_ctrl_dim, self.z_contact_dim = split_latent_dims(z_dim)

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
        return self.token_net(token)

    def encode_posterior(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.build_tokens(states, actions, rewards=rewards)
        _outputs, hidden = self.gru(tokens)
        final_hidden = hidden[-1]
        mean = self.mean_head(final_hidden)
        logvar = torch.clamp(self.logvar_head(final_hidden), -5.0, 2.0)
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
    terminated = windows["terminated"]
    truncated = windows["truncated"]

    current_state = states[:, -2, :]
    next_state = states[:, -1, :]
    delta_state = next_state - current_state
    current_action = actions[:, -1]
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

    return {
        "window_states": states.astype(np.float32),
        "window_actions": actions.astype(np.int64),
        "window_rewards": rewards.astype(np.float32),
        "current_state": current_state.astype(np.float32),
        "current_action": current_action.astype(np.int64),
        "target_delta": delta_state.astype(np.float32),
        "target_env_params": normalize_targets(env_params).astype(np.float32),
        "target_affordances": normalize_targets(target_affordances).astype(np.float32),
        "target_decision": normalize_targets(decision_targets).astype(np.float32),
        "target_return": normalize_targets(return_target).astype(np.float32),
        "target_risk": risk_target.astype(np.float32),
    }


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
    ensemble_size: int = 3,
    action_vocab_size: int | None = None,
    intervention_horizon: int = 12,
    analytic_affordances: bool = True,
    env_name: str | None = None,
) -> tuple[WorldEncoder, DeltaPredictorEnsemble, torch.device]:
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
    current_state = torch.tensor(tensors["current_state"], dtype=torch.float32, device=device)
    current_action = torch.tensor(tensors["current_action"], dtype=torch.long, device=device)
    target_delta = torch.tensor(tensors["target_delta"], dtype=torch.float32, device=device)
    target_env_params = torch.tensor(tensors["target_env_params"], dtype=torch.float32, device=device)
    target_affordances = torch.tensor(tensors["target_affordances"], dtype=torch.float32, device=device)
    target_decision = torch.tensor(tensors["target_decision"], dtype=torch.float32, device=device)
    target_return = torch.tensor(tensors["target_return"], dtype=torch.float32, device=device)
    target_risk = torch.tensor(tensors["target_risk"], dtype=torch.float32, device=device)

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
    physics_predictor = OutcomePredictor(encoder.z_dyn_dim, target_env_params.shape[-1]).to(device)
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

    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(physics_predictor.parameters())
        + list(affordance_predictor.parameters())
        + list(decision_predictor.parameters())
        + list(return_predictor.parameters())
        + list(risk_predictor.parameters()),
        lr=lr,
    )
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    num_windows = window_states.shape[0]

    for epoch in range(epochs):
        permutation = torch.randperm(num_windows, device=device)
        total_loss = 0.0

        for start in range(0, num_windows, batch_size):
            idx = permutation[start:start + batch_size]
            batch_states = window_states[idx]
            batch_actions = window_actions[idx]
            batch_rewards = window_rewards[idx]
            batch_current_state = current_state[idx]
            batch_current_action = current_action[idx]
            batch_target_delta = target_delta[idx]
            batch_target_env_params = target_env_params[idx]
            batch_target_affordances = target_affordances[idx]
            batch_target_decision = target_decision[idx]
            batch_target_return = target_return[idx]
            batch_target_risk = target_risk[idx]

            mean, logvar = encoder.encode_posterior(
                batch_states,
                batch_actions,
                rewards=batch_rewards,
            )
            z = encoder.sample_latent(mean, logvar)
            parts = encoder.split_latent(z)
            delta_preds = predictor.predict_all(batch_current_state, batch_current_action, z)
            env_pred = physics_predictor(parts["dyn"])
            affordance_pred = affordance_predictor(z)
            decision_pred = decision_predictor(torch.cat([parts["ctrl"], parts["contact"]], dim=1))
            return_pred = return_predictor(torch.cat([parts["dyn"], parts["ctrl"]], dim=1))
            risk_pred = risk_predictor(torch.cat([parts["ctrl"], parts["contact"]], dim=1))

            delta_loss = torch.stack(
                [mse_loss(delta_preds[member_idx], batch_target_delta) for member_idx in range(predictor.ensemble_size)],
                dim=0,
            ).mean()
            physics_loss = mse_loss(env_pred, batch_target_env_params)
            affordance_loss = mse_loss(affordance_pred, batch_target_affordances)
            decision_loss = mse_loss(decision_pred, batch_target_decision)
            return_loss = mse_loss(return_pred, batch_target_return)
            risk_loss = bce_loss(risk_pred, batch_target_risk)
            kl_loss = 0.5 * torch.mean(torch.exp(logvar) + mean.pow(2) - 1.0 - logvar)

            loss = (
                delta_loss
                + physics_loss_weight * physics_loss
                + affordance_loss_weight * affordance_loss
                + decision_loss_weight * decision_loss
                + return_loss_weight * return_loss
                + risk_loss_weight * risk_loss
                + kl_loss_weight * kl_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        avg_loss = total_loss / num_windows
        print(f"encoder epoch {epoch + 1:02d} | total loss = {avg_loss:.6f}")

    return encoder, predictor, device
