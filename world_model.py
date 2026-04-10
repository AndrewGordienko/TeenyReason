import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

TAU = 0.02
X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12 * 2 * np.pi / 360


class WorldEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int = 4,
        window_size: int = 8,
        action_vocab_size: int = 2,
        z_dim: int = 8,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(action_vocab_size, 4)
        input_dim = (window_size + 1) * state_dim + window_size * 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        action_emb = self.action_emb(actions)
        x = torch.cat(
            [states.reshape(batch_size, -1), action_emb.reshape(batch_size, -1)],
            dim=1,
        )
        return self.net(x)


class DeltaPredictor(nn.Module):
    def __init__(
        self,
        state_dim: int = 4,
        action_vocab_size: int = 2,
        z_dim: int = 8,
    ):
        super().__init__()
        self.action_emb = nn.Embedding(action_vocab_size, 4)

        self.net = nn.Sequential(
            nn.Linear(state_dim + 4 + z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        action_emb = self.action_emb(action)
        x = torch.cat([state, action_emb, z], dim=1)
        return self.net(x)


class PhysicsPredictor(nn.Module):
    def __init__(self, z_dim: int, param_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, param_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AffordancePredictor(nn.Module):
    def __init__(self, z_dim: int, affordance_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, affordance_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_action_controls(action_vocab_size: int) -> np.ndarray:
    if action_vocab_size <= 1:
        return np.asarray([0.0], dtype=np.float32)
    if action_vocab_size == 2:
        return np.asarray([-1.0, 1.0], dtype=np.float32)
    return np.linspace(-1.0, 1.0, action_vocab_size, dtype=np.float32)


def key_action_indices(action_vocab_size: int) -> dict[str, int]:
    controls = build_action_controls(action_vocab_size)
    center_idx = int(np.argmin(np.abs(controls)))
    offset = max(1, (action_vocab_size - 1) // 4)
    return {
        "left": 0,
        "right": action_vocab_size - 1,
        "center": center_idx,
        "small_left": max(0, center_idx - offset),
        "small_right": min(action_vocab_size - 1, center_idx + offset),
    }


def build_program_action_indices(action_vocab_size: int, horizon: int) -> np.ndarray:
    action_idx = key_action_indices(action_vocab_size)

    programs = [
        np.full(horizon, action_idx["left"], dtype=np.int64),
        np.full(horizon, action_idx["right"], dtype=np.int64),
        np.full(horizon, action_idx["center"], dtype=np.int64),
        np.asarray(
            [
                action_idx["small_left"] if step < 2 else action_idx["center"]
                for step in range(horizon)
            ],
            dtype=np.int64,
        ),
        np.asarray(
            [
                action_idx["small_right"] if step < 2 else action_idx["center"]
                for step in range(horizon)
            ],
            dtype=np.int64,
        ),
        np.asarray(
            [
                action_idx["small_left"] if step < 2 else action_idx["small_right"]
                if step < 4 else action_idx["center"]
                for step in range(horizon)
            ],
            dtype=np.int64,
        ),
        np.asarray(
            [
                (
                    action_idx["small_left"],
                    action_idx["center"],
                    action_idx["small_right"],
                    action_idx["center"],
                )[step % 4]
                for step in range(horizon)
            ],
            dtype=np.int64,
        ),
    ]
    return np.stack(programs, axis=0)


def step_cartpole_dynamics(state: np.ndarray, force: float, env_params: np.ndarray) -> np.ndarray:
    gravity, masscart, masspole, length, _force_mag = env_params
    x, x_dot, theta, theta_dot = state
    total_mass = masspole + masscart
    polemass_length = masspole * length
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    return np.asarray(
        [
            x + TAU * x_dot,
            x_dot + TAU * xacc,
            theta + TAU * theta_dot,
            theta_dot + TAU * thetaacc,
        ],
        dtype=np.float32,
    )


def stability_margin(state: np.ndarray) -> float:
    x_margin = (X_THRESHOLD - abs(float(state[0]))) / X_THRESHOLD
    theta_margin = (THETA_THRESHOLD_RADIANS - abs(float(state[2]))) / THETA_THRESHOLD_RADIANS
    return float(min(x_margin, theta_margin))


def simulate_affordance_program(
    initial_state: np.ndarray,
    env_params: np.ndarray,
    action_indices: np.ndarray,
    action_controls: np.ndarray,
) -> np.ndarray:
    state = np.asarray(initial_state, dtype=np.float32).copy()
    force_mag = float(env_params[4])
    initial_theta = abs(float(initial_state[2]))
    initial_theta_dot = abs(float(initial_state[3]))
    min_margin = stability_margin(state)
    min_abs_theta = initial_theta
    peak_abs_theta = initial_theta
    peak_abs_x = abs(float(initial_state[0]))
    initial_theta_sign = float(np.sign(float(initial_state[2])))
    theta_reversal = 0.0
    steps_survived = 0

    for action_idx in action_indices:
        control = float(action_controls[int(action_idx)])
        force = force_mag * control
        state = step_cartpole_dynamics(state, force, env_params)
        abs_theta = abs(float(state[2]))
        peak_abs_theta = max(peak_abs_theta, abs_theta)
        peak_abs_x = max(peak_abs_x, abs(float(state[0])))
        min_abs_theta = min(min_abs_theta, abs_theta)
        min_margin = min(min_margin, stability_margin(state))
        steps_survived += 1
        theta_sign = float(np.sign(float(state[2])))
        if initial_theta_sign != 0.0 and theta_sign != 0.0 and theta_sign != initial_theta_sign:
            theta_reversal = 1.0
        if abs(float(state[0])) > X_THRESHOLD or abs(float(state[2])) > THETA_THRESHOLD_RADIANS:
            break

    final_theta = abs(float(state[2]))
    final_theta_dot = abs(float(state[3]))
    survival_fraction = steps_survived / len(action_indices)
    recovery_gain = initial_theta - final_theta
    damping_gain = initial_theta_dot - final_theta_dot
    peak_theta_ratio = peak_abs_theta / THETA_THRESHOLD_RADIANS
    peak_x_ratio = peak_abs_x / X_THRESHOLD
    recovered_flag = 1.0 if min_abs_theta <= max(0.02, 0.5 * max(initial_theta, 0.02)) else 0.0

    return np.concatenate(
        [
            state - initial_state,
            np.asarray(
                [
                    survival_fraction,
                    min_margin,
                    peak_theta_ratio,
                    peak_x_ratio,
                    recovery_gain,
                    damping_gain,
                    theta_reversal,
                    recovered_flag,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def summarize_hold_affordances(
    initial_state: np.ndarray,
    env_params: np.ndarray,
    action_controls: np.ndarray,
    intervention_horizon: int,
) -> np.ndarray:
    hold_features = np.stack(
        [
            simulate_affordance_program(
                initial_state=initial_state,
                env_params=env_params,
                action_indices=np.full(intervention_horizon, action_idx, dtype=np.int64),
                action_controls=action_controls,
            )
            for action_idx in range(len(action_controls))
        ],
        axis=0,
    )
    action_idx = key_action_indices(len(action_controls))
    center_features = hold_features[action_idx["center"]]

    survival_idx = 4
    min_margin_idx = 5
    recovery_idx = 8
    damping_idx = 9
    recovered_idx = 11
    theta_delta_idx = 2

    recovered_mask = hold_features[:, recovered_idx] > 0.5
    if np.any(recovered_mask):
        min_recovery_strength = float(
            np.min(np.abs(action_controls[recovered_mask]))
        )
    else:
        min_recovery_strength = 1.0

    controllability_span = float(
        np.max(hold_features[:, theta_delta_idx]) - np.min(hold_features[:, theta_delta_idx])
    )

    return np.asarray(
        [
            center_features[survival_idx],
            center_features[min_margin_idx],
            np.max(hold_features[:, survival_idx]),
            np.max(hold_features[:, recovery_idx]),
            np.max(hold_features[:, damping_idx]),
            min_recovery_strength,
            controllability_span,
        ],
        dtype=np.float32,
    )


def build_affordance_targets(
    current_states: np.ndarray,
    env_params: np.ndarray,
    action_vocab_size: int,
    intervention_horizon: int,
) -> np.ndarray:
    action_controls = build_action_controls(action_vocab_size)
    programs = build_program_action_indices(action_vocab_size, intervention_horizon)
    summary_dim = 7
    feature_dim = programs.shape[0] * 12 + summary_dim
    targets = np.zeros((current_states.shape[0], feature_dim), dtype=np.float32)

    for row_idx in range(current_states.shape[0]):
        program_features = []
        for program in programs:
            program_features.append(
                simulate_affordance_program(
                    initial_state=current_states[row_idx],
                    env_params=env_params[row_idx],
                    action_indices=program,
                    action_controls=action_controls,
                )
            )
        behavior_summary = summarize_hold_affordances(
            initial_state=current_states[row_idx],
            env_params=env_params[row_idx],
            action_controls=action_controls,
            intervention_horizon=intervention_horizon,
        )
        targets[row_idx] = np.concatenate(program_features + [behavior_summary], axis=0)

    return targets


def build_generic_affordance_targets(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    action_vocab_size: int,
) -> np.ndarray:
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


def normalize_targets(values: np.ndarray) -> np.ndarray:
    value_mean = values.mean(axis=0, keepdims=True).astype(np.float32)
    value_std = values.std(axis=0, keepdims=True).astype(np.float32)
    value_std = np.where(value_std < 1e-6, 1.0, value_std)
    return ((values - value_mean) / value_std).astype(np.float32)


def build_training_tensors(
    windows: dict[str, np.ndarray],
    action_vocab_size: int,
    intervention_horizon: int,
    analytic_affordances: bool = True,
) -> dict[str, np.ndarray]:
    states = windows["states"]
    actions = windows["actions"]
    env_params = windows["env_params"]

    current_state = states[:, -2, :]
    next_state = states[:, -1, :]
    delta_state = next_state - current_state
    current_action = actions[:, -1]

    normalized_env_params = normalize_targets(env_params)
    if analytic_affordances:
        target_affordances = build_affordance_targets(
            current_states=current_state,
            env_params=env_params,
            action_vocab_size=action_vocab_size,
            intervention_horizon=intervention_horizon,
        )
    else:
        target_affordances = build_generic_affordance_targets(
            states=states,
            actions=actions,
            rewards=windows["rewards"],
            terminated=windows["terminated"],
            truncated=windows["truncated"],
            action_vocab_size=action_vocab_size,
        )
    normalized_affordances = normalize_targets(target_affordances)

    return {
        "window_states": states.astype(np.float32),
        "window_actions": actions.astype(np.int64),
        "current_state": current_state.astype(np.float32),
        "current_action": current_action.astype(np.int64),
        "target_delta": delta_state.astype(np.float32),
        "target_env_params": normalized_env_params.astype(np.float32),
        "target_affordances": normalized_affordances.astype(np.float32),
    }


def train_encoder_predictor(
    windows: dict[str, np.ndarray],
    z_dim: int = 8,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    physics_loss_weight: float = 0.1,
    affordance_loss_weight: float = 1.0,
    action_vocab_size: int | None = None,
    intervention_horizon: int = 12,
    analytic_affordances: bool = True,
) -> tuple[WorldEncoder, DeltaPredictor, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if action_vocab_size is None:
        action_vocab_size = int(np.max(windows["actions"])) + 1
    tensors = build_training_tensors(
        windows,
        action_vocab_size=action_vocab_size,
        intervention_horizon=intervention_horizon,
        analytic_affordances=analytic_affordances,
    )

    window_states = torch.tensor(tensors["window_states"], dtype=torch.float32, device=device)
    window_actions = torch.tensor(tensors["window_actions"], dtype=torch.long, device=device)
    current_state = torch.tensor(tensors["current_state"], dtype=torch.float32, device=device)
    current_action = torch.tensor(tensors["current_action"], dtype=torch.long, device=device)
    target_delta = torch.tensor(tensors["target_delta"], dtype=torch.float32, device=device)
    target_env_params = torch.tensor(
        tensors["target_env_params"],
        dtype=torch.float32,
        device=device,
    )
    target_affordances = torch.tensor(
        tensors["target_affordances"],
        dtype=torch.float32,
        device=device,
    )

    encoder = WorldEncoder(
        state_dim=window_states.shape[-1],
        window_size=window_actions.shape[1],
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)

    predictor = DeltaPredictor(
        state_dim=window_states.shape[-1],
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    physics_predictor = PhysicsPredictor(
        z_dim=z_dim,
        param_dim=target_env_params.shape[-1],
    ).to(device)
    affordance_predictor = AffordancePredictor(
        z_dim=z_dim,
        affordance_dim=target_affordances.shape[-1],
    ).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(physics_predictor.parameters())
        + list(affordance_predictor.parameters()),
        lr=lr,
    )
    loss_fn = nn.MSELoss()

    num_windows = window_states.shape[0]

    for epoch in range(epochs):
        permutation = torch.randperm(num_windows, device=device)
        total_delta_loss = 0.0
        total_physics_loss = 0.0
        total_affordance_loss = 0.0

        for start in range(0, num_windows, batch_size):
            idx = permutation[start:start + batch_size]

            batch_window_states = window_states[idx]
            batch_window_actions = window_actions[idx]
            batch_current_state = current_state[idx]
            batch_current_action = current_action[idx]
            batch_target_delta = target_delta[idx]
            batch_target_env_params = target_env_params[idx]
            batch_target_affordances = target_affordances[idx]

            z = encoder(batch_window_states, batch_window_actions)
            pred_delta = predictor(batch_current_state, batch_current_action, z)
            pred_env_params = physics_predictor(z)
            pred_affordances = affordance_predictor(z)

            delta_loss = loss_fn(pred_delta, batch_target_delta)
            physics_loss = loss_fn(pred_env_params, batch_target_env_params)
            affordance_loss = loss_fn(pred_affordances, batch_target_affordances)
            loss = (
                delta_loss
                + physics_loss_weight * physics_loss
                + affordance_loss_weight * affordance_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_delta_loss += delta_loss.item() * len(idx)
            total_physics_loss += physics_loss.item() * len(idx)
            total_affordance_loss += affordance_loss.item() * len(idx)

        avg_delta_loss = total_delta_loss / num_windows
        avg_physics_loss = total_physics_loss / num_windows
        avg_affordance_loss = total_affordance_loss / num_windows
        print(
            f"encoder epoch {epoch + 1:02d} | "
            f"delta loss = {avg_delta_loss:.6f} | "
            f"physics loss = {avg_physics_loss:.6f} | "
            f"affordance loss = {avg_affordance_loss:.6f}"
        )

    return encoder, predictor, device
