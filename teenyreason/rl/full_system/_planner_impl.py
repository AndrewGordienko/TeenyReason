"""Belief-driven planning helpers for the full-system breakthrough path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ...crawler import CrawlerModelBundle
from ...probe.probe_latent import aggregate_env_belief
from ._objectives_impl import rollout_supervision_and_alignment
from ..core.ppo_core import (
    BeliefNativeActorCritic,
    action_scale_bias,
    sanitize_numpy,
    sanitize_tensor,
)


@dataclass
class PlanningBeliefState:
    """Planner-facing belief plus recurrent controller cache."""

    context: np.ndarray
    recurrent_hidden: np.ndarray | None
    context_age_steps: int
    last_refresh_step: int


@dataclass(frozen=True)
class BeliefPlannerConfig:
    """Fixed explicit defaults for the CartPole-first planner."""

    horizon: int = 8
    candidates: int = 64
    elites: int = 8
    iterations: int = 3
    planner_only_trust_floor: float = 0.35
    actor_only_trust_floor: float = 0.15
    reward_weight: float = 1.0
    recoverability_weight: float = 0.5
    termination_weight: float = 2.0
    disagreement_weight: float = 0.5
    uncertainty_growth_weight: float = 0.25
    gamma: float = 0.99
    sequence_semantic_horizon: int = 4


class BeliefDynamicsHead(nn.Module):
    """One ensemble member for action-conditioned belief dynamics."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        mechanics_dim: int,
        affordance_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        input_dim = int(state_dim + action_dim + mechanics_dim + affordance_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.delta_head = nn.Linear(hidden_dim, int(state_dim))
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.term_head = nn.Linear(hidden_dim, 1)
        self.recoverability_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        mechanics_code: torch.Tensor,
        affordance_code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = sanitize_tensor(
            self.net(
                torch.cat(
                    [
                        sanitize_tensor(state),
                        sanitize_tensor(action),
                        sanitize_tensor(mechanics_code),
                        sanitize_tensor(affordance_code),
                    ],
                    dim=-1,
                )
            )
        )
        return (
            sanitize_tensor(self.delta_head(features)),
            sanitize_tensor(self.reward_head(features).squeeze(-1)),
            sanitize_tensor(self.term_head(features).squeeze(-1)),
            sanitize_tensor(
                torch.sigmoid(self.recoverability_head(features).squeeze(-1))
            ),
        )


class BeliefDynamicsModel(nn.Module):
    """Small explicit ensemble used by the full-system planner."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        mechanics_dim: int,
        affordance_dim: int,
        hidden_dim: int = 128,
        ensemble_size: int = 5,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.mechanics_dim = int(mechanics_dim)
        self.affordance_dim = int(affordance_dim)
        self.ensemble_size = int(ensemble_size)
        self.heads = nn.ModuleList(
            [
                BeliefDynamicsHead(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    mechanics_dim=mechanics_dim,
                    affordance_dim=affordance_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(self.ensemble_size)
            ]
        )

    def split_context(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mechanics_end = self.mechanics_dim
        affordance_end = mechanics_end + self.affordance_dim
        mechanics = sanitize_tensor(context[..., :mechanics_end])
        affordance = sanitize_tensor(context[..., mechanics_end:affordance_end])
        confidence = sanitize_tensor(context[..., affordance_end : affordance_end + 1])
        uncertainty = sanitize_tensor(context[..., affordance_end + 1 : affordance_end + 2])
        return mechanics, affordance, confidence, uncertainty

    def predict_all(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mechanics, affordance, _confidence, _uncertainty = self.split_context(context)
        outputs = [
            head(state, action, mechanics, affordance)
            for head in self.heads
        ]
        delta = torch.stack([row[0] for row in outputs], dim=0)
        reward = torch.stack([row[1] for row in outputs], dim=0)
        term = torch.stack([row[2] for row in outputs], dim=0)
        recoverability = torch.stack([row[3] for row in outputs], dim=0)
        return (
            sanitize_tensor(delta),
            sanitize_tensor(reward),
            sanitize_tensor(term),
            sanitize_tensor(recoverability),
        )

    def predict_summary(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta_all, reward_all, term_all, recover_all = self.predict_all(
            state,
            action,
            context,
        )
        disagreement = torch.mean(torch.std(delta_all, dim=0), dim=-1)
        disagreement = disagreement + 0.25 * torch.std(reward_all, dim=0)
        disagreement = disagreement + 0.25 * torch.std(torch.sigmoid(term_all), dim=0)
        return {
            "next_state_delta": sanitize_tensor(delta_all.mean(dim=0)),
            "reward": sanitize_tensor(reward_all.mean(dim=0)),
            "term_logit": sanitize_tensor(term_all.mean(dim=0)),
            "recoverability": sanitize_tensor(recover_all.mean(dim=0)),
            "disagreement": sanitize_tensor(disagreement),
        }


def _normalized_actions_from_bins(
    actions: np.ndarray,
    *,
    action_vocab_size: int,
) -> np.ndarray:
    if action_vocab_size <= 1:
        return np.zeros(actions.shape + (1,), dtype=np.float32)
    normalized = (2.0 * actions.astype(np.float32) / float(action_vocab_size - 1)) - 1.0
    return normalized[..., None].astype(np.float32)


def _cartpole_recoverability(states: np.ndarray) -> np.ndarray:
    if states.shape[-1] < 4:
        return np.ones((states.shape[0],), dtype=np.float32)
    x = np.abs(states[:, 0])
    dx = np.abs(states[:, 1])
    theta = np.abs(states[:, 2])
    dtheta = np.abs(states[:, 3])
    centered = 1.0 - np.clip(x / 2.4, 0.0, 1.0)
    upright = 1.0 - np.clip(theta / 0.35, 0.0, 1.0)
    calm = 1.0 / (1.0 + dx + dtheta)
    return np.clip(0.45 * upright + 0.35 * centered + 0.20 * calm, 0.0, 1.0).astype(np.float32)


def build_planner_probe_dataset(
    *,
    windows: dict[str, np.ndarray],
    crawler_bundle: CrawlerModelBundle,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    sequence_horizon: int = 4,
) -> dict[str, np.ndarray]:
    """Build one explicit transition dataset for planner-model pretraining."""
    states = np.asarray(windows["states"], dtype=np.float32)
    actions = np.asarray(windows["actions"], dtype=np.int64)
    rewards = np.asarray(windows["rewards"], dtype=np.float32)
    terminated = np.asarray(windows["terminated"], dtype=bool)
    truncated = np.asarray(windows["truncated"], dtype=bool)
    env_params = np.asarray(windows["env_params"], dtype=np.float32)
    env_ids = np.asarray(windows["env_instance_id"], dtype=np.int32)
    if state_mean is not None and state_std is not None:
        state_scale = np.maximum(np.asarray(state_std, dtype=np.float32), 1e-6)
        normalized_states = ((states - np.asarray(state_mean, dtype=np.float32)) / state_scale).astype(np.float32)
    else:
        normalized_states = states.astype(np.float32)
    action_inputs = _normalized_actions_from_bins(
        actions,
        action_vocab_size=int(crawler_bundle.action_vocab_size),
    )

    learned_by_env: dict[int, np.ndarray] = {}
    oracle_by_env: dict[int, np.ndarray] = {}
    unique_env_ids = np.unique(env_ids)
    for env_id in unique_env_ids.tolist():
        env_mask = env_ids == int(env_id)
        posterior_views = [
            crawler_bundle.encode_probe_window(
                states[idx],
                actions[idx],
                rewards[idx],
            )
            for idx in np.where(env_mask)[0].tolist()
        ]
        _belief, payload = aggregate_env_belief(
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            device=crawler_bundle.device,
            posterior_views=posterior_views,
        )
        step_result = crawler_bundle.build_step_result(
            payload=payload,
            expected_family_gain={},
            realized_family_gain={},
            stop_reason=None,
            bits_per_dim=0,
            use_residual_sketch=False,
        )
        learned_by_env[int(env_id)] = sanitize_numpy(step_result.controller_context.vector)
        oracle_by_env[int(env_id)] = sanitize_numpy(
            crawler_bundle.build_oracle_controller_context(env_params[env_mask][0]).vector
        )

    sample_states = []
    sample_actions = []
    sample_delta = []
    sample_rewards = []
    sample_term = []
    sample_recoverability = []
    sample_learned = []
    sample_oracle = []
    sequence_initial_state = []
    sequence_actions = []
    sequence_next_states = []
    sequence_rewards = []
    sequence_terminals = []
    sequence_recoverability = []
    sequence_learned = []
    sequence_oracle = []
    rollout_horizon = max(2, min(int(sequence_horizon), int(actions.shape[1])))
    for row_idx in range(states.shape[0]):
        env_id = int(env_ids[row_idx])
        for step_idx in range(actions.shape[1]):
            sample_states.append(normalized_states[row_idx, step_idx])
            sample_actions.append(action_inputs[row_idx, step_idx])
            next_state = normalized_states[row_idx, step_idx + 1]
            sample_delta.append(next_state - normalized_states[row_idx, step_idx])
            sample_rewards.append(rewards[row_idx, step_idx])
            is_terminal = (
                step_idx == actions.shape[1] - 1
                and (bool(terminated[row_idx]) or bool(truncated[row_idx]))
            )
            sample_term.append(float(is_terminal))
            sample_recoverability.append(_cartpole_recoverability(states[row_idx, step_idx + 1][None, :])[0])
            sample_learned.append(learned_by_env[env_id])
            sample_oracle.append(oracle_by_env[env_id])
        max_start = max(1, actions.shape[1] - rollout_horizon + 1)
        for start_idx in range(max_start):
            end_idx = start_idx + rollout_horizon
            sequence_initial_state.append(normalized_states[row_idx, start_idx])
            sequence_actions.append(action_inputs[row_idx, start_idx:end_idx])
            sequence_next_states.append(normalized_states[row_idx, start_idx + 1 : end_idx + 1])
            sequence_rewards.append(rewards[row_idx, start_idx:end_idx].astype(np.float32))
            term_seq = np.zeros((rollout_horizon,), dtype=np.float32)
            if bool(terminated[row_idx]) or bool(truncated[row_idx]):
                final_transition = actions.shape[1] - 1
                if start_idx <= final_transition < end_idx:
                    term_seq[final_transition - start_idx] = 1.0
            sequence_terminals.append(term_seq)
            sequence_recoverability.append(
                _cartpole_recoverability(states[row_idx, start_idx + 1 : end_idx + 1])
            )
            sequence_learned.append(learned_by_env[env_id])
            sequence_oracle.append(oracle_by_env[env_id])
    return {
        "states": np.asarray(sample_states, dtype=np.float32),
        "actions": np.asarray(sample_actions, dtype=np.float32),
        "target_delta": np.asarray(sample_delta, dtype=np.float32),
        "target_reward": np.asarray(sample_rewards, dtype=np.float32),
        "target_term": np.asarray(sample_term, dtype=np.float32),
        "target_recoverability": np.asarray(sample_recoverability, dtype=np.float32),
        "learned_context": np.asarray(sample_learned, dtype=np.float32),
        "oracle_context": np.asarray(sample_oracle, dtype=np.float32),
        "sequence_initial_state": np.asarray(sequence_initial_state, dtype=np.float32),
        "sequence_actions": np.asarray(sequence_actions, dtype=np.float32),
        "sequence_next_states": np.asarray(sequence_next_states, dtype=np.float32),
        "sequence_rewards": np.asarray(sequence_rewards, dtype=np.float32),
        "sequence_terminals": np.asarray(sequence_terminals, dtype=np.float32),
        "sequence_recoverability": np.asarray(sequence_recoverability, dtype=np.float32),
        "sequence_learned_context": np.asarray(sequence_learned, dtype=np.float32),
        "sequence_oracle_context": np.asarray(sequence_oracle, dtype=np.float32),
    }


def fit_belief_dynamics_model(
    *,
    model: BeliefDynamicsModel,
    optimizer: optim.Optimizer,
    dataset: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 256,
    epochs: int = 12,
    config: BeliefPlannerConfig | None = None,
) -> dict[str, float]:
    """Pretrain the planner dynamics on probe windows and semantic score agreement."""
    config = BeliefPlannerConfig() if config is None else config
    states = torch.tensor(dataset["states"], dtype=torch.float32, device=device)
    actions = torch.tensor(dataset["actions"], dtype=torch.float32, device=device)
    target_delta = torch.tensor(dataset["target_delta"], dtype=torch.float32, device=device)
    target_reward = torch.tensor(dataset["target_reward"], dtype=torch.float32, device=device)
    target_term = torch.tensor(dataset["target_term"], dtype=torch.float32, device=device)
    target_recover = torch.tensor(dataset["target_recoverability"], dtype=torch.float32, device=device)
    learned_context = torch.tensor(dataset["learned_context"], dtype=torch.float32, device=device)
    oracle_context = torch.tensor(dataset["oracle_context"], dtype=torch.float32, device=device)
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_score_align = 0.0
    total_rollout_loss = 0.0
    num_rows = int(states.shape[0])
    sequence_initial = torch.tensor(
        dataset.get("sequence_initial_state", np.zeros((0, states.shape[-1]), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    sequence_actions = torch.tensor(
        dataset.get("sequence_actions", np.zeros((0, config.sequence_semantic_horizon, actions.shape[-1]), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    sequence_next_states = torch.tensor(
        dataset.get("sequence_next_states", np.zeros((0, config.sequence_semantic_horizon, states.shape[-1]), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    sequence_rewards = torch.tensor(
        dataset.get("sequence_rewards", np.zeros((0, config.sequence_semantic_horizon), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    sequence_terms = torch.tensor(
        dataset.get("sequence_terminals", np.zeros((0, config.sequence_semantic_horizon), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    sequence_recover = torch.tensor(
        dataset.get("sequence_recoverability", np.zeros((0, config.sequence_semantic_horizon), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    sequence_learned = torch.tensor(
        dataset.get("sequence_learned_context", np.zeros((0, learned_context.shape[-1]), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    sequence_oracle = torch.tensor(
        dataset.get("sequence_oracle_context", np.zeros((0, oracle_context.shape[-1]), dtype=np.float32)),
        dtype=torch.float32,
        device=device,
    )
    num_sequences = int(sequence_initial.shape[0])

    for _epoch in range(max(1, int(epochs))):
        permutation = torch.randperm(num_rows, device=device)
        for start in range(0, num_rows, max(1, int(batch_size))):
            idx = permutation[start:start + max(1, int(batch_size))]
            batch_state = states[idx]
            batch_action = actions[idx]
            batch_delta = target_delta[idx]
            batch_reward = target_reward[idx]
            batch_term = target_term[idx]
            batch_recover = target_recover[idx]
            batch_learned = learned_context[idx]
            batch_oracle = oracle_context[idx]

            learned_pred = model.predict_summary(batch_state, batch_action, batch_learned)
            oracle_pred = model.predict_summary(batch_state, batch_action, batch_oracle)
            loss = (
                mse_loss(learned_pred["next_state_delta"], batch_delta)
                + mse_loss(oracle_pred["next_state_delta"], batch_delta)
                + 0.5 * mse_loss(learned_pred["reward"], batch_reward)
                + 0.5 * mse_loss(oracle_pred["reward"], batch_reward)
                + 0.25 * bce_loss(learned_pred["term_logit"], batch_term)
                + 0.25 * bce_loss(oracle_pred["term_logit"], batch_term)
                + 0.25 * mse_loss(learned_pred["recoverability"], batch_recover)
                + 0.25 * mse_loss(oracle_pred["recoverability"], batch_recover)
            )
            learned_score = (
                learned_pred["reward"]
                + config.recoverability_weight * learned_pred["recoverability"]
                - config.termination_weight * torch.sigmoid(learned_pred["term_logit"])
                - config.disagreement_weight * learned_pred["disagreement"]
            )
            oracle_score = (
                oracle_pred["reward"]
                + config.recoverability_weight * oracle_pred["recoverability"]
                - config.termination_weight * torch.sigmoid(oracle_pred["term_logit"])
                - config.disagreement_weight * oracle_pred["disagreement"]
            )
            score_align_loss = mse_loss(learned_score, oracle_score.detach())
            rollout_loss = loss.new_tensor(0.0)
            if num_sequences > 0:
                sequence_batch = min(int(idx.shape[0]), num_sequences)
                sequence_idx = torch.randperm(num_sequences, device=device)[:sequence_batch]
                rollout_loss, rollout_score_align = rollout_supervision_and_alignment(
                    model=model,
                    initial_state=sequence_initial[sequence_idx],
                    action_sequences=sequence_actions[sequence_idx],
                    learned_context=sequence_learned[sequence_idx],
                    oracle_context=sequence_oracle[sequence_idx],
                    target_next_states=sequence_next_states[sequence_idx],
                    target_rewards=sequence_rewards[sequence_idx],
                    target_terms=sequence_terms[sequence_idx],
                    target_recoverability=sequence_recover[sequence_idx],
                    gamma=config.gamma,
                    reward_weight=config.reward_weight,
                    recoverability_weight=config.recoverability_weight,
                    termination_weight=config.termination_weight,
                    disagreement_weight=config.disagreement_weight,
                )
                score_align_loss = sanitize_tensor(score_align_loss + 0.50 * rollout_score_align)
                loss = sanitize_tensor(loss + 0.20 * rollout_loss)
            loss = sanitize_tensor(loss + 0.10 * score_align_loss)
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            total_score_align += float(score_align_loss.item()) * len(idx)
            total_rollout_loss += float(rollout_loss.item()) * len(idx)

    return {
        "loss": total_loss / max(num_rows * max(1, int(epochs)), 1),
        "score_alignment_loss": total_score_align / max(num_rows * max(1, int(epochs)), 1),
        "rollout_supervision_loss": total_rollout_loss / max(num_rows * max(1, int(epochs)), 1),
    }


def planner_trust_from_context(context: np.ndarray) -> float:
    flat = sanitize_numpy(context).reshape(-1)
    if flat.size < 2:
        return 0.0
    confidence = float(np.clip(flat[-2], 0.0, 1.0))
    uncertainty = max(float(flat[-1]), 0.0)
    return float(np.clip(confidence * np.exp(-0.5 * uncertainty), 0.0, 1.0))


def _simulate_sequence_scores(
    *,
    model: BeliefDynamicsModel,
    policy: BeliefNativeActorCritic,
    initial_state: torch.Tensor,
    action_sequences: torch.Tensor,
    context: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
    config: BeliefPlannerConfig,
) -> torch.Tensor:
    batch_size, horizon, _action_dim = action_sequences.shape
    current_state = initial_state.expand(batch_size, -1)
    repeated_context = context.expand(batch_size, -1)
    total_score = torch.zeros((batch_size,), dtype=torch.float32, device=initial_state.device)
    scale, bias = action_scale_bias(action_low, action_high, initial_state.device)
    uncertainty = sanitize_tensor(repeated_context[:, -1])
    for step_idx in range(horizon):
        normalized_action = torch.clamp(action_sequences[:, step_idx, :], -1.0, 1.0)
        env_action = sanitize_tensor(bias + scale * normalized_action)
        prediction = model.predict_summary(current_state, env_action, repeated_context)
        discount = float(config.gamma ** step_idx)
        total_score = total_score + discount * (
            config.reward_weight * prediction["reward"]
            + config.recoverability_weight * prediction["recoverability"]
            - config.termination_weight * torch.sigmoid(prediction["term_logit"])
            - config.disagreement_weight * prediction["disagreement"]
            - config.uncertainty_growth_weight
            * uncertainty
            * float(step_idx + 1)
            / float(max(horizon, 1))
        )
        current_state = sanitize_tensor(current_state + prediction["next_state_delta"])
    with torch.no_grad():
        _terminal_mean, terminal_value = policy(current_state, repeated_context)
    total_score = total_score + float(config.gamma ** horizon) * sanitize_tensor(terminal_value)
    return sanitize_tensor(total_score)


def plan_cem_action(
    *,
    policy: BeliefNativeActorCritic,
    model: BeliefDynamicsModel,
    state_t: torch.Tensor,
    context_t: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
    previous_plan: np.ndarray | None,
    hidden_state: torch.Tensor | None = None,
    rng: np.random.Generator | None = None,
    config: BeliefPlannerConfig | None = None,
) -> dict[str, np.ndarray | float]:
    """Run a small continuous-action CEM loop and return the selected action."""
    config = BeliefPlannerConfig() if config is None else config
    rng = np.random.default_rng(0) if rng is None else rng
    with torch.no_grad():
        if hidden_state is None:
            actor_mean, actor_value = policy(state_t, context_t)
            next_hidden = policy.init_recurrent_state(context_t)
        else:
            actor_mean, actor_value, next_hidden, _aux = policy.forward_with_hidden(
                state_t,
                context_t,
                hidden_state=hidden_state,
            )
        scale, bias = action_scale_bias(action_low, action_high, actor_mean.device)
    actor_action = sanitize_numpy(
        (bias + scale * torch.tanh(actor_mean))
        .squeeze(0)
        .cpu()
        .numpy()
    )
    action_dim = actor_action.shape[0]
    actor_norm = sanitize_numpy(torch.tanh(actor_mean).squeeze(0).cpu().numpy())
    raw_std = np.clip(np.exp(policy.log_std.detach().cpu().numpy()), 0.05, 0.75).astype(np.float32)
    plan_mean = np.repeat(actor_norm[None, :], config.horizon, axis=0)
    if previous_plan is not None and previous_plan.shape == (config.horizon, action_dim):
        shifted = np.vstack([previous_plan[1:], previous_plan[-1:]])
        plan_mean = shifted.astype(np.float32)
    plan_std = np.repeat(raw_std[None, :], config.horizon, axis=0).astype(np.float32)

    best_sequence = plan_mean.copy()
    best_score = -float("inf")
    for _iter in range(config.iterations):
        candidate_sequences = rng.normal(
            loc=plan_mean[None, :, :],
            scale=plan_std[None, :, :],
            size=(config.candidates, config.horizon, action_dim),
        ).astype(np.float32)
        candidate_sequences = np.clip(candidate_sequences, -1.0, 1.0)
        if previous_plan is not None:
            candidate_sequences[0] = previous_plan.astype(np.float32)
        candidate_sequences[1] = plan_mean.astype(np.float32)
        candidate_t = torch.tensor(candidate_sequences, dtype=torch.float32, device=state_t.device)
        scores_t = _simulate_sequence_scores(
            model=model,
            policy=policy,
            initial_state=state_t,
            action_sequences=candidate_t,
            context=context_t,
            action_low=action_low,
            action_high=action_high,
            config=config,
        )
        scores = sanitize_numpy(scores_t.detach().cpu().numpy())
        elite_idx = np.argsort(scores)[-config.elites :]
        elites = candidate_sequences[elite_idx]
        plan_mean = elites.mean(axis=0).astype(np.float32)
        plan_std = np.clip(elites.std(axis=0).astype(np.float32), 0.05, 0.60)
        if float(scores[elite_idx[-1]]) > best_score:
            best_score = float(scores[elite_idx[-1]])
            best_sequence = candidate_sequences[elite_idx[-1]].copy()

    planner_action = sanitize_numpy(
        (bias + scale * torch.tensor(best_sequence[0], dtype=torch.float32, device=state_t.device))
        .cpu()
        .numpy()
    )
    with torch.no_grad():
        first_prediction = model.predict_summary(
            state_t,
            torch.tensor(planner_action[None, :], dtype=torch.float32, device=state_t.device),
            context_t,
        )
    first_step_disagreement = float(first_prediction["disagreement"].item())
    self_consistency = float(np.exp(-1.5 * max(first_step_disagreement, 0.0)))
    trust = planner_trust_from_context(context_t.squeeze(0).detach().cpu().numpy()) * self_consistency
    if trust < config.actor_only_trust_floor:
        chosen_action = actor_action
        planner_used = 0.0
    elif trust < config.planner_only_trust_floor:
        blend = (trust - config.actor_only_trust_floor) / max(
            config.planner_only_trust_floor - config.actor_only_trust_floor,
            1e-6,
        )
        chosen_action = sanitize_numpy(
            (1.0 - blend) * actor_action + blend * planner_action
        )
        planner_used = float(blend)
    else:
        chosen_action = planner_action
        planner_used = 1.0
    return {
        "action": chosen_action,
        "actor_action": actor_action,
        "planner_action": planner_action,
        "planner_sequence": sanitize_numpy(best_sequence),
        "planner_score": float(best_score),
        "planner_trust": float(trust),
        "planner_used": float(planner_used),
        "planner_disagreement": float(first_step_disagreement),
        "actor_value": float(actor_value.item()),
        "action_divergence": float(np.mean(np.abs(planner_action - actor_action))),
        "next_hidden": sanitize_numpy(next_hidden.squeeze(0).detach().cpu().numpy()),
    }


def update_planner_prior(
    *,
    policy: BeliefNativeActorCritic,
    optimizer: optim.Optimizer,
    batch_states: np.ndarray,
    batch_contexts: np.ndarray,
    batch_actions: np.ndarray,
    batch_returns: np.ndarray,
    batch_hidden: np.ndarray | None,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_grad_norm: float,
) -> float:
    """Train the actor/value prior to imitate planner actions and bootstrap returns."""
    device = next(policy.parameters()).device
    states = torch.tensor(sanitize_numpy(batch_states), dtype=torch.float32, device=device)
    contexts = torch.tensor(sanitize_numpy(batch_contexts), dtype=torch.float32, device=device)
    target_actions = torch.tensor(sanitize_numpy(batch_actions), dtype=torch.float32, device=device)
    target_returns = torch.tensor(sanitize_numpy(batch_returns), dtype=torch.float32, device=device)
    if batch_hidden is not None:
        hidden = torch.tensor(sanitize_numpy(batch_hidden), dtype=torch.float32, device=device)
        mean, value, _next_hidden, _aux = policy.forward_sequence(
            states,
            contexts,
            hidden[:, 0, :],
            mask=torch.ones(states.shape[:2], dtype=torch.float32, device=device),
        )
        mean = mean.reshape(-1, mean.shape[-1])
        value = value.reshape(-1)
        target_actions = target_actions.reshape(-1, target_actions.shape[-1])
        target_returns = target_returns.reshape(-1)
    else:
        mean, value = policy(states.reshape(-1, states.shape[-1]), contexts.reshape(-1, contexts.shape[-1]))
        target_actions = target_actions.reshape(-1, target_actions.shape[-1])
        target_returns = target_returns.reshape(-1)
    scale, bias = action_scale_bias(action_low, action_high, device)
    predicted_actions = sanitize_tensor(bias + scale * torch.tanh(mean))
    loss = nn.functional.mse_loss(predicted_actions, target_actions) + 0.5 * nn.functional.mse_loss(value, target_returns)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.item())


def replay_batch_mean_error(
    *,
    model: BeliefDynamicsModel,
    states: np.ndarray,
    actions: np.ndarray,
    contexts: np.ndarray,
    next_states: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
) -> float:
    """Return one simple held-out-ish online model error summary."""
    if states.shape[0] == 0:
        return 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        prediction = model.predict_summary(
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.float32, device=device),
            torch.tensor(contexts, dtype=torch.float32, device=device),
        )
    predicted_next = sanitize_numpy(
        (torch.tensor(states, dtype=torch.float32, device=device) + prediction["next_state_delta"])
        .cpu()
        .numpy()
    )
    reward_error = np.mean(np.abs(sanitize_numpy(prediction["reward"].cpu().numpy()) - rewards.reshape(-1)))
    term_error = np.mean(
        np.abs(sanitize_numpy(torch.sigmoid(prediction["term_logit"]).cpu().numpy()) - terminals.reshape(-1))
    )
    state_error = np.mean(np.abs(predicted_next - next_states))
    return float(state_error + 0.5 * reward_error + 0.5 * term_error)


def update_belief_dynamics_from_replay(
    *,
    model: BeliefDynamicsModel,
    optimizer: optim.Optimizer,
    states: np.ndarray,
    actions: np.ndarray,
    contexts: np.ndarray,
    next_states: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    recoverability: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
    updates: int = 2,
) -> float:
    """Run a few small replay updates on planner dynamics using control rollouts."""
    if states.shape[0] <= 0:
        return 0.0
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    num_rows = int(states.shape[0])
    for _ in range(max(1, int(updates))):
        batch_idx = np.random.choice(
            num_rows,
            size=min(max(1, int(batch_size)), num_rows),
            replace=(num_rows < max(1, int(batch_size))),
        )
        state_t = torch.tensor(states[batch_idx], dtype=torch.float32, device=device)
        action_t = torch.tensor(actions[batch_idx], dtype=torch.float32, device=device)
        context_t = torch.tensor(contexts[batch_idx], dtype=torch.float32, device=device)
        next_state_t = torch.tensor(next_states[batch_idx], dtype=torch.float32, device=device)
        reward_t = torch.tensor(
            rewards[batch_idx].reshape(-1),
            dtype=torch.float32,
            device=device,
        )
        term_t = torch.tensor(
            terminals[batch_idx].reshape(-1),
            dtype=torch.float32,
            device=device,
        )
        recover_t = torch.tensor(
            recoverability[batch_idx].reshape(-1),
            dtype=torch.float32,
            device=device,
        )
        prediction = model.predict_summary(state_t, action_t, context_t)
        delta_target = sanitize_tensor(next_state_t - state_t)
        loss = (
            mse_loss(prediction["next_state_delta"], delta_target)
            + 0.5 * mse_loss(prediction["reward"], reward_t)
            + 0.25 * bce_loss(prediction["term_logit"], term_t)
            + 0.25 * mse_loss(prediction["recoverability"], recover_t)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / float(max(1, int(updates)))


__all__ = [
    "BeliefDynamicsModel",
    "BeliefPlannerConfig",
    "PlanningBeliefState",
    "build_planner_probe_dataset",
    "fit_belief_dynamics_model",
    "plan_cem_action",
    "planner_trust_from_context",
    "replay_batch_mean_error",
    "update_belief_dynamics_from_replay",
    "update_planner_prior",
]
