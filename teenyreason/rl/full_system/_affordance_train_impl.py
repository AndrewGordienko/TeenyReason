"""Cheap belief-conditioned controller training for the full-system path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ...crawler import CrawlerModelBundle
from ...envs import CONTINUOUS_CARTPOLE_NAME, get_action_values, make_env
from ...models.env_belief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ...probe.probe_data import apply_env_params, default_env_params
from ...probe.probe_latent import (
    aggregate_env_belief,
    maybe_update_online_belief,
    nearest_probe_action_idx,
    select_episode_physics,
)
from ...representation import DeltaPredictorEnsemble, WorldEncoder
from .affordance import (
    BeliefAffordanceController,
    choose_affordance_action,
    generate_candidate_actions,
    mean_to_action,
)
from .context_support import (
    collect_support_context as _collect_support_context,
    controller_context_for_episode as _controller_context_for_episode,
)
from .curriculum import (
    DEFAULT_FULL_SYSTEM_CURRICULUM_SCHEDULE,
    full_system_oracle_weight_for_episode,
    normalize_oracle_curriculum_schedule,
    should_stop_belief_planner_plateau,
)
from ..core.ppo_core import (
    RunningNormalizer,
    action_scale_bias,
    build_episode_batch,
    evaluate_continuous_actions,
    prepare_recurrent_minibatch,
    sanitize_numpy,
    sanitize_tensor,
    set_optimizer_lr,
    validate_continuous_env,
)
from ..probe_policy.eval import compute_control_surprise, transform_controller_context_input
from ..probe_policy.logging import print_belief_episode_status, print_solve_event
from ..probe_policy.types import MatchedEvalSummary, TrainingRunResult
from ..probe_policy.reporting import (
    format_peer_solve_status,
    restore_normalizer_state,
    snapshot_normalizer_state,
    snapshot_policy_state_dict,
)
from .simulator_fanout import SimulatorFanoutAdapter, candidate_score
from .simulator_fanout import FanoutLabel, PersistentFanoutLabelCache


@dataclass(frozen=True)
class EvaluationEpisodeFixture:
    """One reusable eval episode fixture shared across controller ablations."""

    episode_physics: object
    reset_seed: int
    support: dict | None
    learned_context: np.ndarray
    oracle_context: np.ndarray | None


def normalize_full_system_curriculum_schedule(
    schedule: list[tuple[int, float]] | tuple[tuple[int, float], ...] | None,
) -> tuple[tuple[int, float], ...]:
    """Return the explicit full-system curriculum schedule used by the controller."""
    return normalize_oracle_curriculum_schedule(
        schedule,
        default_schedule=DEFAULT_FULL_SYSTEM_CURRICULUM_SCHEDULE,
    )


def _zero_context_like(context_dim: int) -> np.ndarray:
    return np.zeros((int(context_dim),), dtype=np.float32)


def _clone_support_state(support: dict | None):
    if support is None:
        return None, None, []
    belief = None if support["belief"] is None else support["belief"].copy()
    belief_hidden = support["belief_hidden"]
    if isinstance(belief_hidden, torch.Tensor):
        belief_hidden = belief_hidden.detach().clone()
    belief_posteriors = [
        (posterior_mean.copy(), posterior_logvar.copy())
        for posterior_mean, posterior_logvar in support["belief_posteriors"]
    ]
    return belief, belief_hidden, belief_posteriors


def _build_evaluation_fixtures(
    *,
    crawler_bundle: CrawlerModelBundle,
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    env_future_predictor,
    predictor: DeltaPredictorEnsemble | None,
    env_name: str,
    action_values: np.ndarray,
    window_size: int,
    randomize_physics: bool,
    base_physics,
    base_probe_episodes: int,
    max_probe_episodes: int,
    probe_adaptive_budget: bool,
    uncertainty_probe_threshold: float,
    surprise_probe_threshold: float,
    eval_episodes: int,
    seed: int,
    use_context: bool,
) -> list[EvaluationEpisodeFixture]:
    """Collect support once per eval episode and reuse it across all ablations."""
    probe_env = make_env(env_name)
    rng = np.random.default_rng(seed)
    fixtures: list[EvaluationEpisodeFixture] = []
    for eval_episode in range(eval_episodes):
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        support = None
        learned_context = _zero_context_like(crawler_bundle.full_system_controller_dim)
        oracle_context = None
        if use_context:
            support = _collect_support_context(
                probe_env=probe_env,
                crawler_bundle=crawler_bundle,
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                env_future_predictor=env_future_predictor,
                predictor=predictor,
                rng=rng,
                env_name=env_name,
                episode_physics=episode_physics,
                action_values=action_values,
                window_size=window_size,
                base_probe_episodes=base_probe_episodes,
                max_probe_episodes=max_probe_episodes,
                probe_adaptive_budget=probe_adaptive_budget,
                uncertainty_probe_threshold=uncertainty_probe_threshold,
                surprise_probe_threshold=surprise_probe_threshold,
                trace_writer=None,
                episode=eval_episode + 1,
                variant_label="belief-controller-eval",
            )
            if support is None:
                continue
            learned_context = sanitize_numpy(support["step_result"].controller_context.vector)
            oracle_context = sanitize_numpy(
                crawler_bundle.build_oracle_controller_context(
                    episode_physics.as_array()
                ).vector
            )
        fixtures.append(
            EvaluationEpisodeFixture(
                episode_physics=episode_physics,
                reset_seed=int(seed + 10_000 + eval_episode),
                support=support,
                learned_context=learned_context,
                oracle_context=oracle_context,
            )
        )
    probe_env.close()
    return fixtures


def _pack_auxiliary_sequences(values: list[np.ndarray], sequence_length: int) -> np.ndarray:
    total_steps = len(values)
    if total_steps == 0:
        return np.zeros((0, sequence_length), dtype=np.float32)
    num_sequences = int(np.ceil(float(total_steps) / float(sequence_length)))
    tail_shape = tuple(np.asarray(values[0]).shape)
    packed = np.zeros((num_sequences, sequence_length) + tail_shape, dtype=np.float32)
    for seq_idx, start in enumerate(range(0, total_steps, sequence_length)):
        end = min(start + sequence_length, total_steps)
        valid = end - start
        packed[seq_idx, :valid] = np.asarray(values[start:end], dtype=np.float32)
    return packed.astype(np.float32)


def _affordance_training_stage(
    *,
    current_episode: int,
    total_episodes: int,
) -> str:
    """Stage the cheap controller around teacher imitation before PPO calibration."""
    if total_episodes <= 0:
        return "state_student"
    progress = float(current_episode) / float(max(total_episodes, 1))
    if progress <= 0.40:
        return "state_student"
    if progress <= 0.85:
        return "belief_residual"
    return "ppo_calibration"


def _squashed_action_from_mean(
    mean: torch.Tensor,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> torch.Tensor:
    """Map controller means to bounded actions while staying in Torch for losses."""
    scale, bias = action_scale_bias(action_low, action_high, mean.device)
    return sanitize_tensor(bias + scale * torch.tanh(sanitize_tensor(mean)))


def _teacher_label_for_candidates(
    *,
    env,
    adapter: SimulatorFanoutAdapter,
    label_cache: PersistentFanoutLabelCache,
    candidate_actions: np.ndarray,
    gamma: float,
    snapshot=None,
) -> FanoutLabel:
    """Fetch one cached fan-out label or compute it once from the live simulator."""
    return label_cache.get_or_compute(
        env=env,
        adapter=adapter,
        candidate_actions=candidate_actions,
        horizon=4,
        gamma=gamma,
        snapshot=snapshot,
    )


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad_(bool(trainable))


def _configure_affordance_training_stage(
    controller: BeliefAffordanceController,
    training_stage: str,
) -> None:
    """Freeze the controller around a strict state-student then belief-residual split."""
    is_state_student = training_stage == "state_student"
    state_backbone_trainable = is_state_student
    residual_side_trainable = not is_state_student

    _set_module_trainable(controller.state_encoder, state_backbone_trainable)
    _set_module_trainable(controller.recurrent, state_backbone_trainable)
    _set_module_trainable(controller.actor_head, state_backbone_trainable)
    _set_module_trainable(controller.state_candidate_head, state_backbone_trainable)
    controller.log_std.requires_grad_(state_backbone_trainable)

    _set_module_trainable(controller.context_encoder, residual_side_trainable)
    _set_module_trainable(controller.belief_residual_head, residual_side_trainable)
    _set_module_trainable(controller.trust_head, residual_side_trainable)
    _set_module_trainable(controller.value_head, True)


def _top2_margin(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] < 2:
        return torch.zeros(
            values.shape[0],
            dtype=values.dtype,
            device=values.device,
        )
    top2 = torch.topk(values, k=2, dim=1).values
    return sanitize_tensor(top2[:, 0] - top2[:, 1])


def _summarize_eval_solve(
    *,
    returns: list[float] | None,
    episode_total_steps: list[int] | None,
    solved_return: float,
) -> tuple[int | None, int | None, int | None, int]:
    if not returns or not episode_total_steps:
        return None, None, None, 0
    cumulative_steps = 0
    solved_episode = None
    solved_env_steps = None
    for episode_idx, (episode_return, episode_steps) in enumerate(
        zip(returns, episode_total_steps, strict=False),
        start=1,
    ):
        cumulative_steps += int(episode_steps)
        if solved_episode is None and float(episode_return) >= float(solved_return):
            solved_episode = int(episode_idx)
            solved_env_steps = int(cumulative_steps)
    return (
        solved_episode,
        solved_env_steps,
        int(cumulative_steps),
        int(len(returns)),
    )


def _build_matched_eval_summary(
    *,
    returns: list[float] | None,
    episode_total_steps: list[int] | None,
    solved_return: float,
) -> MatchedEvalSummary | None:
    """Collapse one fixed-fixture evaluation into a compact matched summary."""
    if not returns or not episode_total_steps:
        return None
    returns_arr = np.asarray(returns, dtype=np.float32)
    steps_arr = np.asarray(episode_total_steps, dtype=np.int64)
    fixture_count = int(min(len(returns_arr), len(steps_arr)))
    if fixture_count <= 0:
        return None
    returns_list = [float(value) for value in returns_arr[:fixture_count].tolist()]
    steps_list = [int(value) for value in steps_arr[:fixture_count].tolist()]
    return MatchedEvalSummary(
        returns=returns_list,
        episode_total_env_steps=steps_list,
        mean_return=float(np.mean(returns_arr[:fixture_count])),
        mean_total_env_steps=float(np.mean(steps_arr[:fixture_count])),
        solved_count=int(np.sum(returns_arr[:fixture_count] >= float(solved_return))),
        fixture_count=fixture_count,
    )


def _matched_eval_mean(summary: MatchedEvalSummary | None) -> float:
    if summary is None:
        return float("-inf")
    return float(summary.mean_return)


def _matched_eval_step_mean(summary: MatchedEvalSummary | None) -> float:
    if summary is None:
        return float("inf")
    return float(summary.mean_total_env_steps)


def _matched_eval_delta(
    learned: MatchedEvalSummary | None,
    ablation: MatchedEvalSummary | None,
) -> float | None:
    if learned is None or ablation is None:
        return None
    return float(learned.mean_return - ablation.mean_return)


def _checkpoint_selection_key(
    *,
    learned_summary: MatchedEvalSummary | None,
    state_only_summary: MatchedEvalSummary | None,
    zero_context_summary: MatchedEvalSummary | None,
    stale_context_summary: MatchedEvalSummary | None,
    shuffled_context_summary: MatchedEvalSummary | None,
    training_return: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Lexicographic attribution-aware checkpoint objective."""
    required_context_margin = 50.0
    learned_mean = _matched_eval_mean(learned_summary)
    if not np.isfinite(learned_mean):
        return (
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float("-inf"),
            float(training_return),
        )
    state_only_gain = learned_mean - _matched_eval_mean(state_only_summary)
    zero_context_gain = learned_mean - _matched_eval_mean(zero_context_summary)
    stale_context_gain = learned_mean - _matched_eval_mean(stale_context_summary)
    shuffled_context_gain = learned_mean - _matched_eval_mean(shuffled_context_summary)
    finite_gains = [
        float(gain)
        for gain in (
            state_only_gain,
            zero_context_gain,
            stale_context_gain,
            shuffled_context_gain,
        )
        if np.isfinite(gain)
    ]
    if not finite_gains:
        finite_gains = [float("-inf")]
    gain_floor = float(min(finite_gains))
    gain_mean = float(np.mean(np.asarray(finite_gains, dtype=np.float32)))
    gate_margin_values = [
        float(state_only_gain),
        float(zero_context_gain),
        float(stale_context_gain),
        float(shuffled_context_gain),
    ]
    passing_margin_count = sum(
        1 for gain in gate_margin_values if np.isfinite(gain) and gain >= required_context_margin
    )
    attribution_gate_pass = 1.0 if passing_margin_count == len(gate_margin_values) else 0.0
    attribution_margin_penalty = float(
        sum(
            max(0.0, required_context_margin - gain)
            for gain in gate_margin_values
            if np.isfinite(gain)
        )
    )
    effective_learned_mean = float(learned_mean - 0.20 * attribution_margin_penalty)
    effective_gain_mean = float(gain_mean - 0.10 * attribution_margin_penalty)
    return (
        attribution_gate_pass,
        gain_floor,
        effective_learned_mean,
        effective_gain_mean,
        float(state_only_gain),
        float(zero_context_gain),
        float(stale_context_gain),
        float(shuffled_context_gain),
        float(-_matched_eval_step_mean(learned_summary)),
        float(training_return),
    )


def _evaluate_counterfactual_context(
    *,
    controller: BeliefAffordanceController,
    state_trunk: torch.Tensor,
    candidate_actions: torch.Tensor,
    context: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Score one counterfactual controller context on the same state/candidate batch."""
    context = sanitize_tensor(context)
    context_features = controller.encode_context(context)
    _mechanics, _affordance, confidence, uncertainty = controller.split_controller_context(context)
    trust_input = torch.cat(
        [state_trunk, context_features, confidence, uncertainty],
        dim=-1,
    )
    trust_prior = torch.clamp(confidence, 0.0, 1.0) * torch.exp(
        -torch.clamp(uncertainty, min=0.0)
    )
    trust = sanitize_tensor(
        torch.clamp(
            torch.sigmoid(controller.trust_head(trust_input)).squeeze(-1) * trust_prior.squeeze(-1),
            0.0,
            1.0,
        )
    )
    score_outputs = controller.evaluate_candidate_scores(
        state_trunk=state_trunk,
        context_features=context_features,
        candidate_actions=candidate_actions,
        trust=trust,
        confidence=confidence.squeeze(-1),
        uncertainty=uncertainty.squeeze(-1),
    )
    return {
        "trust": trust,
        "belief_residual": score_outputs["belief_residual"],
        "final_scores": score_outputs["final_scores"],
    }


def _counterfactual_context_penalties(
    *,
    controller: BeliefAffordanceController,
    state_trunk: torch.Tensor,
    candidate_actions: torch.Tensor,
    state_scores: torch.Tensor,
    context: torch.Tensor,
    include_shuffled: bool = True,
) -> dict[str, torch.Tensor]:
    """Push trust and residuals toward zero when context is missing, stale, or permuted."""
    if context.shape[0] <= 0:
        zero = state_scores.sum() * 0.0
        return {
            "counterfactual_trust_loss": zero,
            "counterfactual_residual_loss": zero,
            "counterfactual_state_alignment_loss": zero,
        }
    detached_context = sanitize_tensor(context.detach())
    zero_context = torch.zeros_like(detached_context)
    stale_context = (
        torch.roll(detached_context, shifts=1, dims=0)
        if detached_context.shape[0] > 1
        else torch.zeros_like(detached_context)
    )
    context_variants = [zero_context, stale_context]
    if include_shuffled:
        if detached_context.shape[0] > 1:
            shuffled_indices = torch.randperm(detached_context.shape[0], device=detached_context.device)
            shuffled_context = detached_context[shuffled_indices]
        else:
            shuffled_context = torch.zeros_like(detached_context)
        context_variants.append(shuffled_context)
    detached_state_scores = sanitize_tensor(state_scores.detach())
    trust_losses = []
    residual_losses = []
    state_alignment_losses = []
    for variant_context in context_variants:
        outputs = _evaluate_counterfactual_context(
            controller=controller,
            state_trunk=state_trunk,
            candidate_actions=candidate_actions,
            context=variant_context,
        )
        trust_losses.append(torch.mean(outputs["trust"].pow(2)))
        residual_losses.append(torch.mean(outputs["belief_residual"].pow(2)))
        state_alignment_losses.append(F.mse_loss(outputs["final_scores"], detached_state_scores))
    return {
        "counterfactual_trust_loss": torch.mean(torch.stack(trust_losses)),
        "counterfactual_residual_loss": torch.mean(torch.stack(residual_losses)),
        "counterfactual_state_alignment_loss": torch.mean(torch.stack(state_alignment_losses)),
    }


def _update_affordance_controller(
    *,
    controller: BeliefAffordanceController,
    optimizer: torch.optim.Optimizer,
    batch,
    candidate_actions: list[np.ndarray],
    candidate_returns: list[np.ndarray],
    candidate_risks: list[np.ndarray],
    candidate_recoverability: list[np.ndarray],
    candidate_scores: list[np.ndarray],
    candidate_best_idx: list[int],
    candidate_best_vs_actor_margin: list[float],
    context_confidences: list[float],
    action_low: np.ndarray,
    action_high: np.ndarray,
    clip_ratio: float,
    value_loss_weight: float,
    entropy_coef: float,
    ppo_epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    target_kl: float,
    sequence_length: int,
    current_episode: int,
    total_episodes: int,
) -> None:
    """Run staged teacher-student updates before light PPO calibration."""
    device = next(controller.parameters()).device
    training_stage = _affordance_training_stage(
        current_episode=current_episode,
        total_episodes=total_episodes,
    )
    _configure_affordance_training_stage(controller, training_stage)
    states = torch.tensor(sanitize_numpy(batch.states), dtype=torch.float32, device=device)
    actions = torch.tensor(sanitize_numpy(batch.actions), dtype=torch.float32, device=device)
    old_log_probs = torch.tensor(sanitize_numpy(batch.old_log_probs), dtype=torch.float32, device=device)
    old_values = torch.tensor(sanitize_numpy(batch.old_values), dtype=torch.float32, device=device)
    returns = torch.tensor(sanitize_numpy(batch.returns), dtype=torch.float32, device=device)
    advantages = torch.tensor(sanitize_numpy(batch.advantages), dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    beliefs = torch.tensor(sanitize_numpy(batch.beliefs), dtype=torch.float32, device=device)
    recurrent_hidden_states = torch.tensor(
        sanitize_numpy(batch.recurrent_hidden_states),
        dtype=torch.float32,
        device=device,
    )
    sequence_batch = prepare_recurrent_minibatch(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        beliefs=beliefs,
        recurrent_hidden_states=recurrent_hidden_states,
        sequence_length=sequence_length,
    )
    packed_candidate_actions = torch.tensor(
        _pack_auxiliary_sequences(candidate_actions, sequence_length),
        dtype=torch.float32,
        device=device,
    )
    packed_candidate_returns = torch.tensor(
        _pack_auxiliary_sequences(candidate_returns, sequence_length),
        dtype=torch.float32,
        device=device,
    )
    packed_candidate_risks = torch.tensor(
        _pack_auxiliary_sequences(candidate_risks, sequence_length),
        dtype=torch.float32,
        device=device,
    )
    packed_candidate_recoverability = torch.tensor(
        _pack_auxiliary_sequences(candidate_recoverability, sequence_length),
        dtype=torch.float32,
        device=device,
    )
    packed_candidate_scores = torch.tensor(
        _pack_auxiliary_sequences(candidate_scores, sequence_length),
        dtype=torch.float32,
        device=device,
    )
    packed_candidate_best_idx = torch.tensor(
        _pack_auxiliary_sequences([np.asarray([idx], dtype=np.float32) for idx in candidate_best_idx], sequence_length)[..., 0],
        dtype=torch.long,
        device=device,
    )
    packed_best_vs_actor_margin = torch.tensor(
        _pack_auxiliary_sequences(
            [np.asarray([margin], dtype=np.float32) for margin in candidate_best_vs_actor_margin],
            sequence_length,
        )[..., 0],
        dtype=torch.float32,
        device=device,
    )
    packed_context_confidences = torch.tensor(
        _pack_auxiliary_sequences(
            [np.asarray([confidence], dtype=np.float32) for confidence in context_confidences],
            sequence_length,
        )[..., 0],
        dtype=torch.float32,
        device=device,
    )
    total_sequences = int(sequence_batch["states"].shape[0])
    minibatch_size = min(int(minibatch_size), total_sequences)
    state_student_stage = training_stage == "state_student"

    for _ in range(int(ppo_epochs)):
        permutation = torch.randperm(total_sequences, device=device)
        stop_early = False
        for start in range(0, total_sequences, minibatch_size):
            idx = permutation[start:start + minibatch_size]
            batch_states = sequence_batch["states"][idx]
            batch_actions = sequence_batch["actions"][idx]
            batch_old_log_probs = sequence_batch["old_log_probs"][idx]
            batch_old_values = sequence_batch["old_values"][idx]
            batch_returns_seq = sequence_batch["returns"][idx]
            batch_advantages = sequence_batch["advantages"][idx]
            batch_beliefs = sequence_batch["beliefs"][idx]
            batch_hidden = sequence_batch["hidden"][idx]
            batch_mask = sequence_batch["mask"][idx]
            batch_candidate_actions = packed_candidate_actions[idx]
            batch_candidate_returns = packed_candidate_returns[idx]
            batch_candidate_risks = packed_candidate_risks[idx]
            batch_candidate_recoverability = packed_candidate_recoverability[idx]
            batch_candidate_scores = packed_candidate_scores[idx]
            batch_candidate_best_idx = packed_candidate_best_idx[idx]
            batch_best_vs_actor_margin = packed_best_vs_actor_margin[idx]
            batch_context_confidences = packed_context_confidences[idx]

            mean, value, _next_hidden, aux = controller.forward_sequence(
                batch_states,
                batch_beliefs,
                batch_hidden,
                mask=batch_mask,
            )
            flat_mask = batch_mask.reshape(-1) > 0
            flat_mean = mean.reshape(-1, mean.shape[-1])[flat_mask]
            flat_value = value.reshape(-1)[flat_mask]
            flat_actions = batch_actions.reshape(-1, batch_actions.shape[-1])[flat_mask]
            flat_old_log_probs = batch_old_log_probs.reshape(-1)[flat_mask]
            flat_old_values = batch_old_values.reshape(-1)[flat_mask]
            flat_returns = batch_returns_seq.reshape(-1)[flat_mask]
            flat_advantages = batch_advantages.reshape(-1)[flat_mask]
            flat_trust = aux["trust"].reshape(-1)[flat_mask]
            flat_state_trunk = aux["state_trunk"].reshape(
                -1, aux["state_trunk"].shape[-1]
            )[flat_mask]
            flat_context = batch_beliefs.reshape(-1, batch_beliefs.shape[-1])[flat_mask]
            flat_context_features = aux["context_features"].reshape(
                -1, aux["context_features"].shape[-1]
            )[flat_mask]
            flat_confidence = aux["confidence"].reshape(-1)[flat_mask]
            flat_uncertainty = aux["uncertainty"].reshape(-1)[flat_mask]
            flat_candidate_actions = batch_candidate_actions.reshape(
                -1,
                batch_candidate_actions.shape[-2],
                batch_candidate_actions.shape[-1],
            )[flat_mask]
            flat_candidate_returns = batch_candidate_returns.reshape(
                -1,
                batch_candidate_returns.shape[-1],
            )[flat_mask]
            flat_candidate_risks = batch_candidate_risks.reshape(
                -1,
                batch_candidate_risks.shape[-1],
            )[flat_mask]
            flat_candidate_recoverability = batch_candidate_recoverability.reshape(
                -1,
                batch_candidate_recoverability.shape[-1],
            )[flat_mask]
            flat_candidate_scores = batch_candidate_scores.reshape(
                -1,
                batch_candidate_scores.shape[-1],
            )[flat_mask]
            flat_candidate_best_idx = batch_candidate_best_idx.reshape(-1)[flat_mask]
            flat_best_vs_actor_margin = batch_best_vs_actor_margin.reshape(-1)[flat_mask]
            flat_context_confidences = batch_context_confidences.reshape(-1)[flat_mask]

            new_log_prob, entropy = evaluate_continuous_actions(
                mean=flat_mean,
                log_std=controller.log_std,
                actions=flat_actions,
                action_low=action_low,
                action_high=action_high,
            )
            log_ratio = new_log_prob - flat_old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped = ratio * flat_advantages
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * flat_advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(flat_value, flat_returns)

            score_outputs = controller.evaluate_candidate_scores(
                state_trunk=flat_state_trunk,
                context_features=flat_context_features,
                candidate_actions=flat_candidate_actions,
                trust=flat_trust,
                confidence=flat_confidence,
                uncertainty=flat_uncertainty,
            )
            pred_return = score_outputs["return"]
            pred_risk = score_outputs["risk"]
            pred_recoverability = score_outputs["recoverability"]
            state_scores = score_outputs["state_scores"]
            belief_residual = score_outputs["belief_residual"]
            final_scores = score_outputs["final_scores"]
            return_loss = F.mse_loss(pred_return, flat_candidate_returns)
            risk_loss = F.mse_loss(pred_risk, flat_candidate_risks)
            recoverability_loss = F.mse_loss(
                pred_recoverability,
                flat_candidate_recoverability,
            )
            state_score_loss = F.mse_loss(state_scores, flat_candidate_scores)
            state_rank_loss = F.cross_entropy(state_scores, flat_candidate_best_idx)
            state_margin_loss = F.smooth_l1_loss(
                _top2_margin(state_scores),
                _top2_margin(flat_candidate_scores),
            )
            detached_state_scores = state_scores.detach()
            state_best_idx = detached_state_scores.argmax(dim=1)
            state_selected_teacher_scores = flat_candidate_scores.gather(
                1,
                state_best_idx.unsqueeze(-1),
            ).squeeze(-1)
            teacher_best_scores = flat_candidate_scores.gather(
                1,
                flat_candidate_best_idx.unsqueeze(-1),
            ).squeeze(-1)
            teacher_margin = _top2_margin(flat_candidate_scores)
            teacher_gain = torch.relu(teacher_best_scores - state_selected_teacher_scores)
            teacher_mismatch = flat_candidate_best_idx != state_best_idx
            positive_teacher_gain = torch.logical_and(
                teacher_gain >= 0.05,
                teacher_mismatch,
            )
            trust_target = torch.where(
                positive_teacher_gain,
                torch.clamp(teacher_gain / teacher_margin.clamp_min(1e-4), 0.0, 1.0),
                torch.zeros_like(teacher_gain),
            )
            residual_target = flat_candidate_scores - detached_state_scores
            residual_loss = F.mse_loss(belief_residual, residual_target)
            detached_final_scores = detached_state_scores + flat_trust.unsqueeze(-1) * belief_residual
            final_score_loss = F.mse_loss(detached_final_scores, flat_candidate_scores)
            final_rank_loss = F.cross_entropy(detached_final_scores, flat_candidate_best_idx)
            final_margin_loss = F.smooth_l1_loss(
                _top2_margin(detached_final_scores),
                teacher_margin,
            )
            trust_loss = F.smooth_l1_loss(flat_trust, trust_target)
            no_gain_mask = (~positive_teacher_gain).to(dtype=belief_residual.dtype)
            positive_gain_mask = positive_teacher_gain.to(dtype=belief_residual.dtype)
            residual_shrink_loss = torch.mean(
                no_gain_mask.unsqueeze(-1) * belief_residual.pow(2)
            )
            trust_shrink_loss = torch.mean(no_gain_mask * flat_trust.pow(2))
            teacher_best_final_scores = detached_final_scores.gather(
                1,
                flat_candidate_best_idx.unsqueeze(-1),
            ).squeeze(-1)
            state_best_final_scores = detached_final_scores.gather(
                1,
                state_best_idx.unsqueeze(-1),
            ).squeeze(-1)
            ranking_hinge_loss = (
                torch.sum(
                    positive_gain_mask
                    * torch.relu(0.10 - (teacher_best_final_scores - state_best_final_scores))
                )
                / positive_gain_mask.sum().clamp_min(1.0)
            )
            counterfactual_losses = _counterfactual_context_penalties(
                controller=controller,
                state_trunk=flat_state_trunk,
                candidate_actions=flat_candidate_actions,
                state_scores=state_scores,
                context=flat_context,
                include_shuffled=True,
            )
            counterfactual_trust_loss = counterfactual_losses["counterfactual_trust_loss"]
            counterfactual_residual_loss = counterfactual_losses["counterfactual_residual_loss"]
            counterfactual_state_alignment_loss = counterfactual_losses["counterfactual_state_alignment_loss"]
            squashed_actor_action = _squashed_action_from_mean(
                flat_mean,
                action_low=action_low,
                action_high=action_high,
            )
            teacher_best_actions = flat_candidate_actions[
                torch.arange(flat_candidate_actions.shape[0], device=device),
                flat_candidate_best_idx,
            ]
            actor_bc_loss = F.mse_loss(squashed_actor_action, teacher_best_actions)

            if state_student_stage:
                loss = (
                    0.42 * state_score_loss
                    + 0.20 * state_rank_loss
                    + 0.10 * state_margin_loss
                    + 0.13 * actor_bc_loss
                    + 0.07 * return_loss
                    + 0.04 * risk_loss
                    + 0.04 * recoverability_loss
                    + 0.05 * value_loss
                )
            elif training_stage == "belief_residual":
                loss = (
                    0.24 * residual_loss
                    + 0.28 * final_score_loss
                    + 0.18 * final_rank_loss
                    + 0.10 * final_margin_loss
                    + 0.12 * ranking_hinge_loss
                    + 0.12 * trust_loss
                    + 0.08 * residual_shrink_loss
                    + 0.05 * trust_shrink_loss
                    + 0.10 * counterfactual_residual_loss
                    + 0.08 * counterfactual_trust_loss
                    + 0.08 * counterfactual_state_alignment_loss
                    + 0.05 * value_loss
                )
            else:
                loss = (
                    0.18 * residual_loss
                    + 0.22 * final_score_loss
                    + 0.14 * final_rank_loss
                    + 0.10 * final_margin_loss
                    + 0.10 * ranking_hinge_loss
                    + 0.14 * trust_loss
                    + 0.08 * residual_shrink_loss
                    + 0.05 * trust_shrink_loss
                    + 0.10 * counterfactual_residual_loss
                    + 0.08 * counterfactual_trust_loss
                    + 0.08 * counterfactual_state_alignment_loss
                    + 0.04 * return_loss
                    + 0.03 * risk_loss
                    + 0.03 * recoverability_loss
                    + 0.12 * float(value_loss_weight) * value_loss
                )
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_grad_norm)
            optimizer.step()
            approx_kl = float(sanitize_tensor((flat_old_log_probs - new_log_prob).mean()).item())
            if approx_kl > 1.5 * float(target_kl):
                stop_early = True
                break
        if stop_early:
            break


def _run_attribution_aux_update(
    *,
    controller: BeliefAffordanceController,
    optimizer: torch.optim.Optimizer,
    replay_buffer: deque[dict[str, np.ndarray | float | int]],
    minibatch_size: int,
    max_grad_norm: float,
) -> None:
    """Run one lightweight residual/trust-only replay update on cached labels."""
    if not replay_buffer:
        return
    sample_count = min(int(minibatch_size), len(replay_buffer))
    if sample_count <= 0:
        return
    device = next(controller.parameters()).device
    sample_indices = np.random.choice(len(replay_buffer), size=sample_count, replace=False)
    batch_rows = [list(replay_buffer)[int(idx)] for idx in sample_indices.tolist()]
    states = torch.tensor(
        np.stack([np.asarray(row["state"], dtype=np.float32) for row in batch_rows], axis=0),
        dtype=torch.float32,
        device=device,
    )
    contexts = torch.tensor(
        np.stack([np.asarray(row["context"], dtype=np.float32) for row in batch_rows], axis=0),
        dtype=torch.float32,
        device=device,
    )
    candidate_actions = torch.tensor(
        np.stack(
            [np.asarray(row["candidate_actions"], dtype=np.float32) for row in batch_rows],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )
    candidate_scores = torch.tensor(
        np.stack(
            [np.asarray(row["candidate_scores"], dtype=np.float32) for row in batch_rows],
            axis=0,
        ),
        dtype=torch.float32,
        device=device,
    )
    candidate_best_idx = torch.tensor(
        [int(row["candidate_best_idx"]) for row in batch_rows],
        dtype=torch.long,
        device=device,
    )
    raw_context = torch.tensor(
        np.stack([np.asarray(row["context"], dtype=np.float32) for row in batch_rows], axis=0),
        dtype=torch.float32,
        device=device,
    )

    _mean, _value, _next_hidden, aux = controller.forward_with_hidden(
        states,
        contexts,
        hidden_state=None,
    )
    score_outputs = controller.evaluate_candidate_scores(
        state_trunk=aux["state_trunk"],
        context_features=aux["context_features"],
        candidate_actions=candidate_actions,
        trust=aux["trust"],
        confidence=aux["confidence"],
        uncertainty=aux["uncertainty"],
    )
    state_scores = score_outputs["state_scores"]
    belief_residual = score_outputs["belief_residual"]
    final_scores = score_outputs["final_scores"]
    trust = aux["trust"]
    detached_state_scores = state_scores.detach()
    state_best_idx = detached_state_scores.argmax(dim=1)
    state_selected_teacher_scores = candidate_scores.gather(
        1,
        state_best_idx.unsqueeze(-1),
    ).squeeze(-1)
    teacher_best_scores = candidate_scores.gather(
        1,
        candidate_best_idx.unsqueeze(-1),
    ).squeeze(-1)
    teacher_margin = _top2_margin(candidate_scores)
    teacher_gain = torch.relu(teacher_best_scores - state_selected_teacher_scores)
    positive_teacher_gain = torch.logical_and(
        teacher_gain >= 0.05,
        candidate_best_idx != state_best_idx,
    )
    trust_target = torch.where(
        positive_teacher_gain,
        torch.clamp(teacher_gain / teacher_margin.clamp_min(1e-4), 0.0, 1.0),
        torch.zeros_like(teacher_gain),
    )
    residual_target = candidate_scores - detached_state_scores
    residual_loss = F.mse_loss(belief_residual, residual_target)
    final_rank_loss = F.cross_entropy(final_scores, candidate_best_idx)
    trust_loss = F.smooth_l1_loss(trust, trust_target)
    no_gain_mask = (~positive_teacher_gain).to(dtype=belief_residual.dtype)
    positive_gain_mask = positive_teacher_gain.to(dtype=belief_residual.dtype)
    residual_shrink_loss = torch.mean(no_gain_mask.unsqueeze(-1) * belief_residual.pow(2))
    trust_shrink_loss = torch.mean(no_gain_mask * trust.pow(2))
    teacher_best_final_scores = final_scores.gather(
        1,
        candidate_best_idx.unsqueeze(-1),
    ).squeeze(-1)
    state_best_final_scores = final_scores.gather(
        1,
        state_best_idx.unsqueeze(-1),
    ).squeeze(-1)
    ranking_hinge_loss = (
        torch.sum(
            positive_gain_mask
            * torch.relu(0.10 - (teacher_best_final_scores - state_best_final_scores))
        )
        / positive_gain_mask.sum().clamp_min(1.0)
    )
    counterfactual_losses = _counterfactual_context_penalties(
        controller=controller,
        state_trunk=aux["state_trunk"],
        candidate_actions=candidate_actions,
        state_scores=state_scores,
        context=raw_context,
        include_shuffled=True,
    )
    loss = (
        0.32 * residual_loss
        + 0.24 * final_rank_loss
        + 0.18 * trust_loss
        + 0.12 * ranking_hinge_loss
        + 0.09 * residual_shrink_loss
        + 0.05 * trust_shrink_loss
        + 0.10 * counterfactual_losses["counterfactual_residual_loss"]
        + 0.08 * counterfactual_losses["counterfactual_trust_loss"]
        + 0.08 * counterfactual_losses["counterfactual_state_alignment_loss"]
    )
    if not torch.isfinite(loss):
        return
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_grad_norm)
    optimizer.step()


def _select_sim_fanout_action(
    *,
    controller: BeliefAffordanceController,
    env,
    adapter: SimulatorFanoutAdapter,
    label_cache: PersistentFanoutLabelCache,
    state_t: torch.Tensor,
    context_t: torch.Tensor,
    action_low: np.ndarray,
    action_high: np.ndarray,
    hidden_state: torch.Tensor | None,
    gamma: float,
    snapshot,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, np.ndarray | float | int]]:
    with torch.no_grad():
        mean, value, next_hidden, aux = controller.forward_with_hidden(
            state_t,
            context_t,
            hidden_state=hidden_state,
        )
    actor_action = mean_to_action(mean, action_low=action_low, action_high=action_high)
    candidate_actions = generate_candidate_actions(
        mean_action=actor_action,
        action_low=action_low,
        action_high=action_high,
    )
    fanout = _teacher_label_for_candidates(
        env=env,
        adapter=adapter,
        label_cache=label_cache,
        candidate_actions=candidate_actions,
        gamma=gamma,
        snapshot=snapshot,
    )
    best_idx = int(fanout.best_idx)
    selected_action = sanitize_numpy(candidate_actions[best_idx].reshape(-1))
    return selected_action, mean.detach(), value.detach(), next_hidden.detach(), {
        "trust": 1.0,
        "controller_used": 1.0,
        "action_divergence": float(np.mean(np.abs(selected_action - actor_action.reshape(-1)))),
        "candidate_actions": sanitize_numpy(candidate_actions),
        "candidate_returns": sanitize_numpy(fanout.returns),
        "candidate_risks": sanitize_numpy(fanout.risks),
        "candidate_recoverability": sanitize_numpy(fanout.recoverability),
        "candidate_scores": sanitize_numpy(fanout.scores),
        "best_vs_actor_margin": float(fanout.best_vs_actor_margin),
        "best_idx": best_idx,
    }


def evaluate_belief_affordance_fixtures(
    *,
    controller: BeliefAffordanceController,
    fixtures: list[EvaluationEpisodeFixture],
    crawler_bundle: CrawlerModelBundle,
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    predictor: DeltaPredictorEnsemble | None,
    state_normalizer: RunningNormalizer,
    env_name: str,
    action_values: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    gamma: float,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    full_system_surprise_refresh_threshold: float,
    context_source: str,
    selector_mode: str,
    disable_context: bool = False,
    shuffle_context: bool = False,
    use_stale_previous_context: bool = False,
    disable_online_refinement: bool = False,
    actor_only: bool = False,
    state_only_student: bool = False,
) -> tuple[list[float], int, float, list[int]]:
    """Evaluate one controller or baseline on reusable episode fixtures."""
    env = make_env(env_name)
    adapter = SimulatorFanoutAdapter(env_name)
    label_cache = PersistentFanoutLabelCache(env_name=env_name)
    device = next(controller.parameters()).device
    rng = np.random.default_rng(0)
    returns: list[float] = []
    episode_total_steps: list[int] = []
    total_steps = 0
    total_probe_windows = 0
    stale_context_input: np.ndarray | None = None

    for fixture in fixtures:
        apply_env_params(env, fixture.episode_physics)
        raw_state, _info = env.reset(seed=fixture.reset_seed)
        raw_state = np.asarray(raw_state, dtype=np.float32)
        base_context = fixture.learned_context
        if context_source == "oracle" and fixture.oracle_context is not None:
            base_context = fixture.oracle_context
        context_input = transform_controller_context_input(
            base_context,
            disable_controller_context=disable_context,
            shuffle_controller_context=shuffle_context,
            rng=rng,
            stale_context_input=stale_context_input if use_stale_previous_context else None,
        )
        stale_context_input = sanitize_numpy(base_context.copy())
        belief, belief_hidden, belief_posteriors = _clone_support_state(fixture.support)
        episode_total = 0
        if fixture.support is not None:
            total_steps += int(fixture.support["probe_steps_total"])
            episode_total += int(fixture.support["probe_steps_total"])
            total_probe_windows += int(fixture.support["probe_windows_total"])
        done = False
        episode_return = 0.0
        episode_step = 0
        controller_hidden = controller.init_recurrent_state(
            torch.tensor(context_input[None, :], dtype=torch.float32, device=device)
        )

        while not done:
            state = sanitize_numpy(state_normalizer.normalize(raw_state))
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            context_t = torch.tensor(context_input[None, :], dtype=torch.float32, device=device)
            baseline_snapshot = adapter.snapshot(env)
            if actor_only:
                with torch.no_grad():
                    mean, _value, controller_hidden, _aux = controller.forward_with_hidden(
                        state_t,
                        context_t,
                        hidden_state=controller_hidden,
                    )
                action = mean_to_action(mean, action_low=action_low, action_high=action_high)
            elif selector_mode == "sim_fanout":
                action, _mean, _value, controller_hidden, _aux = _select_sim_fanout_action(
                    controller=controller,
                    env=env,
                    adapter=adapter,
                    label_cache=label_cache,
                    state_t=state_t,
                    context_t=context_t,
                    action_low=action_low,
                    action_high=action_high,
                    hidden_state=controller_hidden,
                    gamma=gamma,
                    snapshot=baseline_snapshot,
                )
            else:
                selection = choose_affordance_action(
                    controller=controller,
                    state_t=state_t,
                    context_t=context_t,
                    action_low=action_low,
                    action_high=action_high,
                    hidden_state=controller_hidden,
                    force_state_only=state_only_student,
                )
                action = selection.action
                controller_hidden = selection.next_hidden

            prev_raw_state = raw_state.copy()
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            total_steps += 1
            episode_total += 1
            raw_state = np.asarray(next_raw_state, dtype=np.float32)
            episode_return += float(reward)
            episode_step += 1
            done = bool(terminated or truncated)
            if (
                fixture.support is not None
                and (not disable_online_refinement)
                and belief_posteriors
            ):
                control_surprise = compute_control_surprise(
                    predictor=predictor,
                    belief=belief,
                    prev_state=prev_raw_state,
                    action_idx=nearest_probe_action_idx(action, action_values),
                    next_state=raw_state,
                    device=device,
                )
                belief, belief_hidden, belief_posteriors = maybe_update_online_belief(
                    encoder=encoder,
                    belief_aggregator=belief_aggregator,
                    env_param_predictor=env_param_predictor,
                    device=device,
                    belief_hidden=belief_hidden,
                    belief_posteriors=belief_posteriors,
                    prev_state=prev_raw_state,
                    action_idx=nearest_probe_action_idx(action, action_values),
                    reward=float(reward),
                    next_state=raw_state,
                    belief=belief,
                    online_z_update_alpha=online_z_update_alpha,
                    online_z_update_freq=online_z_update_freq,
                    episode_step=episode_step,
                )
                if (
                    episode_step % max(1, int(online_z_update_freq)) == 0
                    or float(control_surprise) >= float(full_system_surprise_refresh_threshold)
                ):
                    belief, payload = aggregate_env_belief(
                        belief_aggregator=belief_aggregator,
                        env_param_predictor=env_param_predictor,
                        device=device,
                        posterior_views=belief_posteriors,
                    )
                    refreshed = crawler_bundle.build_step_result(
                        payload=payload,
                        expected_family_gain={},
                        realized_family_gain={},
                        stop_reason=None,
                        bits_per_dim=0,
                        use_residual_sketch=False,
                    )
                    context_input = sanitize_numpy(
                        _controller_context_for_episode(
                            crawler_bundle=crawler_bundle,
                            step_result=refreshed,
                            episode_physics=fixture.episode_physics,
                            context_source=context_source,
                            oracle_weight=1.0 if context_source == "oracle" else 0.0,
                        ).vector
                    )
                    controller_hidden = controller.refresh_recurrent_state(
                        torch.tensor(context_input[None, :], dtype=torch.float32, device=device),
                        controller_hidden,
                    )
        returns.append(float(episode_return))
        episode_total_steps.append(int(episode_total))

    env.close()
    avg_probe_windows = float(total_probe_windows) / float(max(len(fixtures), 1))
    return returns, total_steps, avg_probe_windows, episode_total_steps


def _evaluate_affordance_checkpoint(
    *,
    policy_state_dict: dict[str, torch.Tensor],
    normalizer_state: dict[str, np.ndarray | float],
    fixtures: list[EvaluationEpisodeFixture],
    crawler_bundle: CrawlerModelBundle,
    hidden_dim: int,
    env_name: str,
    action_values: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    gamma: float,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    full_system_surprise_refresh_threshold: float,
    context_source: str,
    selector_mode: str,
    solved_return: float,
    evaluation_profile: str,
) -> dict[str, MatchedEvalSummary | float | None]:
    """Evaluate one saved checkpoint on a fixed fixture set."""
    if not fixtures:
        return {
            "learned_eval_summary": None,
            "state_only_eval_summary": None,
            "zero_context_eval_summary": None,
            "shuffled_context_eval_summary": None,
            "stale_context_eval_summary": None,
            "no_online_refinement_eval_summary": None,
            "frozen_context_eval_summary": None,
            "actor_only_eval_summary": None,
            "state_only_ablation_delta": None,
            "zero_context_ablation_delta": None,
            "shuffled_context_ablation_delta": None,
            "stale_context_ablation_delta": None,
            "online_refinement_ablation_delta": None,
            "frozen_context_ablation_delta": None,
            "actor_only_ablation_delta": None,
        }

    device = crawler_bundle.device
    eval_env = make_env(env_name)
    try:
        eval_controller = BeliefAffordanceController(
            state_dim=int(np.prod(eval_env.observation_space.shape)),
            action_dim=int(np.prod(eval_env.action_space.shape)),
            mechanics_dim=int(crawler_bundle.z_dim),
            affordance_dim=int(crawler_bundle.z_dim),
            hidden_dim=hidden_dim,
        ).to(device)
    finally:
        eval_env.close()
    eval_controller.load_state_dict(policy_state_dict)
    eval_controller.eval()
    eval_normalizer = restore_normalizer_state(
        int(len(normalizer_state["mean"])),
        normalizer_state,
    )

    learned_returns, _learned_total_steps, _avg_probe_windows, learned_episode_steps = (
        evaluate_belief_affordance_fixtures(
            controller=eval_controller,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            predictor=crawler_bundle.predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=context_source,
            selector_mode=selector_mode,
        )
    )
    learned_summary = _build_matched_eval_summary(
        returns=learned_returns,
        episode_total_steps=learned_episode_steps,
        solved_return=solved_return,
    )
    result: dict[str, MatchedEvalSummary | float | None] = {
        "learned_eval_summary": learned_summary,
        "state_only_eval_summary": None,
        "zero_context_eval_summary": None,
        "shuffled_context_eval_summary": None,
        "stale_context_eval_summary": None,
        "no_online_refinement_eval_summary": None,
        "frozen_context_eval_summary": None,
        "actor_only_eval_summary": None,
        "state_only_ablation_delta": None,
        "zero_context_ablation_delta": None,
        "shuffled_context_ablation_delta": None,
        "stale_context_ablation_delta": None,
        "online_refinement_ablation_delta": None,
        "frozen_context_ablation_delta": None,
        "actor_only_ablation_delta": None,
    }
    if selector_mode != "learned_heads":
        return result

    state_only_returns, _state_only_total_steps, _avg_probe_windows, state_only_episode_steps = (
        evaluate_belief_affordance_fixtures(
            controller=eval_controller,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            predictor=crawler_bundle.predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=context_source,
            selector_mode=selector_mode,
            state_only_student=True,
        )
    )
    zero_returns, _zero_total_steps, _avg_probe_windows, zero_episode_steps = (
        evaluate_belief_affordance_fixtures(
            controller=eval_controller,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            predictor=crawler_bundle.predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=context_source,
            selector_mode=selector_mode,
            disable_context=True,
        )
    )
    stale_returns, _stale_total_steps, _avg_probe_windows, stale_episode_steps = (
        evaluate_belief_affordance_fixtures(
            controller=eval_controller,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            predictor=crawler_bundle.predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=context_source,
            selector_mode=selector_mode,
            use_stale_previous_context=True,
        )
    )
    actor_returns, _actor_total_steps, _avg_probe_windows, actor_episode_steps = (
        evaluate_belief_affordance_fixtures(
            controller=eval_controller,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            predictor=crawler_bundle.predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=context_source,
            selector_mode=selector_mode,
            actor_only=True,
        )
    )
    state_only_summary = _build_matched_eval_summary(
        returns=state_only_returns,
        episode_total_steps=state_only_episode_steps,
        solved_return=solved_return,
    )
    zero_summary = _build_matched_eval_summary(
        returns=zero_returns,
        episode_total_steps=zero_episode_steps,
        solved_return=solved_return,
    )
    stale_summary = _build_matched_eval_summary(
        returns=stale_returns,
        episode_total_steps=stale_episode_steps,
        solved_return=solved_return,
    )
    actor_summary = _build_matched_eval_summary(
        returns=actor_returns,
        episode_total_steps=actor_episode_steps,
        solved_return=solved_return,
    )
    shuffled_returns, _shuffled_total_steps, _avg_probe_windows, shuffled_episode_steps = (
        evaluate_belief_affordance_fixtures(
            controller=eval_controller,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            predictor=crawler_bundle.predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=context_source,
            selector_mode=selector_mode,
            shuffle_context=True,
        )
    )
    shuffled_summary = _build_matched_eval_summary(
        returns=shuffled_returns,
        episode_total_steps=shuffled_episode_steps,
        solved_return=solved_return,
    )
    no_refresh_returns, _no_refresh_total_steps, _avg_probe_windows, no_refresh_episode_steps = (
        evaluate_belief_affordance_fixtures(
            controller=eval_controller,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            predictor=crawler_bundle.predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=context_source,
            selector_mode=selector_mode,
            disable_online_refinement=True,
        )
    )
    no_refresh_summary = _build_matched_eval_summary(
        returns=no_refresh_returns,
        episode_total_steps=no_refresh_episode_steps,
        solved_return=solved_return,
    )
    frozen_returns, _frozen_total_steps, _avg_probe_windows, frozen_episode_steps = (
        evaluate_belief_affordance_fixtures(
            controller=eval_controller,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            predictor=crawler_bundle.predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=context_source,
            selector_mode=selector_mode,
            use_stale_previous_context=True,
            disable_online_refinement=True,
        )
    )
    frozen_summary = _build_matched_eval_summary(
        returns=frozen_returns,
        episode_total_steps=frozen_episode_steps,
        solved_return=solved_return,
    )
    result.update(
        {
            "state_only_eval_summary": state_only_summary,
            "zero_context_eval_summary": zero_summary,
            "shuffled_context_eval_summary": shuffled_summary,
            "stale_context_eval_summary": stale_summary,
            "no_online_refinement_eval_summary": no_refresh_summary,
            "frozen_context_eval_summary": frozen_summary,
            "actor_only_eval_summary": actor_summary,
            "state_only_ablation_delta": _matched_eval_delta(learned_summary, state_only_summary),
            "zero_context_ablation_delta": _matched_eval_delta(learned_summary, zero_summary),
            "shuffled_context_ablation_delta": _matched_eval_delta(learned_summary, shuffled_summary),
            "stale_context_ablation_delta": _matched_eval_delta(learned_summary, stale_summary),
            "online_refinement_ablation_delta": _matched_eval_delta(learned_summary, no_refresh_summary),
            "frozen_context_ablation_delta": _matched_eval_delta(learned_summary, frozen_summary),
            "actor_only_ablation_delta": _matched_eval_delta(learned_summary, actor_summary),
        }
    )
    return result


def _build_failure_debug_bundle(
    *,
    controller: BeliefAffordanceController,
    fixture: EvaluationEpisodeFixture,
    crawler_bundle: CrawlerModelBundle,
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    predictor: DeltaPredictorEnsemble | None,
    state_normalizer: RunningNormalizer,
    env_name: str,
    action_values: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    gamma: float,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    full_system_surprise_refresh_threshold: float,
    context_source: str,
    max_steps: int = 128,
) -> dict[str, object]:
    """Capture one reusable seed-level debug bundle for learned-vs-state-only attribution."""
    env = make_env(env_name)
    adapter = SimulatorFanoutAdapter(env_name)
    label_cache = PersistentFanoutLabelCache(env_name=env_name)
    device = next(controller.parameters()).device
    try:
        apply_env_params(env, fixture.episode_physics)
        raw_state, _info = env.reset(seed=fixture.reset_seed)
        raw_state = np.asarray(raw_state, dtype=np.float32)
        base_context = fixture.learned_context
        if context_source == "oracle" and fixture.oracle_context is not None:
            base_context = fixture.oracle_context
        context_input = sanitize_numpy(base_context.copy())
        belief, belief_hidden, belief_posteriors = _clone_support_state(fixture.support)
        controller_hidden = controller.init_recurrent_state(
            torch.tensor(context_input[None, :], dtype=torch.float32, device=device)
        )
        trace_rows: list[dict[str, object]] = []
        total_steps = 0
        episode_return = 0.0
        done = False

        while not done and total_steps < max(1, int(max_steps)):
            state = sanitize_numpy(state_normalizer.normalize(raw_state))
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            context_t = torch.tensor(context_input[None, :], dtype=torch.float32, device=device)
            baseline_snapshot = adapter.snapshot(env)
            with torch.no_grad():
                mean, _value, next_hidden, aux = controller.forward_with_hidden(
                    state_t,
                    context_t,
                    hidden_state=controller_hidden,
                )
                actor_action = np.asarray(
                    mean_to_action(mean, action_low=action_low, action_high=action_high),
                    dtype=np.float32,
                ).reshape(-1)
                candidate_actions = generate_candidate_actions(
                    mean_action=actor_action,
                    action_low=action_low,
                    action_high=action_high,
                )
                candidate_t = torch.tensor(
                    candidate_actions[None, :, :],
                    dtype=torch.float32,
                    device=device,
                )
                score_outputs = controller.evaluate_candidate_scores(
                    state_trunk=aux["state_trunk"],
                    context_features=aux["context_features"],
                    candidate_actions=candidate_t,
                    trust=aux["trust"],
                    confidence=aux.get("confidence"),
                    uncertainty=aux.get("uncertainty"),
                )
            state_scores = sanitize_numpy(score_outputs["state_scores"].squeeze(0).cpu().numpy())
            final_scores = sanitize_numpy(score_outputs["final_scores"].squeeze(0).cpu().numpy())
            belief_residual = sanitize_numpy(score_outputs["belief_residual"].squeeze(0).cpu().numpy())
            teacher_label = _teacher_label_for_candidates(
                env=env,
                adapter=adapter,
                label_cache=label_cache,
                candidate_actions=candidate_actions,
                gamma=gamma,
                snapshot=baseline_snapshot,
            )
            trust = float(aux["trust"].squeeze(0).item())
            state_best_idx = int(np.argmax(state_scores))
            learned_best_idx = int(np.argmax(final_scores))
            best_candidate = candidate_actions[learned_best_idx].reshape(-1)
            if trust < 0.15:
                action = actor_action.copy()
                controller_used = 0.0
            elif trust < 0.35:
                action = sanitize_numpy(0.5 * actor_action + 0.5 * best_candidate).reshape(-1)
                controller_used = 0.5
            else:
                action = best_candidate.copy()
                controller_used = 1.0
            trace_rows.append(
                {
                    "step": int(total_steps),
                    "state": state.astype(np.float32),
                    "raw_state": raw_state.astype(np.float32),
                    "actor_action": actor_action.astype(np.float32),
                    "selected_action": np.asarray(action, dtype=np.float32),
                    "candidate_actions": candidate_actions.astype(np.float32),
                    "state_scores": state_scores.astype(np.float32),
                    "final_scores": final_scores.astype(np.float32),
                    "teacher_scores": teacher_label.scores.astype(np.float32),
                    "teacher_best_idx": int(teacher_label.best_idx),
                    "state_best_idx": int(state_best_idx),
                    "learned_best_idx": int(learned_best_idx),
                    "teacher_rank_agreement_state_only": bool(state_best_idx == int(teacher_label.best_idx)),
                    "teacher_rank_agreement_learned": bool(learned_best_idx == int(teacher_label.best_idx)),
                    "trust": float(trust),
                    "controller_used": float(controller_used),
                    "residual_norm": float(np.linalg.norm(belief_residual)),
                    "belief_residual": belief_residual.astype(np.float32),
                }
            )

            prev_raw_state = raw_state.copy()
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            raw_state = np.asarray(next_raw_state, dtype=np.float32)
            episode_return += float(reward)
            total_steps += 1
            done = bool(terminated or truncated)
            controller_hidden = next_hidden.detach()

            if fixture.support is None or predictor is None or not belief_posteriors:
                continue
            control_surprise = compute_control_surprise(
                predictor=predictor,
                belief=belief,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                next_state=raw_state,
                device=device,
            )
            belief, belief_hidden, belief_posteriors = maybe_update_online_belief(
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                device=device,
                belief_hidden=belief_hidden,
                belief_posteriors=belief_posteriors,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                reward=float(reward),
                next_state=raw_state,
                belief=belief,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                episode_step=total_steps,
            )
            if (
                total_steps % max(1, int(online_z_update_freq)) != 0
                and float(control_surprise) < float(full_system_surprise_refresh_threshold)
            ):
                continue
            belief, payload = aggregate_env_belief(
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                device=device,
                posterior_views=belief_posteriors,
            )
            refreshed = crawler_bundle.build_step_result(
                payload=payload,
                expected_family_gain={},
                realized_family_gain={},
                stop_reason=None,
                bits_per_dim=0,
                use_residual_sketch=False,
            )
            context_input = sanitize_numpy(
                _controller_context_for_episode(
                    crawler_bundle=crawler_bundle,
                    step_result=refreshed,
                    episode_physics=fixture.episode_physics,
                    context_source=context_source,
                    oracle_weight=1.0 if context_source == "oracle" else 0.0,
                ).vector
            )
            controller_hidden = controller.refresh_recurrent_state(
                torch.tensor(context_input[None, :], dtype=torch.float32, device=device),
                controller_hidden,
            )

        return {
            "reset_seed": int(fixture.reset_seed),
            "episode_return": float(episode_return),
            "episode_steps": int(total_steps),
            "probe_support_steps": 0 if fixture.support is None else int(fixture.support["probe_steps_total"]),
            "probe_support_windows": 0 if fixture.support is None else int(fixture.support["probe_windows_total"]),
            "probe_support_count": 0 if fixture.support is None else int(fixture.support["probe_count"]),
            "support_windows": [] if fixture.support is None else list(fixture.support.get("probe_windows", [])),
            "learned_context": np.asarray(fixture.learned_context, dtype=np.float32),
            "oracle_context": None if fixture.oracle_context is None else np.asarray(fixture.oracle_context, dtype=np.float32),
            "trace_rows": trace_rows,
        }
    finally:
        env.close()


def train_belief_affordance_controller(
    *,
    env_name: str,
    crawler_bundle: CrawlerModelBundle,
    num_episodes: int,
    window_size: int,
    gamma: float,
    lr: float,
    max_grad_norm: float,
    hidden_dim: int,
    normalize_rewards: bool,
    seed: int,
    randomize_physics: bool,
    base_probe_episodes: int,
    max_probe_episodes: int,
    probe_adaptive_budget: bool,
    uncertainty_probe_threshold: float,
    surprise_probe_threshold: float,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    solved_return: float,
    solve_eval_episodes: int,
    run_index: int,
    total_runs: int,
    variant_label: str,
    peer_variant_label: str,
    peer_solved_episode: int | None,
    full_system_online_refinement: bool,
    full_system_surprise_refresh_threshold: float,
    full_system_context_source: str,
    full_system_context_chunk_len: int,
    full_system_curriculum_schedule,
    full_system_plateau_warmup_episodes: int,
    full_system_plateau_patience: int,
    full_system_plateau_best_return_delta: float,
    full_system_plateau_avg50_delta: float,
    full_system_ablation_eval_episodes: int,
    belief_controller_eval_interval: int,
    evaluation_profile: str = "full",
    gae_lambda: float = 0.95,
    ppo_epochs: int = 4,
    minibatch_size: int = 64,
    value_loss_weight: float = 0.5,
    clip_ratio: float = 0.2,
    entropy_coef: float = 2e-3,
    target_kl: float = 0.02,
    use_context: bool = True,
    selector_mode: str = "learned_heads",
    trace_writer=None,
) -> TrainingRunResult:
    """Train the cheap belief-conditioned controller or the sim-fanout baseline."""
    if env_name != CONTINUOUS_CARTPOLE_NAME:
        raise ValueError("Cheap belief-conditioned control is currently CartPole-only")
    train_env = make_env(env_name)
    probe_env = make_env(env_name)
    action_low, action_high = validate_continuous_env(train_env)
    action_values = get_action_values(train_env, crawler_bundle.action_vocab_size, env_name=env_name)
    base_physics = default_env_params(env_name, probe_env)
    device = crawler_bundle.device
    rng = np.random.default_rng(seed)
    curriculum_schedule = normalize_full_system_curriculum_schedule(
        full_system_curriculum_schedule
    )
    evaluation_profile = str(evaluation_profile)
    if evaluation_profile not in {"fast", "full", "archived_planner"}:
        raise ValueError(f"Unsupported evaluation profile: {evaluation_profile}")

    controller = BeliefAffordanceController(
        state_dim=int(np.prod(train_env.observation_space.shape)),
        action_dim=int(np.prod(train_env.action_space.shape)),
        mechanics_dim=int(crawler_bundle.z_dim),
        affordance_dim=int(crawler_bundle.z_dim),
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = optim.Adam(controller.parameters(), lr=lr, eps=1e-5)
    state_normalizer = RunningNormalizer(train_env.observation_space.shape[0], clip=5.0)
    reward_normalizer = RunningNormalizer(1, clip=10.0) if normalize_rewards else None
    adapter = SimulatorFanoutAdapter(env_name)
    label_cache = PersistentFanoutLabelCache(env_name=env_name)

    returns: list[float] = []
    best_return_so_far = float("-inf")
    best_training_policy_state_dict = snapshot_policy_state_dict(controller)
    best_training_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
    best_training_episode = None
    best_avg50_so_far = float("-inf")
    plateau_best_return_marker = float("-inf")
    plateau_best_avg50_marker = float("-inf")
    last_meaningful_progress_episode = None
    solved_episode = None
    solved_env_steps = None
    solve_policy_state_dict = None
    solve_state_normalizer_state = None
    total_env_steps = 0
    total_probe_env_steps = 0
    total_control_env_steps = 0
    total_probe_windows = 0
    trust_history: list[float] = []
    controller_used_history: list[float] = []
    action_divergence_history: list[float] = []
    refresh_count_history: list[float] = []
    rollout_error_history: list[float] = []
    controller_stop_reason = "belief_controller_max_episodes"
    attribution_replay_buffer: deque[dict[str, np.ndarray | float | int]] = deque(maxlen=4096)
    ablation_eval_episodes = max(1, int(full_system_ablation_eval_episodes))
    eval_interval = max(1, int(belief_controller_eval_interval))
    checkpoint_selection_fixtures: list[EvaluationEpisodeFixture] = []
    selected_policy_state_dict = snapshot_policy_state_dict(controller)
    selected_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
    selected_checkpoint_episode = None
    selected_checkpoint_training_return = float("-inf")
    selected_checkpoint_key: tuple[float, float, float, float] | None = None

    if solve_eval_episodes > 0 and use_context and selector_mode == "learned_heads":
        checkpoint_selection_fixtures = _build_evaluation_fixtures(
            crawler_bundle=crawler_bundle,
            encoder=crawler_bundle.encoder,
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            env_future_predictor=crawler_bundle.env_future_predictor,
            predictor=crawler_bundle.predictor,
            env_name=env_name,
            action_values=action_values,
            window_size=window_size,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=surprise_probe_threshold,
            eval_episodes=ablation_eval_episodes,
            seed=seed,
            use_context=use_context,
        )

    if trace_writer is not None:
        trace_writer.set_stage(
            "sim_fanout_training" if selector_mode == "sim_fanout" else "belief_controller_training",
            "Simulator Fan-Out" if selector_mode == "sim_fanout" else "Belief Controller",
            (
                "Scoring a tiny fixed action set on the real simulator as the cheap-sim ceiling."
                if selector_mode == "sim_fanout"
                else "Training a cheap belief-conditioned controller with trust gating and short-horizon affordance heads."
            ),
            run_index=run_index,
            total_runs=total_runs,
            seed=seed,
            variant=variant_label,
        )

    for episode in range(1, num_episodes + 1):
        progress_left = max(0.0, 1.0 - float(episode - 1) / max(float(num_episodes), 1.0))
        set_optimizer_lr(optimizer, lr * progress_left)
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
        support = None
        controller_input = _zero_context_like(crawler_bundle.full_system_controller_dim)
        belief = None
        belief_hidden = None
        belief_posteriors = []
        controller_confidence = 1.0 if (not use_context) else 0.0
        if use_context:
            support = _collect_support_context(
                probe_env=probe_env,
                crawler_bundle=crawler_bundle,
                encoder=crawler_bundle.encoder,
                belief_aggregator=crawler_bundle.belief_aggregator,
                env_param_predictor=crawler_bundle.env_param_predictor,
                env_future_predictor=crawler_bundle.env_future_predictor,
                predictor=crawler_bundle.predictor,
                rng=rng,
                env_name=env_name,
                episode_physics=episode_physics,
                action_values=action_values,
                window_size=window_size,
                base_probe_episodes=base_probe_episodes,
                max_probe_episodes=max_probe_episodes,
                probe_adaptive_budget=probe_adaptive_budget,
                uncertainty_probe_threshold=uncertainty_probe_threshold,
                surprise_probe_threshold=surprise_probe_threshold,
                trace_writer=None,
                episode=episode,
                variant_label=variant_label,
            )
            if support is None:
                returns.append(0.0)
                continue
            total_probe_env_steps += int(support["probe_steps_total"])
            total_probe_windows += int(support["probe_windows_total"])
            total_env_steps += int(support["probe_steps_total"])
            belief = support["belief"]
            belief_hidden = support["belief_hidden"]
            belief_posteriors = list(support["belief_posteriors"])
            oracle_weight = full_system_oracle_weight_for_episode(
                context_source=full_system_context_source,
                current_episode=episode,
                curriculum_schedule=curriculum_schedule,
            )
            selected_context = _controller_context_for_episode(
                crawler_bundle=crawler_bundle,
                step_result=support["step_result"],
                episode_physics=episode_physics,
                context_source=full_system_context_source,
                oracle_weight=oracle_weight,
            )
            controller_input = sanitize_numpy(selected_context.vector)
            controller_confidence = float(selected_context.confidence)

        apply_env_params(train_env, episode_physics)
        raw_state, _info = train_env.reset()
        raw_state = np.asarray(raw_state, dtype=np.float32)
        state_normalizer.update(raw_state)
        controller_hidden = controller.init_recurrent_state(
            torch.tensor(controller_input[None, :], dtype=torch.float32, device=device)
        )
        episode_states = []
        episode_contexts = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_values = []
        episode_terminals = []
        episode_hidden = []
        episode_candidate_actions = []
        episode_candidate_returns = []
        episode_candidate_risks = []
        episode_candidate_recoverability = []
        episode_candidate_scores = []
        episode_candidate_best_idx = []
        episode_candidate_best_vs_actor_margin = []
        episode_context_confidences = []
        episode_return = 0.0
        episode_step = 0
        refresh_count = 0
        done = False
        last_next_state = sanitize_numpy(state_normalizer.normalize(raw_state))
        last_terminated = False

        while not done:
            state = sanitize_numpy(state_normalizer.normalize(raw_state))
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            context_t = torch.tensor(controller_input[None, :], dtype=torch.float32, device=device)
            hidden_input = sanitize_numpy(controller_hidden.detach().cpu().numpy().squeeze(0))
            baseline_snapshot = adapter.snapshot(train_env)
            if selector_mode == "sim_fanout":
                action, mean, value, next_hidden, runtime = _select_sim_fanout_action(
                    controller=controller,
                    env=train_env,
                    adapter=adapter,
                    label_cache=label_cache,
                    state_t=state_t,
                    context_t=context_t,
                    action_low=action_low,
                    action_high=action_high,
                    hidden_state=controller_hidden,
                    gamma=gamma,
                    snapshot=baseline_snapshot,
                )
            else:
                selection = choose_affordance_action(
                    controller=controller,
                    state_t=state_t,
                    context_t=context_t,
                    action_low=action_low,
                    action_high=action_high,
                    hidden_state=controller_hidden,
                )
                action = selection.action
                mean = selection.mean
                value = selection.value
                next_hidden = selection.next_hidden
                runtime = {
                    "trust": selection.trust,
                    "controller_used": selection.controller_used,
                    "action_divergence": selection.action_divergence,
                    "candidate_actions": selection.candidate_actions,
                    "candidate_returns": selection.candidate_returns,
                    "candidate_risks": selection.candidate_risks,
                    "candidate_recoverability": selection.candidate_recoverability,
                    "candidate_scores": selection.candidate_scores,
                    "best_vs_actor_margin": float(
                        selection.candidate_scores[selection.best_idx] - selection.candidate_scores[0]
                    ),
                    "best_idx": selection.best_idx,
                }
            candidate_labels = _teacher_label_for_candidates(
                env=train_env,
                adapter=adapter,
                label_cache=label_cache,
                candidate_actions=runtime["candidate_actions"],
                gamma=gamma,
                snapshot=baseline_snapshot,
            )
            controller_hidden = next_hidden
            action_t = torch.tensor(action[None, :], dtype=torch.float32, device=device)
            log_prob, _entropy = evaluate_continuous_actions(
                mean=mean,
                log_std=controller.log_std,
                actions=action_t,
                action_low=action_low,
                action_high=action_high,
            )

            prev_raw_state = raw_state.copy()
            next_raw_state, reward, terminated, truncated, _info = train_env.step(action)
            raw_state = np.asarray(next_raw_state, dtype=np.float32)
            total_env_steps += 1
            total_control_env_steps += 1
            episode_step += 1
            raw_reward = float(reward)
            train_reward = raw_reward
            if reward_normalizer is not None:
                reward_normalizer.update(np.asarray([[raw_reward]], dtype=np.float32))
                train_reward = float(
                    reward_normalizer.scale_only(np.asarray([raw_reward], dtype=np.float32))[0]
                )
            episode_states.append(state.copy())
            episode_contexts.append(controller_input.copy())
            episode_actions.append(action.copy())
            episode_log_probs.append(float(log_prob.item()))
            episode_rewards.append(train_reward)
            episode_values.append(float(value.item()))
            episode_terminals.append(float(terminated))
            episode_hidden.append(hidden_input.copy())
            episode_candidate_actions.append(runtime["candidate_actions"].copy())
            episode_candidate_returns.append(sanitize_numpy(candidate_labels.returns))
            episode_candidate_risks.append(sanitize_numpy(candidate_labels.risks))
            episode_candidate_recoverability.append(
                sanitize_numpy(candidate_labels.recoverability)
            )
            episode_candidate_scores.append(sanitize_numpy(candidate_labels.scores))
            episode_candidate_best_idx.append(int(candidate_labels.best_idx))
            episode_candidate_best_vs_actor_margin.append(float(candidate_labels.best_vs_actor_margin))
            episode_context_confidences.append(float(controller_confidence if use_context else 0.0))
            trust_history.append(float(runtime["trust"]))
            controller_used_history.append(float(runtime["controller_used"]))
            action_divergence_history.append(float(runtime["action_divergence"]))
            rollout_error_history.append(
                float(
                    np.mean(
                        np.abs(
                            np.asarray(runtime["candidate_returns"], dtype=np.float32)
                            - np.asarray(candidate_labels.returns, dtype=np.float32)
                        )
                    )
                )
            )
            state_normalizer.update(raw_state)
            last_next_state = sanitize_numpy(state_normalizer.normalize(raw_state))
            last_terminated = bool(terminated)
            episode_return += raw_reward
            done = bool(terminated or truncated)

            if use_context and belief_posteriors and full_system_online_refinement:
                control_surprise = compute_control_surprise(
                    predictor=crawler_bundle.predictor,
                    belief=belief,
                    prev_state=prev_raw_state,
                    action_idx=nearest_probe_action_idx(action, action_values),
                    next_state=raw_state,
                    device=device,
                )
                belief, belief_hidden, belief_posteriors = maybe_update_online_belief(
                    encoder=crawler_bundle.encoder,
                    belief_aggregator=crawler_bundle.belief_aggregator,
                    env_param_predictor=crawler_bundle.env_param_predictor,
                    device=device,
                    belief_hidden=belief_hidden,
                    belief_posteriors=belief_posteriors,
                    prev_state=prev_raw_state,
                    action_idx=nearest_probe_action_idx(action, action_values),
                    reward=raw_reward,
                    next_state=raw_state,
                    belief=belief,
                    online_z_update_alpha=online_z_update_alpha,
                    online_z_update_freq=online_z_update_freq,
                    episode_step=episode_step,
                )
                if (
                    episode_step % max(1, int(online_z_update_freq)) == 0
                    or float(control_surprise) >= float(full_system_surprise_refresh_threshold)
                ):
                    belief, payload = aggregate_env_belief(
                        belief_aggregator=crawler_bundle.belief_aggregator,
                        env_param_predictor=crawler_bundle.env_param_predictor,
                        device=device,
                        posterior_views=belief_posteriors,
                    )
                    refreshed_step = crawler_bundle.build_step_result(
                        payload=payload,
                        expected_family_gain={},
                        realized_family_gain={},
                        stop_reason=None,
                        bits_per_dim=0,
                        use_residual_sketch=False,
                    )
                    oracle_weight = full_system_oracle_weight_for_episode(
                        context_source=full_system_context_source,
                        current_episode=episode,
                        curriculum_schedule=curriculum_schedule,
                    )
                    refreshed_context = _controller_context_for_episode(
                        crawler_bundle=crawler_bundle,
                        step_result=refreshed_step,
                        episode_physics=episode_physics,
                        context_source=full_system_context_source,
                        oracle_weight=oracle_weight,
                    )
                    controller_input = sanitize_numpy(refreshed_context.vector)
                    controller_confidence = float(refreshed_context.confidence)
                    controller_hidden = controller.refresh_recurrent_state(
                        torch.tensor(controller_input[None, :], dtype=torch.float32, device=device),
                        controller_hidden,
                    )
                    refresh_count += 1

        refresh_count_history.append(float(refresh_count))
        returns.append(float(episode_return))
        avg_10 = float(np.mean(returns[-10:]))
        avg_50 = float(np.mean(returns[-50:]))
        new_best = bool(episode_return >= best_return_so_far)
        if new_best:
            best_return_so_far = float(episode_return)
            best_training_policy_state_dict = snapshot_policy_state_dict(controller)
            best_training_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
            best_training_episode = int(episode)
        best_avg50_so_far = max(best_avg50_so_far, avg_50)
        if episode == 1 or episode_return >= plateau_best_return_marker + float(full_system_plateau_best_return_delta):
            plateau_best_return_marker = float(episode_return)
            last_meaningful_progress_episode = int(episode)
        if episode == 1 or avg_50 >= plateau_best_avg50_marker + float(full_system_plateau_avg50_delta):
            plateau_best_avg50_marker = float(avg_50)
            last_meaningful_progress_episode = int(episode)

        if episode_states:
            batch = build_episode_batch(
                states=episode_states,
                actions=episode_actions,
                log_probs=episode_log_probs,
                rewards=episode_rewards,
                values=episode_values,
                terminals=episode_terminals,
                bootstrap_value=0.0 if last_terminated else float(episode_values[-1]),
                gamma=gamma,
                gae_lambda=gae_lambda,
                beliefs=episode_contexts,
                recurrent_hidden_states=episode_hidden,
                sequence_length=full_system_context_chunk_len,
            )
            _update_affordance_controller(
                controller=controller,
                optimizer=optimizer,
                batch=batch,
                candidate_actions=episode_candidate_actions,
                candidate_returns=episode_candidate_returns,
                candidate_risks=episode_candidate_risks,
                candidate_recoverability=episode_candidate_recoverability,
                candidate_scores=episode_candidate_scores,
                candidate_best_idx=episode_candidate_best_idx,
                candidate_best_vs_actor_margin=episode_candidate_best_vs_actor_margin,
                context_confidences=episode_context_confidences,
                action_low=action_low,
                action_high=action_high,
                clip_ratio=clip_ratio,
                value_loss_weight=value_loss_weight,
                entropy_coef=entropy_coef,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                max_grad_norm=max_grad_norm,
                target_kl=target_kl,
                sequence_length=full_system_context_chunk_len,
                current_episode=episode,
                total_episodes=num_episodes,
            )
            if selector_mode == "learned_heads":
                for step_idx in range(len(episode_states)):
                    attribution_replay_buffer.append(
                        {
                            "state": sanitize_numpy(episode_states[step_idx]),
                            "context": sanitize_numpy(episode_contexts[step_idx]),
                            "candidate_actions": sanitize_numpy(episode_candidate_actions[step_idx]),
                            "candidate_scores": sanitize_numpy(episode_candidate_scores[step_idx]),
                            "candidate_best_idx": int(episode_candidate_best_idx[step_idx]),
                        }
                    )
        if checkpoint_selection_fixtures and (
            episode % eval_interval == 0 or episode == num_episodes
        ):
            current_policy_state = snapshot_policy_state_dict(controller)
            current_normalizer_state = snapshot_normalizer_state(state_normalizer)
            checkpoint_eval = _evaluate_affordance_checkpoint(
                policy_state_dict=current_policy_state,
                normalizer_state=current_normalizer_state,
                fixtures=checkpoint_selection_fixtures,
                crawler_bundle=crawler_bundle,
                hidden_dim=hidden_dim,
                env_name=env_name,
                action_values=action_values,
                action_low=action_low,
                action_high=action_high,
                gamma=gamma,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
                context_source="learned" if full_system_context_source == "curriculum" else full_system_context_source,
                selector_mode=selector_mode,
                solved_return=solved_return,
                evaluation_profile=evaluation_profile,
            )
            candidate_key = _checkpoint_selection_key(
                learned_summary=checkpoint_eval["learned_eval_summary"],
                state_only_summary=checkpoint_eval["state_only_eval_summary"],
                zero_context_summary=checkpoint_eval["zero_context_eval_summary"],
                stale_context_summary=checkpoint_eval["stale_context_eval_summary"],
                shuffled_context_summary=checkpoint_eval["shuffled_context_eval_summary"],
                training_return=episode_return,
            )
            if selected_checkpoint_key is None or candidate_key > selected_checkpoint_key:
                selected_checkpoint_key = candidate_key
                selected_policy_state_dict = current_policy_state
                selected_state_normalizer_state = current_normalizer_state
                selected_checkpoint_episode = int(episode)
                selected_checkpoint_training_return = float(episode_return)
        if (
            selector_mode == "learned_heads"
            and _affordance_training_stage(
                current_episode=episode,
                total_episodes=num_episodes,
            )
            != "state_student"
        ):
            _run_attribution_aux_update(
                controller=controller,
                optimizer=optimizer,
                replay_buffer=attribution_replay_buffer,
                minibatch_size=minibatch_size,
                max_grad_norm=max_grad_norm,
            )
        if trace_writer is not None:
            trace_writer.record_episode_summary(
                variant=variant_label,
                episode=episode,
                episode_return=episode_return,
                avg10=avg_10,
                avg50=avg_50,
                total_env_steps=total_env_steps,
                probe_steps=0 if support is None else int(support["probe_steps_total"]),
                probe_count=0 if support is None else int(support["probe_count"]),
                uncertainty=(
                    None
                    if support is None
                    else float(support["step_result"].controller_context.uncertainty_scalar)
                ),
                expression_confidence=float(controller_confidence),
                expression_ready=bool((not use_context) or controller_confidence > 0.0),
            )

        episode_window = max(1, len(episode_states))
        trust_mean = float(np.mean(trust_history[-episode_window:]))
        controller_used_mean = float(np.mean(controller_used_history[-episode_window:]))
        print_belief_episode_status(
            run_index=run_index,
            total_runs=total_runs,
            seed=seed,
            variant_label=variant_label,
            episode=episode,
            episode_return=float(episode_return),
            avg10=avg_10,
            best_return=float(best_return_so_far),
            total_env_steps=int(total_env_steps),
            probe_count=0 if support is None else int(support["probe_count"]),
            episode_probe_steps=0 if support is None else int(support["probe_steps_total"]),
            trust=trust_mean,
            usage=controller_used_mean,
            usage_label="ctrl",
            refresh_count=int(refresh_count),
            avg50=avg_50,
            target_return=float(solved_return),
            solved_episode=solved_episode,
            peer_status=format_peer_solve_status(peer_variant_label, peer_solved_episode),
            new_best=new_best,
        )
        if episode_return >= solved_return:
            solved_episode = int(episode)
            solved_env_steps = int(total_env_steps)
            solve_policy_state_dict = snapshot_policy_state_dict(controller)
            solve_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
            print_solve_event(
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant_label=variant_label,
                episode=episode,
                total_env_steps=total_env_steps,
                episode_return=float(episode_return),
                probe_count=0 if support is None else int(support["probe_count"]),
            )
            controller_stop_reason = "belief_controller_solved"
            break
        if should_stop_belief_planner_plateau(
            current_episode=episode,
            warmup_episodes=full_system_plateau_warmup_episodes,
            patience=full_system_plateau_patience,
            last_meaningful_progress_episode=last_meaningful_progress_episode,
        ):
            controller_stop_reason = "belief_controller_plateau"
            break

    solve_eval_returns = None
    learned_eval_summary = None
    zero_context_eval_returns = None
    shuffled_context_eval_returns = None
    stale_context_eval_returns = None
    no_online_refinement_eval_returns = None
    frozen_context_eval_returns = None
    actor_only_eval_returns = None
    state_only_eval_returns = None
    zero_context_eval_summary = None
    shuffled_context_eval_summary = None
    stale_context_eval_summary = None
    no_online_refinement_eval_summary = None
    frozen_context_eval_summary = None
    actor_only_eval_summary = None
    state_only_eval_summary = None
    zero_context_ablation_delta = None
    shuffled_context_ablation_delta = None
    stale_context_ablation_delta = None
    online_refinement_ablation_delta = None
    frozen_context_ablation_delta = None
    actor_only_ablation_delta = None
    state_only_ablation_delta = None
    state_only_solved_episode = None
    state_only_solved_env_steps = None
    state_only_total_env_steps = None
    state_only_completed_episodes = None
    failure_debug_bundle = None

    best_policy_state_dict = selected_policy_state_dict
    best_state_normalizer_state = selected_state_normalizer_state
    best_episode = selected_checkpoint_episode
    if best_episode is None:
        best_policy_state_dict = best_training_policy_state_dict
        best_state_normalizer_state = best_training_state_normalizer_state
        best_episode = best_training_episode

    if solve_eval_episodes > 0 and best_episode is not None:
        fixtures = checkpoint_selection_fixtures
        if not fixtures:
            fixtures = _build_evaluation_fixtures(
                crawler_bundle=crawler_bundle,
                encoder=crawler_bundle.encoder,
                belief_aggregator=crawler_bundle.belief_aggregator,
                env_param_predictor=crawler_bundle.env_param_predictor,
                env_future_predictor=crawler_bundle.env_future_predictor,
                predictor=crawler_bundle.predictor,
                env_name=env_name,
                action_values=action_values,
                window_size=window_size,
                randomize_physics=randomize_physics,
                base_physics=base_physics,
                base_probe_episodes=base_probe_episodes,
                max_probe_episodes=max_probe_episodes,
                probe_adaptive_budget=probe_adaptive_budget,
                uncertainty_probe_threshold=uncertainty_probe_threshold,
                surprise_probe_threshold=surprise_probe_threshold,
                eval_episodes=ablation_eval_episodes,
                seed=seed,
                use_context=use_context,
            )
        eval_context_source = "learned" if full_system_context_source == "curriculum" else full_system_context_source
        checkpoint_eval = _evaluate_affordance_checkpoint(
            policy_state_dict=best_policy_state_dict,
            normalizer_state=best_state_normalizer_state,
            fixtures=fixtures,
            crawler_bundle=crawler_bundle,
            hidden_dim=hidden_dim,
            env_name=env_name,
            action_values=action_values,
            action_low=action_low,
            action_high=action_high,
            gamma=gamma,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
            context_source=eval_context_source,
            selector_mode=selector_mode,
            solved_return=solved_return,
            evaluation_profile=evaluation_profile,
        )
        learned_eval_summary = checkpoint_eval["learned_eval_summary"]
        state_only_eval_summary = checkpoint_eval["state_only_eval_summary"]
        zero_context_eval_summary = checkpoint_eval["zero_context_eval_summary"]
        shuffled_context_eval_summary = checkpoint_eval["shuffled_context_eval_summary"]
        stale_context_eval_summary = checkpoint_eval["stale_context_eval_summary"]
        no_online_refinement_eval_summary = checkpoint_eval["no_online_refinement_eval_summary"]
        frozen_context_eval_summary = checkpoint_eval["frozen_context_eval_summary"]
        actor_only_eval_summary = checkpoint_eval["actor_only_eval_summary"]
        solve_eval_returns = None if learned_eval_summary is None else list(learned_eval_summary.returns)
        state_only_eval_returns = None if state_only_eval_summary is None else list(state_only_eval_summary.returns)
        zero_context_eval_returns = None if zero_context_eval_summary is None else list(zero_context_eval_summary.returns)
        shuffled_context_eval_returns = None if shuffled_context_eval_summary is None else list(shuffled_context_eval_summary.returns)
        stale_context_eval_returns = None if stale_context_eval_summary is None else list(stale_context_eval_summary.returns)
        no_online_refinement_eval_returns = (
            None
            if no_online_refinement_eval_summary is None
            else list(no_online_refinement_eval_summary.returns)
        )
        frozen_context_eval_returns = (
            None
            if frozen_context_eval_summary is None
            else list(frozen_context_eval_summary.returns)
        )
        actor_only_eval_returns = None if actor_only_eval_summary is None else list(actor_only_eval_summary.returns)
        state_only_ablation_delta = checkpoint_eval["state_only_ablation_delta"]
        zero_context_ablation_delta = checkpoint_eval["zero_context_ablation_delta"]
        shuffled_context_ablation_delta = checkpoint_eval["shuffled_context_ablation_delta"]
        stale_context_ablation_delta = checkpoint_eval["stale_context_ablation_delta"]
        online_refinement_ablation_delta = checkpoint_eval["online_refinement_ablation_delta"]
        frozen_context_ablation_delta = checkpoint_eval["frozen_context_ablation_delta"]
        actor_only_ablation_delta = checkpoint_eval["actor_only_ablation_delta"]
        if state_only_eval_summary is not None:
            (
                state_only_solved_episode,
                state_only_solved_env_steps,
                state_only_total_env_steps,
                state_only_completed_episodes,
            ) = _summarize_eval_solve(
                returns=state_only_eval_summary.returns,
                episode_total_steps=state_only_eval_summary.episode_total_env_steps,
                solved_return=solved_return,
            )
        if (
            int(seed) == 0
            and use_context
            and selector_mode == "learned_heads"
            and fixtures
        ):
            eval_controller = BeliefAffordanceController(
                state_dim=int(np.prod(train_env.observation_space.shape)),
                action_dim=int(np.prod(train_env.action_space.shape)),
                mechanics_dim=int(crawler_bundle.z_dim),
                affordance_dim=int(crawler_bundle.z_dim),
                hidden_dim=hidden_dim,
            ).to(device)
            eval_controller.load_state_dict(best_policy_state_dict)
            eval_controller.eval()
            eval_normalizer = restore_normalizer_state(
                int(np.prod(train_env.observation_space.shape)),
                best_state_normalizer_state,
            )
            failure_debug_bundle = _build_failure_debug_bundle(
                controller=eval_controller,
                fixture=fixtures[0],
                crawler_bundle=crawler_bundle,
                encoder=crawler_bundle.encoder,
                belief_aggregator=crawler_bundle.belief_aggregator,
                env_param_predictor=crawler_bundle.env_param_predictor,
                predictor=crawler_bundle.predictor,
                state_normalizer=eval_normalizer,
                env_name=env_name,
                action_values=action_values,
                action_low=action_low,
                action_high=action_high,
                gamma=gamma,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                full_system_surprise_refresh_threshold=full_system_surprise_refresh_threshold,
                context_source=eval_context_source,
            )

    train_env.close()
    probe_env.close()
    return TrainingRunResult(
        policy=controller,
        returns=returns,
        state_normalizer=state_normalizer,
        solved_episode=solved_episode,
        solved_env_steps=solved_env_steps,
        total_env_steps=total_env_steps,
        best_policy_state_dict=best_policy_state_dict,
        best_state_normalizer_state=best_state_normalizer_state,
        best_return=best_return_so_far if np.isfinite(best_return_so_far) else 0.0,
        best_episode=best_episode,
        solve_policy_state_dict=solve_policy_state_dict,
        solve_state_normalizer_state=solve_state_normalizer_state,
        solve_eval_returns=solve_eval_returns,
        solve_probe_count=0 if not use_context else base_probe_episodes,
        probe_env_steps_total=total_probe_env_steps,
        control_env_steps_total=total_control_env_steps,
        probe_windows_total=total_probe_windows,
        probe_stop_reasons={} if not use_context else {"fair_two_probe_handoff": len(returns)},
        probe_family_expected_gain=None,
        probe_family_realized_gain=None,
        probe_family_future_error=None,
        probe_family_selection_count=None,
        last_probe_stop_reason=None if not use_context else "fair_two_probe_handoff",
        solve_probe_stop_reason=None if not use_context else "fair_two_probe_handoff",
        post_expression_env_steps_total=total_control_env_steps,
        post_expression_episode_count=solved_episode,
        controller_style=(
            "sim_fanout_state_only"
            if selector_mode == "sim_fanout"
            else f"belief_controller_state_residual_{full_system_context_source}"
        ),
        learned_eval_summary=learned_eval_summary,
        zero_context_eval_returns=zero_context_eval_returns,
        shuffled_context_eval_returns=shuffled_context_eval_returns,
        stale_context_eval_returns=stale_context_eval_returns,
        no_online_refinement_eval_returns=no_online_refinement_eval_returns,
        frozen_context_eval_returns=frozen_context_eval_returns,
        zero_context_eval_summary=zero_context_eval_summary,
        shuffled_context_eval_summary=shuffled_context_eval_summary,
        stale_context_eval_summary=stale_context_eval_summary,
        no_online_refinement_eval_summary=no_online_refinement_eval_summary,
        frozen_context_eval_summary=frozen_context_eval_summary,
        zero_context_ablation_delta=zero_context_ablation_delta,
        shuffled_context_ablation_delta=shuffled_context_ablation_delta,
        stale_context_ablation_delta=stale_context_ablation_delta,
        online_refinement_ablation_delta=online_refinement_ablation_delta,
        frozen_context_ablation_delta=frozen_context_ablation_delta,
        actor_only_eval_returns=actor_only_eval_returns,
        actor_only_eval_summary=actor_only_eval_summary,
        actor_only_ablation_delta=actor_only_ablation_delta,
        state_only_eval_returns=state_only_eval_returns,
        state_only_eval_summary=state_only_eval_summary,
        state_only_ablation_delta=state_only_ablation_delta,
        state_only_solved_episode=state_only_solved_episode,
        state_only_solved_env_steps=state_only_solved_env_steps,
        state_only_total_env_steps=state_only_total_env_steps,
        state_only_completed_episodes=state_only_completed_episodes,
        planner_trust_usage_rate=(
            float(np.mean(np.asarray(controller_used_history, dtype=np.float32)))
            if controller_used_history
            else None
        ),
        actor_planner_action_divergence=(
            float(np.mean(np.asarray(action_divergence_history, dtype=np.float32)))
            if action_divergence_history
            else None
        ),
        rollout_model_error_mean=(
            float(np.mean(np.asarray(rollout_error_history, dtype=np.float32)))
            if rollout_error_history
            else None
        ),
        refresh_count_mean=(
            float(np.mean(np.asarray(refresh_count_history, dtype=np.float32)))
            if refresh_count_history
            else None
        ),
        extra_checkpoint_data={
            "selector_mode": selector_mode,
            "use_context": bool(use_context),
            "evaluation_profile": evaluation_profile,
            "stop_reason": controller_stop_reason,
            "selected_checkpoint_episode": best_episode,
            "selected_checkpoint_training_return": (
                None if not np.isfinite(selected_checkpoint_training_return) else selected_checkpoint_training_return
            ),
            "training_best_episode": best_training_episode,
            "training_best_return": (
                None if not np.isfinite(best_return_so_far) else float(best_return_so_far)
            ),
            "learned_eval_summary": (
                None if learned_eval_summary is None else learned_eval_summary.to_dict()
            ),
            "state_only_eval_summary": (
                None if state_only_eval_summary is None else state_only_eval_summary.to_dict()
            ),
            "zero_context_eval_summary": (
                None if zero_context_eval_summary is None else zero_context_eval_summary.to_dict()
            ),
            "shuffled_context_eval_summary": (
                None if shuffled_context_eval_summary is None else shuffled_context_eval_summary.to_dict()
            ),
            "stale_context_eval_summary": (
                None if stale_context_eval_summary is None else stale_context_eval_summary.to_dict()
            ),
            "no_online_refinement_eval_summary": (
                None
                if no_online_refinement_eval_summary is None
                else no_online_refinement_eval_summary.to_dict()
            ),
            "frozen_context_eval_summary": (
                None
                if frozen_context_eval_summary is None
                else frozen_context_eval_summary.to_dict()
            ),
            "actor_only_eval_summary": (
                None if actor_only_eval_summary is None else actor_only_eval_summary.to_dict()
            ),
            "state_only_ablation_delta": state_only_ablation_delta,
            "zero_context_ablation_delta": zero_context_ablation_delta,
            "shuffled_context_ablation_delta": shuffled_context_ablation_delta,
            "stale_context_ablation_delta": stale_context_ablation_delta,
            "online_refinement_ablation_delta": online_refinement_ablation_delta,
            "frozen_context_ablation_delta": frozen_context_ablation_delta,
            "actor_only_ablation_delta": actor_only_ablation_delta,
            "training_stage_final": _affordance_training_stage(
                current_episode=len(returns),
                total_episodes=num_episodes,
            ),
            "seed0_failure_debug_bundle": failure_debug_bundle,
        },
    )


__all__ = [
    "EvaluationEpisodeFixture",
    "evaluate_belief_affordance_fixtures",
    "train_belief_affordance_controller",
]
