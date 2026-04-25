"""Belief-planner training and evaluation for the breakthrough full-system path."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.optim as optim

from ...crawler import CrawlerModelBundle
from ...crawler.types import ControllerBeliefContext
from ...envs import get_action_values, make_env
from ...models.env_belief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ...probe.explorer import build_probe_planner
from ...probe.probe_data import apply_env_params, default_env_params
from ...probe.probe_latent import (
    aggregate_env_belief,
    collect_adaptive_probe_window,
    encode_window_posterior,
    init_recurrent_belief_hidden,
    maybe_update_online_belief,
    nearest_probe_action_idx,
    sanitize_belief_vector,
    select_episode_physics,
    update_recurrent_belief_from_window,
)
from ...representation import DeltaPredictorEnsemble, WorldEncoder
from .planner import (
    BeliefDynamicsModel,
    BeliefPlannerConfig,
    PlanningBeliefState,
    build_planner_probe_dataset,
    fit_belief_dynamics_model,
    plan_cem_action,
    replay_batch_mean_error,
    update_belief_dynamics_from_replay,
    update_planner_prior,
)
from .objectives import planner_prediction_surprise, surprise_z_score
from .context_support import (
    collect_support_context as _shared_collect_support_context,
    controller_context_for_episode as _shared_controller_context_for_episode,
    mix_controller_contexts as _shared_mix_controller_contexts,
)
from .curriculum import (
    DEFAULT_FULL_SYSTEM_CURRICULUM_SCHEDULE,
    full_system_oracle_weight_for_episode as _shared_full_system_oracle_weight_for_episode,
    normalize_oracle_curriculum_schedule,
    should_stop_belief_planner_plateau as _shared_should_stop_belief_planner_plateau,
)
from ..core.ppo_core import (
    BeliefNativeActorCritic,
    RunningNormalizer,
    mean_to_continuous_action,
    sanitize_numpy,
    set_optimizer_lr,
    validate_continuous_env,
)
from ..probe_policy.budget import choose_fair_probe_family
from ..probe_policy.eval import (
    compute_control_surprise,
    compute_probe_surprise,
    transform_controller_context_input,
)
from ..probe_policy.logging import print_belief_episode_status, print_solve_event
from ..probe_policy.reporting import (
    format_peer_solve_status,
    restore_normalizer_state,
    snapshot_normalizer_state,
    snapshot_policy_state_dict,
)
from ..probe_policy.types import TrainingRunResult


def normalize_full_system_curriculum_schedule(
    schedule: list[tuple[int, float]] | tuple[tuple[int, float], ...] | None,
) -> tuple[tuple[int, float], ...]:
    """Return one explicit oracle-weight schedule for curriculum mode."""
    return normalize_oracle_curriculum_schedule(
        schedule,
        default_schedule=DEFAULT_FULL_SYSTEM_CURRICULUM_SCHEDULE,
    )


def full_system_oracle_weight_for_episode(
    *,
    context_source: str,
    current_episode: int,
    curriculum_schedule: tuple[tuple[int, float], ...],
) -> float:
    """Resolve oracle context weight for this episode."""
    return _shared_full_system_oracle_weight_for_episode(
        context_source=context_source,
        current_episode=current_episode,
        curriculum_schedule=curriculum_schedule,
    )


def mix_controller_contexts(
    learned_context: ControllerBeliefContext,
    oracle_context: ControllerBeliefContext,
    *,
    oracle_weight: float,
) -> ControllerBeliefContext:
    """Blend learned and oracle context for curriculum training."""
    return _shared_mix_controller_contexts(
        learned_context,
        oracle_context,
        oracle_weight=oracle_weight,
    )


def _cartpole_recoverability(next_state: np.ndarray) -> float:
    state = sanitize_numpy(np.asarray(next_state, dtype=np.float32).reshape(-1))
    if state.shape[0] < 4:
        return 1.0
    x = abs(float(state[0]))
    dx = abs(float(state[1]))
    theta = abs(float(state[2]))
    dtheta = abs(float(state[3]))
    centered = 1.0 - np.clip(x / 2.4, 0.0, 1.0)
    upright = 1.0 - np.clip(theta / 0.35, 0.0, 1.0)
    calm = 1.0 / (1.0 + dx + dtheta)
    return float(np.clip(0.45 * upright + 0.35 * centered + 0.20 * calm, 0.0, 1.0))


def should_stop_belief_planner_plateau(
    *,
    current_episode: int,
    warmup_episodes: int,
    patience: int,
    last_meaningful_progress_episode: int | None,
) -> bool:
    """Return whether the planner has stalled long enough to stop early."""
    return _shared_should_stop_belief_planner_plateau(
        current_episode=current_episode,
        warmup_episodes=warmup_episodes,
        patience=patience,
        last_meaningful_progress_episode=last_meaningful_progress_episode,
    )


def _discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros((len(rewards),), dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + float(gamma) * running
        returns[idx] = running
    return returns


def _planner_sequence_batch(
    *,
    states: list[np.ndarray],
    contexts: list[np.ndarray],
    actions: list[np.ndarray],
    rewards: list[float],
    hidden_states: list[np.ndarray],
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    state_arr = sanitize_numpy(np.asarray(states, dtype=np.float32))[None, :, :]
    context_arr = sanitize_numpy(np.asarray(contexts, dtype=np.float32))[None, :, :]
    action_arr = sanitize_numpy(np.asarray(actions, dtype=np.float32))[None, :, :]
    return_arr = _discounted_returns(rewards, gamma=gamma)[None, :]
    if not hidden_states:
        return state_arr, context_arr, action_arr, return_arr, None
    hidden_arr = sanitize_numpy(np.asarray(hidden_states, dtype=np.float32))[None, :, :]
    return state_arr, context_arr, action_arr, return_arr, hidden_arr


def _collect_support_context(
    *,
    probe_env,
    crawler_bundle: CrawlerModelBundle,
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    env_future_predictor,
    predictor: DeltaPredictorEnsemble | None,
    rng: np.random.Generator,
    env_name: str,
    episode_physics,
    action_values: np.ndarray,
    window_size: int,
    base_probe_episodes: int,
    max_probe_episodes: int,
    probe_adaptive_budget: bool,
    uncertainty_probe_threshold: float,
    surprise_probe_threshold: float,
    trace_writer,
    episode: int,
    variant_label: str,
) -> dict | None:
    """Collect one compact support set and return the latest crawler step result."""
    return _shared_collect_support_context(
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
        trace_writer=trace_writer,
        episode=episode,
        variant_label=variant_label,
    )


def _controller_context_for_episode(
    *,
    crawler_bundle: CrawlerModelBundle,
    step_result,
    episode_physics,
    context_source: str,
    oracle_weight: float,
) -> ControllerBeliefContext:
    return _shared_controller_context_for_episode(
        crawler_bundle=crawler_bundle,
        step_result=step_result,
        episode_physics=episode_physics,
        context_source=context_source,
        oracle_weight=oracle_weight,
    )


def evaluate_belief_planner(
    *,
    policy: BeliefNativeActorCritic,
    dynamics_model: BeliefDynamicsModel,
    crawler_bundle: CrawlerModelBundle,
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    env_future_predictor,
    predictor: DeltaPredictorEnsemble | None,
    state_normalizer: RunningNormalizer,
    env_name: str,
    action_values: np.ndarray,
    window_size: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    randomize_physics: bool,
    base_physics,
    base_probe_episodes: int,
    max_probe_episodes: int,
    probe_adaptive_budget: bool,
    uncertainty_probe_threshold: float,
    surprise_probe_threshold: float,
    online_z_update_alpha: float,
    online_z_update_freq: int,
    eval_episodes: int,
    seed: int,
    context_source: str,
    planner_config: BeliefPlannerConfig,
    disable_controller_context: bool = False,
    shuffle_controller_context: bool = False,
    use_stale_previous_context: bool = False,
    disable_online_refinement: bool = False,
    actor_only: bool = False,
) -> tuple[list[float], int, float]:
    """Run deterministic planner eval episodes with optional context ablations."""
    env = make_env(env_name)
    probe_env = make_env(env_name)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    total_steps = 0
    total_probe_windows = 0
    stale_context_input: np.ndarray | None = None

    for eval_episode in range(eval_episodes):
        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
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
            variant_label="belief-planner-eval",
        )
        if support is None:
            returns.append(0.0)
            continue
        total_steps += int(support["probe_steps_total"])
        total_probe_windows += int(support["probe_windows_total"])
        oracle_weight = 1.0 if context_source == "oracle" else 0.0
        selected_context = _controller_context_for_episode(
            crawler_bundle=crawler_bundle,
            step_result=support["step_result"],
            episode_physics=episode_physics,
            context_source=context_source,
            oracle_weight=oracle_weight,
        )
        context_input = transform_controller_context_input(
            selected_context.vector,
            disable_controller_context=disable_controller_context,
            shuffle_controller_context=shuffle_controller_context,
            rng=rng,
            stale_context_input=stale_context_input if use_stale_previous_context else None,
        )
        planner_state = PlanningBeliefState(
            context=sanitize_numpy(context_input),
            recurrent_hidden=sanitize_numpy(
                policy.init_recurrent_state(
                    torch.tensor(context_input[None, :], dtype=torch.float32, device=crawler_bundle.device)
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            ),
            context_age_steps=0,
            last_refresh_step=0,
        )

        apply_env_params(env, episode_physics)
        raw_state, _info = env.reset(seed=seed + eval_episode)
        raw_state = np.asarray(raw_state, dtype=np.float32)
        done = False
        episode_return = 0.0
        episode_step = 0
        previous_plan = None
        stale_context_input = sanitize_numpy(planner_state.context.copy())
        belief = support["belief"]
        belief_hidden = support["belief_hidden"]
        belief_posteriors = list(support["belief_posteriors"])
        planner_surprise_window: deque[float] = deque(maxlen=64)

        while not done:
            state = sanitize_numpy(state_normalizer.normalize(raw_state))
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=crawler_bundle.device)
            context_t = torch.tensor(
                planner_state.context[None, :],
                dtype=torch.float32,
                device=crawler_bundle.device,
            )
            hidden_t = None
            if planner_state.recurrent_hidden is not None:
                hidden_t = torch.tensor(
                    planner_state.recurrent_hidden[None, :],
                    dtype=torch.float32,
                    device=crawler_bundle.device,
                )
            if actor_only:
                with torch.no_grad():
                    mean, _value, next_hidden, _aux = policy.forward_with_hidden(
                        state_t,
                        context_t,
                        hidden_state=hidden_t,
                    )
                action = mean_to_continuous_action(mean, action_low, action_high)
                planner_output = {
                    "planner_trust": 0.0,
                    "planner_used": 0.0,
                    "action_divergence": 0.0,
                    "planner_sequence": None,
                    "next_hidden": sanitize_numpy(next_hidden.squeeze(0).cpu().numpy()),
                }
            else:
                planner_output = plan_cem_action(
                    policy=policy,
                    model=dynamics_model,
                    state_t=state_t,
                    context_t=context_t,
                    action_low=action_low,
                    action_high=action_high,
                    previous_plan=previous_plan,
                    hidden_state=hidden_t,
                    rng=rng,
                    config=planner_config,
                )
                action = sanitize_numpy(planner_output["action"])
                previous_plan = (
                    None
                    if planner_output["planner_sequence"] is None
                    else sanitize_numpy(planner_output["planner_sequence"])
                )
            planner_state.recurrent_hidden = sanitize_numpy(planner_output["next_hidden"])

            with torch.no_grad():
                prediction = dynamics_model.predict_summary(
                    state_t,
                    torch.tensor(action[None, :], dtype=torch.float32, device=crawler_bundle.device),
                    context_t,
                )
                predicted_term_prob = float(
                    torch.sigmoid(prediction["term_logit"]).item()
                )
            prev_raw_state = raw_state.copy()
            next_raw_state, reward, terminated, truncated, _info = env.step(action)
            next_raw_state = np.asarray(next_raw_state, dtype=np.float32)
            total_steps += 1
            episode_step += 1
            next_state = sanitize_numpy(state_normalizer.normalize(next_raw_state))
            control_surprise = compute_control_surprise(
                predictor=predictor,
                belief=belief,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                next_state=next_raw_state,
                device=crawler_bundle.device,
            )
            planner_surprise = planner_prediction_surprise(
                normalized_state=state,
                prediction=prediction,
                normalized_next_state=next_state,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            planner_surprise_z = surprise_z_score(planner_surprise, planner_surprise_window)
            planner_surprise_window.append(float(planner_surprise))
            belief, belief_hidden, belief_posteriors = maybe_update_online_belief(
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                device=crawler_bundle.device,
                belief_hidden=belief_hidden,
                belief_posteriors=belief_posteriors,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                reward=float(reward),
                next_state=next_raw_state,
                belief=belief,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                episode_step=episode_step,
            )
            refresh_due_to_risk = abs(predicted_term_prob - float(terminated or truncated)) > 0.15
            refresh_due_to_surprise = (
                control_surprise >= surprise_probe_threshold
                or planner_surprise_z >= 2.0
            )
            refresh_due_to_schedule = episode_step % max(1, int(online_z_update_freq)) == 0
            if (
                (not disable_online_refinement)
                and belief_posteriors
                and (refresh_due_to_schedule or refresh_due_to_surprise or refresh_due_to_risk)
            ):
                belief, payload = aggregate_env_belief(
                    belief_aggregator=belief_aggregator,
                    env_param_predictor=env_param_predictor,
                    device=crawler_bundle.device,
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
                refreshed_context = _controller_context_for_episode(
                    crawler_bundle=crawler_bundle,
                    step_result=refreshed_step,
                    episode_physics=episode_physics,
                    context_source=context_source,
                    oracle_weight=oracle_weight,
                )
                planner_state.context = transform_controller_context_input(
                    refreshed_context.vector,
                    disable_controller_context=disable_controller_context,
                    shuffle_controller_context=shuffle_controller_context,
                    rng=rng,
                    stale_context_input=stale_context_input if use_stale_previous_context else None,
                )
                planner_state.context_age_steps = 0
                planner_state.last_refresh_step = episode_step
                previous_plan = None
            else:
                planner_state.context_age_steps += 1
            raw_state = next_raw_state
            episode_return += float(reward)
            done = bool(terminated or truncated)

        returns.append(episode_return)

    env.close()
    probe_env.close()
    return returns, total_steps, float(total_probe_windows / max(eval_episodes, 1))


def train_belief_planner(
    *,
    env_name: str,
    crawler_bundle: CrawlerModelBundle,
    planner_windows: dict[str, np.ndarray],
    num_episodes: int = 300,
    window_size: int = 8,
    gamma: float = 0.99,
    lr: float = 3e-4,
    max_grad_norm: float = 0.5,
    hidden_dim: int = 128,
    normalize_rewards: bool = False,
    seed: int = 0,
    randomize_physics: bool = False,
    base_probe_episodes: int = 1,
    max_probe_episodes: int = 3,
    probe_adaptive_budget: bool = False,
    uncertainty_probe_threshold: float = 0.20,
    surprise_probe_threshold: float = 0.75,
    online_z_update_alpha: float = 0.25,
    online_z_update_freq: int = 4,
    solved_return: float = 500.0,
    solve_eval_episodes: int = 3,
    run_index: int = 1,
    total_runs: int = 1,
    variant_label: str = "belief-planner",
    peer_variant_label: str = "probe",
    peer_solved_episode: int | None = None,
    full_system_online_refinement: bool = True,
    full_system_surprise_refresh_threshold: float | None = None,
    full_system_context_source: str = "learned",
    full_system_context_chunk_len: int = 32,
    full_system_curriculum_schedule: list[tuple[int, float]] | tuple[tuple[int, float], ...] | None = None,
    full_system_plateau_warmup_episodes: int = 200,
    full_system_plateau_patience: int = 125,
    full_system_plateau_best_return_delta: float = 10.0,
    full_system_plateau_avg50_delta: float = 5.0,
    trace_writer=None,
) -> TrainingRunResult:
    """Train the belief-driven planner track with oracle bridge and CEM control."""
    encoder = crawler_bundle.encoder
    belief_aggregator = crawler_bundle.belief_aggregator
    env_param_predictor = crawler_bundle.env_param_predictor
    env_future_predictor = crawler_bundle.env_future_predictor
    predictor = crawler_bundle.predictor
    device = crawler_bundle.device
    train_env = make_env(env_name)
    probe_env = make_env(env_name)
    action_low, action_high = validate_continuous_env(train_env)
    action_values = get_action_values(train_env, crawler_bundle.action_vocab_size, env_name=env_name)
    if action_values is None:
        raise ValueError("Belief planner expects a continuous control env")
    rng = np.random.default_rng(seed)
    base_physics = default_env_params(env_name, train_env)
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    if full_system_context_source not in {"learned", "oracle", "curriculum"}:
        raise ValueError(f"Unsupported planner context source: {full_system_context_source}")
    curriculum_schedule = normalize_full_system_curriculum_schedule(
        full_system_curriculum_schedule
    )
    refresh_threshold = (
        float(surprise_probe_threshold)
        if full_system_surprise_refresh_threshold is None
        else float(full_system_surprise_refresh_threshold)
    )
    planner_config = BeliefPlannerConfig()

    policy = BeliefNativeActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        mechanics_dim=int(encoder.z_dim),
        affordance_dim=int(encoder.z_dim),
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    dynamics_model = BeliefDynamicsModel(
        state_dim=state_dim,
        action_dim=action_dim,
        mechanics_dim=int(encoder.z_dim),
        affordance_dim=int(encoder.z_dim),
        hidden_dim=hidden_dim,
        ensemble_size=5,
    ).to(device)
    dynamics_optimizer = optim.Adam(dynamics_model.parameters(), lr=1e-3)
    state_normalizer = RunningNormalizer(state_dim)
    planner_state_windows = np.asarray(planner_windows["states"], dtype=np.float32).reshape(-1, state_dim)
    if planner_state_windows.size > 0:
        state_normalizer.update(planner_state_windows)
    dynamics_dataset = build_planner_probe_dataset(
        windows=planner_windows,
        crawler_bundle=crawler_bundle,
        state_mean=state_normalizer.mean.astype(np.float32),
        state_std=np.sqrt(state_normalizer.var.astype(np.float32) + 1e-8),
        sequence_horizon=planner_config.sequence_semantic_horizon,
    )
    dynamics_pretrain = fit_belief_dynamics_model(
        model=dynamics_model,
        optimizer=dynamics_optimizer,
        dataset=dynamics_dataset,
        device=device,
        config=planner_config,
    )

    reward_normalizer = RunningNormalizer(1, clip=10.0) if normalize_rewards else None
    returns: list[float] = []
    best_return_so_far = 0.0
    best_avg50_so_far = float("-inf")
    plateau_best_return_marker = float("-inf")
    plateau_best_avg50_marker = float("-inf")
    best_policy_state_dict = snapshot_policy_state_dict(policy)
    best_dynamics_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in dynamics_model.state_dict().items()
    }
    best_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
    best_episode = None
    best_avg50_episode = None
    last_meaningful_progress_episode = None
    solved_episode = None
    solved_env_steps = None
    planner_stop_reason = "belief_planner_max_episodes"
    solve_policy_state_dict = None
    solve_state_normalizer_state = None
    solve_eval_returns = None
    zero_context_eval_returns = None
    shuffled_context_eval_returns = None
    stale_context_eval_returns = None
    no_online_refinement_eval_returns = None
    actor_only_eval_returns = None
    zero_context_ablation_delta = None
    shuffled_context_ablation_delta = None
    stale_context_ablation_delta = None
    online_refinement_ablation_delta = None
    actor_only_ablation_delta = None
    total_env_steps = 0
    total_probe_env_steps = 0
    total_control_env_steps = 0
    total_probe_windows = 0
    planner_trust_history: list[float] = []
    planner_usage_history: list[float] = []
    action_divergence_history: list[float] = []
    refresh_count_history: list[float] = []
    rollout_error_history: list[float] = []
    score_agreement_history: list[float] = [1.0 / (1.0 + float(dynamics_pretrain["score_alignment_loss"]))]

    replay_states: deque[np.ndarray] = deque(maxlen=8192)
    replay_actions: deque[np.ndarray] = deque(maxlen=8192)
    replay_contexts: deque[np.ndarray] = deque(maxlen=8192)
    replay_next_states: deque[np.ndarray] = deque(maxlen=8192)
    replay_rewards: deque[float] = deque(maxlen=8192)
    replay_terminals: deque[float] = deque(maxlen=8192)
    replay_recoverability: deque[float] = deque(maxlen=8192)

    for episode in range(1, num_episodes + 1):
        if trace_writer is not None and episode == 1:
            trace_writer.set_stage(
                "belief_planner_training",
                "Belief Planner",
                "Planning over action sequences with a learned belief-conditioned rollout model and actor fallback prior.",
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant=variant_label,
            )
        progress_left = max(0.0, 1.0 - float(episode - 1) / max(float(num_episodes), 1.0))
        set_optimizer_lr(optimizer, lr * progress_left)

        episode_physics = select_episode_physics(rng, randomize_physics, base_physics)
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
            trace_writer=trace_writer,
            episode=episode,
            variant_label=variant_label,
        )
        if support is None:
            returns.append(0.0)
            continue

        total_probe_env_steps += int(support["probe_steps_total"])
        total_probe_windows += int(support["probe_windows_total"])
        total_env_steps += int(support["probe_steps_total"])
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
        context_input = sanitize_numpy(selected_context.vector)
        planner_state = PlanningBeliefState(
            context=context_input,
            recurrent_hidden=sanitize_numpy(
                policy.init_recurrent_state(
                    torch.tensor(context_input[None, :], dtype=torch.float32, device=device)
                )
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            ),
            context_age_steps=0,
            last_refresh_step=0,
        )
        belief = support["belief"]
        belief_hidden = support["belief_hidden"]
        belief_posteriors = list(support["belief_posteriors"])
        uncertainty = float(support["step_result"].uncertainty.scalar)
        expression_confidence = float(selected_context.confidence)

        apply_env_params(train_env, episode_physics)
        raw_state, _info = train_env.reset()
        raw_state = np.asarray(raw_state, dtype=np.float32)
        state_normalizer.update(raw_state)
        done = False
        episode_step = 0
        episode_return = 0.0
        previous_plan = None
        refresh_count = 0
        episode_states: list[np.ndarray] = []
        episode_contexts: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
        episode_rewards: list[float] = []
        episode_hidden_states: list[np.ndarray] = []
        last_next_state = sanitize_numpy(state_normalizer.normalize(raw_state))
        last_terminated = False
        planner_surprise_window: deque[float] = deque(maxlen=64)

        while not done:
            state = sanitize_numpy(state_normalizer.normalize(raw_state))
            pre_action_context = sanitize_numpy(planner_state.context.copy())
            state_t = torch.tensor(state[None, :], dtype=torch.float32, device=device)
            context_t = torch.tensor(pre_action_context[None, :], dtype=torch.float32, device=device)
            hidden_t = torch.tensor(
                planner_state.recurrent_hidden[None, :],
                dtype=torch.float32,
                device=device,
            )
            planner_output = plan_cem_action(
                policy=policy,
                model=dynamics_model,
                state_t=state_t,
                context_t=context_t,
                action_low=action_low,
                action_high=action_high,
                previous_plan=previous_plan,
                hidden_state=hidden_t,
                rng=rng,
                config=planner_config,
            )
            action = sanitize_numpy(planner_output["action"])
            previous_plan = (
                None
                if planner_output["planner_sequence"] is None
                else sanitize_numpy(planner_output["planner_sequence"])
            )
            controller_hidden_input = sanitize_numpy(planner_state.recurrent_hidden.copy())
            planner_state.recurrent_hidden = sanitize_numpy(planner_output["next_hidden"])
            planner_trust_history.append(float(planner_output["planner_trust"]))
            planner_usage_history.append(float(planner_output["planner_used"]))
            action_divergence_history.append(float(planner_output["action_divergence"]))

            with torch.no_grad():
                prediction = dynamics_model.predict_summary(
                    state_t,
                    torch.tensor(action[None, :], dtype=torch.float32, device=device),
                    context_t,
                )
                predicted_term_prob = float(torch.sigmoid(prediction["term_logit"]).item())

            prev_raw_state = raw_state.copy()
            next_raw_state, reward, terminated, truncated, _info = train_env.step(action)
            next_raw_state = np.asarray(next_raw_state, dtype=np.float32)
            total_env_steps += 1
            total_control_env_steps += 1
            episode_step += 1
            raw_reward = float(reward)
            next_state = sanitize_numpy(state_normalizer.normalize(next_raw_state))
            train_reward = raw_reward
            if reward_normalizer is not None:
                reward_normalizer.update(np.asarray([[raw_reward]], dtype=np.float32))
                train_reward = float(
                    reward_normalizer.scale_only(np.asarray([raw_reward], dtype=np.float32))[0]
                )
            if trace_writer is not None:
                trace_writer.record_policy_step(
                    phase="belief_planner_control",
                    variant=variant_label,
                    state=prev_raw_state,
                    action_value=float(np.asarray(action, dtype=np.float32).reshape(-1)[0]),
                    reward=raw_reward,
                    episode=episode,
                    step_idx=episode_step,
                    episode_return=episode_return + raw_reward,
                    probe_count=int(support["probe_count"]),
                    uncertainty=uncertainty,
                    message_scale=float(planner_output["planner_trust"]),
                    expression_confidence=expression_confidence,
                    expression_ready=bool(expression_confidence >= 0.15),
                    focus_label="belief_planner",
                )

            control_surprise = compute_control_surprise(
                predictor=predictor,
                belief=belief,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                next_state=next_raw_state,
                device=device,
            )
            planner_surprise = planner_prediction_surprise(
                normalized_state=state,
                prediction=prediction,
                normalized_next_state=next_state,
                reward=raw_reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            planner_surprise_z = surprise_z_score(planner_surprise, planner_surprise_window)
            planner_surprise_window.append(float(planner_surprise))
            belief, belief_hidden, belief_posteriors = maybe_update_online_belief(
                encoder=encoder,
                belief_aggregator=belief_aggregator,
                env_param_predictor=env_param_predictor,
                device=device,
                belief_hidden=belief_hidden,
                belief_posteriors=belief_posteriors,
                prev_state=prev_raw_state,
                action_idx=nearest_probe_action_idx(action, action_values),
                reward=raw_reward,
                next_state=next_raw_state,
                belief=belief,
                online_z_update_alpha=online_z_update_alpha,
                online_z_update_freq=online_z_update_freq,
                episode_step=episode_step,
            )
            refresh_due_to_risk = abs(predicted_term_prob - float(terminated or truncated)) > 0.15
            refresh_due_to_surprise = (
                control_surprise >= refresh_threshold
                or planner_surprise_z >= 2.0
            )
            refresh_due_to_schedule = episode_step % max(1, int(online_z_update_freq)) == 0
            if (
                full_system_online_refinement
                and belief_posteriors
                and (refresh_due_to_schedule or refresh_due_to_surprise or refresh_due_to_risk)
            ):
                belief, payload = aggregate_env_belief(
                    belief_aggregator=belief_aggregator,
                    env_param_predictor=env_param_predictor,
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
                selected_context = _controller_context_for_episode(
                    crawler_bundle=crawler_bundle,
                    step_result=refreshed_step,
                    episode_physics=episode_physics,
                    context_source=full_system_context_source,
                    oracle_weight=oracle_weight,
                )
                planner_state.context = sanitize_numpy(selected_context.vector)
                planner_state.recurrent_hidden = sanitize_numpy(
                    policy.refresh_recurrent_state(
                        torch.tensor(
                            planner_state.context[None, :],
                            dtype=torch.float32,
                            device=device,
                        ),
                        torch.tensor(
                            planner_state.recurrent_hidden[None, :],
                            dtype=torch.float32,
                            device=device,
                        ),
                    )
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                )
                planner_state.context_age_steps = 0
                planner_state.last_refresh_step = episode_step
                previous_plan = None
                refresh_count += 1
                expression_confidence = float(selected_context.confidence)
                uncertainty = float(refreshed_step.uncertainty.scalar)
            else:
                planner_state.context_age_steps += 1

            recoverability = _cartpole_recoverability(next_raw_state)
            episode_states.append(state.copy())
            episode_contexts.append(pre_action_context.copy())
            episode_actions.append(action.copy())
            episode_rewards.append(train_reward)
            episode_hidden_states.append(controller_hidden_input.copy())
            replay_states.append(state.copy())
            replay_actions.append(action.copy())
            replay_contexts.append(pre_action_context.copy())
            replay_next_states.append(next_state.copy())
            replay_rewards.append(raw_reward)
            replay_terminals.append(float(terminated or truncated))
            replay_recoverability.append(recoverability)
            state_normalizer.update(next_raw_state)

            raw_state = next_raw_state
            last_next_state = next_state.copy()
            last_terminated = bool(terminated)
            episode_return += raw_reward
            done = bool(terminated or truncated)

        refresh_count_history.append(float(refresh_count))
        returns.append(episode_return)
        new_best = bool(episode_return >= best_return_so_far)
        if new_best:
            best_return_so_far = episode_return
            best_policy_state_dict = snapshot_policy_state_dict(policy)
            best_dynamics_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in dynamics_model.state_dict().items()
            }
            best_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
            best_episode = episode

        if episode_states:
            batch_states, batch_contexts, batch_actions, batch_returns, batch_hidden = _planner_sequence_batch(
                states=episode_states,
                contexts=episode_contexts,
                actions=episode_actions,
                rewards=episode_rewards,
                hidden_states=episode_hidden_states,
                gamma=gamma,
            )
            for _ in range(2):
                update_planner_prior(
                    policy=policy,
                    optimizer=optimizer,
                    batch_states=batch_states,
                    batch_contexts=batch_contexts,
                    batch_actions=batch_actions,
                    batch_returns=batch_returns,
                    batch_hidden=batch_hidden,
                    action_low=action_low,
                    action_high=action_high,
                    max_grad_norm=max_grad_norm,
                )

        if len(replay_states) >= 64:
            replay_loss = update_belief_dynamics_from_replay(
                model=dynamics_model,
                optimizer=dynamics_optimizer,
                states=np.asarray(replay_states, dtype=np.float32),
                actions=np.asarray(replay_actions, dtype=np.float32),
                contexts=np.asarray(replay_contexts, dtype=np.float32),
                next_states=np.asarray(replay_next_states, dtype=np.float32),
                rewards=np.asarray(replay_rewards, dtype=np.float32),
                terminals=np.asarray(replay_terminals, dtype=np.float32),
                recoverability=np.asarray(replay_recoverability, dtype=np.float32),
                device=device,
            )
            recent_count = min(256, len(replay_states))
            rollout_error = replay_batch_mean_error(
                model=dynamics_model,
                states=np.asarray(list(replay_states)[-recent_count:], dtype=np.float32),
                actions=np.asarray(list(replay_actions)[-recent_count:], dtype=np.float32),
                contexts=np.asarray(list(replay_contexts)[-recent_count:], dtype=np.float32),
                next_states=np.asarray(list(replay_next_states)[-recent_count:], dtype=np.float32),
                rewards=np.asarray(list(replay_rewards)[-recent_count:], dtype=np.float32),
                terminals=np.asarray(list(replay_terminals)[-recent_count:], dtype=np.float32),
            )
            rollout_error_history.append(float(rollout_error))
            score_agreement_history.append(float(1.0 / (1.0 + replay_loss)))

        avg_10 = float(np.mean(returns[-10:]))
        avg_50 = float(np.mean(returns[-50:]))
        if avg_50 >= best_avg50_so_far:
            best_avg50_so_far = avg_50
        meaningful_progress = False
        if episode == 1 or episode_return >= plateau_best_return_marker + float(full_system_plateau_best_return_delta):
            plateau_best_return_marker = episode_return
            meaningful_progress = True
        if episode == 1 or avg_50 >= plateau_best_avg50_marker + float(full_system_plateau_avg50_delta):
            plateau_best_avg50_marker = avg_50
            best_avg50_episode = episode
            meaningful_progress = True
        if meaningful_progress:
            last_meaningful_progress_episode = episode
        episode_window = max(1, len(episode_states))
        planner_trust_mean = float(np.mean(planner_trust_history[-episode_window:]))
        planner_usage_mean = float(np.mean(planner_usage_history[-episode_window:]))
        print_belief_episode_status(
            run_index=run_index,
            total_runs=total_runs,
            seed=seed,
            variant_label=variant_label,
            episode=episode,
            episode_return=float(episode_return),
            avg10=float(avg_10),
            best_return=float(best_return_so_far),
            total_env_steps=int(total_env_steps),
            probe_count=int(support["probe_count"]),
            episode_probe_steps=int(support["probe_steps_total"]),
            trust=planner_trust_mean,
            usage=planner_usage_mean,
            usage_label="plan",
            refresh_count=int(refresh_count),
            avg50=float(avg_50),
            target_return=float(solved_return),
            solved_episode=solved_episode,
            peer_status=format_peer_solve_status(peer_variant_label, peer_solved_episode),
            new_best=new_best,
        )
        if trace_writer is not None:
            trace_writer.record_episode_summary(
                variant=variant_label,
                episode=episode,
                episode_return=episode_return,
                avg10=avg_10,
                avg50=avg_50,
                total_env_steps=total_env_steps,
                probe_steps=int(support["probe_steps_total"]),
                probe_count=int(support["probe_count"]),
                uncertainty=uncertainty,
                message_scale=float(np.mean(planner_usage_history[-max(1, len(episode_states)):])),
                expression_confidence=expression_confidence,
                expression_ready=bool(expression_confidence >= 0.15),
                stop_reason="belief_planner",
            )

        if episode_return >= solved_return:
            solved_episode = episode
            solved_env_steps = total_env_steps
            planner_stop_reason = "belief_planner_solved"
            solve_policy_state_dict = snapshot_policy_state_dict(policy)
            solve_state_normalizer_state = snapshot_normalizer_state(state_normalizer)
            print_solve_event(
                run_index=run_index,
                total_runs=total_runs,
                seed=seed,
                variant_label=variant_label,
                episode=episode,
                total_env_steps=total_env_steps,
                episode_return=float(episode_return),
                probe_count=int(support["probe_count"]),
            )
            break
        if should_stop_belief_planner_plateau(
            current_episode=episode,
            warmup_episodes=full_system_plateau_warmup_episodes,
            patience=full_system_plateau_patience,
            last_meaningful_progress_episode=last_meaningful_progress_episode,
        ):
            planner_stop_reason = "belief_planner_plateau"
            print(
                f"[run {run_index}/{total_runs} | seed {seed} | {variant_label}] "
                f"early stop | reason=plateau | warmup={full_system_plateau_warmup_episodes} | "
                f"patience={full_system_plateau_patience} | "
                f"last-progress={last_meaningful_progress_episode} | "
                f"best={best_return_so_far:7.2f} | avg50={avg_50:7.2f}"
            )
            break

    if solve_eval_episodes > 0 and best_episode is not None:
        eval_policy = BeliefNativeActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            mechanics_dim=int(encoder.z_dim),
            affordance_dim=int(encoder.z_dim),
            hidden_dim=hidden_dim,
        ).to(device)
        eval_policy.load_state_dict(best_policy_state_dict)
        eval_policy.eval()
        eval_model = BeliefDynamicsModel(
            state_dim=state_dim,
            action_dim=action_dim,
            mechanics_dim=int(encoder.z_dim),
            affordance_dim=int(encoder.z_dim),
            hidden_dim=hidden_dim,
            ensemble_size=5,
        ).to(device)
        eval_model.load_state_dict(best_dynamics_state_dict)
        eval_model.eval()
        eval_normalizer = restore_normalizer_state(state_dim, best_state_normalizer_state)
        eval_context_source = (
            "learned"
            if full_system_context_source == "curriculum"
            else full_system_context_source
        )
        solve_eval_returns, _eval_steps, avg_probe_windows = evaluate_belief_planner(
            policy=eval_policy,
            dynamics_model=eval_model,
            crawler_bundle=crawler_bundle,
            encoder=encoder,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            env_future_predictor=env_future_predictor,
            predictor=predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            window_size=window_size,
            action_low=action_low,
            action_high=action_high,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=refresh_threshold,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            eval_episodes=solve_eval_episodes,
            seed=seed,
            context_source=eval_context_source,
            planner_config=planner_config,
        )
        zero_context_eval_returns, _zero_steps, _zero_probe_windows = evaluate_belief_planner(
            policy=eval_policy,
            dynamics_model=eval_model,
            crawler_bundle=crawler_bundle,
            encoder=encoder,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            env_future_predictor=env_future_predictor,
            predictor=predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            window_size=window_size,
            action_low=action_low,
            action_high=action_high,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=refresh_threshold,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            eval_episodes=solve_eval_episodes,
            seed=seed,
            context_source=eval_context_source,
            planner_config=planner_config,
            disable_controller_context=True,
        )
        shuffled_context_eval_returns, _shuffle_steps, _shuffle_probe_windows = evaluate_belief_planner(
            policy=eval_policy,
            dynamics_model=eval_model,
            crawler_bundle=crawler_bundle,
            encoder=encoder,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            env_future_predictor=env_future_predictor,
            predictor=predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            window_size=window_size,
            action_low=action_low,
            action_high=action_high,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=refresh_threshold,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            eval_episodes=solve_eval_episodes,
            seed=seed,
            context_source=eval_context_source,
            planner_config=planner_config,
            shuffle_controller_context=True,
        )
        stale_context_eval_returns, _stale_steps, _stale_probe_windows = evaluate_belief_planner(
            policy=eval_policy,
            dynamics_model=eval_model,
            crawler_bundle=crawler_bundle,
            encoder=encoder,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            env_future_predictor=env_future_predictor,
            predictor=predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            window_size=window_size,
            action_low=action_low,
            action_high=action_high,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=refresh_threshold,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            eval_episodes=solve_eval_episodes,
            seed=seed,
            context_source=eval_context_source,
            planner_config=planner_config,
            use_stale_previous_context=True,
        )
        no_online_refinement_eval_returns, _frozen_steps, _frozen_probe_windows = evaluate_belief_planner(
            policy=eval_policy,
            dynamics_model=eval_model,
            crawler_bundle=crawler_bundle,
            encoder=encoder,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            env_future_predictor=env_future_predictor,
            predictor=predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            window_size=window_size,
            action_low=action_low,
            action_high=action_high,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=refresh_threshold,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            eval_episodes=solve_eval_episodes,
            seed=seed,
            context_source=eval_context_source,
            planner_config=planner_config,
            disable_online_refinement=True,
        )
        actor_only_eval_returns, _actor_steps, _actor_probe_windows = evaluate_belief_planner(
            policy=eval_policy,
            dynamics_model=eval_model,
            crawler_bundle=crawler_bundle,
            encoder=encoder,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            env_future_predictor=env_future_predictor,
            predictor=predictor,
            state_normalizer=eval_normalizer,
            env_name=env_name,
            action_values=action_values,
            window_size=window_size,
            action_low=action_low,
            action_high=action_high,
            randomize_physics=randomize_physics,
            base_physics=base_physics,
            base_probe_episodes=base_probe_episodes,
            max_probe_episodes=max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=refresh_threshold,
            online_z_update_alpha=online_z_update_alpha,
            online_z_update_freq=online_z_update_freq,
            eval_episodes=solve_eval_episodes,
            seed=seed,
            context_source=eval_context_source,
            planner_config=planner_config,
            actor_only=True,
        )
        solve_eval_mean = float(np.mean(np.asarray(solve_eval_returns, dtype=np.float32)))
        zero_context_ablation_delta = solve_eval_mean - float(np.mean(np.asarray(zero_context_eval_returns, dtype=np.float32)))
        shuffled_context_ablation_delta = solve_eval_mean - float(np.mean(np.asarray(shuffled_context_eval_returns, dtype=np.float32)))
        stale_context_ablation_delta = solve_eval_mean - float(np.mean(np.asarray(stale_context_eval_returns, dtype=np.float32)))
        online_refinement_ablation_delta = solve_eval_mean - float(np.mean(np.asarray(no_online_refinement_eval_returns, dtype=np.float32)))
        actor_only_ablation_delta = solve_eval_mean - float(np.mean(np.asarray(actor_only_eval_returns, dtype=np.float32)))
        print(
            f"[run {run_index}/{total_runs} | seed {seed} | {variant_label}] "
            f"planner eval | full={solve_eval_mean:7.2f} | "
            f"zero={np.mean(zero_context_eval_returns):7.2f} | "
            f"stale={np.mean(stale_context_eval_returns):7.2f} | "
            f"shuffle={np.mean(shuffled_context_eval_returns):7.2f} | "
            f"frozen={np.mean(no_online_refinement_eval_returns):7.2f} | "
            f"actor={np.mean(actor_only_eval_returns):7.2f} | "
            f"avg-probes={avg_probe_windows:4.2f}"
        )

    train_env.close()
    probe_env.close()
    planner_trust_usage_rate = None
    if planner_usage_history:
        planner_trust_usage_rate = float(np.mean(np.asarray(planner_usage_history, dtype=np.float32)))
    return TrainingRunResult(
        policy=policy,
        returns=returns,
        state_normalizer=state_normalizer,
        solved_episode=solved_episode,
        solved_env_steps=solved_env_steps,
        total_env_steps=total_env_steps,
        best_policy_state_dict=best_policy_state_dict,
        best_state_normalizer_state=best_state_normalizer_state,
        best_return=best_return_so_far,
        best_episode=best_episode,
        solve_policy_state_dict=solve_policy_state_dict,
        solve_state_normalizer_state=solve_state_normalizer_state,
        solve_eval_returns=solve_eval_returns,
        solve_probe_count=base_probe_episodes,
        probe_env_steps_total=total_probe_env_steps,
        control_env_steps_total=total_control_env_steps,
        probe_windows_total=total_probe_windows,
        probe_stop_reasons=None,
        probe_family_expected_gain=None,
        probe_family_realized_gain=None,
        probe_family_future_error=None,
        probe_family_selection_count=None,
        last_probe_stop_reason=planner_stop_reason,
        solve_probe_stop_reason=planner_stop_reason,
        env_expression_eval_returns=solve_eval_returns,
        no_env_expression_eval_returns=None,
        env_expression_ablation_delta=None,
        post_expression_env_steps_total=total_control_env_steps,
        post_expression_episode_count=solved_episode,
        controller_style=f"belief_planner_context_{full_system_context_source}",
        zero_context_eval_returns=zero_context_eval_returns,
        shuffled_context_eval_returns=shuffled_context_eval_returns,
        stale_context_eval_returns=stale_context_eval_returns,
        no_online_refinement_eval_returns=no_online_refinement_eval_returns,
        zero_context_ablation_delta=zero_context_ablation_delta,
        shuffled_context_ablation_delta=shuffled_context_ablation_delta,
        stale_context_ablation_delta=stale_context_ablation_delta,
        online_refinement_ablation_delta=online_refinement_ablation_delta,
        actor_only_eval_returns=actor_only_eval_returns,
        actor_only_ablation_delta=actor_only_ablation_delta,
        planner_trust_usage_rate=planner_trust_usage_rate,
        actor_planner_action_divergence=(
            None
            if not action_divergence_history
            else float(np.mean(np.asarray(action_divergence_history, dtype=np.float32)))
        ),
        rollout_model_error_mean=(
            None
            if not rollout_error_history
            else float(np.mean(np.asarray(rollout_error_history, dtype=np.float32)))
        ),
        refresh_count_mean=(
            None
            if not refresh_count_history
            else float(np.mean(np.asarray(refresh_count_history, dtype=np.float32)))
        ),
        oracle_score_agreement_mean=(
            None
            if not score_agreement_history
            else float(np.mean(np.asarray(score_agreement_history, dtype=np.float32)))
        ),
        extra_checkpoint_data={
            "planner_model_state_dict": dynamics_model.state_dict(),
            "best_planner_model_state_dict": best_dynamics_state_dict,
            "planner_config": planner_config.__dict__,
            "dynamics_pretrain": dynamics_pretrain,
            "planner_stop_reason": planner_stop_reason,
            "planner_plateau_warmup_episodes": int(full_system_plateau_warmup_episodes),
            "planner_plateau_patience": int(full_system_plateau_patience),
            "planner_plateau_best_return_delta": float(full_system_plateau_best_return_delta),
            "planner_plateau_avg50_delta": float(full_system_plateau_avg50_delta),
            "last_meaningful_progress_episode": None
            if last_meaningful_progress_episode is None
            else int(last_meaningful_progress_episode),
            "best_avg50_return": None
            if not np.isfinite(best_avg50_so_far)
            else float(best_avg50_so_far),
            "best_avg50_episode": None
            if best_avg50_episode is None
            else int(best_avg50_episode),
        },
    )


__all__ = [
    "evaluate_belief_planner",
    "should_stop_belief_planner_plateau",
    "train_belief_planner",
]
