"""Support-context collection used by the runtime crawler facade."""

from __future__ import annotations

import numpy as np

from ...models.envbelief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ..probes.explorer import build_probe_planner
from ..probes.latent import (
    aggregate_env_belief,
    collect_adaptive_probe_window,
    encode_window_posterior,
    init_recurrent_belief_hidden,
    probe_group_ids_from_families,
    update_recurrent_belief_from_window,
)
from ...cognition.representation import DeltaPredictorEnsemble, WorldEncoder
from ...rl.core import sanitize_numpy
from ...rl.probe_policy.budgeting import choose_fair_probe_family
from ...rl.probe_policy.training.evaluation.surprise import compute_probe_surprise
from ..library import CrawlerModelBundle
from ..types import ControllerBeliefContext


def cheap_mechanics_controller_context(
    mechanics_vector: np.ndarray,
    *,
    confidence: float,
    uncertainty: float,
    affordance_dim: int | None = None,
    source_kind: str = "cheap_dual_use",
) -> ControllerBeliefContext:
    """Build a controller context from a cheap/passive mechanics belief."""
    mechanics = sanitize_numpy(np.asarray(mechanics_vector, dtype=np.float32).reshape(-1))
    if affordance_dim is None:
        affordance_dim = int(mechanics.size)
    affordance = np.zeros((max(0, int(affordance_dim)),), dtype=np.float32)
    return ControllerBeliefContext(
        mechanics_code=mechanics,
        affordance_code=sanitize_numpy(affordance),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        uncertainty_scalar=float(max(0.0, uncertainty)),
        metadata={
            "source_kind": str(source_kind),
            "belief_source": str(source_kind),
            "solver_message_source": str(source_kind),
            "handoff": "cheap_mechanics_controller_context",
        },
    )


def collect_support_context(
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
    belief_bits_per_dim: int = 0,
    belief_use_residual_sketch: bool = False,
) -> dict | None:
    """Collect one compact support set and return the latest crawler step result."""
    probe_target_count = max(1, int(base_probe_episodes))
    probe_count = 0
    probe_steps_total = 0
    total_probe_windows = 0
    belief = None
    belief_hidden = init_recurrent_belief_hidden(encoder=encoder, device=crawler_bundle.device)
    belief_posteriors: list[tuple[np.ndarray, np.ndarray]] = []
    probe_families: list[str | None] = []
    probe_windows: list[dict[str, object]] = []
    step_result = None
    probe_planner = build_probe_planner(
        action_space=probe_env.action_space,
        action_values=action_values,
        rng=rng,
        env_name=env_name,
    )
    if probe_planner is not None:
        probe_planner.begin_env_instance()
    family_counts = {family: 0 for family in getattr(crawler_bundle, "family_names", ())}

    while probe_count < probe_target_count:
        chosen_family = _choose_probe_family(
            probe_planner=probe_planner,
            crawler_bundle=crawler_bundle,
            family_counts=family_counts,
            probe_count=probe_count,
            step_result=step_result,
        )
        prior_belief = None if belief is None else belief.copy()
        window_states, window_actions, window_rewards, probe_failed, probe_steps_used = (
            collect_adaptive_probe_window(
                env=probe_env,
                encoder=encoder,
                predictor=predictor,
                device=crawler_bundle.device,
                rng=rng,
                window_size=window_size,
                episode_physics=episode_physics,
                action_values=action_values,
                env_name=env_name,
                prior_belief=belief,
                prior_hidden=belief_hidden,
                planner=probe_planner,
                trace_writer=trace_writer,
                trace_context={
                    "episode_id": episode,
                    "env_instance_id": int(getattr(episode_physics, "seed", episode)),
                    "step_offset": probe_steps_total,
                },
            )
        )
        probe_steps_total += int(probe_steps_used)
        if probe_failed:
            return None

        probe_surprise = compute_probe_surprise(
            env_future_predictor=env_future_predictor,
            belief=prior_belief,
            window_states=window_states,
            window_actions=window_actions,
            window_rewards=window_rewards,
            action_vocab_size=int(crawler_bundle.action_vocab_size),
            device=crawler_bundle.device,
        )
        window_posterior = encode_window_posterior(
            encoder=encoder,
            device=crawler_bundle.device,
            window_states=window_states,
            window_actions=window_actions,
            window_rewards=window_rewards,
        )
        _window_belief, belief_hidden, _posterior = update_recurrent_belief_from_window(
            encoder=encoder,
            device=crawler_bundle.device,
            belief_hidden=belief_hidden,
            window_states=window_states,
            window_actions=window_actions,
            window_rewards=window_rewards,
            prior_belief=None,
            alpha=1.0,
        )
        belief_posteriors.append(window_posterior)
        observed_family = _observed_probe_family(chosen_family, probe_planner)
        probe_families.append(observed_family)
        probe_windows.append(
            _probe_window_record(
                window_states=window_states,
                window_actions=window_actions,
                window_rewards=window_rewards,
                observed_family=observed_family,
                probe_surprise=probe_surprise,
                probe_steps_used=probe_steps_used,
            )
        )
        belief, payload = _aggregate_current_belief(
            crawler_bundle=crawler_bundle,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            belief_posteriors=belief_posteriors,
            probe_families=probe_families,
            probe_windows=probe_windows,
            belief_bits_per_dim=belief_bits_per_dim,
            belief_use_residual_sketch=belief_use_residual_sketch,
        )
        step_result = _build_step_result(
            crawler_bundle=crawler_bundle,
            payload=payload,
            family_counts=family_counts,
            belief_bits_per_dim=belief_bits_per_dim,
            belief_use_residual_sketch=belief_use_residual_sketch,
        )
        total_probe_windows += 1
        probe_count += 1
        if chosen_family is not None:
            family_counts[chosen_family] = family_counts.get(chosen_family, 0) + 1
        _record_probe_decision(
            trace_writer=trace_writer,
            episode=episode,
            probe_count=probe_count,
            max_probe_episodes=max_probe_episodes,
            chosen_family=chosen_family,
            probe_surprise=probe_surprise,
            step_result=step_result,
        )
        probe_target_count = _maybe_extend_probe_budget(
            probe_target_count=probe_target_count,
            probe_count=probe_count,
            max_probe_episodes=max_probe_episodes,
            probe_adaptive_budget=probe_adaptive_budget,
            uncertainty_probe_threshold=uncertainty_probe_threshold,
            surprise_probe_threshold=surprise_probe_threshold,
            probe_surprise=probe_surprise,
            step_result=step_result,
        )

    if belief is None or step_result is None:
        return None
    return {
        "belief": belief,
        "belief_hidden": belief_hidden,
        "belief_posteriors": belief_posteriors,
        "step_result": step_result,
        "probe_count": int(probe_count),
        "probe_steps_total": int(probe_steps_total),
        "probe_windows_total": int(total_probe_windows),
        "probe_windows": probe_windows,
    }


def _choose_probe_family(
    *,
    probe_planner,
    crawler_bundle: CrawlerModelBundle,
    family_counts: dict[str, int],
    probe_count: int,
    step_result,
) -> str | None:
    if probe_planner is None or not crawler_bundle.family_names:
        return None
    chosen_family = choose_fair_probe_family(
        family_names=crawler_bundle.family_names,
        expected_family_gain={} if step_result is None else step_result.expected_family_gain,
        family_counts=family_counts,
        probe_count=probe_count,
        probe_surprise=0.0,
    )
    probe_planner.begin_rollout(primary_goal=chosen_family)
    return chosen_family


def _observed_probe_family(chosen_family: str | None, probe_planner) -> str | None:
    if chosen_family is not None:
        return chosen_family
    if probe_planner is None:
        return None
    return getattr(probe_planner, "current_goal", None)


def _probe_window_record(
    *,
    window_states,
    window_actions,
    window_rewards,
    observed_family: str | None,
    probe_surprise: float,
    probe_steps_used: int,
) -> dict[str, object]:
    family = "" if observed_family is None else str(observed_family)
    return {
        "states": np.asarray(window_states, dtype=np.float32),
        "actions": np.asarray(window_actions, dtype=np.int64),
        "rewards": np.asarray(window_rewards, dtype=np.float32),
        "terminated": False,
        "truncated": False,
        "probe_family": family,
        "chosen_family": family,
        "probe_surprise": float(probe_surprise),
        "probe_steps_used": int(probe_steps_used),
    }


def _aggregate_current_belief(
    *,
    crawler_bundle: CrawlerModelBundle,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble | None,
    belief_posteriors: list[tuple[np.ndarray, np.ndarray]],
    probe_families: list[str | None],
    probe_windows: list[dict[str, object]],
    belief_bits_per_dim: int,
    belief_use_residual_sketch: bool,
) -> tuple[np.ndarray, dict]:
    if str(getattr(crawler_bundle, "belief_mode", "latent_pool")) == "particle_sysid":
        return crawler_bundle.build_particle_env_belief(
            probe_windows,
            bits_per_dim=int(belief_bits_per_dim),
            use_residual_sketch=bool(belief_use_residual_sketch),
        )
    probe_group_ids = probe_group_ids_from_families(
        probe_families,
        family_names=crawler_bundle.family_names,
    )
    return aggregate_env_belief(
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        device=crawler_bundle.device,
        posterior_views=belief_posteriors,
        probe_group_ids=probe_group_ids,
    )


def _build_step_result(
    *,
    crawler_bundle: CrawlerModelBundle,
    payload: dict,
    family_counts: dict[str, int],
    belief_bits_per_dim: int,
    belief_use_residual_sketch: bool,
):
    predictive_belief = crawler_bundle.build_predictive_belief(payload)
    uncertainty_estimate = crawler_bundle.build_uncertainty_estimate(payload)
    expected_family_gain = crawler_bundle.score_probe_families(
        predictive_belief,
        uncertainty_estimate,
        family_counts=family_counts,
        global_family_counts=family_counts,
        family_error_history={},
        family_realized_gain_history={},
        use_learned_family_value=True,
    )
    return crawler_bundle.build_step_result(
        payload=payload,
        expected_family_gain=expected_family_gain,
        realized_family_gain={},
        stop_reason=None,
        bits_per_dim=int(belief_bits_per_dim),
        use_residual_sketch=bool(belief_use_residual_sketch),
    )


def _record_probe_decision(
    *,
    trace_writer,
    episode: int,
    probe_count: int,
    max_probe_episodes: int,
    chosen_family: str | None,
    probe_surprise: float,
    step_result,
) -> None:
    if trace_writer is None:
        return
    trace_writer.record_probe_decision(
        episode=episode,
        probe_count=probe_count,
        max_probe_episodes=max_probe_episodes,
        chosen_family=chosen_family,
        uncertainty=float(step_result.uncertainty.scalar),
        surprise=float(probe_surprise),
        message_scale=float(step_result.controller_context.confidence),
        expression_confidence=float(step_result.controller_context.confidence),
        expression_ready=bool(step_result.controller_context.confidence >= 0.15),
        expected_family_gain=step_result.expected_family_gain,
        realized_family_gain={},
        stop_reason=None,
    )


def _maybe_extend_probe_budget(
    *,
    probe_target_count: int,
    probe_count: int,
    max_probe_episodes: int,
    probe_adaptive_budget: bool,
    uncertainty_probe_threshold: float,
    surprise_probe_threshold: float,
    probe_surprise: float,
    step_result,
) -> int:
    if not probe_adaptive_budget:
        return int(probe_target_count)
    if probe_count < probe_target_count:
        return int(probe_target_count)
    if probe_target_count >= max(1, int(max_probe_episodes)):
        return int(probe_target_count)
    if float(step_result.uncertainty.scalar) >= float(uncertainty_probe_threshold):
        return int(probe_target_count) + 1
    if float(probe_surprise) >= float(surprise_probe_threshold):
        return int(probe_target_count) + 1
    return int(probe_target_count)


__all__ = [
    "cheap_mechanics_controller_context",
    "collect_support_context",
]
