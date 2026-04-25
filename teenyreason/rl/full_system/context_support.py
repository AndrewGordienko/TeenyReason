"""Shared support-collection and controller-context helpers."""

from __future__ import annotations

import numpy as np

from ...crawler import CrawlerModelBundle
from ...crawler.types import ControllerBeliefContext
from ...models.env_belief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ...probe.explorer import build_probe_planner
from ...probe.probe_latent import (
    aggregate_env_belief,
    collect_adaptive_probe_window,
    encode_window_posterior,
    init_recurrent_belief_hidden,
    probe_group_ids_from_families,
    update_recurrent_belief_from_window,
)
from ...representation import DeltaPredictorEnsemble, WorldEncoder
from ..core.ppo_core import sanitize_numpy
from ..probe_policy.budget import choose_fair_probe_family
from ..probe_policy.eval import compute_probe_surprise


def mix_controller_contexts(
    learned_context: ControllerBeliefContext,
    oracle_context: ControllerBeliefContext,
    *,
    oracle_weight: float,
) -> ControllerBeliefContext:
    """Blend learned and oracle context for curriculum-style handoff."""
    oracle_weight = float(np.clip(oracle_weight, 0.0, 1.0))
    if oracle_weight >= 1.0:
        return oracle_context
    if oracle_weight <= 0.0:
        return learned_context
    learned_weight = 1.0 - oracle_weight
    return ControllerBeliefContext(
        mechanics_code=sanitize_numpy(
            learned_weight * learned_context.mechanics_code
            + oracle_weight * oracle_context.mechanics_code
        ),
        affordance_code=sanitize_numpy(
            learned_weight * learned_context.affordance_code
            + oracle_weight * oracle_context.affordance_code
        ),
        confidence=float(
            learned_weight * float(learned_context.confidence)
            + oracle_weight * float(oracle_context.confidence)
        ),
        uncertainty_scalar=float(
            learned_weight * float(learned_context.uncertainty_scalar)
            + oracle_weight * float(oracle_context.uncertainty_scalar)
        ),
        metadata={
            "source_kind": "mixed",
            "oracle_weight": oracle_weight,
            "learned_weight": learned_weight,
            "learned_source_kind": str(learned_context.metadata.get("source_kind", "learned")),
            "oracle_source_kind": str(oracle_context.metadata.get("source_kind", "oracle")),
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
    )
    if probe_planner is not None:
        probe_planner.begin_env_instance()
    family_counts = {family: 0 for family in getattr(crawler_bundle, "family_names", ())}

    while probe_count < max(1, int(probe_target_count)):
        chosen_family = None
        if probe_planner is not None and crawler_bundle.family_names:
            chosen_family = choose_fair_probe_family(
                family_names=crawler_bundle.family_names,
                expected_family_gain={} if step_result is None else step_result.expected_family_gain,
                family_counts=family_counts,
                probe_count=probe_count,
                probe_surprise=0.0,
            )
            probe_planner.begin_rollout(primary_goal=chosen_family)
        prior_belief = None if belief is None else belief.copy()
        window_states, window_actions, window_rewards, probe_failed, probe_steps_used = collect_adaptive_probe_window(
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
        observed_family = (
            chosen_family
            if chosen_family is not None
            else (
                None
                if probe_planner is None
                else getattr(probe_planner, "current_goal", None)
            )
        )
        probe_families.append(observed_family)
        probe_group_ids = probe_group_ids_from_families(
            probe_families,
            family_names=crawler_bundle.family_names,
        )
        belief, payload = aggregate_env_belief(
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            device=crawler_bundle.device,
            posterior_views=belief_posteriors,
            probe_group_ids=probe_group_ids,
        )
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
        step_result = crawler_bundle.build_step_result(
            payload=payload,
            expected_family_gain=expected_family_gain,
            realized_family_gain={},
            stop_reason=None,
            bits_per_dim=0,
            use_residual_sketch=False,
        )
        total_probe_windows += 1
        probe_count += 1
        probe_windows.append(
            {
                "states": np.asarray(window_states, dtype=np.float32),
                "actions": np.asarray(window_actions, dtype=np.int64),
                "rewards": np.asarray(window_rewards, dtype=np.float32),
                "chosen_family": "" if observed_family is None else str(observed_family),
                "probe_surprise": float(probe_surprise),
                "probe_steps_used": int(probe_steps_used),
            }
        )
        if chosen_family is not None:
            family_counts[chosen_family] = family_counts.get(chosen_family, 0) + 1
        if trace_writer is not None:
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
        if (
            probe_adaptive_budget
            and probe_count >= probe_target_count
            and probe_target_count < max(1, int(max_probe_episodes))
            and (
                float(step_result.uncertainty.scalar) >= float(uncertainty_probe_threshold)
                or float(probe_surprise) >= float(surprise_probe_threshold)
            )
        ):
            probe_target_count += 1

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


def controller_context_for_episode(
    *,
    crawler_bundle: CrawlerModelBundle,
    step_result,
    episode_physics,
    context_source: str,
    oracle_weight: float,
) -> ControllerBeliefContext:
    """Resolve the controller context used for one episode."""
    learned_context = step_result.controller_context
    if context_source == "learned":
        return learned_context
    oracle_context = crawler_bundle.build_oracle_controller_context(episode_physics.as_array())
    if context_source == "oracle":
        return oracle_context
    return mix_controller_contexts(
        learned_context,
        oracle_context,
        oracle_weight=oracle_weight,
    )
