"""CEM planner for generic continuous-action world models."""

from __future__ import annotations

import numpy as np

from .control import ActorPolicyModel
from .model import ActionValueModel, EnsembleMLPWorldModel, ValueBootstrapModel


class CEMPlanner:
    """Cross-entropy method planner with uncertainty gating."""

    def __init__(
        self,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int,
        candidate_count: int,
        iterations: int,
        elite_fraction: float,
        noise_floor: float,
        temporal_chunk_size: int = 1,
        temporal_chunk_candidates: int = 0,
        temporal_smoothness_penalty: float = 0.0,
    ):
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        self.horizon = max(1, int(horizon))
        self.candidate_count = max(4, int(candidate_count))
        self.iterations = max(1, int(iterations))
        self.elite_fraction = float(np.clip(elite_fraction, 0.05, 0.8))
        self.noise_floor = float(max(0.0, noise_floor))
        self.temporal_chunk_size = max(1, int(temporal_chunk_size))
        self.temporal_chunk_candidates = max(0, int(temporal_chunk_candidates))
        self.temporal_smoothness_penalty = float(max(0.0, temporal_smoothness_penalty))

    def choose_action(
        self,
        model: EnsembleMLPWorldModel,
        observation: np.ndarray,
        *,
        seed: int,
        step: int,
        discount: float,
        done_penalty: float,
        uncertainty_penalty: float,
        uncertainty_gate_quantile: float = 0.85,
        value_model: ValueBootstrapModel | None = None,
        action_value_model: ActionValueModel | None = None,
        action_value_weight: float = 0.0,
        actor_model: ActorPolicyModel | None = None,
        actor_center_prior: bool = False,
        actor_prior_candidates: int = 0,
        actor_noise: float = 0.0,
        uncertainty_bonus: float = 0.0,
    ) -> np.ndarray:
        plan = self.choose_plan(
            model,
            observation,
            seed=seed,
            step=step,
            discount=discount,
            done_penalty=done_penalty,
            uncertainty_penalty=uncertainty_penalty,
            uncertainty_gate_quantile=uncertainty_gate_quantile,
            value_model=value_model,
            action_value_model=action_value_model,
            action_value_weight=action_value_weight,
            actor_model=actor_model,
            actor_center_prior=actor_center_prior,
            actor_prior_candidates=actor_prior_candidates,
            actor_noise=actor_noise,
            uncertainty_bonus=uncertainty_bonus,
        )
        return np.asarray(plan["action"], dtype=np.float32)

    def choose_sequence(
        self,
        model: EnsembleMLPWorldModel,
        observation: np.ndarray,
        *,
        seed: int,
        step: int,
        discount: float,
        done_penalty: float,
        uncertainty_penalty: float,
        uncertainty_gate_quantile: float = 0.85,
        value_model: ValueBootstrapModel | None = None,
        action_value_model: ActionValueModel | None = None,
        action_value_weight: float = 0.0,
        actor_model: ActorPolicyModel | None = None,
        actor_center_prior: bool = False,
        actor_prior_candidates: int = 0,
        actor_noise: float = 0.0,
        uncertainty_bonus: float = 0.0,
    ) -> np.ndarray:
        plan = self.choose_plan(
            model,
            observation,
            seed=seed,
            step=step,
            discount=discount,
            done_penalty=done_penalty,
            uncertainty_penalty=uncertainty_penalty,
            uncertainty_gate_quantile=uncertainty_gate_quantile,
            value_model=value_model,
            action_value_model=action_value_model,
            action_value_weight=action_value_weight,
            actor_model=actor_model,
            actor_center_prior=actor_center_prior,
            actor_prior_candidates=actor_prior_candidates,
            actor_noise=actor_noise,
            uncertainty_bonus=uncertainty_bonus,
        )
        return np.asarray(plan["sequence"], dtype=np.float32)

    def choose_plan(
        self,
        model: EnsembleMLPWorldModel,
        observation: np.ndarray,
        *,
        seed: int,
        step: int,
        discount: float,
        done_penalty: float,
        uncertainty_penalty: float,
        uncertainty_gate_quantile: float = 0.85,
        value_model: ValueBootstrapModel | None = None,
        action_value_model: ActionValueModel | None = None,
        action_value_weight: float = 0.0,
        actor_model: ActorPolicyModel | None = None,
        actor_center_prior: bool = False,
        actor_prior_candidates: int = 0,
        actor_noise: float = 0.0,
        uncertainty_bonus: float = 0.0,
        goal_control_prior=None,
        goal_prior_candidates: int = 0,
        goal_reachability_floor: float = 0.0,
        manifold_model=None,
        off_manifold_penalty: float = 0.0,
        value_overestimate_penalty: float = 0.0,
        value_calibration: bool = False,
        value_reachability_power: float = 1.0,
        value_manifold_power: float = 1.0,
    ) -> dict[str, object]:
        rng = np.random.default_rng(71_000 + int(seed) * 997 + int(step) * 37)
        action_dim = int(self.action_low.shape[0])
        center = np.clip(np.zeros((action_dim,), dtype=np.float32), self.action_low, self.action_high)
        actor_sequence = actor_prior_sequence(
            actor_model,
            model,
            observation,
            horizon=self.horizon,
            action_low=self.action_low,
            action_high=self.action_high,
        )
        goal_sequence, goal_stats = goal_prior_sequence(
            goal_control_prior,
            model,
            observation,
            horizon=self.horizon,
            reachability_floor=float(goal_reachability_floor),
        )
        if manifold_model is None and goal_control_prior is not None:
            manifold_model = getattr(goal_control_prior, "manifold", None)
        confidence = value_confidence(
            goal_stats,
            manifold_model,
            observation,
            has_goal_prior=goal_control_prior is not None,
            reachability_power=float(value_reachability_power),
            manifold_power=float(value_manifold_power),
        )
        mean = np.tile(center, (self.horizon, 1)).astype(np.float32)
        if actor_sequence is not None and bool(actor_center_prior):
            mean = actor_sequence.copy()
        std = np.tile((self.action_high - self.action_low) * 0.5, (self.horizon, 1)).astype(np.float32)
        best_sequence = mean.copy()
        best_score = -float("inf")
        best_uncertainty = 0.0
        final_uncertainties = np.zeros((self.candidate_count,), dtype=np.float32)
        for _iteration in range(self.iterations):
            candidates = rng.normal(mean, std, size=(self.candidate_count, self.horizon, action_dim))
            candidates = np.clip(candidates, self.action_low, self.action_high).astype(np.float32)
            candidates = with_scripted_sequences(candidates, self.action_low, self.action_high)
            candidates = with_temporal_chunk_sequences(
                candidates,
                self.temporal_chunk_size,
                candidate_count=self.temporal_chunk_candidates,
                start_offset=len(scripted_sequences(self.action_low, self.action_high, self.horizon)),
            )
            candidates = with_actor_prior_sequences(
                candidates,
                actor_sequence,
                self.action_low,
                self.action_high,
                rng,
                actor_prior_candidates=int(actor_prior_candidates),
                actor_noise=float(actor_noise),
            )
            candidates = with_goal_prior_sequences(
                candidates,
                goal_sequence,
                self.action_low,
                self.action_high,
                rng,
                candidate_count=int(goal_prior_candidates),
                noise=float(actor_noise),
            )
            scores, uncertainties = model.score_sequences(
                observation,
                candidates,
                discount=float(discount),
                done_penalty=float(done_penalty),
                uncertainty_penalty=float(uncertainty_penalty),
                value_model=value_model,
                action_value_model=action_value_model,
                action_value_weight=float(action_value_weight),
                manifold_model=manifold_model,
                off_manifold_penalty=float(off_manifold_penalty),
                value_overestimate_penalty=float(value_overestimate_penalty),
                value_confidence=confidence,
                value_calibration=bool(value_calibration),
            )
            scores = penalize_temporal_roughness(
                scores,
                candidates,
                self.action_low,
                self.action_high,
                self.temporal_smoothness_penalty,
            )
            scores = add_uncertainty_bonus(scores, uncertainties, uncertainty_bonus)
            scores = gate_uncertain_scores(scores, uncertainties, uncertainty_penalty)
            final_uncertainties = uncertainties
            order = np.argsort(scores)[::-1]
            elite_count = max(2, int(round(self.candidate_count * self.elite_fraction)))
            elite = candidates[order[:elite_count]]
            if float(scores[order[0]]) > best_score:
                best_score = float(scores[order[0]])
                best_uncertainty = float(uncertainties[order[0]])
                best_sequence = candidates[order[0]].copy()
            mean = np.mean(elite, axis=0)
            std = np.maximum(np.std(elite, axis=0), self.noise_floor * (self.action_high - self.action_low))
        summary = model.sequence_summary(
            observation,
            best_sequence,
            discount=discount,
            done_penalty=done_penalty,
            uncertainty_penalty=uncertainty_penalty,
            value_model=value_model,
            action_value_model=action_value_model,
            action_value_weight=float(action_value_weight),
            manifold_model=manifold_model,
            off_manifold_penalty=float(off_manifold_penalty),
            value_overestimate_penalty=float(value_overestimate_penalty),
            value_confidence=confidence,
            value_calibration=bool(value_calibration),
        )
        return {
            "sequence": best_sequence.astype(np.float32),
            "action": best_sequence[0].astype(np.float32),
            "score": float(best_score),
            "uncertainty": float(best_uncertainty),
            "candidate_uncertainty_median": percentile_or_zero(final_uncertainties, 50.0),
            "candidate_uncertainty_p75": percentile_or_zero(final_uncertainties, 75.0),
            "candidate_uncertainty_gate": percentile_or_zero(
                final_uncertainties,
                float(np.clip(uncertainty_gate_quantile, 0.5, 0.99)) * 100.0,
            ),
            "value_confidence": float(confidence),
            **summary,
            **goal_stats,
        }


def gate_uncertain_scores(
    scores: np.ndarray,
    uncertainties: np.ndarray,
    uncertainty_penalty: float,
) -> np.ndarray:
    """Apply a hard relative penalty to high-uncertainty fantasy plans."""
    scores = np.asarray(scores, dtype=np.float32)
    uncertainties = np.asarray(uncertainties, dtype=np.float32)
    if scores.size < 4 or float(uncertainty_penalty) <= 0.0:
        return scores
    threshold = float(np.percentile(uncertainties, 75.0))
    excess = np.maximum(uncertainties - threshold, 0.0)
    scale = float(np.std(scores) + 1e-4)
    return (scores - float(uncertainty_penalty) * scale * excess).astype(np.float32)


def add_uncertainty_bonus(
    scores: np.ndarray,
    uncertainties: np.ndarray,
    uncertainty_bonus: float,
) -> np.ndarray:
    if float(uncertainty_bonus) == 0.0:
        return np.asarray(scores, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    uncertainties = np.asarray(uncertainties, dtype=np.float32)
    if scores.size < 2 or float(np.std(uncertainties)) < 1e-8:
        return scores
    score_scale = float(np.std(scores) + 1e-4)
    uncertainty_z = (uncertainties - float(np.mean(uncertainties))) / float(np.std(uncertainties) + 1e-4)
    return (scores + float(uncertainty_bonus) * score_scale * uncertainty_z).astype(np.float32)


def with_scripted_sequences(
    candidates: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    scripted = scripted_sequences(action_low, action_high, int(candidates.shape[1]))
    count = min(int(scripted.shape[0]), int(candidates.shape[0]))
    if count:
        candidates[:count] = scripted[:count]
    return candidates


def with_actor_prior_sequences(
    candidates: np.ndarray,
    actor_sequence: np.ndarray | None,
    action_low: np.ndarray,
    action_high: np.ndarray,
    rng: np.random.Generator,
    *,
    actor_prior_candidates: int,
    actor_noise: float,
) -> np.ndarray:
    if actor_sequence is None or int(actor_prior_candidates) <= 0:
        return candidates
    action_low = np.asarray(action_low, dtype=np.float32).reshape(1, 1, -1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(1, 1, -1)
    sequence = np.asarray(actor_sequence, dtype=np.float32).reshape(1, candidates.shape[1], candidates.shape[2])
    count = min(int(actor_prior_candidates), int(candidates.shape[0]))
    rows = np.repeat(sequence, count, axis=0)
    if count > 1 and float(actor_noise) > 0.0:
        scale = np.maximum(action_high - action_low, 1e-6)
        noise = rng.normal(0.0, float(actor_noise), size=rows.shape).astype(np.float32)
        rows[1:] = np.clip(rows[1:] + noise[1:] * scale, action_low, action_high)
    candidates[-count:] = np.clip(rows, action_low, action_high).astype(np.float32)
    return candidates


def value_confidence(
    goal_stats: dict[str, float],
    manifold_model,
    observation: np.ndarray,
    *,
    has_goal_prior: bool,
    reachability_power: float,
    manifold_power: float,
) -> float:
    reachability = float(goal_stats.get("goal_reachability", 1.0)) if bool(has_goal_prior) else 1.0
    reachability = float(np.clip(reachability, 0.0, 1.0)) ** float(max(0.0, reachability_power))
    if manifold_model is None:
        return float(np.clip(reachability, 0.0, 1.0))
    distance = float(manifold_model.distance(np.asarray(observation, dtype=np.float32).reshape(1, -1))[0])
    gate = max(float(getattr(manifold_model, "distance_gate", 1.0)), 1e-4)
    manifold_score = float(np.exp(-distance / gate)) ** float(max(0.0, manifold_power))
    return float(np.clip(reachability * manifold_score, 0.0, 1.0))


def goal_prior_sequence(
    goal_control_prior,
    model: EnsembleMLPWorldModel,
    observation: np.ndarray,
    *,
    horizon: int,
    reachability_floor: float,
) -> tuple[np.ndarray | None, dict[str, float]]:
    if goal_control_prior is None:
        return None, {
            "goal_reachability": 0.0,
            "goal_target_norm": 0.0,
            "goal_actor_train_loss": 0.0,
            "goal_actor_rows": 0.0,
        }
    try:
        sequence, stats = goal_control_prior.plan_sequence(model, observation, horizon=max(1, int(horizon)))
    except (RuntimeError, ValueError, FloatingPointError):
        return None, {
            "goal_reachability": 0.0,
            "goal_target_norm": 0.0,
            "goal_actor_train_loss": 0.0,
            "goal_actor_rows": 0.0,
        }
    if float(stats.get("goal_reachability", 0.0)) < float(reachability_floor):
        return None, stats
    return np.asarray(sequence, dtype=np.float32), stats


def with_goal_prior_sequences(
    candidates: np.ndarray,
    goal_sequence: np.ndarray | None,
    action_low: np.ndarray,
    action_high: np.ndarray,
    rng: np.random.Generator,
    *,
    candidate_count: int,
    noise: float,
) -> np.ndarray:
    if goal_sequence is None or int(candidate_count) <= 0:
        return candidates
    action_low = np.asarray(action_low, dtype=np.float32).reshape(1, 1, -1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(1, 1, -1)
    sequence = np.asarray(goal_sequence, dtype=np.float32).reshape(1, candidates.shape[1], candidates.shape[2])
    count = min(int(candidate_count), int(candidates.shape[0]))
    rows = np.repeat(sequence, count, axis=0)
    if count > 1 and float(noise) > 0.0:
        scale = np.maximum(action_high - action_low, 1e-6)
        eps = rng.normal(0.0, float(noise), size=rows.shape).astype(np.float32)
        rows[1:] = np.clip(rows[1:] + eps[1:] * scale, action_low, action_high)
    candidates[-count:] = np.clip(rows, action_low, action_high).astype(np.float32)
    return candidates


def with_temporal_chunk_sequences(
    candidates: np.ndarray,
    chunk_size: int,
    *,
    candidate_count: int,
    start_offset: int,
) -> np.ndarray:
    if int(chunk_size) <= 1 or int(candidate_count) <= 0:
        return candidates
    start = max(0, int(start_offset))
    available = max(0, int(candidates.shape[0]) - start)
    count = min(int(candidate_count), available)
    if count <= 0:
        return candidates
    end = start + count
    candidates[start:end] = temporal_chunked_sequences(candidates[start:end], chunk_size=int(chunk_size))
    return candidates


def temporal_chunked_sequences(sequences: np.ndarray, *, chunk_size: int) -> np.ndarray:
    rows = np.asarray(sequences, dtype=np.float32).copy()
    chunk = max(1, int(chunk_size))
    horizon = int(rows.shape[1])
    for start in range(0, horizon, chunk):
        end = min(horizon, start + chunk)
        rows[:, start:end, :] = rows[:, start : start + 1, :]
    return rows.astype(np.float32)


def penalize_temporal_roughness(
    scores: np.ndarray,
    candidates: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    weight: float,
) -> np.ndarray:
    if float(weight) <= 0.0 or int(candidates.shape[1]) <= 1:
        return np.asarray(scores, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(1, 1, -1)
    high = np.asarray(action_high, dtype=np.float32).reshape(1, 1, -1)
    scale = np.maximum(high - low, 1e-6)
    normalized = 2.0 * (np.asarray(candidates, dtype=np.float32) - low) / scale - 1.0
    roughness = np.mean(np.abs(np.diff(normalized, axis=1)), axis=(1, 2))
    score_scale = float(np.std(scores) + 1e-4)
    return (np.asarray(scores, dtype=np.float32) - float(weight) * score_scale * roughness).astype(np.float32)


def actor_prior_sequence(
    actor_model: ActorPolicyModel | None,
    model: EnsembleMLPWorldModel,
    observation: np.ndarray,
    *,
    horizon: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray | None:
    if actor_model is None:
        return None
    try:
        sequence = actor_model.plan_sequence(model, observation, horizon=horizon)
    except (RuntimeError, ValueError):
        return None
    sequence = np.asarray(sequence, dtype=np.float32)
    expected = (max(1, int(horizon)), int(np.asarray(action_low).reshape(-1).shape[0]))
    if sequence.shape != expected:
        return None
    return np.clip(sequence, action_low, action_high).astype(np.float32)


def scripted_sequences(action_low: np.ndarray, action_high: np.ndarray, horizon: int) -> np.ndarray:
    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    action_dim = int(action_low.shape[0])
    center = np.clip(np.zeros((action_dim,), dtype=np.float32), action_low, action_high)
    rows = [np.tile(center, (horizon, 1))]
    for axis in range(action_dim):
        low_action = center.copy()
        high_action = center.copy()
        low_action[axis] = action_low[axis]
        high_action[axis] = action_high[axis]
        rows.append(np.tile(low_action, (horizon, 1)))
        rows.append(np.tile(high_action, (horizon, 1)))
    return np.stack(rows, axis=0).astype(np.float32)


def percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


__all__ = ["CEMPlanner"]
