"""Local scenario imagination around remembered traces."""

from __future__ import annotations

import numpy as np

from teenyreason.multidomain.planning.generic.model import ActionValueModel, EnsembleMLPWorldModel, ValueBootstrapModel

from .schema import ScenarioVariant, ScenarioWindow
from .weights import normalize_variant_weights, score_variant_weights


def generate_variants(
    config,
    model: EnsembleMLPWorldModel,
    value_model: ValueBootstrapModel | None,
    action_value_model: ActionValueModel | None,
    windows: list[ScenarioWindow],
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    round_idx: int,
) -> list[ScenarioVariant]:
    """Create small local variations of remembered action traces."""
    rng = np.random.default_rng(int(config.seed) + 211_000 + int(round_idx) * 997)
    variants: list[ScenarioVariant] = []
    for window_idx, window in enumerate(windows):
        base = fit_horizon(window.actions, max(1, int(config.scenario_variant_horizon)), action_low, action_high)
        for variant_idx, actions in enumerate(action_variants(config, base, action_low, action_high, rng)):
            rows, predicted_return, done_risk, uncertainty = rollout_actions(model, window.start_observation, actions)
            value = final_value(config, model, value_model, action_value_model, window.start_observation, actions)
            predicted_total = float(predicted_return + value)
            predicted_lift = float(predicted_total - window.observed_return)
            weights = score_variant_weights(
                window,
                predicted_lift=predicted_lift,
                uncertainty=uncertainty,
                done_risk=done_risk,
                variant_surprise=window.mean_surprise,
                advantage_temperature=float(config.scenario_advantage_temperature),
                uncertainty_scale=float(config.scenario_uncertainty_scale),
                surprise_scale=float(config.scenario_surprise_scale),
            )
            variants.append(
                ScenarioVariant(
                    window=window,
                    actions=actions,
                    rows=tuple(rows),
                    predicted_return=float(predicted_return),
                    predicted_value=float(value),
                    predicted_lift=predicted_lift,
                    uncertainty=float(uncertainty),
                    done_risk=float(done_risk),
                    weights=weights,
                    variant_kind=f"local_{window_idx}_{variant_idx}",
                )
            )
    return variants


def variants_to_training_rows(
    config,
    variants: list[ScenarioVariant],
    real_obs: np.ndarray,
    real_actions: np.ndarray,
    real_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge real rows with softly weighted imagined rows."""
    variant_rows = [row for variant in variants for row in variant.rows]
    if not variant_rows:
        return real_obs, real_actions, real_weights
    obs = np.asarray([row["observation"] for row in variant_rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in variant_rows], dtype=np.float32)
    per_variant = normalize_variant_weights(variants)
    weights = np.concatenate(
        [
            np.full((len(variant.rows),), per_variant[idx], dtype=np.float32)
            for idx, variant in enumerate(variants)
        ],
        axis=0,
    )
    weights *= float(config.scenario_imagined_weight)
    return (
        np.concatenate([real_obs, obs], axis=0),
        np.concatenate([real_actions, actions], axis=0),
        np.concatenate([real_weights, weights], axis=0),
    )


def action_variants(config, base: np.ndarray, action_low: np.ndarray, action_high: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    variants = [base.astype(np.float32)]
    scale = np.maximum(np.asarray(action_high - action_low, dtype=np.float32), 1e-6).reshape(1, -1)
    while len(variants) < max(1, int(config.scenario_variants_per_window)):
        noise = rng.normal(0.0, float(config.scenario_action_noise), size=base.shape).astype(np.float32) * scale
        if len(variants) % 3 == 0:
            actions = np.roll(base, shift=1, axis=0) + noise
        elif len(variants) % 3 == 1:
            actions = base + noise
        else:
            actions = 0.75 * base + noise
        variants.append(np.clip(actions, action_low, action_high).astype(np.float32))
    return variants[: max(1, int(config.scenario_variants_per_window))]


def rollout_actions(
    model: EnsembleMLPWorldModel,
    observation: np.ndarray,
    actions: np.ndarray,
) -> tuple[list[dict[str, np.ndarray | float]], float, float, float]:
    obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
    rows: list[dict[str, np.ndarray | float]] = []
    rewards: list[float] = []
    risks: list[float] = []
    uncertainties: list[float] = []
    for action in np.asarray(actions, dtype=np.float32):
        pred = model.predict_batch(obs, action.reshape(1, -1))
        reward = float(pred["reward"][0])
        risk = float(pred["done_risk"][0])
        next_obs = np.asarray(pred["next_observation"][0], dtype=np.float32).reshape(-1)
        rows.append({"observation": obs.reshape(-1).copy(), "action": action.copy(), "reward": reward, "next_observation": next_obs.copy(), "done": float(risk >= 0.5)})
        rewards.append(reward)
        risks.append(risk)
        uncertainties.append(float(pred["uncertainty"][0]))
        obs = next_obs.reshape(1, -1)
        if risk >= 0.5:
            break
    return rows, sum(rewards), max(risks, default=0.0), mean(uncertainties)


def final_value(
    config,
    model: EnsembleMLPWorldModel,
    value_model: ValueBootstrapModel | None,
    action_value_model: ActionValueModel | None,
    observation: np.ndarray,
    actions: np.ndarray,
) -> float:
    summary = model.sequence_summary(
        observation,
        actions,
        discount=float(config.discount),
        done_penalty=float(config.done_penalty),
        uncertainty_penalty=float(config.uncertainty_penalty),
        value_model=value_model if bool(config.value_bootstrap) else None,
        action_value_model=action_value_model if bool(config.action_value_bootstrap) else None,
        action_value_weight=float(config.action_value_score_weight),
    )
    return float(summary.get("predicted_value_total", 0.0) + summary.get("predicted_action_value_total", 0.0))


def fit_horizon(actions: np.ndarray, horizon: int, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    rows = np.asarray(actions, dtype=np.float32).reshape(-1, int(action_low.shape[0]))
    if rows.shape[0] == 0:
        center = np.clip(np.zeros_like(action_low, dtype=np.float32), action_low, action_high)
        return np.repeat(center.reshape(1, -1), max(1, int(horizon)), axis=0)
    if rows.shape[0] >= int(horizon):
        return np.clip(rows[: int(horizon)], action_low, action_high).astype(np.float32)
    tail = np.repeat(rows[-1:].copy(), int(horizon) - rows.shape[0], axis=0)
    return np.clip(np.concatenate([rows, tail], axis=0), action_low, action_high).astype(np.float32)


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


__all__ = ["generate_variants", "variants_to_training_rows"]
