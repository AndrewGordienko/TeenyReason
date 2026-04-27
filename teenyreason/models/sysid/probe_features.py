"""Feature construction for particle system-identification probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _safe_array(values: Any, *, dtype=np.float32) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.asarray(values, dtype=np.float32)
    return np.where(np.abs(std) < 1e-3, 1e-3, std).astype(np.float32)


def _normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / _safe_std(std)).astype(np.float32)


@dataclass(frozen=True)
class SysIdFeatureStats:
    """Normalizers and static metadata for system-ID probe features."""

    query_mean: np.ndarray
    query_std: np.ndarray
    outcome_mean: np.ndarray
    outcome_std: np.ndarray
    env_param_mean: np.ndarray
    env_param_std: np.ndarray
    env_param_min: np.ndarray
    env_param_max: np.ndarray
    family_names: tuple[str, ...]
    action_vocab_size: int
    state_dim: int
    window_size: int
    family_query_mean_norm: np.ndarray
    validation_nll_good: float = 0.0
    validation_nll_bad: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats into checkpoint-safe primitives."""
        return {
            "query_mean": np.asarray(self.query_mean, dtype=np.float32),
            "query_std": np.asarray(self.query_std, dtype=np.float32),
            "outcome_mean": np.asarray(self.outcome_mean, dtype=np.float32),
            "outcome_std": np.asarray(self.outcome_std, dtype=np.float32),
            "env_param_mean": np.asarray(self.env_param_mean, dtype=np.float32),
            "env_param_std": np.asarray(self.env_param_std, dtype=np.float32),
            "env_param_min": np.asarray(self.env_param_min, dtype=np.float32),
            "env_param_max": np.asarray(self.env_param_max, dtype=np.float32),
            "family_names": np.asarray(self.family_names, dtype="U"),
            "action_vocab_size": int(self.action_vocab_size),
            "state_dim": int(self.state_dim),
            "window_size": int(self.window_size),
            "family_query_mean_norm": np.asarray(self.family_query_mean_norm, dtype=np.float32),
            "validation_nll_good": float(self.validation_nll_good),
            "validation_nll_bad": float(self.validation_nll_bad),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SysIdFeatureStats":
        """Rebuild stats saved by :meth:`to_dict`."""
        family_names = tuple(str(item) for item in np.asarray(payload["family_names"], dtype="U").tolist())
        return cls(
            query_mean=_safe_array(payload["query_mean"]),
            query_std=_safe_std(payload["query_std"]),
            outcome_mean=_safe_array(payload["outcome_mean"]),
            outcome_std=_safe_std(payload["outcome_std"]),
            env_param_mean=_safe_array(payload["env_param_mean"]),
            env_param_std=_safe_std(payload["env_param_std"]),
            env_param_min=_safe_array(payload["env_param_min"]),
            env_param_max=_safe_array(payload["env_param_max"]),
            family_names=family_names,
            action_vocab_size=int(payload["action_vocab_size"]),
            state_dim=int(payload["state_dim"]),
            window_size=int(payload["window_size"]),
            family_query_mean_norm=_safe_array(payload.get("family_query_mean_norm", np.zeros((len(family_names), 0), dtype=np.float32))),
            validation_nll_good=float(payload.get("validation_nll_good", 0.0)),
            validation_nll_bad=float(payload.get("validation_nll_bad", 1.0)),
        )


@dataclass(frozen=True)
class SysIdFeatureBatch:
    """Normalized system-ID features for every saved probe window."""

    query_features: np.ndarray
    outcome_features: np.ndarray
    env_params_raw: np.ndarray
    env_params_norm: np.ndarray
    family_ids: np.ndarray
    env_instance_id: np.ndarray
    stats: SysIdFeatureStats


def _family_names_from_windows(windows: dict[str, np.ndarray], family_names: Sequence[str] | None) -> tuple[str, ...]:
    if family_names is not None:
        return tuple(str(name) for name in family_names)
    modes = np.asarray(windows["probe_mode"], dtype="U")
    return tuple(sorted({str(mode) for mode in modes.tolist()}))


def _family_ids(modes: np.ndarray, family_names: Sequence[str]) -> np.ndarray:
    family_to_id = {str(name): idx for idx, name in enumerate(family_names)}
    return np.asarray([family_to_id.get(str(mode), -1) for mode in modes.tolist()], dtype=np.int64)


def _query_features(states: np.ndarray, actions: np.ndarray, action_vocab_size: int) -> np.ndarray:
    state0 = states[:, 0, :].astype(np.float32)
    actions_i = np.asarray(actions, dtype=np.int64)
    denom = float(max(int(action_vocab_size) - 1, 1))
    hist = np.zeros((actions_i.shape[0], int(action_vocab_size)), dtype=np.float32)
    for row_idx in range(actions_i.shape[0]):
        clipped = np.clip(actions_i[row_idx], 0, int(action_vocab_size) - 1)
        counts = np.bincount(clipped, minlength=int(action_vocab_size)).astype(np.float32)
        hist[row_idx] = counts / float(max(clipped.shape[0], 1))
    action_mean = (actions_i.mean(axis=1, dtype=np.float32) / denom).reshape(-1, 1)
    action_std = (actions_i.std(axis=1, dtype=np.float32) / denom).reshape(-1, 1)
    window_fraction = np.ones((states.shape[0], 1), dtype=np.float32)
    return np.concatenate([state0, hist, action_mean, action_std, window_fraction], axis=1).astype(np.float32)


def _outcome_features(
    states: np.ndarray,
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> np.ndarray:
    state0 = states[:, 0, :].astype(np.float32)
    state_t = states[:, -1, :].astype(np.float32)
    delta = np.diff(states.astype(np.float32), axis=1)
    net_delta = state_t - state0
    delta_mean = delta.mean(axis=1)
    delta_std = delta.std(axis=1)
    abs_delta = np.abs(delta)
    abs_delta_mean = abs_delta.mean(axis=1)
    abs_delta_max = abs_delta.max(axis=1)
    reward_sum = rewards.sum(axis=1, dtype=np.float32).reshape(-1, 1)
    reward_mean = rewards.mean(axis=1, dtype=np.float32).reshape(-1, 1)
    reward_std = rewards.std(axis=1, dtype=np.float32).reshape(-1, 1)
    terminated_f = np.asarray(terminated, dtype=np.float32).reshape(-1, 1)
    truncated_f = np.asarray(truncated, dtype=np.float32).reshape(-1, 1)
    return np.concatenate(
        [
            state_t,
            net_delta,
            delta_mean,
            delta_std,
            abs_delta_mean,
            abs_delta_max,
            reward_sum,
            reward_mean,
            reward_std,
            terminated_f,
            truncated_f,
        ],
        axis=1,
    ).astype(np.float32)


def _family_query_means(query_norm: np.ndarray, family_ids: np.ndarray, family_count: int) -> np.ndarray:
    result = np.zeros((family_count, query_norm.shape[1]), dtype=np.float32)
    global_mean = query_norm.mean(axis=0).astype(np.float32)
    for family_idx in range(family_count):
        mask = family_ids == family_idx
        result[family_idx] = query_norm[mask].mean(axis=0) if np.any(mask) else global_mean
    return result


def build_probe_sysid_features(
    windows: dict[str, np.ndarray],
    action_vocab_size: int,
    family_names: Sequence[str] | None = None,
    fit_mask: np.ndarray | None = None,
) -> SysIdFeatureBatch:
    """Build normalized system-ID features from stored probe windows."""
    states = _safe_array(windows["states"])
    actions = np.asarray(windows["actions"], dtype=np.int64)
    rewards = _safe_array(windows["rewards"])
    env_params = _safe_array(windows["env_params"])
    terminated = np.asarray(windows.get("terminated", np.zeros((states.shape[0],), dtype=np.bool_)), dtype=np.bool_)
    truncated = np.asarray(windows.get("truncated", np.zeros((states.shape[0],), dtype=np.bool_)), dtype=np.bool_)
    modes = np.asarray(windows["probe_mode"], dtype="U")
    names = _family_names_from_windows(windows, family_names)
    family_ids = _family_ids(modes, names)

    query_raw = _query_features(states, actions, action_vocab_size)
    outcome_raw = _outcome_features(states, rewards, terminated, truncated)
    if fit_mask is None:
        fit_rows = np.ones((states.shape[0],), dtype=np.bool_)
    else:
        fit_rows = np.asarray(fit_mask, dtype=np.bool_).reshape(-1)
        if fit_rows.shape[0] != states.shape[0] or not np.any(fit_rows):
            fit_rows = np.ones((states.shape[0],), dtype=np.bool_)
    query_fit = query_raw[fit_rows]
    outcome_fit = outcome_raw[fit_rows]
    env_param_fit = env_params[fit_rows]
    query_mean = query_fit.mean(axis=0).astype(np.float32)
    query_std = _safe_std(query_fit.std(axis=0).astype(np.float32))
    outcome_mean = outcome_fit.mean(axis=0).astype(np.float32)
    outcome_std = _safe_std(outcome_fit.std(axis=0).astype(np.float32))
    env_param_mean = env_param_fit.mean(axis=0).astype(np.float32)
    env_param_std = _safe_std(env_param_fit.std(axis=0).astype(np.float32))

    query_norm = _normalize(query_raw, query_mean, query_std)
    outcome_norm = _normalize(outcome_raw, outcome_mean, outcome_std)
    env_params_norm = _normalize(env_params, env_param_mean, env_param_std)
    stats = SysIdFeatureStats(
        query_mean=query_mean,
        query_std=query_std,
        outcome_mean=outcome_mean,
        outcome_std=outcome_std,
        env_param_mean=env_param_mean,
        env_param_std=env_param_std,
        env_param_min=env_param_fit.min(axis=0).astype(np.float32),
        env_param_max=env_param_fit.max(axis=0).astype(np.float32),
        family_names=names,
        action_vocab_size=int(action_vocab_size),
        state_dim=int(states.shape[-1]),
        window_size=int(actions.shape[1]),
        family_query_mean_norm=_family_query_means(query_norm[fit_rows], family_ids[fit_rows], len(names)),
    )
    return SysIdFeatureBatch(
        query_features=query_norm.astype(np.float32),
        outcome_features=outcome_norm.astype(np.float32),
        env_params_raw=env_params.astype(np.float32),
        env_params_norm=env_params_norm.astype(np.float32),
        family_ids=family_ids,
        env_instance_id=np.asarray(windows["env_instance_id"], dtype=np.int64),
        stats=stats,
    )


def probe_record_features(record: dict[str, Any], stats: SysIdFeatureStats) -> tuple[np.ndarray, np.ndarray, int]:
    """Build normalized query/outcome features for one runtime probe record."""
    states = _safe_array(record["states"])[None, ...]
    actions = np.asarray(record["actions"], dtype=np.int64)[None, ...]
    rewards = _safe_array(record.get("rewards", np.ones((actions.shape[1],), dtype=np.float32)))[None, ...]
    terminated = np.asarray([bool(record.get("terminated", False))], dtype=np.bool_)
    truncated = np.asarray([bool(record.get("truncated", False))], dtype=np.bool_)
    query_raw = _query_features(states, actions, stats.action_vocab_size)
    outcome_raw = _outcome_features(states, rewards, terminated, truncated)
    family = str(record.get("probe_family", record.get("chosen_family", "")))
    family_to_id = {name: idx for idx, name in enumerate(stats.family_names)}
    family_id = int(family_to_id.get(family, -1))
    query = _normalize(query_raw, stats.query_mean, stats.query_std).reshape(-1).astype(np.float32)
    outcome = _normalize(outcome_raw, stats.outcome_mean, stats.outcome_std).reshape(-1).astype(np.float32)
    return query, outcome, family_id
