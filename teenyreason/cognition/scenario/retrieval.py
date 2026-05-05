"""Retrieve familiar windows from scenario memory."""

from __future__ import annotations

import numpy as np

from .memory import ScenarioMemory
from .schema import ScenarioTracelet, ScenarioWindow


def retrieve_windows(
    memory: ScenarioMemory,
    *,
    count: int,
    window_size: int,
    focus_observations: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray | None = None,
    focus_weight: float = 0.0,
) -> list[ScenarioWindow]:
    """Rank short real windows by value, familiarity, and failure-frontier usefulness."""
    mean, std = memory.observation_stats()
    focus = normalize_focus(focus_observations, mean.shape[0])
    windows = candidate_windows(memory, window_size=max(1, int(window_size)))
    scored = [(window_score(window, mean, std, focus, focus_weight=float(focus_weight)), window) for window in windows]
    ranked = sorted(scored, key=lambda item: item[0], reverse=True)
    return [window for _score, window in ranked[: max(0, int(count))]]


def candidate_windows(memory: ScenarioMemory, *, window_size: int) -> list[ScenarioWindow]:
    rows: list[ScenarioWindow] = []
    for tracelets in memory.by_trajectory().values():
        real = [item for item in tracelets if item.source == "real"]
        if not real:
            continue
        starts = start_indices(real)
        for start in starts:
            chunk = tuple(real[start : min(len(real), start + int(window_size))])
            if chunk:
                rows.append(make_window(chunk))
    return rows


def start_indices(tracelets: list[ScenarioTracelet]) -> list[int]:
    if len(tracelets) <= 1:
        return [0]
    returns = np.asarray([item.return_to_go for item in tracelets], dtype=np.float32)
    dones = np.asarray([item.done for item in tracelets], dtype=np.float32)
    terminal = int(np.argmax(dones)) if float(np.max(dones)) > 0.5 else len(tracelets) - 1
    raw = [
        int(np.argmax(returns)),
        max(0, terminal - 12),
        max(0, terminal - 6),
        max(0, int(round(0.5 * (len(tracelets) - 1)))),
    ]
    out: list[int] = []
    for value in raw:
        idx = max(0, min(int(value), len(tracelets) - 1))
        if idx not in out:
            out.append(idx)
    return out


def make_window(tracelets: tuple[ScenarioTracelet, ...]) -> ScenarioWindow:
    terminal_distance = 1.0
    for idx, item in enumerate(tracelets):
        if float(item.done) > 0.5:
            terminal_distance = float(idx / max(1, len(tracelets) - 1))
            break
    surprise = float(np.mean([item.surprise for item in tracelets])) if tracelets else 0.0
    familiarity = float(1.0 / (1.0 + surprise))
    return ScenarioWindow(tracelets=tracelets, familiarity=familiarity, terminal_distance=terminal_distance, mean_surprise=surprise)


def window_score(
    window: ScenarioWindow,
    mean: np.ndarray,
    std: np.ndarray,
    focus: np.ndarray | None,
    *,
    focus_weight: float,
) -> float:
    start = (window.start_observation - mean) / std
    spread_penalty = float(np.mean(np.square(start))) * 0.01
    survival_bonus = float(window.terminal_distance)
    focus_bonus = 0.0
    if focus is not None and float(focus_weight) > 0.0:
        focus_z = (focus - mean.reshape(1, -1)) / std.reshape(1, -1)
        distance = float(np.min(np.mean(np.square(focus_z - start.reshape(1, -1)), axis=1)))
        focus_bonus = float(focus_weight) / (1.0 + distance)
    return float(window.best_return_to_go + 0.25 * window.observed_return + survival_bonus + focus_bonus - spread_penalty - window.mean_surprise)


def normalize_focus(
    focus_observations: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray | None,
    obs_dim: int,
) -> np.ndarray | None:
    if focus_observations is None:
        return None
    focus = np.asarray(focus_observations, dtype=np.float32)
    if focus.size == 0:
        return None
    return focus.reshape(-1, int(obs_dim)).astype(np.float32)


__all__ = ["retrieve_windows"]
