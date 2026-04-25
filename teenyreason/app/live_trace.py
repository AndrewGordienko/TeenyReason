"""Lightweight real-time training trace for the local dashboard.

The training code writes one bounded JSON file in ``artifacts/``. The dashboard
polls that file and renders a live "crawler theater" without needing websocket
infrastructure or a second app server.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ..envs import CONTINUOUS_CARTPOLE_NAME, get_env_display_name
from ..probe.data.probe_env import CARTPOLE_PARAM_NAMES


LIVE_TRACE_FILENAME = "live_training_trace.json"
LIVE_TRACE_HISTORY_FILENAME = "live_training_history.json"


def load_live_trace_payload(artifact_dir: str | Path = "artifacts") -> dict[str, Any]:
    """Load the current live-trace payload or return an empty default."""
    artifact_root = Path(artifact_dir)
    trace_path = artifact_root / LIVE_TRACE_FILENAME
    history_runs = load_live_trace_history(artifact_root)
    if not trace_path.exists():
        return {
            "active": False,
            "finished": False,
            "available": False,
            "env_name": None,
            "env_display_name": None,
            "stage": {"id": "idle", "title": "Awaiting Run", "detail": ""},
            "run": {},
            "focus": {},
            "histories": {},
            "family_scores": [],
            "recent_windows": [],
            "recent_events": [],
            "cartpole_history": [],
            "history_runs": history_runs,
        }
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        payload["history_runs"] = history_runs
        return payload
    except (OSError, json.JSONDecodeError):
        return {
            "active": False,
            "finished": False,
            "available": False,
            "env_name": None,
            "env_display_name": None,
            "stage": {"id": "error", "title": "Live Trace Unavailable", "detail": ""},
            "run": {},
            "focus": {},
            "histories": {},
            "family_scores": [],
            "recent_windows": [],
            "recent_events": [],
            "cartpole_history": [],
            "history_runs": history_runs,
        }


def load_live_trace_history(artifact_dir: str | Path = "artifacts") -> list[dict[str, Any]]:
    """Load archived live-trace sessions for debugging after a run finishes."""
    artifact_root = Path(artifact_dir)
    history_path = artifact_root / LIVE_TRACE_HISTORY_FILENAME
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _summary_solve_values(summary: dict[str, Any], key: str) -> list[int]:
    values = summary.get(key, [])
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for item in values:
        if item is None:
            continue
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            result.append(value)
    return result


def _archive_solve_display(run_variant: str | None, summary: dict[str, Any]) -> tuple[str, list[int]]:
    """Choose one exact archived solve list for the finished variant."""
    variant = "" if run_variant is None else str(run_variant).lower()
    candidates: list[tuple[str, str]] = []
    if "state-only" in variant:
        candidates.append(("state-only eval means", "full_system_state_only_eval_mean_returns"))
    elif "sim-fanout" in variant:
        candidates.append(("sim-fanout solves", "sim_fanout_episode_solves"))
    elif "belief-controller-oracle" in variant or "belief-planner-oracle" in variant:
        candidates.append(("controller oracle solves", "full_system_oracle_episode_solves"))
    elif "belief-controller" in variant or "belief-native" in variant or "belief-planner" in variant:
        candidates.append(("controller solves", "full_system_episode_solves"))
    elif "probe-noexpr" in variant:
        candidates.append(("probe no-expr solves", "probe_no_expression_episode_solves"))
    elif "probe-shadow" in variant:
        candidates.append(("probe shadow solves", "probe_shadow_episode_solves"))
    candidates.append(("probe solves", "probe_episode_solves"))
    for label, key in candidates:
        values = _summary_solve_values(summary, key)
        if values:
            return label, values
    return "saved for debugging", []


class LiveTrainingTraceWriter:
    """Keep a compact, dashboard-friendly snapshot of the current training run."""

    def __init__(
        self,
        artifact_dir: str | Path = "artifacts",
        *,
        enabled: bool = True,
        min_write_interval: float = 0.12,
        max_events: int = 18,
        max_windows: int = 12,
        max_state_history: int = 56,
        max_curve_points: int = 96,
        max_archived_runs: int = 10,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.path = self.artifact_dir / LIVE_TRACE_FILENAME
        self.history_path = self.artifact_dir / LIVE_TRACE_HISTORY_FILENAME
        self.enabled = bool(enabled)
        self.min_write_interval = float(min_write_interval)
        self.max_events = int(max_events)
        self.max_windows = int(max_windows)
        self.max_state_history = int(max_state_history)
        self.max_curve_points = int(max_curve_points)
        self.max_archived_runs = int(max_archived_runs)
        self._last_write_time = 0.0
        self._payload: dict[str, Any] = {
            "available": False,
            "active": False,
            "finished": False,
        }

    def reset_session(
        self,
        *,
        env_name: str,
        benchmark_tag: str,
        seeds: list[int],
        total_runs: int,
    ) -> None:
        if not self.enabled:
            return
        now = time.time()
        self._payload = {
            "version": 1,
            "available": True,
            "active": True,
            "finished": False,
            "session_id": f"{benchmark_tag}-{int(now * 1000)}",
            "started_at": now,
            "updated_at": now,
            "env_name": env_name,
            "env_display_name": get_env_display_name(env_name),
            "benchmark_tag": benchmark_tag,
            "seeds": list(seeds),
            "total_runs": int(total_runs),
            "stage": {
                "id": "boot",
                "title": "Preparing Run",
                "detail": "Waiting for the crawler to enter the first hidden world.",
            },
            "run": {
                "run_index": 0,
                "total_runs": int(total_runs),
                "seed": None,
                "variant": None,
            },
            "focus": {},
            "histories": {
                "encoder_epochs": [],
                "baseline_returns": [],
                "probe_returns": [],
                "probe_shadow_returns": [],
                "probe_noexpr_returns": [],
                "uncertainty": [],
                "message_scale": [],
                "expression_confidence": [],
            },
            "family_scores": [],
            "recent_windows": [],
            "recent_events": [],
            "cartpole_history": [],
            "history_runs": load_live_trace_history(self.artifact_dir),
        }
        self._write(force=True)

    def begin_seed(self, *, run_index: int, total_runs: int, seed: int) -> None:
        self._set_run_context(run_index=run_index, total_runs=total_runs, seed=seed, variant=None)
        self.set_stage(
            "seed_boot",
            "Sampling Hidden World",
            f"Starting seed {seed} and preparing the CartPole world fingerprint.",
            run_index=run_index,
            total_runs=total_runs,
            seed=seed,
        )

    def set_stage(
        self,
        stage_id: str,
        title: str,
        detail: str,
        *,
        run_index: int | None = None,
        total_runs: int | None = None,
        seed: int | None = None,
        variant: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._set_run_context(run_index=run_index, total_runs=total_runs, seed=seed, variant=variant)
        self._payload["stage"] = {
            "id": str(stage_id),
            "title": str(title),
            "detail": str(detail),
        }
        self._append_event(kind="stage", label=title, detail=detail)
        self._write(force=True)

    def record_probe_collection_step(
        self,
        *,
        phase: str,
        state: np.ndarray,
        action_value: float,
        action_index: int | None,
        probe_mode: str,
        env_params: np.ndarray | None,
        env_instance_id: int,
        episode_id: int,
        step_idx: int,
        reward: float | None = None,
    ) -> None:
        if not self._should_visualize_cartpole():
            return
        self._set_cartpole_focus(
            phase=phase,
            state=state,
            action_value=action_value,
            action_index=action_index,
            focus_label=probe_mode,
            reward=reward,
            step_idx=step_idx,
            episode_id=episode_id,
            env_instance_id=env_instance_id,
            env_params=env_params,
        )
        self._write()

    def record_probe_window(
        self,
        *,
        probe_mode: str,
        episode_id: int,
        env_instance_id: int,
        reward_sum: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not self.enabled:
            return
        windows = self._payload.setdefault("recent_windows", [])
        windows.insert(
            0,
            {
                "probe_mode": str(probe_mode),
                "episode_id": int(episode_id),
                "env_instance_id": int(env_instance_id),
                "reward_sum": float(reward_sum),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        )
        del windows[self.max_windows :]
        self._append_event(
            kind="window",
            label=f"Support window: {probe_mode}",
            detail=f"env {env_instance_id} · episode {episode_id} · reward {reward_sum:.2f}",
        )
        self._write()

    def record_encoder_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        window_metrics: dict[str, float],
        env_metrics: dict[str, float],
    ) -> None:
        if not self.enabled:
            return
        self._payload["focus"] = {
            "phase": "encoder_training",
            "epoch": int(epoch),
            "total_epochs": int(total_epochs),
            "total_loss": float(window_metrics.get("loss", 0.0)),
            "env_param_loss": float(env_metrics.get("env_param_loss", 0.0)),
            "env_retrieval_loss": float(env_metrics.get("env_retrieval_loss", 0.0)),
            "uncertainty_calibration_loss": float(env_metrics.get("uncertainty_calibration_loss", 0.0)),
            "dominant_term": str(env_metrics.get("env_dominant_term_name", "unknown")),
            "dominant_value": float(env_metrics.get("env_dominant_term_value", 0.0)),
        }
        history = self._payload.setdefault("histories", {}).setdefault("encoder_epochs", [])
        history.append(
            {
                "epoch": int(epoch),
                "total_loss": float(window_metrics.get("loss", 0.0)),
                "env_param_loss": float(env_metrics.get("env_param_loss", 0.0)),
                "env_retrieval_loss": float(env_metrics.get("env_retrieval_loss", 0.0)),
                "uncertainty_calibration_loss": float(env_metrics.get("uncertainty_calibration_loss", 0.0)),
            }
        )
        del history[:-self.max_curve_points]
        self._write(force=(epoch == total_epochs))

    def record_policy_step(
        self,
        *,
        phase: str,
        variant: str,
        state: np.ndarray,
        action_value: float,
        reward: float,
        episode: int,
        step_idx: int,
        episode_return: float,
        probe_count: int | None = None,
        uncertainty: float | None = None,
        message_scale: float | None = None,
        expression_confidence: float | None = None,
        expression_ready: bool | None = None,
        focus_label: str | None = None,
        expected_return: float | None = None,
        novelty: float | None = None,
        expression_muted_by_policy: bool | None = None,
        expression_mute_reason: str | None = None,
    ) -> None:
        if not self._should_visualize_cartpole():
            return
        self._set_run_context(variant=variant)
        self._payload["focus"] = {
            "phase": str(phase),
            "variant": str(variant),
            "episode": int(episode),
            "step_idx": int(step_idx),
            "episode_return": float(episode_return),
            "probe_count": None if probe_count is None else int(probe_count),
            "uncertainty": None if uncertainty is None else float(uncertainty),
            "message_scale": None if message_scale is None else float(message_scale),
            "expression_confidence": None if expression_confidence is None else float(expression_confidence),
            "expression_ready": None if expression_ready is None else bool(expression_ready),
            "expression_muted_by_policy": (
                None
                if expression_muted_by_policy is None
                else bool(expression_muted_by_policy)
            ),
            "expression_mute_reason": (
                None if expression_mute_reason is None else str(expression_mute_reason)
            ),
            "focus_label": None if focus_label is None else str(focus_label),
            "expected_return": None if expected_return is None else float(expected_return),
            "novelty": None if novelty is None else float(novelty),
        }
        self._set_cartpole_snapshot(
            phase=phase,
            state=state,
            action_value=action_value,
            action_index=None,
            focus_label=focus_label,
            reward=reward,
            step_idx=step_idx,
            episode_id=episode,
            env_instance_id=None,
            env_params=None,
        )
        self._write()

    def record_probe_decision(
        self,
        *,
        episode: int,
        probe_count: int,
        max_probe_episodes: int,
        chosen_family: str | None,
        uncertainty: float,
        surprise: float,
        message_scale: float | None,
        expected_family_gain: dict[str, dict[str, float]],
        realized_family_gain: dict[str, float],
        stop_reason: str | None,
        expression_confidence: float | None = None,
        expression_ready: bool | None = None,
        expression_muted_by_policy: bool | None = None,
        expression_mute_reason: str | None = None,
        chosen_raw_future_estimate: float | None = None,
        chosen_future_estimate: float | None = None,
        chosen_future_gain_for_choice: float | None = None,
        chosen_entropy_reduction: float | None = None,
        chosen_hypothesis_separation: float | None = None,
        chosen_estimated_cost: float | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._payload["focus"] = {
            "phase": "probe_reasoning",
            "episode": int(episode),
            "probe_count": int(probe_count),
            "max_probe_episodes": int(max_probe_episodes),
            "focus_label": None if chosen_family is None else str(chosen_family),
            "uncertainty": float(uncertainty),
            "surprise": float(surprise),
            "message_scale": None if message_scale is None else float(message_scale),
            "expression_confidence": None if expression_confidence is None else float(expression_confidence),
            "expression_ready": None if expression_ready is None else bool(expression_ready),
            "expression_muted_by_policy": (
                None
                if expression_muted_by_policy is None
                else bool(expression_muted_by_policy)
            ),
            "expression_mute_reason": (
                None if expression_mute_reason is None else str(expression_mute_reason)
            ),
            "chosen_raw_future_estimate": (
                None
                if chosen_raw_future_estimate is None
                else float(chosen_raw_future_estimate)
            ),
            "chosen_future_estimate": (
                None if chosen_future_estimate is None else float(chosen_future_estimate)
            ),
            "chosen_future_gain_for_choice": (
                None
                if chosen_future_gain_for_choice is None
                else float(chosen_future_gain_for_choice)
            ),
            "chosen_entropy_reduction": (
                None if chosen_entropy_reduction is None else float(chosen_entropy_reduction)
            ),
            "chosen_hypothesis_separation": (
                None
                if chosen_hypothesis_separation is None
                else float(chosen_hypothesis_separation)
            ),
            "chosen_estimated_cost": (
                None if chosen_estimated_cost is None else float(chosen_estimated_cost)
            ),
            "stop_reason": None if stop_reason is None else str(stop_reason),
        }
        family_rows = []
        for family, metrics in expected_family_gain.items():
            family_rows.append(
                {
                    "family": str(family),
                    "selection_score": float(metrics.get("selection_score", metrics.get("score", 0.0))),
                    "value_per_probe_step": float(metrics.get("value_per_probe_step", 0.0)),
                    "predicted_marginal_value": float(metrics.get("predicted_marginal_value", 0.0)),
                    "predicted_entropy_reduction": float(metrics.get("predicted_entropy_reduction", 0.0)),
                    "predicted_hypothesis_separation": float(metrics.get("predicted_hypothesis_separation", 0.0)),
                    "raw_future_error_estimate": float(metrics.get("raw_future_error_estimate", metrics.get("future_error_estimate", 0.0))),
                    "future_error_estimate": float(metrics.get("future_error_estimate", 0.0)),
                    "future_gain_for_choice": float(metrics.get("future_gain_for_choice", 0.0)),
                    "estimated_probe_cost": float(metrics.get("estimated_probe_cost", 0.0)),
                    "realized_gain": float(realized_family_gain.get(family, 0.0)),
                }
            )
        family_rows.sort(key=lambda row: row["selection_score"], reverse=True)
        self._payload["family_scores"] = family_rows[:8]
        if uncertainty is not None:
            hist = self._payload.setdefault("histories", {}).setdefault("uncertainty", [])
            hist.append(float(uncertainty))
            del hist[:-self.max_curve_points]
        if message_scale is not None:
            hist = self._payload.setdefault("histories", {}).setdefault("message_scale", [])
            hist.append(float(message_scale))
            del hist[:-self.max_curve_points]
        if expression_confidence is not None:
            hist = self._payload.setdefault("histories", {}).setdefault("expression_confidence", [])
            hist.append(float(expression_confidence))
            del hist[:-self.max_curve_points]
        self._write(force=bool(stop_reason))

    def record_episode_summary(
        self,
        *,
        variant: str,
        episode: int,
        episode_return: float,
        avg10: float,
        avg50: float,
        total_env_steps: int,
        probe_steps: int | None = None,
        probe_count: int | None = None,
        uncertainty: float | None = None,
        message_scale: float | None = None,
        expression_confidence: float | None = None,
        expression_ready: bool | None = None,
        stop_reason: str | None = None,
        expression_muted_by_policy: bool | None = None,
        expression_mute_reason: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._set_run_context(variant=variant)
        if variant == "probe":
            key = "probe_returns"
        elif variant == "probe-shadowexpr":
            key = "probe_shadow_returns"
        elif variant == "probe-noexpr":
            key = "probe_noexpr_returns"
        elif variant in {
            "belief-planner",
            "belief-planner-oracle",
            "belief-native",
            "belief-native-oracle",
            "belief-controller",
            "belief-controller-oracle",
            "sim-fanout",
        }:
            key = "probe_returns"
        else:
            key = "baseline_returns"
        history = self._payload.setdefault("histories", {}).setdefault(key, [])
        history.append(
            {
                "episode": int(episode),
                "return": float(episode_return),
                "avg10": float(avg10),
                "avg50": float(avg50),
            }
        )
        del history[:-self.max_curve_points]
        self._append_event(
            kind="episode",
            label=f"{variant} episode {episode:04d}",
            detail=(
                f"return {episode_return:.2f} · avg10 {avg10:.2f} · "
                f"env steps {int(total_env_steps)}"
            ),
        )
        self._payload["focus"] = {
            "phase": f"{variant}_episode_summary",
            "variant": str(variant),
            "episode": int(episode),
            "episode_return": float(episode_return),
            "avg10": float(avg10),
            "avg50": float(avg50),
            "total_env_steps": int(total_env_steps),
            "probe_steps": None if probe_steps is None else int(probe_steps),
            "probe_count": None if probe_count is None else int(probe_count),
            "uncertainty": None if uncertainty is None else float(uncertainty),
            "message_scale": None if message_scale is None else float(message_scale),
            "expression_confidence": None if expression_confidence is None else float(expression_confidence),
            "expression_ready": None if expression_ready is None else bool(expression_ready),
            "expression_muted_by_policy": (
                None
                if expression_muted_by_policy is None
                else bool(expression_muted_by_policy)
            ),
            "expression_mute_reason": (
                None if expression_mute_reason is None else str(expression_mute_reason)
            ),
            "stop_reason": None if stop_reason is None else str(stop_reason),
        }
        self._write(force=True)

    def finish(self, *, summary: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self._payload["active"] = False
        self._payload["finished"] = True
        self._payload["finished_at"] = time.time()
        if summary is not None:
            self._payload["summary"] = _sanitize_json_value(summary)
            archive_label, archive_values = _archive_solve_display(
                self._payload.get("run", {}).get("variant"),
                self._payload["summary"],
            )
            self._payload["archive_solve_label"] = archive_label
            self._payload["archive_solve_values"] = archive_values
        self._append_event(kind="done", label="Run finished", detail="Training pipeline completed.")
        self._write(force=True)
        self._archive_current_payload()

    def _set_run_context(
        self,
        *,
        run_index: int | None = None,
        total_runs: int | None = None,
        seed: int | None = None,
        variant: str | None = None,
    ) -> None:
        if not self.enabled or "run" not in self._payload:
            return
        run = self._payload.setdefault("run", {})
        if run_index is not None:
            run["run_index"] = int(run_index)
        if total_runs is not None:
            run["total_runs"] = int(total_runs)
        if seed is not None:
            run["seed"] = int(seed)
        if variant is not None:
            run["variant"] = str(variant)

    def _should_visualize_cartpole(self) -> bool:
        return bool(self.enabled and self._payload.get("env_name") == CONTINUOUS_CARTPOLE_NAME)

    def _set_cartpole_focus(
        self,
        *,
        phase: str,
        state: np.ndarray,
        action_value: float,
        action_index: int | None,
        focus_label: str,
        reward: float | None,
        step_idx: int,
        episode_id: int,
        env_instance_id: int,
        env_params: np.ndarray | None,
    ) -> None:
        self._payload["focus"] = {
            "phase": str(phase),
            "focus_label": str(focus_label),
            "episode": int(episode_id),
            "env_instance_id": int(env_instance_id),
            "step_idx": int(step_idx),
            "reward": None if reward is None else float(reward),
        }
        self._set_cartpole_snapshot(
            phase=phase,
            state=state,
            action_value=action_value,
            action_index=action_index,
            focus_label=focus_label,
            reward=reward,
            step_idx=step_idx,
            episode_id=episode_id,
            env_instance_id=env_instance_id,
            env_params=env_params,
        )

    def _set_cartpole_snapshot(
        self,
        *,
        phase: str,
        state: np.ndarray,
        action_value: float,
        action_index: int | None,
        focus_label: str | None,
        reward: float | None,
        step_idx: int,
        episode_id: int,
        env_instance_id: int | None,
        env_params: np.ndarray | None,
    ) -> None:
        state_np = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_np.shape[0] < 4:
            return
        snapshot = {
            "phase": str(phase),
            "x": float(state_np[0]),
            "x_dot": float(state_np[1]),
            "theta": float(state_np[2]),
            "theta_dot": float(state_np[3]),
            "action_value": float(action_value),
            "action_index": None if action_index is None else int(action_index),
            "focus_label": None if focus_label is None else str(focus_label),
            "reward": None if reward is None else float(reward),
            "step_idx": int(step_idx),
            "episode_id": int(episode_id),
            "env_instance_id": None if env_instance_id is None else int(env_instance_id),
        }
        if env_params is not None:
            params = np.asarray(env_params, dtype=np.float32).reshape(-1)
            snapshot["env_params"] = {
                name: float(params[idx])
                for idx, name in enumerate(CARTPOLE_PARAM_NAMES[: params.shape[0]])
            }
        self._payload["cartpole"] = snapshot
        history = self._payload.setdefault("cartpole_history", [])
        history.append(
            {
                "phase": str(phase),
                "x": float(state_np[0]),
                "theta": float(state_np[2]),
                "focus_label": None if focus_label is None else str(focus_label),
            }
        )
        del history[:-self.max_state_history]

    def _append_event(self, *, kind: str, label: str, detail: str) -> None:
        events = self._payload.setdefault("recent_events", [])
        events.insert(
            0,
            {
                "kind": str(kind),
                "label": str(label),
                "detail": str(detail),
                "timestamp": time.time(),
            },
        )
        del events[self.max_events :]

    def _write(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and (now - self._last_write_time) < self.min_write_interval:
            return
        self._last_write_time = now
        self._payload["updated_at"] = time.time()
        self.artifact_dir.mkdir(exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(_sanitize_json_value(self._payload), indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def _archive_current_payload(self) -> None:
        if not self.enabled:
            return
        archived_runs = load_live_trace_history(self.artifact_dir)
        session_id = self._payload.get("session_id")
        archived_runs = [row for row in archived_runs if row.get("session_id") != session_id]
        archived_runs.insert(0, _sanitize_json_value({
            "session_id": session_id,
            "started_at": self._payload.get("started_at"),
            "finished_at": self._payload.get("finished_at", time.time()),
            "available": True,
            "active": False,
            "finished": True,
            "env_name": self._payload.get("env_name"),
            "env_display_name": self._payload.get("env_display_name"),
            "benchmark_tag": self._payload.get("benchmark_tag"),
            "stage": self._payload.get("stage", {}),
            "run": self._payload.get("run", {}),
            "focus": self._payload.get("focus", {}),
            "summary": self._payload.get("summary", {}),
            "archive_solve_label": self._payload.get("archive_solve_label"),
            "archive_solve_values": self._payload.get("archive_solve_values", []),
            "histories": self._payload.get("histories", {}),
            "family_scores": self._payload.get("family_scores", []),
            "recent_windows": self._payload.get("recent_windows", []),
            "recent_events": self._payload.get("recent_events", []),
            "cartpole": self._payload.get("cartpole", {}),
            "cartpole_history": self._payload.get("cartpole_history", []),
        }))
        del archived_runs[self.max_archived_runs :]
        self._payload["history_runs"] = archived_runs
        tmp_path = self.history_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(archived_runs, indent=2), encoding="utf-8")
        tmp_path.replace(self.history_path)


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, deque):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _sanitize_json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else 0.0
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


__all__ = [
    "LIVE_TRACE_FILENAME",
    "LIVE_TRACE_HISTORY_FILENAME",
    "LiveTrainingTraceWriter",
    "load_live_trace_history",
    "load_live_trace_payload",
]
