"""Live dashboard trace writer for training runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ...envs import get_env_display_name
from .cartpole import LiveTraceCartPoleMixin
from .recording import LiveTraceRecordingMixin
from .state import (
    LIVE_TRACE_FILENAME,
    LIVE_TRACE_HISTORY_FILENAME,
    _archive_solve_display,
    _comparison_encoder_probe_steps,
    _first_numeric_summary_value,
    _sanitize_json_value,
    clear_live_trace_history,
    load_live_trace_history,
    load_live_trace_payload,
)


class LiveTrainingTraceWriter(LiveTraceRecordingMixin, LiveTraceCartPoleMixin):
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
        comparison_suite_id: str | None = None,
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
            "comparison_suite_id": comparison_suite_id,
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

    def update_summary(self, summary: dict[str, Any]) -> None:
        """Merge a partial run summary while the current session is still active."""
        if not self.enabled:
            return
        current = self._payload.setdefault("summary", {})
        if not isinstance(current, dict):
            current = {}
        current.update(_sanitize_json_value(summary))
        self._payload["summary"] = current
        self._write(force=True)

    def _update_comparison_summary_from_episode(
        self,
        *,
        variant: str,
        episode: int,
        episode_return: float,
        total_env_steps: int,
    ) -> None:
        """Keep comparison peak metrics stable beyond the bounded curve buffer."""
        if not self._payload.get("comparison_suite_id"):
            return
        if variant not in {"baseline", "probe"}:
            return

        summary = self._payload.setdefault("summary", {})
        if not isinstance(summary, dict):
            summary = {}
            self._payload["summary"] = summary

        best_key = f"{variant}_best_returns"
        current_best = _first_numeric_summary_value(summary, best_key)
        if current_best is not None and float(episode_return) <= current_best:
            return

        if variant == "baseline":
            step_key = "baseline_peak_env_steps"
            seed_step_key = "baseline_best_env_steps"
            peak_steps = int(total_env_steps)
        else:
            step_key = "probe_peak_env_steps_with_encoder"
            seed_step_key = "probe_best_env_steps_with_encoder"
            peak_steps = int(total_env_steps) + _comparison_encoder_probe_steps(summary)

        summary[best_key] = [float(episode_return)]
        summary[step_key] = [int(peak_steps)]
        seed_rows = summary.setdefault("seed_results", [{}])
        if isinstance(seed_rows, list) and seed_rows:
            row = seed_rows[0]
            if isinstance(row, dict):
                row[f"{variant}_best_return"] = float(episode_return)
                row[f"{variant}_best_episode"] = int(episode)
                row[seed_step_key] = int(peak_steps)

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
            "comparison_suite_id": self._payload.get("comparison_suite_id"),
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


__all__ = [
    "LIVE_TRACE_FILENAME",
    "LIVE_TRACE_HISTORY_FILENAME",
    "LiveTrainingTraceWriter",
    "clear_live_trace_history",
    "load_live_trace_history",
    "load_live_trace_payload",
]
