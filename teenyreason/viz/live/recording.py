"""Live trace recording methods for training events."""

from __future__ import annotations

from collections import deque

import numpy as np


class LiveTraceRecordingMixin:
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
                "total_env_steps": int(total_env_steps),
                "probe_steps": None if probe_steps is None else int(probe_steps),
                "probe_count": None if probe_count is None else int(probe_count),
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
        self._update_comparison_summary_from_episode(
            variant=variant,
            episode=episode,
            episode_return=episode_return,
            total_env_steps=total_env_steps,
        )
        self._write(force=True)


