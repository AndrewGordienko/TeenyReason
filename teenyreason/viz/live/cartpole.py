"""CartPole visualization helpers for live traces."""

from __future__ import annotations

import numpy as np

from ...envs import CONTINUOUS_CARTPOLE_NAME
from ...crawler.probes.data.probe_env import CARTPOLE_PARAM_NAMES


class LiveTraceCartPoleMixin:
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


