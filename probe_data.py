"""Probe data collection for both one-step transitions and short history windows."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from envs import (
    BIPEDAL_WALKER_NAME,
    CONTINUOUS_CARTPOLE_NAME,
    CONTINUOUS_LUNAR_LANDER_NAME,
    action_index_to_env_action,
    get_action_dim,
    get_action_values,
    make_env,
)


PROBE_MODES = (
    "random",
    "hold_left",
    "hold_right",
    "center_hold",
    "pulse_left",
    "pulse_right",
    "reverse",
    "sweep",
    "sticky_random",
)

# These names are mostly for readability when inspecting stored parameter arrays.
CARTPOLE_PARAM_NAMES = (
    "gravity",
    "masscart",
    "masspole",
    "length",
    "force_mag",
)

LUNAR_LANDER_PARAM_NAMES = (
    "gravity",
    "wind_power",
    "turbulence_power",
    "enable_wind",
)

BIPEDAL_WALKER_PARAM_NAMES = (
    "motors_torque",
    "speed_hip",
    "speed_knee",
    "lidar_range",
    "friction",
)


@dataclass(frozen=True)
class CartPolePhysics:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.gravity,
                self.masscart,
                self.masspole,
                self.length,
                self.force_mag,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class LunarLanderPhysics:
    gravity: float = -10.0
    wind_power: float = 15.0
    turbulence_power: float = 1.5
    enable_wind: bool = False

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.gravity,
                self.wind_power,
                self.turbulence_power,
                float(self.enable_wind),
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class BipedalWalkerPhysics:
    motors_torque: float = 80.0
    speed_hip: float = 4.0
    speed_knee: float = 6.0
    lidar_range: float = 160.0 / 30.0
    friction: float = 2.5

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.motors_torque,
                self.speed_hip,
                self.speed_knee,
                self.lidar_range,
                self.friction,
            ],
            dtype=np.float32,
        )


def default_cartpole_physics(env=None) -> CartPolePhysics:
    if env is None:
        return CartPolePhysics()

    base_env = env.unwrapped
    return CartPolePhysics(
        gravity=float(base_env.gravity),
        masscart=float(base_env.masscart),
        masspole=float(base_env.masspole),
        length=float(base_env.length),
        force_mag=float(base_env.force_mag),
    )


def default_lunar_lander_physics(env=None) -> LunarLanderPhysics:
    if env is None:
        return LunarLanderPhysics()

    base_env = env.unwrapped
    return LunarLanderPhysics(
        gravity=float(getattr(base_env, "gravity", -10.0)),
        wind_power=float(getattr(base_env, "wind_power", 15.0)),
        turbulence_power=float(getattr(base_env, "turbulence_power", 1.5)),
        enable_wind=bool(getattr(base_env, "enable_wind", False)),
    )


def default_bipedal_walker_physics(env=None) -> BipedalWalkerPhysics:
    del env
    # Gymnasium's BipedalWalker uses module-level Box2D constants rather than
    # per-instance physics fields, so we track reference values here even
    # though the current environment path does not mutate them online.
    return BipedalWalkerPhysics()


def sample_cartpole_physics(
    rng: np.random.Generator,
    base_physics: CartPolePhysics | None = None,
) -> CartPolePhysics:
    base = base_physics or CartPolePhysics()
    return CartPolePhysics(
        gravity=float(rng.uniform(0.75 * base.gravity, 1.25 * base.gravity)),
        masscart=float(rng.uniform(0.8 * base.masscart, 1.6 * base.masscart)),
        masspole=float(rng.uniform(0.5 * base.masspole, 2.0 * base.masspole)),
        length=float(rng.uniform(0.7 * base.length, 1.5 * base.length)),
        force_mag=float(rng.uniform(0.6 * base.force_mag, 1.4 * base.force_mag)),
    )


def sample_lunar_lander_physics(
    rng: np.random.Generator,
    base_physics: LunarLanderPhysics | None = None,
) -> LunarLanderPhysics:
    base = base_physics or LunarLanderPhysics()
    gravity = float(np.clip(rng.uniform(base.gravity - 1.5, base.gravity + 1.5), -11.8, -2.0))
    wind_power = float(np.clip(rng.uniform(0.0, max(4.0, 1.5 * base.wind_power)), 0.0, 20.0))
    turbulence_power = float(
        np.clip(rng.uniform(0.0, max(0.5, 1.5 * base.turbulence_power)), 0.0, 2.0)
    )
    enable_wind = bool(rng.random() < 0.5) if not base.enable_wind else bool(rng.random() < 0.8)
    return LunarLanderPhysics(
        gravity=gravity,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
        enable_wind=enable_wind,
    )


def sample_bipedal_walker_physics(
    rng: np.random.Generator,
    base_physics: BipedalWalkerPhysics | None = None,
) -> BipedalWalkerPhysics:
    del rng
    return base_physics or BipedalWalkerPhysics()


def apply_cartpole_physics(env, physics: CartPolePhysics):
    base_env = env.unwrapped
    base_env.gravity = physics.gravity
    base_env.masscart = physics.masscart
    base_env.masspole = physics.masspole
    base_env.length = physics.length
    base_env.force_mag = physics.force_mag
    base_env.total_mass = base_env.masspole + base_env.masscart
    base_env.polemass_length = base_env.masspole * base_env.length


def apply_lunar_lander_physics(env, physics: LunarLanderPhysics):
    base_env = env.unwrapped
    base_env.gravity = physics.gravity
    base_env.wind_power = physics.wind_power
    base_env.turbulence_power = physics.turbulence_power
    base_env.enable_wind = physics.enable_wind


def apply_bipedal_walker_physics(env, physics: BipedalWalkerPhysics):
    del env
    del physics


def default_env_params(env_name: str, env=None):
    if env_name == CONTINUOUS_CARTPOLE_NAME:
        return default_cartpole_physics(env)
    if env_name == CONTINUOUS_LUNAR_LANDER_NAME:
        return default_lunar_lander_physics(env)
    if env_name == BIPEDAL_WALKER_NAME:
        return default_bipedal_walker_physics(env)
    raise ValueError(f"Unsupported environment for probe parameters: {env_name}")


def sample_env_params(rng: np.random.Generator, base_params):
    if isinstance(base_params, CartPolePhysics):
        return sample_cartpole_physics(rng, base_params)
    if isinstance(base_params, LunarLanderPhysics):
        return sample_lunar_lander_physics(rng, base_params)
    if isinstance(base_params, BipedalWalkerPhysics):
        return sample_bipedal_walker_physics(rng, base_params)
    raise ValueError(f"Unsupported environment parameter type: {type(base_params)!r}")


def apply_env_params(env, env_params):
    if isinstance(env_params, CartPolePhysics):
        apply_cartpole_physics(env, env_params)
        return
    if isinstance(env_params, LunarLanderPhysics):
        apply_lunar_lander_physics(env, env_params)
        return
    if isinstance(env_params, BipedalWalkerPhysics):
        apply_bipedal_walker_physics(env, env_params)
        return
    raise ValueError(f"Unsupported environment parameter type: {type(env_params)!r}")


@dataclass
class Transition:
    episode_id: int
    step_idx: int
    probe_mode: str
    env_params: np.ndarray
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    terminated: bool
    truncated: bool


@dataclass
class WindowRecord:
    episode_id: int
    end_step_idx: int
    probe_mode: str
    env_params: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool
    truncated: bool


class ProbePolicy:
    def __init__(self, action_space_n: int, profile: str = "scalar"):
        self.n = action_space_n
        self.profile = profile
        self._sticky = 0

    def _center_action(self) -> int:
        if self.profile == "lunar_lander":
            return 1
        if self.profile == "bipedal":
            return 0
        return (self.n - 1) // 2

    def _small_left_action(self) -> int:
        if self.profile == "lunar_lander":
            return 5
        if self.profile == "bipedal":
            return 1
        center = self._center_action()
        offset = max(1, (self.n - 1) // 4)
        return max(0, center - offset)

    def _small_right_action(self) -> int:
        if self.profile == "lunar_lander":
            return 6
        if self.profile == "bipedal":
            return 2
        center = self._center_action()
        offset = max(1, (self.n - 1) // 4)
        return min(self.n - 1, center + offset)

    def _hard_left_action(self) -> int:
        if self.profile == "lunar_lander":
            return 3
        if self.profile == "bipedal":
            return 3
        return 0

    def _hard_right_action(self) -> int:
        if self.profile == "lunar_lander":
            return 4
        if self.profile == "bipedal":
            return 4
        return self.n - 1

    def sample_action(
        self,
        mode: str,
        step_idx: int,
        rng: np.random.Generator,
    ) -> int:
        # The probe library is intentionally simple and repeatable so the encoder
        # can compare environments under a small set of consistent "questions."
        if mode == "random":
            return int(rng.integers(0, self.n))

        if mode == "hold_left":
            return self._hard_left_action()

        if mode == "hold_right":
            return self._hard_right_action()

        if mode == "center_hold":
            return self._center_action()

        if mode == "alternate":
            return step_idx % self.n

        if mode == "pulse_left":
            if step_idx < 2:
                return self._small_left_action()
            return self._center_action()

        if mode == "pulse_right":
            if step_idx < 2:
                return self._small_right_action()
            return self._center_action()

        if mode == "reverse":
            if step_idx < 2:
                return self._small_left_action()
            if step_idx < 4:
                return self._small_right_action()
            return self._center_action()

        if mode == "sweep":
            pattern = (
                self._small_left_action(),
                self._center_action(),
                self._small_right_action(),
                self._center_action(),
            )
            return int(pattern[step_idx % len(pattern)])

        if mode == "sticky_random":
            burst_len = 4
            if step_idx % burst_len == 0:
                self._sticky = int(rng.integers(0, self.n))
            return self._sticky

        raise ValueError(f"Unknown probe mode: {mode}")


class CartPoleCrawler:
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        window_size: int = 8,
        seed: int = 0,
        randomize_physics: bool = True,
        action_bins: int = 9,
    ):
        self.env = make_env(env_name)
        self.env_name = env_name
        self.window_size = window_size
        self.rng = np.random.default_rng(seed)
        self.randomize_physics = randomize_physics
        self.base_physics = default_env_params(env_name, self.env)
        self.action_values = get_action_values(self.env, action_bins, env_name=env_name)
        self.action_dim = (
            int(self.env.action_space.n)
            if self.action_values is None
            else int(len(self.action_values))
        )
        probe_profile = "scalar"
        if env_name == CONTINUOUS_LUNAR_LANDER_NAME:
            probe_profile = "lunar_lander"
        if env_name == BIPEDAL_WALKER_NAME:
            probe_profile = "bipedal"
        self.probe_policy = ProbePolicy(self.action_dim, profile=probe_profile)

        # Keep both granular transitions and fixed-length windows because the
        # encoder cares about short temporal patterns, not just one-step effects.
        self.transitions: list[Transition] = []
        self.windows: list[WindowRecord] = []

    def run_episode(
        self,
        episode_id: int,
        probe_mode: str,
        max_steps: int = 200,
        reset_options: Optional[dict[str, Any]] = None,
    ):
        if self.randomize_physics:
            episode_physics = sample_env_params(self.rng, self.base_physics)
        else:
            episode_physics = self.base_physics

        apply_env_params(self.env, episode_physics)
        env_params = episode_physics.as_array()

        state, _info = self.env.reset(options=reset_options)

        state_window = deque(maxlen=self.window_size + 1)
        action_window = deque(maxlen=self.window_size)
        reward_window = deque(maxlen=self.window_size)

        state_window.append(np.asarray(state, dtype=np.float32))

        for step_idx in range(max_steps):
            action_idx = self.probe_policy.sample_action(probe_mode, step_idx, self.rng)
            env_action = action_index_to_env_action(action_idx, self.action_values)
            next_state, reward, terminated, truncated, _info = self.env.step(env_action)

            state_np = np.asarray(state, dtype=np.float32)
            next_state_np = np.asarray(next_state, dtype=np.float32)

            self.transitions.append(
                Transition(
                    episode_id=episode_id,
                    step_idx=step_idx,
                    probe_mode=probe_mode,
                    env_params=env_params.copy(),
                    state=state_np.copy(),
                    action=int(action_idx),
                    next_state=next_state_np.copy(),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )

            action_window.append(int(action_idx))
            reward_window.append(float(reward))
            state_window.append(next_state_np.copy())

            if len(action_window) == self.window_size and len(state_window) == self.window_size + 1:
                # Each window becomes one training example for the world encoder.
                self.windows.append(
                    WindowRecord(
                        episode_id=episode_id,
                        end_step_idx=step_idx,
                        probe_mode=probe_mode,
                        env_params=env_params.copy(),
                        states=np.stack(state_window, axis=0),
                        actions=np.asarray(action_window, dtype=np.int64),
                        rewards=np.asarray(reward_window, dtype=np.float32),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                )

            state = next_state

            if terminated or truncated:
                break

    def collect(self, episodes_per_mode: int = 20, max_steps: int = 200):
        episode_id = 0
        # Sweep through the small probe library so every mode contributes data.
        for mode in PROBE_MODES:
            for _ in range(episodes_per_mode):
                self.run_episode(
                    episode_id=episode_id,
                    probe_mode=mode,
                    max_steps=max_steps,
                )
                episode_id += 1

    def get_transition_arrays(self) -> dict[str, np.ndarray]:
        return {
            "episode_id": np.asarray([t.episode_id for t in self.transitions], dtype=np.int32),
            "step_idx": np.asarray([t.step_idx for t in self.transitions], dtype=np.int32),
            "probe_mode": np.asarray([t.probe_mode for t in self.transitions], dtype=object),
            "env_params": np.stack([t.env_params for t in self.transitions], axis=0),
            "state": np.stack([t.state for t in self.transitions], axis=0),
            "action": np.asarray([t.action for t in self.transitions], dtype=np.int64),
            "next_state": np.stack([t.next_state for t in self.transitions], axis=0),
            "reward": np.asarray([t.reward for t in self.transitions], dtype=np.float32),
            "terminated": np.asarray([t.terminated for t in self.transitions], dtype=np.bool_),
            "truncated": np.asarray([t.truncated for t in self.transitions], dtype=np.bool_),
        }

    def get_window_arrays(self) -> dict[str, np.ndarray]:
        return {
            "episode_id": np.asarray([w.episode_id for w in self.windows], dtype=np.int32),
            "end_step_idx": np.asarray([w.end_step_idx for w in self.windows], dtype=np.int32),
            "probe_mode": np.asarray([w.probe_mode for w in self.windows], dtype=object),
            "env_params": np.stack([w.env_params for w in self.windows], axis=0),
            "states": np.stack([w.states for w in self.windows], axis=0),
            "actions": np.stack([w.actions for w in self.windows], axis=0),
            "rewards": np.stack([w.rewards for w in self.windows], axis=0),
            "terminated": np.asarray([w.terminated for w in self.windows], dtype=np.bool_),
            "truncated": np.asarray([w.truncated for w in self.windows], dtype=np.bool_),
        }

    def close(self):
        self.env.close()


ProbeCrawler = CartPoleCrawler
