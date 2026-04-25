"""Environment-parameter helpers for probe collection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...envs import BIPEDAL_WALKER_NAME, CONTINUOUS_CARTPOLE_NAME, CONTINUOUS_LUNAR_LANDER_NAME


CARTPOLE_PARAM_NAMES = ("gravity", "masscart", "masspole", "length", "force_mag")
LUNAR_LANDER_PARAM_NAMES = ("gravity", "wind_power", "turbulence_power", "enable_wind")
BIPEDAL_WALKER_PARAM_NAMES = ("motors_torque", "speed_hip", "speed_knee", "lidar_range", "friction")
STATIC_ENV_PARAM_NAMES = ("observation_dim", "action_dim", "action_kind")


def _env_name_contains(env_name: str | None, needle: str) -> bool:
    return needle in str(env_name or "").lower()


def _space_dim(space) -> float:
    shape = getattr(space, "shape", None)
    if shape:
        return float(np.prod(shape))
    if hasattr(space, "n"):
        return float(space.n)
    return 0.0


def get_env_param_names(env_name: str | None, param_dim: int) -> tuple[str, ...]:
    """Return readable env-parameter names for dashboard diagnostics."""
    if env_name == CONTINUOUS_CARTPOLE_NAME or _env_name_contains(env_name, "cartpole"):
        names = CARTPOLE_PARAM_NAMES
    elif env_name == CONTINUOUS_LUNAR_LANDER_NAME:
        names = LUNAR_LANDER_PARAM_NAMES
    elif env_name == BIPEDAL_WALKER_NAME:
        names = BIPEDAL_WALKER_PARAM_NAMES
    else:
        names = STATIC_ENV_PARAM_NAMES

    if len(names) >= param_dim:
        return tuple(names[:param_dim])
    suffix = tuple(f"param_{idx + 1}" for idx in range(len(names), param_dim))
    return tuple(names) + suffix


@dataclass(frozen=True)
class CartPolePhysics:
    """Explicit snapshot of the CartPole parameters the probe code cares about."""

    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0

    def as_array(self) -> np.ndarray:
        return np.asarray([self.gravity, self.masscart, self.masspole, self.length, self.force_mag], dtype=np.float32)


@dataclass(frozen=True)
class LunarLanderPhysics:
    """Environment parameters used as the LunarLander fingerprint target."""

    gravity: float = -10.0
    wind_power: float = 15.0
    turbulence_power: float = 1.5
    enable_wind: bool = False

    def as_array(self) -> np.ndarray:
        return np.asarray([self.gravity, self.wind_power, self.turbulence_power, float(self.enable_wind)], dtype=np.float32)


@dataclass(frozen=True)
class BipedalWalkerPhysics:
    """Reference parameter bundle for BipedalWalker-style experiments."""

    motors_torque: float = 80.0
    speed_hip: float = 4.0
    speed_knee: float = 6.0
    lidar_range: float = 160.0 / 30.0
    friction: float = 2.5

    def as_array(self) -> np.ndarray:
        return np.asarray([self.motors_torque, self.speed_hip, self.speed_knee, self.lidar_range, self.friction], dtype=np.float32)


@dataclass(frozen=True)
class StaticEnvPhysics:
    """No-op parameter bundle for envs without registered mutable mechanics."""

    observation_dim: float = 0.0
    action_dim: float = 0.0
    action_kind: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray([self.observation_dim, self.action_dim, self.action_kind], dtype=np.float32)


def default_cartpole_physics(env=None) -> CartPolePhysics:
    """Read the environment's current CartPole parameters or use defaults."""
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
    """Read the environment's current LunarLander parameters or use defaults."""
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
    """Return the reference BipedalWalker parameter bundle."""
    del env
    return BipedalWalkerPhysics()


def default_static_env_physics(env=None) -> StaticEnvPhysics:
    """Return a stable no-op fingerprint for unsupported Gym-style envs."""
    if env is None:
        return StaticEnvPhysics()
    action_space = env.action_space
    action_kind = 0.0 if hasattr(action_space, "n") else 1.0
    return StaticEnvPhysics(
        observation_dim=_space_dim(env.observation_space),
        action_dim=_space_dim(action_space),
        action_kind=action_kind,
    )


def sample_cartpole_physics(rng: np.random.Generator, base_physics: CartPolePhysics | None = None) -> CartPolePhysics:
    """Randomize CartPole around a base parameter setting."""
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
    """Randomize LunarLander around a base parameter setting."""
    base = base_physics or LunarLanderPhysics()
    gravity = float(np.clip(rng.uniform(base.gravity - 1.5, base.gravity + 1.5), -11.8, -2.0))
    wind_power = float(np.clip(rng.uniform(0.0, max(4.0, 1.5 * base.wind_power)), 0.0, 20.0))
    turbulence_power = float(np.clip(rng.uniform(0.0, max(0.5, 1.5 * base.turbulence_power)), 0.0, 2.0))
    enable_wind = bool(rng.random() < 0.5) if not base.enable_wind else bool(rng.random() < 0.8)
    return LunarLanderPhysics(gravity=gravity, wind_power=wind_power, turbulence_power=turbulence_power, enable_wind=enable_wind)


def sample_bipedal_walker_physics(
    rng: np.random.Generator,
    base_physics: BipedalWalkerPhysics | None = None,
) -> BipedalWalkerPhysics:
    """Return the current BipedalWalker reference parameters."""
    del rng
    return base_physics or BipedalWalkerPhysics()


def apply_cartpole_physics(env, physics: CartPolePhysics):
    """Write a CartPole parameter bundle back into the live environment."""
    base_env = env.unwrapped
    base_env.gravity = physics.gravity
    base_env.masscart = physics.masscart
    base_env.masspole = physics.masspole
    base_env.length = physics.length
    base_env.force_mag = physics.force_mag
    base_env.total_mass = base_env.masspole + base_env.masscart
    base_env.polemass_length = base_env.masspole * base_env.length


def apply_lunar_lander_physics(env, physics: LunarLanderPhysics):
    """Write a LunarLander parameter bundle back into the live environment."""
    base_env = env.unwrapped
    base_env.gravity = physics.gravity
    base_env.wind_power = physics.wind_power
    base_env.turbulence_power = physics.turbulence_power
    base_env.enable_wind = physics.enable_wind


def apply_bipedal_walker_physics(env, physics: BipedalWalkerPhysics):
    """Placeholder for walker physics mutation."""
    del env
    del physics


def default_env_params(env_name: str, env=None):
    """Dispatch to the environment-specific default parameter reader."""
    if env_name == CONTINUOUS_CARTPOLE_NAME or _env_name_contains(env_name, "cartpole"):
        return default_cartpole_physics(env)
    if env_name == CONTINUOUS_LUNAR_LANDER_NAME:
        return default_lunar_lander_physics(env)
    if env_name == BIPEDAL_WALKER_NAME:
        return default_bipedal_walker_physics(env)
    return default_static_env_physics(env)


def sample_env_params(rng: np.random.Generator, base_params):
    """Dispatch to the environment-specific parameter sampler."""
    if isinstance(base_params, CartPolePhysics):
        return sample_cartpole_physics(rng, base_params)
    if isinstance(base_params, LunarLanderPhysics):
        return sample_lunar_lander_physics(rng, base_params)
    if isinstance(base_params, BipedalWalkerPhysics):
        return sample_bipedal_walker_physics(rng, base_params)
    if isinstance(base_params, StaticEnvPhysics):
        return base_params
    raise ValueError(f"Unsupported environment parameter type: {type(base_params)!r}")


def apply_env_params(env, env_params):
    """Dispatch to the environment-specific parameter writer."""
    if isinstance(env_params, CartPolePhysics):
        apply_cartpole_physics(env, env_params)
        return
    if isinstance(env_params, LunarLanderPhysics):
        apply_lunar_lander_physics(env, env_params)
        return
    if isinstance(env_params, BipedalWalkerPhysics):
        apply_bipedal_walker_physics(env, env_params)
        return
    if isinstance(env_params, StaticEnvPhysics):
        return
    raise ValueError(f"Unsupported environment parameter type: {type(env_params)!r}")
