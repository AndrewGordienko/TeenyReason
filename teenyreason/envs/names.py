"""Canonical environment names used across demos, probes, and dashboards."""

CONTINUOUS_CARTPOLE_NAME = "ContinuousCartPole-v0"
CONTINUOUS_LUNAR_LANDER_NAME = "LunarLanderContinuous-v3"
BIPEDAL_WALKER_NAME = "BipedalWalker-v3"

ENV_DISPLAY_NAMES = {
    CONTINUOUS_CARTPOLE_NAME: "Continuous CartPole",
    CONTINUOUS_LUNAR_LANDER_NAME: "Continuous LunarLander",
    BIPEDAL_WALKER_NAME: "Bipedal Walker",
}


def get_env_display_name(env_name: str) -> str:
    """Human-friendly label used by logs and the dashboard."""
    return ENV_DISPLAY_NAMES.get(env_name, env_name)


__all__ = [
    "BIPEDAL_WALKER_NAME",
    "CONTINUOUS_CARTPOLE_NAME",
    "CONTINUOUS_LUNAR_LANDER_NAME",
    "ENV_DISPLAY_NAMES",
    "get_env_display_name",
]
