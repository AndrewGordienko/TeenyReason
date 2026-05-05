"""Small preset layer for continuous-control experiments."""

from __future__ import annotations

CONTROL_PRESET_CHOICES = (
    "default",
    "calibrated",
    "no_value",
    "no_action_value",
    "value_only",
    "action_value_only",
    "no_goal",
    "pessimistic_heavy",
)

PRESET_OVERRIDES: dict[str, dict[str, object]] = {
    "default": {},
    "calibrated": {
        "value_calibration": True,
        "value_reachability_power": 1.25,
        "value_manifold_power": 1.00,
    },
    "no_value": {
        "value_bootstrap": False,
        "action_value_bootstrap": False,
        "value_calibration": False,
    },
    "no_action_value": {"action_value_bootstrap": False},
    "value_only": {"value_bootstrap": True, "action_value_bootstrap": False},
    "action_value_only": {"value_bootstrap": False, "action_value_bootstrap": True},
    "no_goal": {"value_calibration": False},
    "pessimistic_heavy": {
        "off_manifold_penalty": 0.75,
        "value_overestimate_penalty": 0.35,
        "value_reachability_power": 1.50,
        "value_manifold_power": 1.25,
    },
}


def control_config_kwargs(args) -> dict[str, object]:
    """Return AdvancedGymMPCConfig kwargs from CLI args plus a named ablation preset."""
    preset = str(getattr(args, "control_preset", "default"))
    values: dict[str, object] = {
        "online_refit": bool(args.online),
        "value_bootstrap": bool(args.value_bootstrap),
        "action_value_bootstrap": bool(args.action_value_bootstrap),
        "actor_policy": bool(args.actor_policy),
        "actor_collection_prior": bool(args.actor_collection_prior),
        "actor_center_prior": bool(args.actor_center_prior),
        "control_preset": preset,
    }
    values.update(PRESET_OVERRIDES[preset])
    return values


def control_preset_overrides(preset: str) -> dict[str, object]:
    values = {"control_preset": str(preset)}
    values.update(PRESET_OVERRIDES[str(preset)])
    return values


__all__ = ["CONTROL_PRESET_CHOICES", "control_config_kwargs", "control_preset_overrides"]
