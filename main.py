"""Editable generic continuous-control experiment."""
from __future__ import annotations

import argparse

from teenyreason.app.demos.continuous_control import (
    print_generic_summary,
    print_header,
    print_performance_table,
)
from teenyreason.app.demos.control_presets import CONTROL_PRESET_CHOICES, control_config_kwargs
from teenyreason.envs import BIPEDAL_WALKER_NAME, CONTINUOUS_LUNAR_LANDER_NAME
from teenyreason.multidomain.planning import (
    AdvancedGymMPCConfig,
    ScenarioActorConfig,
    run_scenario_actor,
)

ENV_NAME = CONTINUOUS_LUNAR_LANDER_NAME  # Change this one line for another continuous Gym env.
SEED = 0
RENDER_MODE = "human"  # Use "human", "rgb_array", or None.
SOLVED_RETURNS = {"ContinuousCartPole-v0": 475.0, "LunarLanderContinuous-v3": 200.0, BIPEDAL_WALKER_NAME: 300.0}
ONLINE_REFIT, VALUE_BOOTSTRAP, ACTION_VALUE_BOOTSTRAP, ACTOR_POLICY = False, True, True, True
ACTOR_COLLECTION_PRIOR, ACTOR_CENTER_PRIOR, TRAINING_ROUNDS = False, False, 1
COLLECTOR, CONTROL_PRESET, METHOD = "replay_frontier", "no_goal", "scenario_actor"
METHOD_CHOICES = ("scenario_actor",)
COLLECTOR_CHOICES = ("random", "frontier", "replay_frontier", "success_archive", "reflex_archive", "option_archive", "curriculum_repair_archive")
MPC = dict(
    probe_episodes=8, probe_steps=200, control_steps=500,
    horizon=8, candidate_count=96, cem_iterations=4,
    ensemble_size=4, hidden_dim=128, epochs=80,
    temporal_chunk_size=2, temporal_chunk_candidates=24,
    temporal_execution_chunk=2, temporal_smoothness_penalty=0.03,
)
def main() -> None:
    args = parse_args(); env_name = str(args.env)
    render_mode, solve_return = None if args.no_render else args.render_mode, float(SOLVED_RETURNS.get(env_name, 0.0))
    print_header(f"{env_name} Experiment"); print("Edit ENV_NAME near the top of main.py to switch environments.")
    print(f"method={args.method}")
    config = build_neural_config(env_name, solve_return, args)
    result = run_scenario_actor(ScenarioActorConfig(config, int(args.rounds)), render_mode=render_mode)
    print_generic_summary(result)
    print_performance_table(result.summary())


def build_neural_config(env_name: str, solve_return: float, args: argparse.Namespace) -> AdvancedGymMPCConfig:
    return AdvancedGymMPCConfig(
        env_name=env_name,
        seed=SEED,
        solve_return=solve_return,
        collector=str(args.collector),
        **control_config_kwargs(args),
        **MPC,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the editable continuous-control experiment.")
    parser.add_argument("--env", default=ENV_NAME)
    parser.add_argument("--render-mode", default=RENDER_MODE)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--online", action=argparse.BooleanOptionalAction, default=ONLINE_REFIT)
    parser.add_argument("--value-bootstrap", action=argparse.BooleanOptionalAction, default=VALUE_BOOTSTRAP)
    parser.add_argument("--action-value-bootstrap", action=argparse.BooleanOptionalAction, default=ACTION_VALUE_BOOTSTRAP)
    parser.add_argument("--actor-policy", action=argparse.BooleanOptionalAction, default=ACTOR_POLICY)
    parser.add_argument("--actor-collection-prior", action=argparse.BooleanOptionalAction, default=ACTOR_COLLECTION_PRIOR)
    parser.add_argument("--actor-center-prior", action=argparse.BooleanOptionalAction, default=ACTOR_CENTER_PRIOR)
    parser.add_argument("--collector", default=COLLECTOR, choices=COLLECTOR_CHOICES)
    parser.add_argument("--control-preset", default=CONTROL_PRESET, choices=CONTROL_PRESET_CHOICES)
    parser.add_argument("--method", default=METHOD, choices=METHOD_CHOICES)
    parser.add_argument("--rounds", type=int, default=TRAINING_ROUNDS)
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    main()
