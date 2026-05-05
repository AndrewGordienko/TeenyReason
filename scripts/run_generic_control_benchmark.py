"""Run multi-seed generic continuous-control benchmarks."""

from __future__ import annotations

import argparse
from statistics import median

from teenyreason.app.demos.control_presets import CONTROL_PRESET_CHOICES, control_preset_overrides
from teenyreason.envs import BIPEDAL_WALKER_NAME
from teenyreason.multidomain.planning import AdvancedGymMPCConfig, ScenarioActorConfig, run_scenario_actor


SOLVED_RETURNS = {
    "ContinuousCartPole-v0": 475.0,
    "LunarLanderContinuous-v3": 200.0,
    BIPEDAL_WALKER_NAME: 300.0,
}


def main() -> None:
    args = parse_args()
    rows = [run_seed(args, seed) for seed in parse_seeds(args.seeds)]
    print_table(args, rows)


def run_seed(args: argparse.Namespace, seed: int) -> dict[str, object]:
    config = AdvancedGymMPCConfig(
        env_name=str(args.env),
        seed=int(seed),
        solve_return=float(SOLVED_RETURNS.get(str(args.env), args.solve_return)),
        collector=str(args.collector),
        probe_episodes=int(args.probe_episodes),
        probe_steps=int(args.probe_steps),
        control_steps=int(args.control_steps),
        horizon=int(args.horizon),
        candidate_count=int(args.candidates),
        cem_iterations=int(args.cem_iterations),
        ensemble_size=int(args.ensemble_size),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        **control_preset_overrides(str(args.control_preset)),
    )
    result = run_scenario_actor(ScenarioActorConfig(config, rounds=int(args.rounds)), render_mode=None)
    summary = result.summary()
    curve = [float(row["best_return"]) for row in summary["rows"]]
    return {
        "seed": int(seed),
        "best_return": float(summary["best_return"]),
        "samples_to_peak": int(summary["samples_to_peak"]),
        "samples_to_solve": summary["samples_to_solve"],
        "solved": bool(summary["solved"]),
        "auc": mean(curve),
    }


def print_table(args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    solve_samples = [float(row["samples_to_solve"]) for row in rows if row["samples_to_solve"] is not None]
    peak_samples = [float(row["samples_to_peak"]) for row in rows]
    returns = [float(row["best_return"]) for row in rows]
    aucs = [float(row["auc"]) for row in rows]
    print(f"env={args.env} collector={args.collector} method={args.method} preset={args.control_preset} rounds={args.rounds} seeds={len(rows)}")
    print("seed | best_return | samples_to_peak | samples_to_solve | solved | auc")
    for row in rows:
        solve = row["samples_to_solve"] if row["samples_to_solve"] is not None else "n/a"
        print(f"{row['seed']} | {float(row['best_return']):.2f} | {row['samples_to_peak']} | {solve} | {row['solved']} | {float(row['auc']):.2f}")
    print("summary:")
    print(f"  solve_rate: {mean([1.0 if row['solved'] else 0.0 for row in rows]):.3f}")
    print(f"  median_samples_to_peak: {median(peak_samples):.1f}")
    print(f"  median_samples_to_solve: {median(solve_samples):.1f}" if solve_samples else "  median_samples_to_solve: n/a")
    print(f"  best_return_mean: {mean(returns):.2f}")
    print(f"  best_return_median: {median(returns):.2f}")
    print(f"  auc_mean: {mean(aucs):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generic continuous-control multi-seed benchmark.")
    parser.add_argument("--env", default="ContinuousCartPole-v0")
    parser.add_argument("--collector", default="replay_frontier")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--method", default="scenario_actor", choices=("scenario_actor",))
    parser.add_argument("--control-preset", default="no_goal", choices=CONTROL_PRESET_CHOICES)
    parser.add_argument("--solve-return", type=float, default=0.0)
    parser.add_argument("--probe-episodes", type=int, default=8)
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--control-steps", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=96)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


if __name__ == "__main__":
    main()
