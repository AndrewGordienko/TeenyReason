"""Run the canonical CartPole representation-repair benchmark batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teenyreason.app.benchmark import run_training_pipeline
from teenyreason.envs import CONTINUOUS_CARTPOLE_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="cartpole_repr_repair_v2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--representation-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after the representation gate fails instead of running PPO.",
    )
    return parser.parse_args()


def main() -> None:
    """Produce matched snapshot and benchmark artifacts for the repair pass."""
    args = parse_args()
    seeds = [int(value.strip()) for value in str(args.seeds).split(",") if value.strip()]
    run_training_pipeline(
        env_name=CONTINUOUS_CARTPOLE_NAME,
        seeds=seeds,
        config_override={
            "benchmark_tag": str(args.tag),
            "benchmark_profile": "fast",
            "benchmark_mode": "fair",
            "probe_budget_mode": "fair_two_probe_handoff",
            "belief_mode": "particle_sysid",
            "representation_repair_mode": True,
            "representation_gate_enabled": True,
            "representation_only_until_gate_pass": bool(args.representation_only),
        },
    )


if __name__ == "__main__":
    main()
