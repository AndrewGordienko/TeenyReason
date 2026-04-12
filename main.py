"""Thin root wrapper for the benchmark entrypoint.

Change `ENV_NAME` below to choose which environment to train on.
"""

from teenyreason.app.benchmark import run_training_pipeline
from teenyreason.envs import (
    BIPEDAL_WALKER_NAME,
    CONTINUOUS_CARTPOLE_NAME,
    CONTINUOUS_LUNAR_LANDER_NAME,
)


# Pick the environment you want to benchmark here.
ENV_NAME = CONTINUOUS_CARTPOLE_NAME

# Optional: edit the seed list here too if you want a shorter or longer benchmark.
SEEDS = [0, 1, 2, 3, 4]


if __name__ == "__main__":
    run_training_pipeline(env_name=ENV_NAME, seeds=SEEDS)
