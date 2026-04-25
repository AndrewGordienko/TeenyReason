"""Small composition entrypoint for the crawler library."""

from teenyreason import ppo, run


ENV_NAME = "ContinuousCartPole-v0"
SEEDS = 2


def main() -> None:
    """Run one small benchmark composition."""
    run(ENV_NAME, ppo(), seeds=SEEDS, profile="fast")


if __name__ == "__main__":
    main()
