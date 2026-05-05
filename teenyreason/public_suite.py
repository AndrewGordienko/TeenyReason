"""Public multi-domain suite entrypoint."""

from __future__ import annotations

from pathlib import Path

from .multidomain.suite import MultidomainSuiteConfig, run_multidomain_suite


SeedInput = int | list[int] | tuple[int, ...] | None


def _normalize_seed_input(seeds: SeedInput) -> list[int] | None:
    if seeds is None:
        return None
    if isinstance(seeds, int):
        return list(range(max(1, int(seeds))))
    return [int(seed) for seed in seeds]


def run_suite(
    *,
    artifact_dir: str | Path = "artifacts",
    seeds: SeedInput = 3,
    run_rl: bool = True,
    run_cartpole_mechanics: bool = True,
    run_family_bridges: bool = True,
    run_latent_handoff: bool = True,
    run_image: bool = True,
    run_language: bool = True,
    run_board: bool = True,
):
    """Run the CartPole, MNIST, Shakespeare, and board-game belief suite."""
    return run_multidomain_suite(
        MultidomainSuiteConfig(
            artifact_dir=Path(artifact_dir),
            rl_seeds=tuple(_normalize_seed_input(seeds) or [0, 1, 2]),
            run_rl_benchmark=bool(run_rl),
            run_cartpole_mechanics_benchmark=bool(run_cartpole_mechanics),
            run_family_bridge_benchmark=bool(run_family_bridges),
            run_latent_handoff_benchmark=bool(run_latent_handoff),
            run_image_benchmark=bool(run_image),
            run_language_benchmark=bool(run_language),
            run_board_benchmark=bool(run_board),
        )
    )
