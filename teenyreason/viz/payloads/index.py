"""Dashboard artifact index selection and run-context loading."""

from __future__ import annotations

import json
from pathlib import Path

from ...app.config import build_experiment_config
from ...envs import get_env_display_name
from ...cognition.representation import list_latent_snapshot_paths
from .common import load_benchmark_summary, load_optional_string


def list_benchmark_paths(artifact_dir: Path) -> list[Path]:
    """Find all saved benchmark summary artifacts."""
    return sorted(artifact_dir.glob("*_solve_benchmark.npz"))


def load_benchmark_profile_name(path: Path) -> str | None:
    """Read the benchmark profile stored in one summary artifact."""
    try:
        summary = load_benchmark_summary(path)
    except OSError:
        return None
    return load_optional_string(summary, "benchmark_profile")


def order_benchmark_paths(benchmark_paths: list[Path]) -> list[Path]:
    """Keep non-archived benchmark summaries ahead of archived planner runs."""
    profile_cache = {
        path.name: (load_benchmark_profile_name(path) or "")
        for path in benchmark_paths
    }
    return sorted(
        benchmark_paths,
        key=lambda path: (
            profile_cache.get(path.name) == "archived_planner",
            -float(path.stat().st_mtime),
            path.name,
        ),
    )


def preferred_benchmark_summary_name(
    benchmark_paths: list[Path],
    context: dict | None,
) -> str | None:
    """Prefer the newest non-archived benchmark for the default dashboard headline."""
    if not benchmark_paths:
        return None

    ordered_paths = order_benchmark_paths(benchmark_paths)
    name_to_path = {path.name: path for path in ordered_paths}
    context_tag = None if context is None else context.get("benchmark_tag")
    preferred_name = None if context is None else context.get("default_benchmark_summary")
    if isinstance(preferred_name, str) and preferred_name in name_to_path:
        preferred_profile = load_benchmark_profile_name(name_to_path[preferred_name])
        if preferred_profile != "archived_planner":
            return preferred_name

    if isinstance(context_tag, str) and context_tag:
        matching_paths = [path for path in ordered_paths if path.name.startswith(context_tag)]
        for path in matching_paths:
            if load_benchmark_profile_name(path) != "archived_planner":
                return path.name
        if matching_paths:
            return matching_paths[0].name
        return None

    for path in ordered_paths:
        if load_benchmark_profile_name(path) != "archived_planner":
            return path.name

    if isinstance(preferred_name, str) and preferred_name in name_to_path:
        return preferred_name
    return ordered_paths[0].name


def preferred_latent_snapshot_name(
    latent_paths: list[Path],
    context: dict | None,
) -> str | None:
    """Prefer the requested snapshot, falling back to the newest active-tag snapshot."""
    if not latent_paths:
        return None
    name_to_path = {path.name: path for path in latent_paths}
    preferred_name = None if context is None else context.get("default_latent_snapshot")
    if isinstance(preferred_name, str) and preferred_name in name_to_path:
        return preferred_name

    context_tag = None if context is None else context.get("benchmark_tag")
    if isinstance(context_tag, str) and context_tag:
        matching_paths = [path for path in latent_paths if path.name.startswith(f"{context_tag}_seed_")]
        if matching_paths:
            newest_path = max(matching_paths, key=lambda path: float(path.stat().st_mtime))
            return newest_path.name
        return None

    newest_path = max(latent_paths, key=lambda path: float(path.stat().st_mtime))
    return newest_path.name


def build_index_payload(artifact_dir: Path) -> dict:
    """List the available dashboard artifacts."""
    context = load_dashboard_context(artifact_dir)
    latent_paths = list_latent_snapshot_paths(artifact_dir)
    benchmark_paths = order_benchmark_paths(list_benchmark_paths(artifact_dir))
    preferred_benchmark = preferred_benchmark_summary_name(benchmark_paths, context)
    preferred_latent = preferred_latent_snapshot_name(latent_paths, context)
    if context is not None and (preferred_benchmark or preferred_latent):
        context = {
            **context,
            "default_benchmark_summary": preferred_benchmark or "",
            "default_latent_snapshot": preferred_latent or "",
        }
    return {
        "artifact_dir": str(artifact_dir),
        "latent_snapshots": [path.name for path in latent_paths],
        "benchmark_summaries": [path.name for path in benchmark_paths],
        "latent_snapshot_mtimes": {
            path.name: float(path.stat().st_mtime)
            for path in latent_paths
        },
        "benchmark_summary_mtimes": {
            path.name: float(path.stat().st_mtime)
            for path in benchmark_paths
        },
        "run_context": context,
    }


def load_dashboard_context(artifact_dir: Path) -> dict | None:
    """Load the most recent training selection written by the benchmark entrypoint."""
    context_path = artifact_dir / "dashboard_context.json"
    if not context_path.exists():
        return load_main_module_context()
    try:
        return json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return load_main_module_context()


def load_main_module_context() -> dict | None:
    """Fallback to the root main.py selection when no saved dashboard context exists yet."""
    try:
        import main as root_main
    except Exception:
        return None

    env_name = getattr(root_main, "ENV_NAME", None)
    seeds = getattr(root_main, "SEEDS", None)
    if not isinstance(env_name, str):
        return None

    try:
        benchmark_tag = build_experiment_config(env_name).benchmark_tag
    except Exception:
        benchmark_tag = env_name.replace("-", "_").replace("/", "_").lower()

    if not isinstance(seeds, list) or not seeds:
        seeds = [0, 1, 2, 3, 4]

    return {
        "env_name": env_name,
        "env_display_name": get_env_display_name(env_name),
        "benchmark_tag": benchmark_tag,
        "default_benchmark_summary": f"{benchmark_tag}_solve_benchmark.npz",
        "default_latent_snapshot": f"{benchmark_tag}_seed_{seeds[-1]}_latent_snapshot.npz",
        "seeds": seeds,
    }
