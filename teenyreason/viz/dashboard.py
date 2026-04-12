"""Small localhost dashboard for latent snapshots and benchmark summaries."""

from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template

from ..representation import list_latent_snapshot_paths, load_latent_snapshot


def list_benchmark_paths(artifact_dir: Path) -> list[Path]:
    """Find all saved benchmark summary artifacts."""
    return sorted(artifact_dir.glob("*_solve_benchmark.npz"))


def load_benchmark_summary(path: Path) -> dict[str, np.ndarray]:
    """Load one saved benchmark summary artifact."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def downsample_indices(count: int, max_points: int = 1500) -> np.ndarray:
    """Keep the browser payload small enough to stay responsive."""
    if count <= max_points:
        return np.arange(count, dtype=np.int32)
    return np.linspace(0, count - 1, num=max_points, dtype=np.int32)


def build_index_payload(artifact_dir: Path) -> dict:
    """List the available dashboard artifacts."""
    return {
        "artifact_dir": str(artifact_dir),
        "latent_snapshots": [path.name for path in list_latent_snapshot_paths(artifact_dir)],
        "benchmark_summaries": [path.name for path in list_benchmark_paths(artifact_dir)],
    }


def build_latent_payload(path: Path) -> dict:
    """Convert one latent snapshot artifact into a JSON-friendly payload."""
    snapshot = load_latent_snapshot(path)
    indices = downsample_indices(int(snapshot["latent_mean"].shape[0]))
    projection = snapshot["projection_2d"][indices]
    reward_sum = snapshot["reward_sum"][indices]
    uncertainty = snapshot["uncertainty"][indices]
    probe_mode = snapshot["probe_mode"][indices]
    episode_id = snapshot["episode_id"][indices]
    terminated = snapshot["terminated"][indices]
    env_params = snapshot["env_params"][indices]

    unique_modes, counts = np.unique(snapshot["probe_mode"], return_counts=True)
    mode_counts = [
        {"probe_mode": str(mode), "count": int(count)}
        for mode, count in sorted(zip(unique_modes.tolist(), counts.tolist()), key=lambda item: item[0])
    ]

    points = []
    for idx in range(len(indices)):
        points.append(
            {
                "x": float(projection[idx, 0]),
                "y": float(projection[idx, 1]),
                "reward_sum": float(reward_sum[idx]),
                "uncertainty": float(uncertainty[idx]),
                "probe_mode": str(probe_mode[idx]),
                "episode_id": int(episode_id[idx]),
                "terminated": bool(int(terminated[idx])),
                "env_param_mean": float(np.mean(env_params[idx])),
            }
        )

    return {
        "name": path.name,
        "summary": {
            "num_windows": int(snapshot["latent_mean"].shape[0]),
            "latent_dim": int(snapshot["latent_mean"].shape[1]),
            "env_param_dim": int(snapshot["env_params"].shape[1]),
            "reward_min": float(snapshot["reward_sum"].min()),
            "reward_max": float(snapshot["reward_sum"].max()),
            "reward_mean": float(snapshot["reward_sum"].mean()),
            "uncertainty_mean": float(snapshot["uncertainty"].mean()),
            "terminated_rate": float(snapshot["terminated"].mean()),
            "pca_explained": [float(value) for value in snapshot["pca_explained"].tolist()],
            "sampled_points": int(len(points)),
        },
        "mode_counts": mode_counts,
        "points": points,
    }


def summarize_solve_array(values: np.ndarray, caps: np.ndarray) -> dict:
    """Mirror the console benchmark summary inside the dashboard."""
    values_list = values.astype(np.int64).tolist()
    caps_list = caps.astype(np.int64).tolist()
    solved_values = [value if value > 0 else None for value in values_list]
    capped = [value if value is not None else cap for value, cap in zip(solved_values, caps_list)]
    return {
        "success_rate": int(sum(1 for value in solved_values if value is not None)),
        "count": int(len(values_list)),
        "median": float(np.median(np.asarray(capped, dtype=np.float32))),
        "mean": float(np.mean(np.asarray(capped, dtype=np.float32))),
        "values": values_list,
    }


def build_benchmark_payload(path: Path) -> dict:
    """Convert one benchmark summary artifact into a JSON-friendly payload."""
    summary = load_benchmark_summary(path)
    seeds = summary["seeds"].astype(np.int64).tolist()
    baseline_episode_solves = summary.get("baseline_episode_solves", summary["baseline_solves"]).astype(np.int64)
    probe_episode_solves = summary.get("probe_episode_solves", summary["probe_solves"]).astype(np.int64)
    baseline_step_solves = summary.get(
        "baseline_step_solves",
        np.full_like(baseline_episode_solves, -1),
    ).astype(np.int64)
    probe_step_solves = summary.get(
        "probe_step_solves",
        np.full_like(probe_episode_solves, -1),
    ).astype(np.int64)
    baseline_total_env_steps = summary.get(
        "baseline_total_env_steps",
        np.full_like(baseline_episode_solves, 0),
    ).astype(np.int64)
    probe_total_env_steps = summary.get(
        "probe_total_env_steps",
        np.full_like(probe_episode_solves, 0),
    ).astype(np.int64)
    baseline_completed_episodes = summary.get(
        "baseline_completed_episodes",
        np.full_like(baseline_episode_solves, 0),
    ).astype(np.int64)
    probe_completed_episodes = summary.get(
        "probe_completed_episodes",
        np.full_like(probe_episode_solves, 0),
    ).astype(np.int64)
    probe_encoder_steps = summary.get(
        "probe_encoder_steps",
        np.zeros_like(probe_episode_solves),
    ).astype(np.int64)

    rows = []
    for idx, seed in enumerate(seeds):
        rows.append(
            {
                "seed": int(seed),
                "baseline_episode_solve": int(baseline_episode_solves[idx]),
                "probe_episode_solve": int(probe_episode_solves[idx]),
                "baseline_step_solve": int(baseline_step_solves[idx]),
                "probe_step_solve": int(probe_step_solves[idx]),
                "baseline_total_env_steps": int(baseline_total_env_steps[idx]),
                "probe_total_env_steps": int(probe_total_env_steps[idx]),
                "probe_encoder_steps": int(probe_encoder_steps[idx]),
            }
        )

    return {
        "name": path.name,
        "rows": rows,
        "summaries": {
            "baseline_episode": summarize_solve_array(
                baseline_episode_solves,
                baseline_completed_episodes,
            ),
            "probe_episode": summarize_solve_array(
                probe_episode_solves,
                probe_completed_episodes,
            ),
            "baseline_steps": summarize_solve_array(
                baseline_step_solves,
                baseline_total_env_steps,
            ),
            "probe_steps": summarize_solve_array(
                probe_step_solves,
                probe_total_env_steps,
            ),
        },
    }


def create_dashboard_app(artifact_dir: str | Path = "artifacts") -> Flask:
    """Build the Flask app used for local latent-space inspection."""
    artifact_root = Path(artifact_dir).resolve()
    template_dir = Path(__file__).with_name("templates")
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["ARTIFACT_DIR"] = str(artifact_root)

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.get("/api/index")
    def api_index():
        return jsonify(build_index_payload(artifact_root))

    @app.get("/api/latent/<path:name>")
    def api_latent(name: str):
        path = artifact_root / name
        if not path.exists() or path.suffix != ".npz":
            return jsonify({"error": f"Unknown latent snapshot: {name}"}), 404
        return jsonify(build_latent_payload(path))

    @app.get("/api/benchmark/<path:name>")
    def api_benchmark(name: str):
        path = artifact_root / name
        if not path.exists() or path.suffix != ".npz":
            return jsonify({"error": f"Unknown benchmark summary: {name}"}), 404
        return jsonify(build_benchmark_payload(path))

    return app
