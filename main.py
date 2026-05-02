"""Live PPO comparison entrypoint for the local dashboard."""

from pathlib import Path
import socket
from threading import Thread
import time

from werkzeug.serving import make_server

import teenyreason as tr
from teenyreason.viz.dashboard import create_dashboard_app


PROFILE = "fast"
SEEDS = 1
ARTIFACT_DIR = Path("artifacts")
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 5050

ENVIRONMENTS = (
    "ContinuousCartPole-v0",
    "LunarLanderContinuous-v3",
    "BipedalWalker-v3",
)

COMMON_OVERRIDES = {
    "benchmark_profile": PROFILE,
}

ENV_OVERRIDES = {
    "BipedalWalker-v3": {
        "initial_log_std": -0.5,
        "entropy_coef": 0.005,
        "min_elite_return": 0.0,
    },
}


def first_open_port(start_port: int, attempts: int = 20) -> int:
    """Find a localhost port for the debug dashboard."""
    for port in range(int(start_port), int(start_port) + int(attempts)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((DASHBOARD_HOST, port)) != 0:
                return port
    raise RuntimeError(f"No open dashboard port found near {start_port}.")


def start_dashboard() -> None:
    """Serve the local dashboard beside the comparison run."""
    port = first_open_port(DASHBOARD_PORT)
    app = create_dashboard_app(artifact_dir=ARTIFACT_DIR)
    server = make_server(DASHBOARD_HOST, port, app)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Dashboard: http://{DASHBOARD_HOST}:{port}")


start_dashboard()
time.sleep(0.2)

result = tr.compare_ppo(
    envs=ENVIRONMENTS,
    seeds=SEEDS,
    profile=PROFILE,
    overrides=COMMON_OVERRIDES,
    env_overrides=ENV_OVERRIDES,
)
print(f"Comparison summary: {result.summary_path}")
