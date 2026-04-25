"""Small localhost dashboard for latent snapshots and benchmark summaries."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template

from ..app.live_trace import load_live_trace_payload
from .payloads import (
    build_benchmark_payload,
    build_index_payload,
    build_latent_payload,
)


def sanitize_json_value(value):
    """Convert nested payload values into JSON-safe finite primitives."""
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else 0.0
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def create_dashboard_app(artifact_dir: str | Path = "artifacts") -> Flask:
    """Build the Flask app used for local latent-space inspection."""
    artifact_root = Path(artifact_dir).resolve()
    template_dir = Path(__file__).with_name("templates")
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["ARTIFACT_DIR"] = str(artifact_root)

    @app.after_request
    def add_no_store_headers(response):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.get("/api/index")
    def api_index():
        return jsonify(sanitize_json_value(build_index_payload(artifact_root)))

    @app.get("/api/latent/<path:name>")
    def api_latent(name: str):
        path = artifact_root / name
        if not path.exists() or path.suffix != ".npz":
            return jsonify({"error": f"Unknown latent snapshot: {name}"}), 404
        return jsonify(sanitize_json_value(build_latent_payload(path)))

    @app.get("/api/benchmark/<path:name>")
    def api_benchmark(name: str):
        path = artifact_root / name
        if not path.exists() or path.suffix != ".npz":
            return jsonify({"error": f"Unknown benchmark summary: {name}"}), 404
        return jsonify(sanitize_json_value(build_benchmark_payload(path)))

    @app.get("/api/live")
    def api_live():
        return jsonify(sanitize_json_value(load_live_trace_payload(artifact_root)))

    return app


__all__ = [
    "build_benchmark_payload",
    "build_index_payload",
    "build_latent_payload",
    "create_dashboard_app",
    "sanitize_json_value",
]
