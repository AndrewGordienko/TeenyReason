"""Run the local latent-space dashboard."""

import argparse

from ..viz.dashboard import create_dashboard_app


def main():
    parser = argparse.ArgumentParser(description="Serve the TeenyReason latent dashboard locally.")
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory containing saved benchmark and latent snapshot artifacts.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=5050, help="Port to listen on.")
    args = parser.parse_args()

    app = create_dashboard_app(artifact_dir=args.artifact_dir)
    print(f"Serving latent dashboard at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
