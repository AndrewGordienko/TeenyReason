"""Side-by-side sample-efficiency suite runner."""

from __future__ import annotations

import time
from pathlib import Path

from ...consumers import (
    BoardProbeBenchmarkConsumer,
    ImageProbeBenchmarkConsumer,
    LanguageProbeBenchmarkConsumer,
    PPOBenchmarkConsumer,
)
from ...envs import CONTINUOUS_CARTPOLE_NAME, CONTINUOUS_LUNAR_LANDER_NAME
from ...recipes import build_benchmark_recipe, build_board_recipe, build_language_recipe, build_mnist_recipe
from .. import (
    CartPolePlannerComparisonConfig,
    run_affordance_crawler_suite,
    run_board_adapter_bridge,
    run_cartpole_adapter_bridge,
    run_cartpole_latent_mpc_benchmark,
    run_cartpole_mechanics_benchmark,
    run_cartpole_planner_comparison,
    run_decision_local_crawler_suite,
    run_image_adapter_bridge,
    run_language_adapter_bridge,
    run_latent_control_handoff,
    run_real_causal_adapter_suite,
)
from .config import MultidomainSuiteConfig
from .reporting.io import summarize_rows, write_json
from .reporting.payloads import build_suite_payload


def run_multidomain_suite(
    config: MultidomainSuiteConfig | None = None,
) -> dict[str, object]:
    """Run the current RL benchmark plus image, language, and board-game checks."""
    config = config or MultidomainSuiteConfig()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    run_id = f"{config.suite_name}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(started_at))}"
    results: dict[str, object] = {
        "rl": {},
        "image": None,
        "language": None,
        "board": None,
        "cartpole_mechanics": None,
        "cartpole_controller_bridge": None,
        "cartpole_latent_mpc": None,
        "cartpole_planner_comparison": None,
        "decision_local_crawler": None,
        "affordance_crawler": None,
        "latent_handoff": None,
        "language_bridge": None,
        "image_bridge": None,
        "board_bridge": None,
        "real_causal": None,
    }
    detail_paths: dict[str, Path] = {}

    if config.run_rl_benchmark:
        print("\n=== Multi-Domain Suite | RL ===")
        rl_consumer = PPOBenchmarkConsumer()
        for env_name in config.rl_envs:
            override = None
            if env_name == CONTINUOUS_LUNAR_LANDER_NAME and config.lunar_lander_randomize_physics:
                override = {"randomize_physics": True}
            print(f"\n--- RL benchmark | env={env_name} ---")
            rl_recipe = build_benchmark_recipe(env_name)
            rl_result = rl_consumer.run(
                rl_recipe,
                seeds=list(config.rl_seeds),
                config_override=override,
            )
            results["rl"][env_name] = rl_result
        if CONTINUOUS_CARTPOLE_NAME in results["rl"]:
            detail_paths["cartpole"] = write_json(
                config.artifact_dir / f"{run_id}_cartpole_detail.json",
                {"domain": "cartpole", "result": results["rl"][CONTINUOUS_CARTPOLE_NAME]},
            )

    if config.run_cartpole_mechanics_benchmark:
        print("\n=== Multi-Domain Suite | CartPole Mechanics Ladder ===")
        mechanics_result = run_cartpole_mechanics_benchmark(config.cartpole_mechanics)
        print(
            "CartPole mechanics ladder | "
            f"decode_acc={mechanics_result['mechanics_decode_accuracy']:.4f} | "
            f"r2={mechanics_result['mechanics_r2']:.4f} | "
            f"transition_mse_lift={mechanics_result['mean_content_lift']:.6f}"
        )
        results["cartpole_mechanics"] = mechanics_result
        detail_paths["cartpole_mechanics"] = write_json(
            config.artifact_dir / f"{run_id}_cartpole_mechanics_detail.json",
            {"domain": "cartpole", "result": mechanics_result},
        )

    if config.run_family_bridge_benchmark:
        print("\n=== Multi-Domain Suite | Family-Real Bridges ===")
        cartpole_bridge = run_cartpole_adapter_bridge(config.cartpole_controller_bridge)
        language_bridge = run_language_adapter_bridge(config.family_bridge)
        image_bridge = run_image_adapter_bridge(config.family_bridge)
        board_bridge = run_board_adapter_bridge(config.board.seeds)
        print(
            "Bridge checks | "
            f"cartpole_lift={cartpole_bridge['content_lift']:.4f} | "
            f"language_lift={language_bridge['content_lift']:.4f} | "
            f"image_lift={image_bridge['content_lift']:.4f} | "
            f"board_lift={board_bridge['content_lift']:.4f}"
        )
        results["cartpole_controller_bridge"] = cartpole_bridge
        results["language_bridge"] = language_bridge
        results["image_bridge"] = image_bridge
        results["board_bridge"] = board_bridge
        detail_paths["family_bridges"] = write_json(
            config.artifact_dir / f"{run_id}_family_bridges_detail.json",
            {
                "cartpole": cartpole_bridge,
                "language": language_bridge,
                "image": image_bridge,
                "board": board_bridge,
            },
        )

    if config.run_latent_handoff_benchmark:
        print("\n=== Multi-Domain Suite | Latent Control Handoff ===")
        handoff_result = run_latent_control_handoff(config.latent_handoff)
        print(
            "Latent handoff v1 | "
            f"cheap_lift={handoff_result['cheap_content_lift']:.4f} | "
            f"cheap_decode={handoff_result['cheap_decode_accuracy']:.4f} | "
            f"action_change={handoff_result['action_change_fraction']:.4f} | "
            f"dedicated_saved={handoff_result['dedicated_probe_steps_saved']:.0f}"
        )
        results["latent_handoff"] = handoff_result
        detail_paths["latent_handoff"] = write_json(
            config.artifact_dir / f"{run_id}_latent_handoff_detail.json",
            {"domain": "cartpole", "result": handoff_result},
        )

    if config.run_cartpole_latent_mpc_benchmark:
        print("\n=== Multi-Domain Suite | Predictive CartPole MPC ===")
        planner_result = run_cartpole_latent_mpc_benchmark(config.cartpole_latent_mpc)
        print(
            "Predictive CartPole MPC | "
            f"belief_return={planner_result['belief_mpc_return']:.4f} | "
            f"no_belief_return={planner_result['no_belief_return']:.4f} | "
            f"solver_gain={planner_result['solver_gain']:.4f} | "
            f"content_lift={planner_result['content_lift']:.4f} | "
            f"net_sample_savings={planner_result['net_env_sample_savings']:.1f} | "
            f"action_match_oracle={planner_result['belief_action_match_oracle']:.4f}"
        )
        results["cartpole_latent_mpc"] = planner_result
        detail_paths["cartpole_latent_mpc"] = write_json(
            config.artifact_dir / f"{run_id}_cartpole_latent_mpc_detail.json",
            {"domain": "cartpole", "result": planner_result},
        )

    if config.run_cartpole_planner_comparison:
        print("\n=== Multi-Domain Suite | Matched Planner Comparison ===")
        comparison_result = run_cartpole_planner_comparison(config.planner_comparison)
        print(
            "Planner comparison | "
            f"profile={comparison_result['profile']} | "
            f"solver_gain={comparison_result['solver_gain']:.4f} | "
            f"regret_reduction={comparison_result['action_regret_reduction']:.4f} | "
            f"probe_roi={comparison_result['probe_roi']:.6f} | "
            f"crawler_vs_no_belief_samples="
            f"{comparison_result['crawler_vs_no_belief_mpc_sample_savings']} | "
            "persistent_amortized_vs_no_belief="
            f"{comparison_result['persistent_affordance_amortized_vs_no_belief_mpc_sample_savings']} | "
            f"state={comparison_result['diagnostic_state']}"
        )
        results["cartpole_planner_comparison"] = comparison_result
        detail_paths["cartpole_planner_comparison"] = write_json(
            config.artifact_dir / f"{run_id}_cartpole_planner_comparison_detail.json",
            {"domain": "cartpole", "result": comparison_result},
        )

    if config.run_decision_local_crawler_benchmark:
        print("\n=== Multi-Domain Suite | Decision-Local Curiosity Crawler ===")
        crawler_result = run_decision_local_crawler_suite(config.decision_local_crawler)
        rows = crawler_result.get("summary_rows", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                print(
                    "Decision-local crawler | "
                    f"domain={row.get('domain', '')} | "
                    f"score={float(row.get('crawler_decision_score', 0.0)):.4f} | "
                    f"regret_delta={float(row.get('regret_reduction', 0.0)):.4f} | "
                    f"content_lift={float(row.get('content_lift', 0.0)):.4f} | "
                    f"voi={float(row.get('voi', 0.0)):.6f} | "
                    f"state={row.get('verdict', '')}"
                )
        results["decision_local_crawler"] = crawler_result
        detail_paths["decision_local_crawler"] = write_json(
            config.artifact_dir / f"{run_id}_decision_local_crawler_detail.json",
            crawler_result,
        )

    if config.run_affordance_crawler_benchmark:
        print("\n=== Multi-Domain Suite | Persistent Affordance Crawler ===")
        affordance_result = run_affordance_crawler_suite(config.affordance_crawler)
        rows = affordance_result.get("summary_rows", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                print(
                    "Persistent affordance crawler | "
                    f"domain={row.get('domain', '')} | "
                    f"score={float(row.get('affordance_decision_score', 0.0)):.4f} | "
                    f"regret_delta={float(row.get('regret_reduction', 0.0)):.4f} | "
                    f"cost={float(row.get('total_probe_cost', 0.0)):.2f} | "
                    f"net={float(row.get('net_value_after_reuse', 0.0)):.4f} | "
                    f"break_even={row.get('break_even_reuse_count')} | "
                    f"state={row.get('verdict', '')}"
                )
        results["affordance_crawler"] = affordance_result
        detail_paths["affordance_crawler"] = write_json(
            config.artifact_dir / f"{run_id}_affordance_crawler_detail.json",
            affordance_result,
        )

    if config.run_real_causal_benchmark:
        print("\n=== Multi-Domain Suite | Real Causal Adapters ===")
        real_causal_result = run_real_causal_adapter_suite(
            config.real_causal,
            language_config=config.language,
            image_config=config.image,
        )
        print(
            "Real causal adapters | "
            f"cartpole_cf={real_causal_result['cartpole']['counterfactual_accuracy']:.4f} | "
            f"language_cf={real_causal_result['language']['counterfactual_accuracy']:.4f} | "
            f"image_cf={real_causal_result['image']['counterfactual_accuracy']:.4f}"
        )
        results["real_causal"] = real_causal_result
        detail_paths["real_causal"] = write_json(
            config.artifact_dir / f"{run_id}_real_causal_detail.json",
            real_causal_result,
        )

    if config.run_image_benchmark:
        print("\n=== Multi-Domain Suite | Images ===")
        image_recipe = build_mnist_recipe(config.image)
        image_result = ImageProbeBenchmarkConsumer().run(image_recipe)
        image_summary = summarize_rows(
            image_result["rows"],
            baseline_key="baseline_accuracy",
            probe_key="probe_accuracy",
        )
        print(
            "MNIST probe benchmark | "
            f"baseline_mean_acc={image_summary['baseline_mean']:.4f} | "
            f"probe_mean_acc={image_summary['probe_mean']:.4f} | "
            f"delta={image_summary['probe_minus_baseline']:.4f}"
        )
        results["image"] = image_result
        detail_paths["image"] = write_json(
            config.artifact_dir / f"{run_id}_image_detail.json",
            {"domain": "image", "result": image_result},
        )

    if config.run_language_benchmark:
        print("\n=== Multi-Domain Suite | Language ===")
        language_recipe = build_language_recipe(config.language)
        language_result = LanguageProbeBenchmarkConsumer().run(language_recipe)
        language_summary = summarize_rows(
            language_result["rows"],
            baseline_key="baseline_bpc",
            probe_key="probe_bpc",
        )
        print(
            "Shakespeare probe benchmark | "
            f"baseline_mean_bpc={language_summary['baseline_mean']:.4f} | "
            f"probe_mean_bpc={language_summary['probe_mean']:.4f} | "
            f"delta={language_summary['probe_minus_baseline']:.4f}"
        )
        results["language"] = language_result
        detail_paths["language"] = write_json(
            config.artifact_dir / f"{run_id}_language_detail.json",
            {"domain": "language", "result": language_result},
        )

    if config.run_board_benchmark:
        print("\n=== Multi-Domain Suite | Board Games ===")
        board_recipe = build_board_recipe(config.board)
        board_result = BoardProbeBenchmarkConsumer().run(board_recipe)
        board_summary = summarize_rows(
            board_result["rows"],
            baseline_key="baseline_move_accuracy",
            probe_key="belief_move_accuracy",
        )
        print(
            "TicTacToe rule benchmark | "
            f"baseline_mean_acc={board_summary['baseline_mean']:.4f} | "
            f"belief_mean_acc={board_summary['probe_mean']:.4f} | "
            f"delta={board_summary['probe_minus_baseline']:.4f}"
        )
        results["board"] = board_result
        detail_paths["board"] = write_json(
            config.artifact_dir / f"{run_id}_board_detail.json",
            {"domain": "board", "result": board_result},
        )

    suite_payload = build_suite_payload(
        config=config,
        run_id=run_id,
        started_at=started_at,
        results=results,
        detail_paths=detail_paths,
    )
    suite_path = write_json(config.artifact_dir / f"{run_id}.json", suite_payload)
    summary_path = config.artifact_dir / "multidomain_probe_suite_summary.json"
    write_json(summary_path, {**results, "suite_path": str(suite_path)})
    print(f"\nSaved multi-domain suite summary to {summary_path}")
    print(f"Saved dashboard-ready suite artifact to {suite_path}")
    return {**results, "suite": suite_payload, "suite_path": str(suite_path)}


def main() -> None:
    """CLI wrapper for the tri-domain suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the TeenyReason multi-domain belief suite.")
    parser.add_argument("--artifact-dir", default="artifacts", help="Directory for suite artifacts.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of RL seeds to run.")
    parser.add_argument("--skip-rl", action="store_true", help="Skip the CartPole PPO benchmark.")
    parser.add_argument("--skip-cartpole-mechanics", action="store_true", help="Skip the CartPole mechanics ladder.")
    parser.add_argument("--skip-bridges", action="store_true", help="Skip generated family-real bridge checks.")
    parser.add_argument("--skip-handoff", action="store_true", help="Skip cheap latent handoff checks.")
    parser.add_argument("--skip-planner", action="store_true", help="Skip predictive CartPole MPC checks.")
    parser.add_argument(
        "--skip-planner-comparison",
        action="store_true",
        help="Skip matched predictive planner comparison checks.",
    )
    parser.add_argument(
        "--planner-profile",
        choices=("smoke", "matched"),
        default="smoke",
        help="Planner comparison profile: smoke is fast, matched is longer-horizon.",
    )
    parser.add_argument(
        "--skip-decision-crawler",
        action="store_true",
        help="Skip decision-local curiosity crawler checks.",
    )
    parser.add_argument(
        "--skip-affordance-crawler",
        action="store_true",
        help="Skip persistent affordance crawler checks.",
    )
    parser.add_argument("--skip-real-causal", action="store_true", help="Skip real-environment causal adapter checks.")
    parser.add_argument("--skip-image", action="store_true", help="Skip the MNIST benchmark.")
    parser.add_argument("--skip-language", action="store_true", help="Skip the Shakespeare benchmark.")
    parser.add_argument("--skip-board", action="store_true", help="Skip the tic-tac-toe benchmark.")
    args = parser.parse_args()

    run_multidomain_suite(
        MultidomainSuiteConfig(
            artifact_dir=Path(args.artifact_dir),
            rl_seeds=tuple(range(max(1, int(args.seeds)))),
            run_rl_benchmark=not args.skip_rl,
            run_cartpole_mechanics_benchmark=not args.skip_cartpole_mechanics,
            run_family_bridge_benchmark=not args.skip_bridges,
            run_latent_handoff_benchmark=not args.skip_handoff,
            run_cartpole_latent_mpc_benchmark=not args.skip_planner,
            run_cartpole_planner_comparison=not args.skip_planner_comparison,
            run_decision_local_crawler_benchmark=not args.skip_decision_crawler,
            run_affordance_crawler_benchmark=not args.skip_affordance_crawler,
            planner_comparison=CartPolePlannerComparisonConfig(profile=args.planner_profile),
            run_real_causal_benchmark=not args.skip_real_causal,
            run_image_benchmark=not args.skip_image,
            run_language_benchmark=not args.skip_language,
            run_board_benchmark=not args.skip_board,
        )
    )


if __name__ == "__main__":
    main()
