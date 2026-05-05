"""Print empirical loss attribution for saved benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from teenyreason.viz.payloads import build_benchmark_payload, list_benchmark_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="?",
        help="Benchmark .npz path. Defaults to the newest *_solve_benchmark.npz.",
    )
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write <artifact>_loss_attribution.json beside the benchmark artifact.",
    )
    return parser.parse_args()


def resolve_artifact(args: argparse.Namespace) -> Path:
    if args.artifact:
        return Path(args.artifact)
    paths = list_benchmark_paths(Path(args.artifact_dir))
    if not paths:
        raise SystemExit(f"No benchmark artifacts found in {args.artifact_dir}.")
    return max(paths, key=lambda path: path.stat().st_mtime)


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:.0f}"
    return f"{number:.{digits}f}"


def print_report(path: Path, payload: dict) -> None:
    report = payload["loss_attribution"]
    sample = report["sample_economics"]
    expression = report["expression_channel"]
    full_context = report.get("full_system_context_channel", {})
    gate = report["latent_win_gate"]
    representation = report["representation_gate"]
    families = report["probe_family_economics"]["families"][:6]

    print(f"Artifact: {path}")
    print(f"Env: {payload.get('env_display_name') or payload.get('env_name')} | profile={payload.get('benchmark_profile')}")
    print("\nSample economics")
    print(
        "  baseline_steps={baseline} | probe_steps={probe} | noexpr_steps={noexpr} | "
        "probe_vs_baseline={delta}".format(
            baseline=fmt(sample["baseline_solve_steps_median"], 0),
            probe=fmt(sample["probe_solve_steps_median"], 0),
            noexpr=fmt(sample["probe_no_expression_solve_steps_median"], 0),
            delta=fmt(sample["probe_step_savings_vs_baseline"], 0),
        )
    )
    print(
        "  capped baseline={baseline} | capped probe={probe} | capped noexpr={noexpr} | "
        "capped_probe_vs_baseline={delta}".format(
            baseline=fmt(sample["baseline_capped_steps_median"], 0),
            probe=fmt(sample["probe_capped_steps_median"], 0),
            noexpr=fmt(sample["probe_no_expression_capped_steps_median"], 0),
            delta=fmt(sample["probe_capped_step_savings_vs_baseline"], 0),
        )
    )
    print(
        "  encoder_frac={encoder} | online_probe_frac={probe} | control_frac={control}".format(
            encoder=fmt(sample["encoder_fraction_of_probe_total"]),
            probe=fmt(sample["online_probe_fraction_of_probe_total"]),
            control=fmt(sample["control_fraction_of_probe_total"]),
        )
    )

    print("\nExpression channel")
    print(
        "  ready={ready} | enabled={enabled} | muted={muted} | expr_delta={delta} | forced_delta={forced}".format(
            ready=fmt(expression["ready_handoff_fraction_mean"]),
            enabled=fmt(expression["expression_enabled_fraction_mean"]),
            muted=fmt(expression["force_muted_fraction_mean"]),
            delta=fmt(expression["expression_delta_median"]),
            forced=fmt(expression["forced_expression_delta_median"]),
        )
    )
    print(f"  readiness_reasons={expression['readiness_reason_counts']}")
    print(f"  fair_stop_blockers={expression['fair_stop_blocker_counts']}")

    if full_context:
        returns = full_context.get("return_mean", {})
        print("\nFull-system context channel")
        print(
            "  learned={learned} | state_only={state} | zero={zero} | shuffled={shuffled} | "
            "stale={stale} | no_refresh={no_refresh} | frozen={frozen}".format(
                learned=fmt(returns.get("learned")),
                state=fmt(returns.get("state_only")),
                zero=fmt(returns.get("zero")),
                shuffled=fmt(returns.get("shuffled")),
                stale=fmt(returns.get("stale")),
                no_refresh=fmt(returns.get("no_refresh")),
                frozen=fmt(returns.get("frozen")),
            )
        )
        print(
            "  state_lift={state_lift} | content_lift={content_lift} | "
            "refresh_penalty={refresh_penalty} | content_causal={content_causal}".format(
                state_lift=fmt(full_context.get("state_channel_lift_mean")),
                content_lift=fmt(full_context.get("context_specific_lift_mean")),
                refresh_penalty=fmt(full_context.get("refresh_penalty_mean")),
                content_causal=full_context.get("context_content_causal"),
            )
        )

    print("\nRepresentation gate margins")
    if representation["available"]:
        print(
            "  pass_frac={pass_frac} | latent_pass_frac={latent_frac} | override_count={override}".format(
                pass_frac=fmt(representation["pass_fraction"]),
                latent_frac=fmt(representation["latent_pass_fraction"]),
                override=representation["override_count"],
            )
        )
        for row in representation["metrics"]:
            if row["target"] is None:
                continue
            print(
                f"  {row['name']}: observed={fmt(row['observed'])} "
                f"{row['direction']} target={fmt(row['target'])} | margin={fmt(row['margin'])} | pass={row['passed']}"
            )
    else:
        print("  not available")

    print("\nLatent-win gate margins")
    for row in gate["metrics"]:
        print(
            f"  {row['name']}: observed={fmt(row['observed'])} "
            f"{row['direction']} target={fmt(row['target'])} | margin={fmt(row['margin'])} | pass={row['passed']}"
        )

    print("\nProbe family economics")
    for row in families:
        print(
            "  {family}: clear2={clear} | sample_eff={sample} | selection={selection} | "
            "value_step={vps} | control={control} | stability={stability} | cost={cost} | selected={selected} | second={second}".format(
                family=row["family"],
                clear=row["clears_fair_second_probe_floor"],
                sample=fmt(row["sample_efficiency_score"]),
                selection=fmt(row["selection_score"]),
                vps=fmt(row["value_per_probe_step"]),
                control=fmt(row["control_utility_value"]),
                stability=fmt(row["stability_confidence"]),
                cost=fmt(row["estimated_probe_cost"]),
                selected=row["selected_count"],
                second=row["second_probe_selected_count"],
            )
        )

    print("\nDecisions")
    for row in report["decisions"]:
        print(f"  {row['priority']}. {row['decision']}")
        print(f"     evidence={row['evidence']}")


def main() -> None:
    args = parse_args()
    path = resolve_artifact(args)
    payload = build_benchmark_payload(path)
    report = payload["loss_attribution"]
    if args.write_json:
        output_path = path.with_name(f"{path.stem}_loss_attribution.json")
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print_report(path, payload)


if __name__ == "__main__":
    main()
