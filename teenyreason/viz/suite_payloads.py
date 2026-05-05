"""Payload helpers for multi-domain suite dashboard artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from ..multidomain.contracts.evidence import (
    attach_standard_domain_blocks,
    causal_metric_row,
    mechanism_check_row,
    transfer_gap_row,
)
from ..multidomain.diagnostics.latent_utility import (
    attach_latent_utility_blocks,
    latent_utility_row,
    wake_up_row,
)
from ..multidomain.diagnostics.world_understanding import (
    attach_world_understanding_blocks,
    world_understanding_row,
)
from ..multidomain.contracts.decision_utility import (
    attach_decision_utility_blocks,
    decision_utility_row,
)
from ..multidomain.contracts.handoff import (
    attach_belief_handoff_blocks,
    belief_handoff_row,
    rate_distortion_row,
)
from ..multidomain.contracts.handoff_repair import (
    attach_handoff_repair_blocks,
    handoff_repair_row,
)
from ..multidomain.suite.reporting.cartpole import cartpole_domain_payload
from ..multidomain.suite.reporting.acceptance import (
    attach_suite_claim_gates,
    suite_acceptance_thresholds,
)


def _is_suite_payload(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and int(payload.get("schema_version", 0)) == 1
        and isinstance(payload.get("domains"), dict)
        and "run_id" in payload
    )


def _suite_artifact_label(path: Path, artifact_dir: Path) -> str:
    """Return the dashboard API path for a suite artifact."""
    try:
        return path.relative_to(artifact_dir).as_posix()
    except ValueError:
        return path.name


def list_suite_paths(artifact_dir: Path) -> list[Path]:
    """List dashboard-ready suite artifacts, newest first."""
    if not artifact_dir.exists():
        return []
    candidates: list[Path] = []
    search_paths = list(artifact_dir.glob("*.json"))
    for run_dir in artifact_dir.glob("*_run*"):
        if run_dir.is_dir():
            search_paths.extend(run_dir.glob("*.json"))
    for path in search_paths:
        if path.name.endswith("_detail.json"):
            continue
        if path.name in {"dashboard_context.json", "live_training_trace.json", "live_training_history.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _is_suite_payload(payload):
            candidates.append(path)
    return sorted(candidates, key=lambda item: float(item.stat().st_mtime), reverse=True)


def build_suite_index_payload(artifact_dir: Path) -> dict[str, object]:
    """Return suite artifact names and the preferred latest run."""
    paths = list_suite_paths(artifact_dir)
    labels = [_suite_artifact_label(path, artifact_dir) for path in paths]
    return {
        "available": bool(paths),
        "suite_runs": labels,
        "latest": "" if not paths else labels[0],
        "suite_mtimes": {
            label: float(path.stat().st_mtime)
            for label, path in zip(labels, paths)
        },
    }


def build_suite_payload(path: Path) -> dict[str, object]:
    """Load one suite artifact by path."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "error": f"Could not read suite artifact: {path.name}"}
    if not _is_suite_payload(payload):
        return {"available": False, "error": f"Not a suite artifact: {path.name}"}
    cross_domain = payload.setdefault("cross_domain", {})
    if isinstance(cross_domain, dict):
        domains = payload.get("domains", {})
        if isinstance(domains, dict):
            _hydrate_cartpole_domain(path=path, payload=payload, domains=domains)
            attach_standard_domain_blocks(domains)
            attach_latent_utility_blocks(domains)
            attach_world_understanding_blocks(domains)
            attach_belief_handoff_blocks(domains)
            attach_handoff_repair_blocks(domains)
            attach_decision_utility_blocks(domains)
            cross_domain["metric_rows"] = _metric_rows_from_domains(domains)
            cross_domain["causal_rows"] = _causal_rows_from_domains(domains)
            cross_domain["mechanism_rows"] = _mechanism_rows_from_domains(domains)
            cross_domain["transfer_rows"] = _transfer_rows_from_domains(domains)
            cross_domain["handoff_rows"] = _handoff_rows_from_domains(domains)
            cross_domain["latent_utility_rows"] = _latent_utility_rows_from_domains(domains)
            cross_domain["wake_up_rows"] = _wake_up_rows_from_domains(domains)
            cross_domain["world_understanding_rows"] = _world_understanding_rows_from_domains(domains)
            cross_domain["belief_handoff_rows"] = _belief_handoff_rows_from_domains(domains)
            cross_domain["rate_distortion_rows"] = _rate_distortion_rows_from_domains(domains)
            cross_domain["handoff_repair_rows"] = _handoff_repair_rows_from_domains(domains)
            cross_domain["decision_utility_rows"] = _decision_utility_rows_from_domains(domains)
            cross_domain["acceptance"] = attach_suite_claim_gates(domains)
        cross_domain["acceptance_thresholds"] = suite_acceptance_thresholds()
    payload["available"] = True
    payload["artifact_name"] = path.name
    return payload


def _metric_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if not isinstance(domain, dict):
            continue
        metrics = domain.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "domain": domain_name,
                "subset_consistency": metrics.get(
                    "subset_consistency",
                    domain.get("trust", 0.0),
                ),
                "solver_gain": domain.get("belief_contribution_margin", 0.0),
                "ablation_gap": domain.get("ablation_gap", 0.0),
                "content_lift": domain.get("causal_ablation", {}).get("content_lift", 0.0),
                "content_causal": domain.get("causal_ablation", {}).get("content_causal", False),
                "belief_bitrate": metrics.get("belief_bitrate", 0),
            }
        )
    return rows


def _causal_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(causal_metric_row(domain_name, domain))
    return rows


def _mechanism_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(mechanism_check_row(domain_name, domain))
    return rows


def _transfer_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(transfer_gap_row(domain_name, domain))
    return rows


def _handoff_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if not isinstance(domain, dict):
            continue
        metrics = domain.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        rows.append(
            {
                "domain": domain_name,
                "hidden_target": metrics.get("handoff_hidden_target", ""),
                "cheap_decode_accuracy": metrics.get("handoff_cheap_decode_accuracy", 0.0),
                "centroid_head_decode_accuracy": metrics.get(
                    "handoff_centroid_head_decode_accuracy",
                    0.0,
                ),
                "cheap_content_lift": metrics.get("handoff_cheap_content_lift", 0.0),
                "action_change_fraction": metrics.get("handoff_action_change_fraction", 0.0),
                "value_delta_correct_vs_shuffled": metrics.get(
                    "handoff_value_delta_correct_vs_shuffled",
                    0.0,
                ),
                "cheap_dedicated_probe_steps": metrics.get("handoff_cheap_dedicated_probe_steps", 0.0),
                "expensive_dedicated_probe_steps": metrics.get(
                    "handoff_expensive_dedicated_probe_steps",
                    0.0,
                ),
                "dual_use_probe_fraction": metrics.get("handoff_dual_use_probe_fraction", 0.0),
                "dedicated_probe_steps_saved": metrics.get("handoff_dedicated_probe_steps_saved", 0.0),
            }
        )
    return rows


def _latent_utility_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(latent_utility_row(domain_name, domain))
    return rows


def _wake_up_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(wake_up_row(domain_name, domain))
    return rows


def _world_understanding_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(world_understanding_row(domain_name, domain))
    return rows


def _belief_handoff_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(belief_handoff_row(domain_name, domain))
    return rows


def _rate_distortion_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(rate_distortion_row(domain_name, domain))
    return rows


def _handoff_repair_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(handoff_repair_row(domain_name, domain))
    return rows


def _decision_utility_rows_from_domains(domains: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            rows.append(decision_utility_row(domain_name, domain))
    return rows


def _hydrate_cartpole_domain(
    *,
    path: Path,
    payload: dict[str, object],
    domains: dict[str, object],
) -> None:
    """Refresh legacy suite CartPole lanes from the richer benchmark payload."""
    artifact_dir = Path(str(payload.get("artifact_dir") or path.parent))
    if not artifact_dir.is_absolute():
        artifact_dir = path.parent if not artifact_dir.exists() else artifact_dir
    candidates = sorted(
        artifact_dir.glob("*cartpole*_solve_benchmark.npz"),
        key=lambda item: float(item.stat().st_mtime),
        reverse=True,
    )
    if not candidates:
        return
    try:
        from .payloads import build_benchmark_payload

        benchmark_payload = build_benchmark_payload(candidates[0])
    except Exception:
        return
    cartpole = domains.get("cartpole")
    if not isinstance(cartpole, dict):
        return
    if not cartpole.get("rows") and not cartpole.get("artifact_ref"):
        return
    detail_ref = None
    if isinstance(cartpole.get("artifact_ref"), str):
        detail_ref = Path(cartpole["artifact_ref"])
    domains["cartpole"] = cartpole_domain_payload(benchmark_payload, detail_ref)


def build_latest_suite_payload(artifact_dir: Path) -> dict[str, object]:
    """Load the newest suite artifact, or return an idle payload."""
    paths = list_suite_paths(artifact_dir)
    if not paths:
        return {
            "available": False,
            "run_id": None,
            "domains": {},
            "cross_domain": {"metric_rows": [], "acceptance": {}},
        }
    return build_suite_payload(paths[0])
