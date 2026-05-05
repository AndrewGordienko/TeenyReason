"""Standard interface and causality blocks for multi-domain crawler reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DomainFeatureSpec:
    """Declared adapter contract between one environment and the crawler."""

    domain: str
    modality: str
    observation: str
    action: str
    hidden_target: str
    belief_dim: int
    message_dim: int
    query_families: tuple[str, ...]
    baseline_algorithm: str
    belief_algorithm: str
    handoff: str


DOMAIN_FEATURE_SPECS: dict[str, DomainFeatureSpec] = {
    "cartpole": DomainFeatureSpec(
        domain="cartpole",
        modality="rl_state",
        observation="4 float state vector plus probe windows",
        action="continuous force",
        hidden_target="environment mechanics and controller affordance",
        belief_dim=0,
        message_dim=0,
        query_families=(
            "impulse_left",
            "impulse_right",
            "chirp",
            "boundary_push",
            "passive_decay",
            "cart_brake",
        ),
        baseline_algorithm="PPO without crawler expression",
        belief_algorithm="Probe-conditioned PPO",
        handoff="controller context vector",
    ),
    "language": DomainFeatureSpec(
        domain="language",
        modality="text",
        observation="character token sequence",
        action="next-character logits",
        hidden_target="source style and local character statistics",
        belief_dim=16,
        message_dim=16,
        query_families=("support_span", "continuation_ranking", "cloze"),
        baseline_algorithm="plain character transformer",
        belief_algorithm="belief-conditioned character transformer",
        handoff="prefix/context embedding",
    ),
    "image": DomainFeatureSpec(
        domain="image",
        modality="image",
        observation="1x28x28 grayscale image",
        action="digit class logits",
        hidden_target="class prototype structure",
        belief_dim=642,
        message_dim=642,
        query_families=("support_example", "crop", "mask", "augment", "contrastive_view"),
        baseline_algorithm="plain CNN",
        belief_algorithm="belief-conditioned CNN",
        handoff="prototype context features",
    ),
    "board": DomainFeatureSpec(
        domain="board",
        modality="board_game",
        observation="tic-tac-toe board plus side to move",
        action="legal move value ranking",
        hidden_target="normal versus misere rule",
        belief_dim=4,
        message_dim=4,
        query_families=("rule_probe_position", "counterfactual_value_probe"),
        baseline_algorithm="normal-rule minimax",
        belief_algorithm="crawler-rule minimax",
        handoff="rule vector into minimax evaluator",
    ),
}


CONTENT_LIFT_FLOORS = {
    "cartpole": 100.0,
    "language": 0.01,
    "image": 0.01,
    "board": 0.05,
}

MECHANISM_CONTENT_LIFT_FLOORS = {
    "cartpole": 0.01,
    "language": 0.01,
    "image": 0.01,
    "board": 0.05,
}


def attach_standard_domain_blocks(domains: dict[str, object]) -> None:
    """Attach interface and causal-ablation blocks to all domain payloads."""
    for domain_name, domain in domains.items():
        if not isinstance(domain, dict):
            continue
        domain["interface"] = build_interface_block(domain_name, domain)
        domain["causal_ablation"] = build_causal_ablation_block(domain_name, domain)
        mechanism = build_mechanism_check_block(domain_name, domain)
        if mechanism:
            domain["mechanism_check"] = mechanism
        transfer = build_transfer_gap_block(domain_name, domain)
        if transfer:
            domain["transfer_gap"] = transfer


def build_interface_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Build the explicit adapter contract for one suite domain."""
    spec = DOMAIN_FEATURE_SPECS.get(domain_name)
    artifact = domain.get("artifact", {})
    if not isinstance(artifact, dict):
        artifact = {}
    metrics = domain.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    query_families = _query_families(spec, artifact)
    bitrate = _float(metrics.get("belief_bitrate", artifact.get("belief_bitrate", 0.0)))
    vector_dim = _message_dim(spec, artifact, bitrate)
    uncertainty = _float(artifact.get("uncertainty_estimate", domain.get("uncertainty_estimate", 0.0)))
    support_size = _support_size(domain, artifact)

    return {
        "schema_version": 1,
        "domain": domain_name,
        "modality": "" if spec is None else spec.modality,
        "input_contract": {
            "observation": "" if spec is None else spec.observation,
            "action": "" if spec is None else spec.action,
            "hidden_target": "" if spec is None else spec.hidden_target,
        },
        "query_families": list(query_families),
        "hidden_targets": _hidden_targets(spec, artifact),
        "belief": {
            "vector_dim": int(vector_dim),
            "message_dim": int(vector_dim),
            "bitrate": int(bitrate),
            "uncertainty": uncertainty,
            "support_size": support_size,
            "ready": _readiness_bool(domain),
            "readiness": str(domain.get("readiness", "")),
            "trust": _float(domain.get("trust", 0.0)),
        },
        "solver": {
            "baseline": "" if spec is None else spec.baseline_algorithm,
            "belief_conditioned": "" if spec is None else spec.belief_algorithm,
            "handoff": "" if spec is None else spec.handoff,
        },
        "ablation_arms": ["zero", "shuffled", "stale"],
    }


def build_causal_ablation_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Normalize learned-vs-ablated belief causality for one domain."""
    headline = domain.get("headline_metric", {})
    baseline_metric = domain.get("baseline_metric", {})
    if not isinstance(headline, dict):
        headline = {}
    if not isinstance(baseline_metric, dict):
        baseline_metric = {}

    lower_is_better = bool(headline.get("lower_is_better", False))
    learned = _float(headline.get("value", 0.0))
    baseline = _float(baseline_metric.get("value", 0.0))
    ablations = _ablation_values(domain_name, domain)
    best_ablation_name, best_ablation = _best_ablation(ablations, learned, lower_is_better)
    solver_gain = _signed_gain(baseline, learned, lower_is_better)
    content_lift = 0.0 if not ablations else _signed_gain(best_ablation, learned, lower_is_better)
    floor = CONTENT_LIFT_FLOORS.get(domain_name, 0.0)
    content_causal = bool(solver_gain > 0.0 and content_lift >= floor)

    return {
        "schema_version": 1,
        "primary_metric": str(headline.get("name", "metric")),
        "lower_is_better": lower_is_better,
        "baseline_value": baseline,
        "learned_value": learned,
        "ablation_values": ablations,
        "best_ablation_name": best_ablation_name,
        "best_ablation_value": best_ablation,
        "solver_gain": solver_gain,
        "content_lift": content_lift,
        "content_lift_floor": floor,
        "content_causal": content_causal,
    }


def causal_metric_row(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Return the dashboard table row for normalized belief causality."""
    causal = domain.get("causal_ablation", {})
    interface = domain.get("interface", {})
    if not isinstance(causal, dict):
        causal = {}
    if not isinstance(interface, dict):
        interface = {}
    belief = interface.get("belief", {})
    if not isinstance(belief, dict):
        belief = {}
    ablations = causal.get("ablation_values", {})
    if not isinstance(ablations, dict):
        ablations = {}
    return {
        "domain": domain_name,
        "metric": causal.get("primary_metric", ""),
        "learned": causal.get("learned_value", 0.0),
        "zero": ablations.get("zero", 0.0),
        "shuffled": ablations.get("shuffled", 0.0),
        "stale": ablations.get("stale", 0.0),
        "content_lift": causal.get("content_lift", 0.0),
        "content_causal": bool(causal.get("content_causal", False)),
        "message_dim": belief.get("message_dim", 0),
        "ready": bool(belief.get("ready", False)),
    }


def build_mechanism_check_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Return controlled hidden-target checks when a domain reports them."""
    metrics = domain.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    if "mechanism_belief_accuracy" not in metrics:
        return {}
    learned = _float(metrics.get("mechanism_belief_accuracy", 0.0))
    baseline = _float(metrics.get("mechanism_baseline_accuracy", 0.0))
    ablations = _present_ablation_values(
        {
            "zero": metrics.get("mechanism_zero_accuracy"),
            "shuffled": metrics.get("mechanism_shuffled_accuracy"),
            "stale": metrics.get("mechanism_stale_accuracy"),
        }
    )
    _best_name, best_ablation = _best_ablation(ablations, learned, lower_is_better=False)
    content_lift = 0.0 if not ablations else learned - best_ablation
    floor = MECHANISM_CONTENT_LIFT_FLOORS.get(domain_name, 0.0)
    return {
        "schema_version": 1,
        "hidden_target": str(metrics.get("mechanism_hidden_target", "")),
        "hidden_rule": str(metrics.get("mechanism_hidden_rule", "")),
        "decoded_rule": str(metrics.get("mechanism_decoded_rule", "")),
        "decode_accuracy": _float(metrics.get("mechanism_decode_accuracy", 0.0)),
        "subset_agreement": _float(metrics.get("mechanism_subset_agreement", 0.0)),
        "baseline_accuracy": baseline,
        "belief_accuracy": learned,
        "ablation_values": ablations,
        "content_lift": content_lift,
        "content_lift_floor": floor,
        "content_causal": bool(learned > baseline and content_lift >= floor),
    }


def mechanism_check_row(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Return one dashboard row for a controlled hidden-target check."""
    mechanism = domain.get("mechanism_check", {})
    if not isinstance(mechanism, dict):
        mechanism = {}
    ablations = mechanism.get("ablation_values", {})
    if not isinstance(ablations, dict):
        ablations = {}
    return {
        "domain": domain_name,
        "hidden_target": mechanism.get("hidden_target", ""),
        "hidden_rule": mechanism.get("hidden_rule", ""),
        "decoded_rule": mechanism.get("decoded_rule", ""),
        "decode_accuracy": mechanism.get("decode_accuracy", 0.0),
        "subset_agreement": mechanism.get("subset_agreement", 0.0),
        "baseline_accuracy": mechanism.get("baseline_accuracy", 0.0),
        "belief_accuracy": mechanism.get("belief_accuracy", 0.0),
        "zero": ablations.get("zero", 0.0),
        "shuffled": ablations.get("shuffled", 0.0),
        "stale": ablations.get("stale", 0.0),
        "content_lift": mechanism.get("content_lift", 0.0),
        "content_causal": bool(mechanism.get("content_causal", False)),
    }


def build_transfer_gap_block(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Compare controlled mechanism lift, bridge lift, and real-task lift."""
    metrics = domain.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    if "bridge_content_lift" not in metrics:
        return {}
    causal = domain.get("causal_ablation", {})
    mechanism = domain.get("mechanism_check", {})
    if not isinstance(causal, dict):
        causal = {}
    if not isinstance(mechanism, dict):
        mechanism = {}
    bridge_lift = _float(metrics.get("bridge_content_lift", 0.0))
    mechanism_lift = _float(mechanism.get("content_lift", metrics.get("mechanism_content_lift", 0.0)))
    real_lift = _float(causal.get("content_lift", 0.0))
    best_ablation = max(
        _float(metrics.get("bridge_zero_value", 0.0)),
        _float(metrics.get("bridge_shuffled_value", 0.0)),
        _float(metrics.get("bridge_stale_value", 0.0)),
    )
    return {
        "schema_version": 1,
        "hidden_target": str(metrics.get("bridge_hidden_target", "")),
        "metric": str(metrics.get("bridge_metric_name", "")),
        "decode_accuracy": _float(metrics.get("bridge_decode_accuracy", 0.0)),
        "subset_agreement": _float(metrics.get("bridge_subset_agreement", 0.0)),
        "baseline_value": _float(metrics.get("bridge_baseline_value", 0.0)),
        "belief_value": _float(metrics.get("bridge_belief_value", 0.0)),
        "best_ablation_value": best_ablation,
        "mechanism_content_lift": mechanism_lift,
        "bridge_content_lift": bridge_lift,
        "real_content_lift": real_lift,
        "mechanism_to_bridge_gap": mechanism_lift - bridge_lift,
        "bridge_to_real_gap": bridge_lift - real_lift,
        "bridge_causal": bool(bridge_lift >= MECHANISM_CONTENT_LIFT_FLOORS.get(domain_name, 0.0)),
        "real_causal": bool(causal.get("content_causal", False)),
    }


def transfer_gap_row(domain_name: str, domain: dict[str, Any]) -> dict[str, Any]:
    """Return one dashboard row for mechanism-to-real transfer."""
    transfer = domain.get("transfer_gap", {})
    if not isinstance(transfer, dict):
        transfer = {}
    return {
        "domain": domain_name,
        "hidden_target": transfer.get("hidden_target", ""),
        "metric": transfer.get("metric", ""),
        "decode_accuracy": transfer.get("decode_accuracy", 0.0),
        "subset_agreement": transfer.get("subset_agreement", 0.0),
        "mechanism_content_lift": transfer.get("mechanism_content_lift", 0.0),
        "bridge_content_lift": transfer.get("bridge_content_lift", 0.0),
        "real_content_lift": transfer.get("real_content_lift", 0.0),
        "mechanism_to_bridge_gap": transfer.get("mechanism_to_bridge_gap", 0.0),
        "bridge_to_real_gap": transfer.get("bridge_to_real_gap", 0.0),
        "bridge_causal": bool(transfer.get("bridge_causal", False)),
        "real_causal": bool(transfer.get("real_causal", False)),
    }


def _query_families(spec: DomainFeatureSpec | None, artifact: dict[str, Any]) -> tuple[str, ...]:
    raw = artifact.get("query_families")
    if isinstance(raw, list) and raw:
        return tuple(str(item) for item in raw)
    if spec is None:
        return ()
    return spec.query_families


def _hidden_targets(spec: DomainFeatureSpec | None, artifact: dict[str, Any]) -> dict[str, Any]:
    raw = artifact.get("hidden_rule_targets")
    if isinstance(raw, dict) and raw:
        return dict(raw)
    if spec is None:
        return {}
    return {"target": spec.hidden_target}


def _message_dim(spec: DomainFeatureSpec | None, artifact: dict[str, Any], bitrate: float) -> int:
    if isinstance(artifact.get("domain_belief"), list):
        return len(artifact["domain_belief"])
    message = artifact.get("crawler_message")
    if isinstance(message, dict) and isinstance(message.get("vector"), list):
        return len(message["vector"])
    if bitrate > 0:
        return int(round(bitrate / 32.0))
    if spec is None:
        return 0
    return int(spec.message_dim)


def _support_size(domain: dict[str, Any], artifact: dict[str, Any]) -> int:
    windows = artifact.get("raw_evidence_windows")
    if isinstance(windows, list):
        return len(windows)
    rows = domain.get("rows")
    if isinstance(rows, list):
        return len(rows)
    return 0


def _readiness_bool(domain: dict[str, Any]) -> bool:
    readiness = str(domain.get("readiness", "")).lower()
    if "unused" in readiness or "blocked" in readiness or "idle" in readiness:
        return False
    if readiness in {"active", "ready", "true"}:
        return True
    if readiness in {"false"}:
        return False
    metric = domain.get("headline_metric", {})
    if isinstance(metric, dict):
        return _float(metric.get("value", 0.0)) > 0.0
    return False


def _ablation_values(domain_name: str, domain: dict[str, Any]) -> dict[str, float]:
    metrics = domain.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    row = {}
    rows = domain.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[-1], dict):
        row = rows[-1]

    if domain_name == "language":
        return _present_ablation_values(
            {
                "zero": row.get("zero_belief_bpc", metrics.get("zero_bpc")),
                "shuffled": row.get("shuffled_belief_bpc", metrics.get("shuffled_bpc")),
                "stale": row.get("stale_belief_bpc", metrics.get("stale_bpc")),
            }
        )
    if domain_name == "image":
        return _present_ablation_values(
            {
                "zero": metrics.get("zero_accuracy", row.get("zero_belief_accuracy")),
                "shuffled": metrics.get("shuffled_accuracy", row.get("shuffled_belief_accuracy")),
                "stale": metrics.get("stale_accuracy", row.get("stale_belief_accuracy")),
            }
        )
    if domain_name == "board":
        return _present_ablation_values(
            {
                "zero": metrics.get("zero_accuracy", row.get("zero_belief_move_accuracy")),
                "shuffled": metrics.get("shuffled_accuracy", row.get("shuffled_belief_move_accuracy")),
                "stale": metrics.get("stale_accuracy", row.get("stale_belief_move_accuracy")),
            }
        )
    if domain_name == "cartpole":
        return _present_ablation_values(
            {
                "zero": metrics.get("zero_eval_return"),
                "shuffled": metrics.get("shuffled_eval_return"),
                "stale": metrics.get("stale_eval_return"),
            }
        )
    return {}


def _present_ablation_values(values: dict[str, object]) -> dict[str, float]:
    ablations: dict[str, float] = {}
    for name, value in values.items():
        if value is None:
            continue
        ablations[name] = _float(value)
    return ablations


def _best_ablation(
    ablations: dict[str, float],
    learned: float,
    lower_is_better: bool,
) -> tuple[str, float]:
    if not ablations:
        return "", float(learned)
    if lower_is_better:
        key = min(ablations, key=lambda name: ablations[name])
    else:
        key = max(ablations, key=lambda name: ablations[name])
    return key, float(ablations[key])


def _signed_gain(baseline: float, learned: float, lower_is_better: bool) -> float:
    if lower_is_better:
        return float(baseline - learned)
    return float(learned - baseline)


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
