"""World-understanding diagnostics for crawler latent expressions."""

from __future__ import annotations


UNDERSTANDING_TARGETS = {
    "cartpole": 0.55,
    "language": 0.35,
    "image": 0.35,
    "board": 0.65,
}


def attach_world_understanding_blocks(domains: dict[str, object]) -> None:
    """Attach empirical world-model diagnostics to every domain payload."""
    for domain_name, domain in domains.items():
        if isinstance(domain, dict):
            domain["world_understanding"] = build_world_understanding_block(domain_name, domain)


def world_understanding_row(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    """Return the dashboard/API row for one domain's understanding metrics."""
    block = domain.get("world_understanding", {})
    if not isinstance(block, dict):
        block = {}
    return {
        "domain": domain_name,
        "score": block.get("score", 0.0),
        "factor_decode": block.get("factor_decode", 0.0),
        "counterfactual": block.get("counterfactual", 0.0),
        "intervention_lift": block.get("intervention_lift", 0.0),
        "compression": block.get("compression", 0.0),
        "transfer": block.get("transfer", 0.0),
        "sample_efficiency": block.get("sample_efficiency", 0.0),
        "passes_target": bool(block.get("passes_target", False)),
        "verdict": block.get("verdict", ""),
        "next_test": block.get("next_test", ""),
    }


def build_world_understanding_block(domain_name: str, domain: dict[str, object]) -> dict[str, object]:
    """Score whether a latent looks like a compact causal world model."""
    metrics = _dict(domain.get("metrics"))
    causal = _dict(domain.get("causal_ablation"))
    transfer = _dict(domain.get("transfer_gap"))
    utility = _dict(domain.get("latent_utility"))
    interface = _dict(domain.get("interface"))
    belief = _dict(interface.get("belief"))

    factor_decode = _factor_decode(metrics, transfer)
    counterfactual = _counterfactual_score(domain_name, domain, metrics)
    intervention_lift = _intervention_lift(causal, metrics, utility)
    compression = _compression_score(
        message_dim=_float(belief.get("message_dim", utility.get("message_dim", 0.0))),
        bitrate=_float(belief.get("bitrate", utility.get("belief_bitrate", 0.0))),
        real_lift=_float(causal.get("content_lift", utility.get("real_content_lift", 0.0))),
        bridge_lift=_float(transfer.get("bridge_content_lift", utility.get("bridge_content_lift", 0.0))),
    )
    transfer_score = _transfer_score(transfer, utility)
    sample_efficiency = _sample_efficiency_score(domain_name, domain, utility)
    real_use = _clip01(_float(causal.get("content_lift", 0.0)) / _real_lift_target(domain_name))

    score = _weighted_mean(
        [
            (factor_decode, 0.18),
            (counterfactual, 0.20),
            (intervention_lift, 0.18),
            (compression, 0.14),
            (transfer_score, 0.16),
            (sample_efficiency, 0.08),
            (real_use, 0.06),
        ]
    )
    target = UNDERSTANDING_TARGETS.get(domain_name, 0.50)
    passes_target = bool(score >= target and transfer_score >= 0.20 and real_use >= 0.20)
    return {
        "schema_version": 1,
        "score": score,
        "target_score": target,
        "passes_target": passes_target,
        "factor_decode": factor_decode,
        "counterfactual": counterfactual,
        "intervention_lift": intervention_lift,
        "compression": compression,
        "transfer": transfer_score,
        "sample_efficiency": sample_efficiency,
        "real_use": real_use,
        "verdict": _verdict(domain_name, score, target, factor_decode, counterfactual, transfer_score, compression),
        "next_test": _next_test(domain_name, factor_decode, counterfactual, transfer_score, compression, real_use),
        "raw": {
            "message_dim": _float(belief.get("message_dim", 0.0)),
            "bitrate": _float(belief.get("bitrate", 0.0)),
            "real_content_lift": _float(causal.get("content_lift", 0.0)),
            "bridge_content_lift": _float(transfer.get("bridge_content_lift", 0.0)),
            "bridge_to_real_gap": _float(transfer.get("bridge_to_real_gap", 0.0)),
        },
    }


def _factor_decode(metrics: dict[str, object], transfer: dict[str, object]) -> float:
    decode = _float(metrics.get("mechanism_decode_accuracy", transfer.get("decode_accuracy", 0.0)))
    subset = _float(metrics.get("mechanism_subset_agreement", transfer.get("subset_agreement", 0.0)))
    real_decode = _float(metrics.get("real_causal_factor_decode", 0.0))
    if decode <= 0.0:
        decode = _float(transfer.get("decode_accuracy", 0.0))
    if subset <= 0.0:
        subset = _float(transfer.get("subset_agreement", 0.0))
    return _clip01(max(real_decode, 0.65 * decode + 0.35 * subset))


def _counterfactual_score(domain_name: str, domain: dict[str, object], metrics: dict[str, object]) -> float:
    if domain_name == "board":
        rows = domain.get("rows")
        if isinstance(rows, list) and rows:
            values = [_float(row.get("belief_value_accuracy", 0.0)) for row in rows if isinstance(row, dict)]
            return _clip01(_mean(values))
        return _clip01(_float(metrics.get("belief_value_accuracy", 0.0)))
    if domain_name == "cartpole":
        real_cf = _float(metrics.get("real_causal_counterfactual_accuracy", 0.0))
        r2 = _float(metrics.get("mechanics_r2", 0.0))
        handoff_delta = _float(metrics.get("handoff_value_delta_correct_vs_shuffled", 0.0))
        return _clip01(max(real_cf, r2, handoff_delta / 10.0))
    if domain_name == "language":
        real_cf = _float(metrics.get("real_causal_counterfactual_accuracy", 0.0))
        continuation = _float(metrics.get("continuation_accuracy", 0.0))
        mechanism = _float(metrics.get("mechanism_belief_accuracy", 0.0))
        return _clip01(max(real_cf, 0.55 * continuation + 0.45 * mechanism))
    if domain_name == "image":
        real_cf = _float(metrics.get("real_causal_counterfactual_accuracy", 0.0))
        mechanism = _float(metrics.get("mechanism_belief_accuracy", 0.0))
        stability = _float(metrics.get("prototype_stability", 0.0))
        return _clip01(max(real_cf, 0.55 * mechanism + 0.45 * stability))
    return 0.0


def _intervention_lift(
    causal: dict[str, object],
    metrics: dict[str, object],
    utility: dict[str, object],
) -> float:
    real_lift = _float(causal.get("content_lift", utility.get("real_content_lift", 0.0)))
    mechanism_lift = _float(metrics.get("mechanism_content_lift", utility.get("mechanism_content_lift", 0.0)))
    bridge_lift = _float(utility.get("bridge_content_lift", 0.0))
    return _clip01(max(real_lift, 0.5 * mechanism_lift, 0.5 * bridge_lift))


def _compression_score(*, message_dim: float, bitrate: float, real_lift: float, bridge_lift: float) -> float:
    useful_lift = max(real_lift, 0.25 * bridge_lift, 0.0)
    dim = max(message_dim, bitrate / 32.0, 1.0)
    raw = useful_lift * 8.0 / (1.0 + dim / 32.0)
    return _clip01(raw)


def _transfer_score(transfer: dict[str, object], utility: dict[str, object]) -> float:
    bridge_lift = _float(transfer.get("bridge_content_lift", utility.get("bridge_content_lift", 0.0)))
    real_lift = _float(transfer.get("real_content_lift", utility.get("real_content_lift", 0.0)))
    if bridge_lift <= 0.0:
        return _clip01(real_lift / 0.05)
    return _clip01(max(real_lift, 0.0) / max(bridge_lift, 1e-9))


def _sample_efficiency_score(domain_name: str, domain: dict[str, object], utility: dict[str, object]) -> float:
    evidence_cost = max(_float(utility.get("evidence_cost", domain.get("evidence_cost", 0.0))), 1.0)
    if domain_name == "board":
        return _clip01(_float(domain.get("belief_contribution_margin", 0.0)) / evidence_cost)
    if domain_name == "cartpole":
        saved = _float(_dict(domain.get("metrics")).get("handoff_dedicated_probe_steps_saved", 0.0))
        old = max(_float(_dict(domain.get("metrics")).get("handoff_expensive_dedicated_probe_steps", 0.0)), 1.0)
        return _clip01(saved / old)
    return _clip01(max(_float(utility.get("budget_gate_mean_gain", 0.0)), 0.0) / evidence_cost)


def _verdict(
    domain_name: str,
    score: float,
    target: float,
    factor_decode: float,
    counterfactual: float,
    transfer: float,
    compression: float,
) -> str:
    if score >= target and transfer >= 0.5:
        return "compact_world_model_candidate"
    if factor_decode >= 0.9 and counterfactual >= 0.8 and transfer < 0.1:
        return "controlled_understanding_not_transferring"
    if compression < 0.1 and domain_name == "image":
        return "too_wide_for_fundamental_latent"
    if counterfactual < 0.4:
        return "missing_counterfactual_model"
    if transfer < 0.1:
        return "solver_interface_gap"
    return "partial_factor_model"


def _next_test(
    domain_name: str,
    factor_decode: float,
    counterfactual: float,
    transfer: float,
    compression: float,
    real_use: float,
) -> str:
    if factor_decode < 0.7:
        return "factor_identification_probe"
    if counterfactual < 0.7:
        return "counterfactual_rollout_probe"
    if compression < 0.2:
        return "compression_curve"
    if transfer < 0.2 or real_use < 0.2:
        if domain_name == "language":
            return "adapter_vs_prefix_handoff"
        if domain_name == "image":
            return "compressed_prototype_gate"
        if domain_name == "cartpole":
            return "cheap_context_ppo_arm"
    if domain_name == "board":
        return "larger_rule_space_board"
    return "harder_transfer_split"


def _real_lift_target(domain_name: str) -> float:
    if domain_name == "cartpole":
        return 100.0
    if domain_name in {"language", "image"}:
        return 0.02
    if domain_name == "board":
        return 0.05
    return 0.10


def _dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _weighted_mean(items: list[tuple[float, float]]) -> float:
    weight_sum = sum(weight for _value, weight in items)
    if weight_sum <= 0.0:
        return 0.0
    return _clip01(sum(_clip01(value) * weight for value, weight in items) / weight_sum)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
