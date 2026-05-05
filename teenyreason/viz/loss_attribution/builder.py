"""Build loss-attribution payloads for benchmark artifacts."""

from __future__ import annotations

from .channels import _decision_rows, _expression_channel, _full_system_context_channel
from .common import _arm_summary
from .economics import _probe_family_economics, _sample_economics
from .gates import _latent_win_gate_summary, _representation_gate_summary


def build_loss_attribution_metrics(
    *,
    benchmark_profile: str | None,
    rows: list[dict],
    latent_win_gate: dict,
) -> dict[str, object]:
    """Build a JSON-ready empirical report for one benchmark artifact."""
    seed_count = len(rows)
    algorithm_arms = {
        "baseline": _arm_summary(rows, "baseline"),
        "probe": _arm_summary(rows, "probe"),
        "probe_no_expression": _arm_summary(rows, "probe_no_expression"),
        "full_system": _arm_summary(rows, "full_system"),
        "sim_fanout": _arm_summary(rows, "sim_fanout"),
    }
    sample_economics = _sample_economics(rows)
    expression_channel = _expression_channel(rows)
    full_system_context = _full_system_context_channel(rows)
    representation_gate = _representation_gate_summary(rows)
    latent_gate = _latent_win_gate_summary(
        rows=rows,
        benchmark_profile=benchmark_profile,
        latent_win_gate=latent_win_gate,
        seed_count=seed_count,
    )
    family_economics = _probe_family_economics(rows)
    decisions = _decision_rows(
        sample_economics=sample_economics,
        expression_channel=expression_channel,
        full_system_context=full_system_context,
        representation_gate=representation_gate,
        latent_win_gate=latent_gate,
        family_economics=family_economics,
    )
    return {
        "available": bool(rows),
        "seed_count": int(seed_count),
        "algorithm_arms": algorithm_arms,
        "sample_economics": sample_economics,
        "expression_channel": expression_channel,
        "full_system_context_channel": full_system_context,
        "representation_gate": representation_gate,
        "latent_win_gate": latent_gate,
        "probe_family_economics": family_economics,
        "decisions": decisions,
    }
