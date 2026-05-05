"""Shared acceptance gates for multi-domain suite payloads."""

from __future__ import annotations


CARTPOLE_MIN_CONTEXT_RETURN_GAIN = 100.0
LANGUAGE_MIN_VISIBLE_BPC_GAIN = 0.01
IMAGE_MIN_VISIBLE_ACCURACY_GAIN = 0.01
BOARD_MIN_VISIBLE_MOVE_GAIN = 0.05


def _domain_value(domains: dict[str, object], domain_name: str, key: str) -> float:
    domain = domains.get(domain_name, {})
    if not isinstance(domain, dict):
        return 0.0
    try:
        return float(domain.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _content_causal(domains: dict[str, object], domain_name: str) -> bool | None:
    domain = domains.get(domain_name, {})
    if not isinstance(domain, dict):
        return None
    causal = domain.get("causal_ablation", {})
    if not isinstance(causal, dict) or "content_causal" not in causal:
        return None
    return bool(causal.get("content_causal", False))


def _claim_allowed_by_content(
    domains: dict[str, object],
    domain_name: str,
    fallback_allowed: bool,
) -> bool:
    content_causal = _content_causal(domains, domain_name)
    if content_causal is None:
        return bool(fallback_allowed)
    return bool(fallback_allowed and content_causal)


def suite_acceptance_from_domains(domains: dict[str, object]) -> dict[str, bool]:
    """Return conservative latent-claim gates from suite domain summaries."""
    cartpole_gain = _domain_value(domains, "cartpole", "belief_contribution_margin")
    cartpole_ablation = _domain_value(domains, "cartpole", "ablation_gap")
    language_gain = _domain_value(domains, "language", "belief_contribution_margin")
    language_ablation = _domain_value(domains, "language", "ablation_gap")
    image_gain = _domain_value(domains, "image", "belief_contribution_margin")
    image_ablation = _domain_value(domains, "image", "ablation_gap")
    board_gain = _domain_value(domains, "board", "belief_contribution_margin")
    board_ablation = _domain_value(domains, "board", "ablation_gap")
    cartpole_allowed = bool(
        cartpole_gain > 0.0
        and cartpole_ablation >= CARTPOLE_MIN_CONTEXT_RETURN_GAIN
    )
    language_allowed = bool(
        language_gain >= LANGUAGE_MIN_VISIBLE_BPC_GAIN
        or language_ablation >= LANGUAGE_MIN_VISIBLE_BPC_GAIN
    )
    image_allowed = bool(
        image_gain >= IMAGE_MIN_VISIBLE_ACCURACY_GAIN
        or image_ablation >= IMAGE_MIN_VISIBLE_ACCURACY_GAIN
    )
    board_allowed = bool(
        board_gain >= BOARD_MIN_VISIBLE_MOVE_GAIN
        or board_ablation >= BOARD_MIN_VISIBLE_MOVE_GAIN
    )
    return {
        "cartpole_latent_claim_allowed": _claim_allowed_by_content(
            domains,
            "cartpole",
            cartpole_allowed,
        ),
        "language_latent_claim_allowed": _claim_allowed_by_content(
            domains,
            "language",
            language_allowed,
        ),
        "image_latent_claim_allowed": _claim_allowed_by_content(
            domains,
            "image",
            image_allowed,
        ),
        "board_latent_claim_allowed": _claim_allowed_by_content(
            domains,
            "board",
            board_allowed,
        ),
    }


def attach_suite_claim_gates(domains: dict[str, object]) -> dict[str, bool]:
    """Attach each domain's conservative latent-claim gate in-place."""
    acceptance = suite_acceptance_from_domains(domains)
    for domain_name in ("cartpole", "language", "image", "board"):
        domain = domains.get(domain_name)
        if not isinstance(domain, dict):
            continue
        key = f"{domain_name}_latent_claim_allowed"
        domain["claim_allowed"] = bool(acceptance.get(key, False))
    return acceptance


def suite_acceptance_thresholds() -> dict[str, float]:
    """Expose gate thresholds so the dashboard can explain blocked claims."""
    return {
        "cartpole_min_context_return_gain": CARTPOLE_MIN_CONTEXT_RETURN_GAIN,
        "language_min_visible_bpc_gain": LANGUAGE_MIN_VISIBLE_BPC_GAIN,
        "image_min_visible_accuracy_gain": IMAGE_MIN_VISIBLE_ACCURACY_GAIN,
        "board_min_visible_move_gain": BOARD_MIN_VISIBLE_MOVE_GAIN,
    }
