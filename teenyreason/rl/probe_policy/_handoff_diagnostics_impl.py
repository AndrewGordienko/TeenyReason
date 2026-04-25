"""Online fair-handoff diagnostics built from the currently observed probe support."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import torch

from ...crawler import CrawlerModelBundle
from ...models.belief_world_model import build_future_summary_targets
from ...probe.probe_latent import aggregate_env_belief


DEFAULT_ONLINE_SPLIT_LATENT_DISAGREEMENT_BAD = 0.08
DEFAULT_ONLINE_SPLIT_RETRIEVAL_MARGIN_DEFICIT_BAD = 0.20
DEFAULT_ONLINE_LEAVEOUT_SHIFT_BAD = 0.18


def _default_handoff_diagnostics() -> dict[str, Any]:
    """Return one stable default payload when diagnostics are unavailable."""
    return {
        "future_probe_error": 0.0,
        "full_future_prediction_error": 0.0,
        "observed_family_future_error": 0.0,
        "heldout_family_future_error": 0.0,
        "support_size_matched_future_error": 0.0,
        "online_offline_gap": 0.0,
        "online_subset_stability": 0.0,
        "online_geometry_complete": False,
        "online_split_latent_disagreement": DEFAULT_ONLINE_SPLIT_LATENT_DISAGREEMENT_BAD,
        "online_split_retrieval_margin_deficit": DEFAULT_ONLINE_SPLIT_RETRIEVAL_MARGIN_DEFICIT_BAD,
        "online_leaveout_shift": DEFAULT_ONLINE_LEAVEOUT_SHIFT_BAD,
        "online_observed_family_count": 0,
        "fair_handoff_probe_families": (),
    }


def _belief_latent_from_payload(payload: dict[str, np.ndarray]) -> np.ndarray:
    """Extract one flat predictive latent from an aggregation payload."""
    values = payload.get("env_mean_raw", payload.get("env_mean", np.zeros((0,), dtype=np.float32)))
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def _probe_window_arrays(
    probe_windows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack raw probe-window records into the tensors used by the future target builder."""
    states = np.stack(
        [np.asarray(item["states"], dtype=np.float32) for item in probe_windows],
        axis=0,
    ).astype(np.float32)
    actions = np.stack(
        [np.asarray(item["actions"], dtype=np.int64) for item in probe_windows],
        axis=0,
    ).astype(np.int64)
    rewards = np.stack(
        [np.asarray(item["rewards"], dtype=np.float32) for item in probe_windows],
        axis=0,
    ).astype(np.float32)
    terminated = np.asarray(
        [bool(item.get("terminated", False)) for item in probe_windows],
        dtype=np.bool_,
    )
    truncated = np.asarray(
        [bool(item.get("truncated", False)) for item in probe_windows],
        dtype=np.bool_,
    )
    families = np.asarray(
        [str(item.get("probe_family", "")) for item in probe_windows],
        dtype="U",
    )
    return states, actions, rewards, terminated, truncated, families


def _family_target_means(
    *,
    families: np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    """Average one target per observed probe family."""
    target_by_family: dict[str, np.ndarray] = {}
    for family in OrderedDict.fromkeys(families.tolist()).keys():
        family_mask = families == family
        target_by_family[str(family)] = np.mean(targets[family_mask], axis=0).astype(np.float32)
    return target_by_family


def _family_index_map(family_names: tuple[str, ...]) -> dict[str, int]:
    """Build one stable family-name lookup for the trained family head."""
    return {
        str(family): idx
        for idx, family in enumerate(family_names)
    }


def _family_group_ids(families: np.ndarray) -> np.ndarray:
    """Map observed family names into one stable integer id array."""
    ordered = OrderedDict.fromkeys(str(family) for family in families.tolist())
    family_to_idx = {
        family: idx
        for idx, family in enumerate(ordered.keys())
    }
    return np.asarray(
        [family_to_idx[str(family)] for family in families.tolist()],
        dtype=np.int64,
    )


def _normalize_ascending(
    value: float | None,
    *,
    good: float,
    bad: float,
) -> float:
    """Map smaller-is-better values into a stable 0-1 quality score."""
    if value is None:
        return 0.0
    if bad <= good:
        return 1.0 if float(value) <= good else 0.0
    return float(np.clip((bad - float(value)) / (bad - good), 0.0, 1.0))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return one bounded cosine similarity for two flat latents."""
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))


def _balanced_family_split_indices(families: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split support windows into two balanced halves while preserving family coverage."""
    families = np.asarray(families, dtype="U").reshape(-1)
    if families.size <= 1:
        single_idx = np.arange(families.size, dtype=np.int64)
        return single_idx, single_idx

    left: list[int] = []
    right: list[int] = []
    leftovers: list[int] = []
    family_to_indices: dict[str, list[int]] = {}
    for idx, family in enumerate(families.tolist()):
        family_to_indices.setdefault(str(family), []).append(int(idx))

    for family in OrderedDict.fromkeys(families.tolist()).keys():
        family_indices = family_to_indices.get(str(family), [])
        if len(family_indices) >= 2:
            left.append(int(family_indices[0]))
            right.append(int(family_indices[1]))
            leftovers.extend(int(item) for item in family_indices[2:])
        elif family_indices:
            leftovers.append(int(family_indices[0]))

    for idx in leftovers:
        target = left if len(left) <= len(right) else right
        target.append(int(idx))

    if not left and right:
        left.append(int(right[0]))
    if not right and left:
        right.append(int(left[-1]))
    if not left or not right:
        ordered_idx = np.arange(families.size, dtype=np.int64)
        split_point = max(1, int(np.ceil(float(ordered_idx.size) / 2.0)))
        left = ordered_idx[:split_point].tolist()
        right = ordered_idx[split_point:].tolist()
        if not right:
            right = ordered_idx[-1:].tolist()
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
    )


def compute_online_geometry_diagnostics(
    *,
    posterior_views: list[tuple[np.ndarray, np.ndarray]],
    probe_families: np.ndarray,
    aggregate_latent_fn: Callable[
        [list[tuple[np.ndarray, np.ndarray]], np.ndarray | None],
        np.ndarray,
    ],
) -> dict[str, Any]:
    """Measure live split/leaveout geometry from the current support set."""
    families = np.asarray(probe_families, dtype="U").reshape(-1)
    observed_family_count = int(len(OrderedDict.fromkeys(families.tolist())))
    diagnostics = {
        "online_subset_stability": 0.0,
        "online_geometry_complete": False,
        "online_split_latent_disagreement": DEFAULT_ONLINE_SPLIT_LATENT_DISAGREEMENT_BAD,
        "online_split_retrieval_margin_deficit": DEFAULT_ONLINE_SPLIT_RETRIEVAL_MARGIN_DEFICIT_BAD,
        "online_leaveout_shift": DEFAULT_ONLINE_LEAVEOUT_SHIFT_BAD,
        "online_observed_family_count": observed_family_count,
    }
    if len(posterior_views) < 2 or observed_family_count < 2:
        return diagnostics

    group_ids = _family_group_ids(families)
    left_idx, right_idx = _balanced_family_split_indices(families)
    if left_idx.size == 0 or right_idx.size == 0:
        return diagnostics

    full_latent = aggregate_latent_fn(posterior_views, group_ids)
    left_latent = aggregate_latent_fn(
        [posterior_views[int(idx)] for idx in left_idx.tolist()],
        group_ids[left_idx],
    )
    right_latent = aggregate_latent_fn(
        [posterior_views[int(idx)] for idx in right_idx.tolist()],
        group_ids[right_idx],
    )
    split_latent_disagreement = float(
        np.linalg.norm(
            np.asarray(left_latent, dtype=np.float32).reshape(-1)
            - np.asarray(right_latent, dtype=np.float32).reshape(-1)
        )
    )
    split_similarity = _cosine_similarity(left_latent, right_latent)
    retrieval_margin_deficit = float(max(0.0, 0.90 - split_similarity))

    leaveout_shifts: list[float] = []
    for family in OrderedDict.fromkeys(families.tolist()).keys():
        keep_idx = np.nonzero(families != str(family))[0]
        if keep_idx.size == 0:
            continue
        leaveout_latent = aggregate_latent_fn(
            [posterior_views[int(idx)] for idx in keep_idx.tolist()],
            group_ids[keep_idx],
        )
        leaveout_shifts.append(
            float(
                np.linalg.norm(
                    np.asarray(leaveout_latent, dtype=np.float32).reshape(-1)
                    - np.asarray(full_latent, dtype=np.float32).reshape(-1)
                )
            )
        )
    if not leaveout_shifts:
        return diagnostics

    leaveout_shift = float(np.mean(np.asarray(leaveout_shifts, dtype=np.float32)))
    disagreement_quality = _normalize_ascending(
        split_latent_disagreement,
        good=0.015,
        bad=DEFAULT_ONLINE_SPLIT_LATENT_DISAGREEMENT_BAD,
    )
    retrieval_quality = _normalize_ascending(
        retrieval_margin_deficit,
        good=0.02,
        bad=DEFAULT_ONLINE_SPLIT_RETRIEVAL_MARGIN_DEFICIT_BAD,
    )
    leaveout_quality = _normalize_ascending(
        leaveout_shift,
        good=0.025,
        bad=DEFAULT_ONLINE_LEAVEOUT_SHIFT_BAD,
    )
    family_coverage_quality = 1.0 if observed_family_count >= 2 else 0.0
    subset_stability = float(
        np.clip(
            0.45 * disagreement_quality
            + 0.30 * retrieval_quality
            + 0.20 * leaveout_quality
            + 0.05 * family_coverage_quality,
            0.0,
            1.0,
        )
    )
    return {
        "online_subset_stability": subset_stability,
        "online_geometry_complete": True,
        "online_split_latent_disagreement": split_latent_disagreement,
        "online_split_retrieval_margin_deficit": retrieval_margin_deficit,
        "online_leaveout_shift": leaveout_shift,
        "online_observed_family_count": observed_family_count,
    }


def _predict_family_future(
    *,
    crawler_bundle: CrawlerModelBundle,
    latent: np.ndarray,
    family_idx: int,
) -> np.ndarray | None:
    """Predict one family-conditioned future signature from the current belief."""
    if crawler_bundle.env_family_future_predictor is None:
        return None
    with torch.no_grad():
        pred_t = crawler_bundle.env_family_future_predictor(
            torch.tensor(latent, dtype=torch.float32, device=crawler_bundle.device),
            torch.tensor([int(family_idx)], dtype=torch.long, device=crawler_bundle.device),
        )
    return np.asarray(pred_t.cpu().numpy(), dtype=np.float32).reshape(-1)


def _predict_full_future(
    *,
    crawler_bundle: CrawlerModelBundle,
    latent: np.ndarray,
) -> np.ndarray | None:
    """Predict the generic future signature from the current full belief."""
    if crawler_bundle.env_future_predictor is None:
        return None
    with torch.no_grad():
        pred_t = crawler_bundle.env_future_predictor(
            torch.tensor(latent, dtype=torch.float32, device=crawler_bundle.device)
        )
    return np.asarray(pred_t.cpu().numpy(), dtype=np.float32).reshape(-1)


def compute_online_future_diagnostics(
    *,
    crawler_bundle: CrawlerModelBundle | None,
    posterior_views: list[tuple[np.ndarray, np.ndarray]],
    probe_windows: list[dict[str, Any]],
    env_name: str,
) -> dict[str, Any]:
    """Estimate how well the current support set predicts and geometrically supports handoff."""
    if crawler_bundle is None or not posterior_views or not probe_windows:
        return _default_handoff_diagnostics()
    if len(posterior_views) != len(probe_windows):
        return _default_handoff_diagnostics()

    states, actions, rewards, terminated, truncated, families = _probe_window_arrays(probe_windows)
    family_group_ids = _family_group_ids(families)
    split_idx = max(2, actions.shape[1] // 2)
    future_targets = build_future_summary_targets(
        states=states[:, split_idx:, :],
        actions=actions[:, split_idx:],
        rewards=rewards[:, split_idx:],
        terminated=terminated,
        truncated=truncated,
        action_vocab_size=int(crawler_bundle.action_vocab_size),
        probe_mode=families,
        env_name=env_name,
    )
    family_target_means = _family_target_means(families=families, targets=future_targets)
    family_index_by_name = _family_index_map(getattr(crawler_bundle, "family_names", ()))

    _belief, full_payload = aggregate_env_belief(
        belief_aggregator=crawler_bundle.belief_aggregator,
        env_param_predictor=crawler_bundle.env_param_predictor,
        device=crawler_bundle.device,
        posterior_views=posterior_views,
        probe_group_ids=family_group_ids,
    )
    full_latent = _belief_latent_from_payload(full_payload)
    geometry_diagnostics = compute_online_geometry_diagnostics(
        posterior_views=posterior_views,
        probe_families=families,
        aggregate_latent_fn=lambda views, group_ids: _belief_latent_from_payload(
            aggregate_env_belief(
                belief_aggregator=crawler_bundle.belief_aggregator,
                env_param_predictor=crawler_bundle.env_param_predictor,
                device=crawler_bundle.device,
                posterior_views=views,
                probe_group_ids=group_ids,
            )[1]
        ),
    )
    full_future_prediction = _predict_full_future(
        crawler_bundle=crawler_bundle,
        latent=full_latent,
    )
    full_future_prediction_error = 0.0
    if full_future_prediction is not None:
        full_target = np.mean(future_targets, axis=0).astype(np.float32)
        full_future_prediction_error = float(
            np.mean(np.abs(full_future_prediction - full_target))
        )

    observed_family_errors: list[float] = []
    for family, target in family_target_means.items():
        family_idx = family_index_by_name.get(str(family))
        if family_idx is None:
            continue
        family_prediction = _predict_family_future(
            crawler_bundle=crawler_bundle,
            latent=full_latent,
            family_idx=family_idx,
        )
        if family_prediction is None:
            continue
        observed_family_errors.append(
            float(np.mean(np.abs(family_prediction - np.asarray(target, dtype=np.float32))))
        )
    observed_family_future_error = (
        float(np.mean(np.asarray(observed_family_errors, dtype=np.float32)))
        if observed_family_errors
        else float(full_future_prediction_error)
    )

    heldout_family_errors: list[float] = []
    unique_families = tuple(OrderedDict.fromkeys(families.tolist()).keys())
    for heldout_family in unique_families:
        family_idx = family_index_by_name.get(str(heldout_family))
        if family_idx is None:
            continue
        subset_indices = [
            idx
            for idx, family in enumerate(families.tolist())
            if str(family) != str(heldout_family)
        ]
        if not subset_indices:
            continue
        subset_posteriors = [posterior_views[idx] for idx in subset_indices]
        _subset_belief, subset_payload = aggregate_env_belief(
            belief_aggregator=crawler_bundle.belief_aggregator,
            env_param_predictor=crawler_bundle.env_param_predictor,
            device=crawler_bundle.device,
            posterior_views=subset_posteriors,
            probe_group_ids=family_group_ids[np.asarray(subset_indices, dtype=np.int64)],
        )
        subset_latent = _belief_latent_from_payload(subset_payload)
        heldout_prediction = _predict_family_future(
            crawler_bundle=crawler_bundle,
            latent=subset_latent,
            family_idx=family_idx,
        )
        if heldout_prediction is None:
            continue
        heldout_target = np.asarray(family_target_means[str(heldout_family)], dtype=np.float32)
        heldout_family_errors.append(
            float(np.mean(np.abs(heldout_prediction - heldout_target)))
        )

    support_size_matched_future_error = (
        float(np.mean(np.asarray(heldout_family_errors, dtype=np.float32)))
        if heldout_family_errors
        else float(observed_family_future_error)
    )
    diagnostics = {
        "future_probe_error": float(support_size_matched_future_error),
        "full_future_prediction_error": float(full_future_prediction_error),
        "observed_family_future_error": float(observed_family_future_error),
        "heldout_family_future_error": float(support_size_matched_future_error),
        "support_size_matched_future_error": float(support_size_matched_future_error),
        "online_offline_gap": float(
            abs(full_future_prediction_error - support_size_matched_future_error)
        ),
        "fair_handoff_probe_families": tuple(str(family) for family in unique_families),
    }
    diagnostics.update(geometry_diagnostics)
    return diagnostics
