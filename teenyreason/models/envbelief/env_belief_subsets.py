"""Subset and support-mask helpers for env-level beliefs."""

from __future__ import annotations

import math

import torch

from .env_belief_models import EnvBeliefAggregator, EnvParamPredictorEnsemble


def build_diverse_support_mask(
    mask: torch.Tensor,
    support_size: int,
    group_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose one canonical support set, preferring early unique groups.

    Probe collection orders windows deliberately. For Continuous CartPole the
    first four named probes are the stable mechanics-identification views, while
    the later stress probes are best treated as held-out evidence. Keeping this
    deterministic makes the belief contract inspectable and avoids silently
    rebuilding the headline belief from different families every epoch.
    """
    if support_size <= 0:
        raise ValueError("support_size must be positive")
    del generator

    mask_bool = mask.bool()
    support_mask = torch.zeros_like(mask_bool, dtype=torch.float32)

    for row_idx in range(mask_bool.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() <= support_size:
            support_mask[row_idx, valid_idx] = 1.0
            continue

        chosen: list[int] = []
        if group_ids is not None:
            valid_groups = group_ids[row_idx, valid_idx]
            seen_groups: set[int] = set()
            for idx, group_value in zip(valid_idx.tolist(), valid_groups.tolist(), strict=False):
                if int(group_value) < 0 or int(group_value) in seen_groups:
                    continue
                seen_groups.add(int(group_value))
                chosen.append(int(idx))
                if len(chosen) >= support_size:
                    break

        if len(chosen) < support_size:
            remaining_idx = torch.tensor(
                [idx for idx in valid_idx.tolist() if idx not in chosen],
                dtype=torch.long,
                device=valid_idx.device,
            )
            if remaining_idx.numel() > 0:
                take = min(int(remaining_idx.numel()), support_size - len(chosen))
                chosen.extend(int(item) for item in remaining_idx[:take].tolist())

        if not chosen:
            support_mask[row_idx, valid_idx[0]] = 1.0
            continue
        support_mask[row_idx, torch.tensor(chosen, dtype=torch.long, device=mask.device)] = 1.0

    return support_mask


def build_support_budget_mask(
    mask: torch.Tensor,
    support_size: int,
    subset_count: int = 1,
    group_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose the canonical support views used to build the env belief.

    `subset_count` used to mean "draw this many supports and union them". That
    made the headline belief quietly average most windows in a world, which hid
    collapse and left little held-out evidence for prediction checks. Keep the
    argument for API compatibility, but spend the support budget exactly once.
    Split and leaveout diagnostics still create their own masks downstream.
    """
    if subset_count <= 0:
        raise ValueError("subset_count must be positive")

    del subset_count
    return build_diverse_support_mask(
        mask=mask,
        support_size=support_size,
        group_ids=group_ids,
        generator=generator,
    )


def build_split_source_mask(
    mask: torch.Tensor,
    support_mask: torch.Tensor,
    min_extra_views: int = 2,
) -> torch.Tensor:
    """Choose which views are allowed to form split-retrieval diagnostics.

    The headline env belief should stay on the small canonical support budget,
    but split retrieval is meant to ask whether two partial evidence sets agree
    about the world. When held-out probe windows exist, use them for the split
    halves so the retrieval task is not starved down to two views per side.
    """
    mask_f = mask.float()
    support_f = support_mask.float()
    available_count = mask_f.sum(dim=1, keepdim=True)
    support_count = support_f.sum(dim=1, keepdim=True)
    use_available = available_count >= support_count + float(max(0, int(min_extra_views)))
    return torch.where(use_available, mask_f, support_f)


def build_env_subset_masks(
    mask: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split valid views into two balanced non-empty subsets per env."""
    mask_bool = mask.bool()
    mask_a = torch.zeros_like(mask_bool)
    mask_b = torch.zeros_like(mask_bool)

    for row_idx in range(mask_bool.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() == 1:
            mask_a[row_idx, valid_idx] = True
            mask_b[row_idx, valid_idx] = True
            continue

        if group_ids is None:
            permutation = valid_idx[
                torch.randperm(valid_idx.numel(), generator=generator, device=valid_idx.device)
            ]
            split_point = max(1, int(math.ceil(permutation.numel() / 2.0)))
            left = permutation[:split_point]
            right = permutation[split_point:]
            if right.numel() == 0:
                right = left[-1:].clone()
            mask_a[row_idx, left] = True
            mask_b[row_idx, right] = True
            continue

        row_groups = group_ids[row_idx, valid_idx]
        chosen_a: list[int] = []
        chosen_b: list[int] = []
        leftovers: list[int] = []
        valid_groups = torch.unique(row_groups[row_groups >= 0], sorted=False)

        if valid_groups.numel() > 0:
            shuffled_groups = valid_groups[
                torch.randperm(
                    valid_groups.numel(),
                    generator=generator,
                    device=valid_groups.device,
                )
            ]
            for group_value in shuffled_groups.tolist():
                group_candidates = valid_idx[row_groups == int(group_value)]
                if group_candidates.numel() == 0:
                    continue
                group_candidates = group_candidates[
                    torch.randperm(
                        group_candidates.numel(),
                        generator=generator,
                        device=group_candidates.device,
                    )
                ]
                if group_candidates.numel() >= 2:
                    chosen_a.append(int(group_candidates[0].item()))
                    chosen_b.append(int(group_candidates[1].item()))
                    leftovers.extend(int(item) for item in group_candidates[2:].tolist())
                else:
                    target = chosen_a if len(chosen_a) <= len(chosen_b) else chosen_b
                    target.append(int(group_candidates[0].item()))

        ungroupped_idx = valid_idx[row_groups < 0]
        if ungroupped_idx.numel() > 0:
            ungroupped_idx = ungroupped_idx[
                torch.randperm(
                    ungroupped_idx.numel(),
                    generator=generator,
                    device=ungroupped_idx.device,
                )
            ]
            leftovers.extend(int(item) for item in ungroupped_idx.tolist())

        assigned = set(chosen_a + chosen_b + leftovers)
        remainder = [int(idx) for idx in valid_idx.tolist() if int(idx) not in assigned]
        if remainder:
            remainder_t = torch.tensor(remainder, dtype=torch.long, device=valid_idx.device)
            remainder_t = remainder_t[
                torch.randperm(
                    remainder_t.numel(),
                    generator=generator,
                    device=remainder_t.device,
                )
            ]
            leftovers.extend(int(item) for item in remainder_t.tolist())

        for idx in leftovers:
            target = chosen_a if len(chosen_a) <= len(chosen_b) else chosen_b
            target.append(int(idx))

        if not chosen_a and chosen_b:
            chosen_a.append(chosen_b.pop())
        if not chosen_b and chosen_a:
            chosen_b.append(chosen_a.pop())
        if not chosen_a and not chosen_b:
            permutation = valid_idx[
                torch.randperm(valid_idx.numel(), generator=generator, device=valid_idx.device)
            ]
            chosen_a.append(int(permutation[0].item()))
            chosen_b.append(int(permutation[-1].item()))

        mask_a[row_idx, torch.tensor(chosen_a, dtype=torch.long, device=mask.device)] = True
        mask_b[row_idx, torch.tensor(chosen_b, dtype=torch.long, device=mask.device)] = True

    return mask_a.float(), mask_b.float()


def build_cross_family_subset_masks(
    mask: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split valid views into disjoint probe-family halves when labels exist."""
    if group_ids is None:
        return build_env_subset_masks(mask, group_ids=None, generator=generator)

    mask_bool = mask.bool()
    mask_a = torch.zeros_like(mask_bool)
    mask_b = torch.zeros_like(mask_bool)

    for row_idx in range(mask_bool.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() <= 1:
            if valid_idx.numel() == 1:
                mask_a[row_idx, valid_idx] = True
                mask_b[row_idx, valid_idx] = True
            continue

        row_groups = group_ids[row_idx, valid_idx]
        valid_groups = torch.unique(row_groups[row_groups >= 0], sorted=False)
        if valid_groups.numel() <= 1:
            fallback_a, fallback_b = build_env_subset_masks(
                mask_bool[row_idx:row_idx + 1].float(),
                group_ids=group_ids[row_idx:row_idx + 1],
                generator=generator,
            )
            mask_a[row_idx] = fallback_a[0].bool()
            mask_b[row_idx] = fallback_b[0].bool()
            continue

        shuffled_groups = valid_groups[
            torch.randperm(
                valid_groups.numel(),
                generator=generator,
                device=valid_groups.device,
            )
        ]
        split_point = max(1, int(math.ceil(shuffled_groups.numel() / 2.0)))
        groups_a = shuffled_groups[:split_point]
        groups_b = shuffled_groups[split_point:]
        if groups_b.numel() == 0:
            groups_b = groups_a[-1:].clone()
            groups_a = groups_a[:-1]
        if groups_a.numel() == 0:
            groups_a = groups_b[:1].clone()
            groups_b = groups_b[1:]

        row_group_ids = group_ids[row_idx]
        for group_value in groups_a.tolist():
            mask_a[row_idx] |= mask_bool[row_idx] & (row_group_ids == int(group_value))
        for group_value in groups_b.tolist():
            mask_b[row_idx] |= mask_bool[row_idx] & (row_group_ids == int(group_value))

        if not torch.any(mask_a[row_idx]) or not torch.any(mask_b[row_idx]):
            fallback_a, fallback_b = build_env_subset_masks(
                mask_bool[row_idx:row_idx + 1].float(),
                group_ids=group_ids[row_idx:row_idx + 1],
                generator=generator,
            )
            mask_a[row_idx] = fallback_a[0].bool()
            mask_b[row_idx] = fallback_b[0].bool()

    return mask_a.float(), mask_b.float()


def compute_disjoint_support_splits(
    aggregator: EnvBeliefAggregator,
    grouped_mean: torch.Tensor,
    grouped_logvar: torch.Tensor,
    support_mask: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    env_param_predictor: EnvParamPredictorEnsemble | None = None,
    split_mode: str = "paired",
) -> dict[str, torch.Tensor]:
    """Build two non-overlapping support-set beliefs for each env instance."""
    if split_mode == "paired":
        split_mask_a, split_mask_b = build_env_subset_masks(
            support_mask,
            group_ids=group_ids,
        )
    elif split_mode == "cross_family":
        split_mask_a, split_mask_b = build_cross_family_subset_masks(
            support_mask,
            group_ids=group_ids,
        )
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    stats_a = aggregator.aggregate_stats(grouped_mean, grouped_logvar, split_mask_a, group_ids)
    stats_b = aggregator.aggregate_stats(grouped_mean, grouped_logvar, split_mask_b, group_ids)
    env_mean_a = stats_a["env_mean_raw"]
    env_mean_b = stats_b["env_mean_raw"]
    split_count = torch.stack([split_mask_a.sum(dim=1), split_mask_b.sum(dim=1)], dim=1)
    if group_ids is None:
        split_group_count_a = split_mask_a.sum(dim=1)
        split_group_count_b = split_mask_b.sum(dim=1)
        split_group_overlap = torch.zeros_like(split_group_count_a)
    else:
        split_group_count_a = torch.zeros((split_mask_a.shape[0],), dtype=torch.float32, device=split_mask_a.device)
        split_group_count_b = torch.zeros((split_mask_b.shape[0],), dtype=torch.float32, device=split_mask_b.device)
        split_group_overlap = torch.zeros((split_mask_a.shape[0],), dtype=torch.float32, device=split_mask_a.device)
        for row_idx in range(split_mask_a.shape[0]):
            idx_a = torch.nonzero(split_mask_a[row_idx] > 0, as_tuple=False).squeeze(-1)
            idx_b = torch.nonzero(split_mask_b[row_idx] > 0, as_tuple=False).squeeze(-1)
            groups_a = group_ids[row_idx, idx_a]
            groups_b = group_ids[row_idx, idx_b]
            groups_a = torch.unique(groups_a[groups_a >= 0], sorted=True)
            groups_b = torch.unique(groups_b[groups_b >= 0], sorted=True)
            split_group_count_a[row_idx] = float(groups_a.numel())
            split_group_count_b[row_idx] = float(groups_b.numel())
            if groups_a.numel() == 0 or groups_b.numel() == 0:
                split_group_overlap[row_idx] = 0.0
                continue
            overlap = torch.isin(groups_a, groups_b).sum().float()
            denom = float(max(min(int(groups_a.numel()), int(groups_b.numel())), 1))
            split_group_overlap[row_idx] = overlap / denom
    split_balanced_half = (torch.abs(split_count[:, 0] - split_count[:, 1]) <= 1).float()
    payload = {
        "mask": torch.stack([split_mask_a, split_mask_b], dim=1),
        "env_mean": torch.stack([env_mean_a, env_mean_b], dim=1),
        "env_mean_unit": torch.stack([stats_a["env_mean"], stats_b["env_mean"]], dim=1),
        "env_logvar": torch.stack([stats_a["env_logvar"], stats_b["env_logvar"]], dim=1),
        "view_spread": torch.stack([stats_a["view_spread"], stats_b["view_spread"]], dim=1),
        "factor_mean": torch.stack([stats_a["factor_mean"], stats_b["factor_mean"]], dim=1),
        "factor_std": torch.stack([stats_a["factor_std"], stats_b["factor_std"]], dim=1),
        "mechanics_posterior_mean": torch.stack(
            [stats_a["mechanics_posterior_mean"], stats_b["mechanics_posterior_mean"]],
            dim=1,
        ),
        "mechanics_posterior_std": torch.stack(
            [stats_a["mechanics_posterior_std"], stats_b["mechanics_posterior_std"]],
            dim=1,
        ),
        "mechanics_posterior_entropy": torch.stack(
            [stats_a["mechanics_posterior_entropy"], stats_b["mechanics_posterior_entropy"]],
            dim=1,
        ),
        "split_count": split_count,
        "split_group_overlap": split_group_overlap,
        "split_balanced_half": split_balanced_half,
        "split_group_count_a": split_group_count_a,
        "split_group_count_b": split_group_count_b,
    }
    if env_param_predictor is not None:
        env_param_mean_a = env_param_predictor.predict_all(env_mean_a).mean(dim=0)
        env_param_mean_b = env_param_predictor.predict_all(env_mean_b).mean(dim=0)
        payload["env_param_mean"] = torch.stack([env_param_mean_a, env_param_mean_b], dim=1)
        payload["env_param_disagreement"] = torch.linalg.norm(env_param_mean_a - env_param_mean_b, dim=-1)
    payload["latent_disagreement"] = torch.linalg.norm(env_mean_a - env_mean_b, dim=-1)
    return payload


def build_random_subset_masks(
    mask: torch.Tensor,
    subset_count: int,
    subset_size: int,
    group_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample several small view subsets for each env instance."""
    if subset_count <= 0:
        raise ValueError("subset_count must be positive")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive")

    subset_masks = torch.zeros((mask.shape[0], subset_count, mask.shape[1]), dtype=torch.float32, device=mask.device)
    for subset_idx in range(subset_count):
        subset_masks[:, subset_idx, :] = _sample_random_subset_mask(
            mask=mask,
            subset_size=subset_size,
            group_ids=group_ids,
            generator=generator,
        )
    return subset_masks


def _sample_random_subset_mask(
    mask: torch.Tensor,
    subset_size: int,
    group_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one diverse subset per row without reusing canonical support order."""
    mask_bool = mask.bool()
    subset_mask = torch.zeros_like(mask_bool, dtype=torch.float32)

    for row_idx in range(mask_bool.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        if valid_idx.numel() <= subset_size:
            subset_mask[row_idx, valid_idx] = 1.0
            continue

        chosen: list[int] = []
        if group_ids is not None:
            row_groups = group_ids[row_idx, valid_idx]
            valid_groups = torch.unique(row_groups[row_groups >= 0], sorted=False)
            if valid_groups.numel() > 0:
                shuffled_groups = valid_groups[
                    torch.randperm(valid_groups.numel(), generator=generator, device=valid_groups.device)
                ]
                for group_value in shuffled_groups.tolist():
                    group_candidates = valid_idx[row_groups == int(group_value)]
                    if group_candidates.numel() == 0:
                        continue
                    group_candidates = group_candidates[
                        torch.randperm(
                            group_candidates.numel(),
                            generator=generator,
                            device=group_candidates.device,
                        )
                    ]
                    chosen.append(int(group_candidates[0].item()))
                    if len(chosen) >= subset_size:
                        break

        if len(chosen) < subset_size:
            remaining = [int(idx) for idx in valid_idx.tolist() if int(idx) not in set(chosen)]
            if remaining:
                remaining_t = torch.tensor(remaining, dtype=torch.long, device=valid_idx.device)
                remaining_t = remaining_t[
                    torch.randperm(remaining_t.numel(), generator=generator, device=remaining_t.device)
                ]
                take = min(int(remaining_t.numel()), subset_size - len(chosen))
                chosen.extend(int(item) for item in remaining_t[:take].tolist())

        if chosen:
            subset_mask[row_idx, torch.tensor(chosen, dtype=torch.long, device=mask.device)] = 1.0

    return subset_mask


def build_leave_one_group_out_masks(
    mask: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build leave-one-group-out masks for each env when multiple groups exist."""
    if group_ids is None:
        empty = torch.zeros((mask.shape[0], 0, mask.shape[1]), dtype=torch.float32, device=mask.device)
        valid = torch.zeros((mask.shape[0], 0), dtype=torch.float32, device=mask.device)
        return empty, valid

    mask_bool = mask.bool()
    max_group_count = 0
    grouped_unique: list[torch.Tensor] = []
    for row_idx in range(mask.shape[0]):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            groups = torch.empty((0,), dtype=torch.long, device=mask.device)
        else:
            valid_groups = group_ids[row_idx, valid_idx]
            groups = torch.unique(valid_groups[valid_groups >= 0], sorted=True)
        grouped_unique.append(groups)
        max_group_count = max(max_group_count, int(groups.numel()))

    leave_masks = torch.zeros((mask.shape[0], max_group_count, mask.shape[1]), dtype=torch.float32, device=mask.device)
    valid_leaveouts = torch.zeros((mask.shape[0], max_group_count), dtype=torch.float32, device=mask.device)
    if max_group_count == 0:
        return leave_masks, valid_leaveouts

    for row_idx, groups in enumerate(grouped_unique):
        valid_idx = torch.nonzero(mask_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0 or groups.numel() <= 1:
            continue
        row_group_ids = group_ids[row_idx]
        for group_pos, group_value in enumerate(groups.tolist()):
            keep_mask = mask_bool[row_idx] & (row_group_ids != int(group_value))
            if torch.any(keep_mask):
                leave_masks[row_idx, group_pos, keep_mask] = 1.0
                valid_leaveouts[row_idx, group_pos] = 1.0

    return leave_masks, valid_leaveouts


def compute_support_group_stats(
    support_mask: torch.Tensor,
    group_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure how diverse the chosen support set is for each env row."""
    support_bool = support_mask.bool()
    support_count = support_bool.sum(dim=1).float().clamp_min(1.0)
    if group_ids is None:
        ones = torch.ones_like(support_count)
        return support_count.clone(), ones

    group_counts = []
    ratios = []
    for row_idx in range(support_bool.shape[0]):
        valid_idx = torch.nonzero(support_bool[row_idx], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            group_counts.append(0.0)
            ratios.append(0.0)
            continue
        row_groups = group_ids[row_idx, valid_idx]
        unique_groups = torch.unique(row_groups[row_groups >= 0], sorted=True)
        count = float(unique_groups.numel())
        group_counts.append(count)
        ratios.append(count / max(float(valid_idx.numel()), 1.0))

    return (
        torch.tensor(group_counts, dtype=torch.float32, device=support_mask.device),
        torch.tensor(ratios, dtype=torch.float32, device=support_mask.device),
    )


def sample_env_belief_subsets(
    aggregator: EnvBeliefAggregator,
    grouped_mean: torch.Tensor,
    grouped_logvar: torch.Tensor,
    grouped_mask: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    env_param_predictor: EnvParamPredictorEnsemble | None = None,
    subset_count: int = 4,
    subset_size: int = 6,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate several random few-view subsets for each env instance."""
    subset_masks = build_random_subset_masks(
        mask=grouped_mask,
        subset_count=subset_count,
        subset_size=subset_size,
        group_ids=group_ids,
        generator=generator,
    )
    batch_size, sampled_subset_count, max_views = subset_masks.shape
    latent_dim = grouped_mean.shape[-1]
    repeated_mean = grouped_mean[:, None, :, :].expand(-1, sampled_subset_count, -1, -1)
    repeated_logvar = grouped_logvar[:, None, :, :].expand(-1, sampled_subset_count, -1, -1)

    env_mean, env_logvar, env_view_spread = aggregator(
        repeated_mean.reshape(batch_size * sampled_subset_count, max_views, latent_dim),
        repeated_logvar.reshape(batch_size * sampled_subset_count, max_views, latent_dim),
        subset_masks.reshape(batch_size * sampled_subset_count, max_views),
        None if group_ids is None else group_ids[:, None, :].expand(-1, sampled_subset_count, -1).reshape(
            batch_size * sampled_subset_count,
            max_views,
        ),
    )

    payload = {
        "mask": subset_masks,
        "env_mean": env_mean.reshape(batch_size, sampled_subset_count, latent_dim),
        "env_logvar": env_logvar.reshape(batch_size, sampled_subset_count, latent_dim),
        "view_spread": env_view_spread.reshape(batch_size, sampled_subset_count, latent_dim),
    }
    if env_param_predictor is not None:
        env_param_preds = env_param_predictor.predict_all(env_mean)
        param_dim = env_param_preds.shape[-1]
        payload["env_param_mean"] = env_param_preds.mean(dim=0).reshape(batch_size, sampled_subset_count, param_dim)
        payload["env_param_std"] = env_param_preds.std(dim=0).reshape(batch_size, sampled_subset_count, param_dim)
    return payload
