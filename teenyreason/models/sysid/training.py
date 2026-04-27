"""Training loop for probe likelihood system identification."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from .likelihood import ProbeLikelihoodModel
from .probe_features import SysIdFeatureStats, build_probe_sysid_features


@dataclass(frozen=True)
class SysIdTrainingResult:
    """Trained likelihood model plus validation diagnostics."""

    model: ProbeLikelihoodModel
    stats: SysIdFeatureStats
    metrics: dict[str, float]
    trusted: bool


def _split_env_ids(env_instance_id: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(np.asarray(env_instance_id, dtype=np.int64))
    if ids.shape[0] <= 1:
        return ids, ids
    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(np.ceil(0.10 * float(shuffled.shape[0]))))
    val_ids = shuffled[:val_count]
    train_ids = shuffled[val_count:]
    if train_ids.shape[0] == 0:
        train_ids = shuffled[val_count - 1 :]
    return train_ids, val_ids


def _tensor(values: np.ndarray, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def _candidate_loglik(
    model: ProbeLikelihoodModel,
    query: torch.Tensor,
    outcome: torch.Tensor,
    family_ids: torch.Tensor,
    true_params: torch.Tensor,
    pool_params: torch.Tensor,
    negative_count: int,
    batch_env_ids: torch.Tensor | None = None,
    pool_env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = int(query.shape[0])
    neg_count = max(1, int(negative_count))
    if pool_params.shape[0] <= 0:
        pool_params = true_params
    if batch_env_ids is None or pool_env_ids is None:
        neg_idx = torch.randint(0, int(pool_params.shape[0]), (batch_size, neg_count), device=query.device)
    else:
        pool_env_ids = pool_env_ids.to(device=query.device).long().reshape(-1)
        batch_env_ids = batch_env_ids.to(device=query.device).long().reshape(-1)
        all_idx = torch.arange(int(pool_params.shape[0]), device=query.device)
        sampled_rows: list[torch.Tensor] = []
        for env_id in batch_env_ids:
            valid = all_idx[pool_env_ids != env_id]
            if int(valid.numel()) <= 0:
                valid = all_idx
            local_idx = torch.randint(0, int(valid.numel()), (neg_count,), device=query.device)
            sampled_rows.append(valid[local_idx])
        neg_idx = torch.stack(sampled_rows, dim=0)
    neg_params = pool_params[neg_idx]
    candidates = torch.cat([true_params.unsqueeze(1), neg_params], dim=1)
    flat_params = candidates.reshape(batch_size * (neg_count + 1), -1)
    flat_query = query.unsqueeze(1).expand(-1, neg_count + 1, -1).reshape(batch_size * (neg_count + 1), -1)
    flat_outcome = outcome.unsqueeze(1).expand(-1, neg_count + 1, -1).reshape(batch_size * (neg_count + 1), -1)
    flat_family = family_ids.unsqueeze(1).expand(-1, neg_count + 1).reshape(-1)
    return model.log_likelihood(flat_params, flat_query, flat_family, flat_outcome).reshape(batch_size, neg_count + 1)


def _evaluate(
    model: ProbeLikelihoodModel,
    query: torch.Tensor,
    outcome: torch.Tensor,
    family_ids: torch.Tensor,
    params: torch.Tensor,
    env_ids: torch.Tensor,
    negative_count: int,
    batch_size: int,
) -> dict[str, float | np.ndarray]:
    model.eval()
    nll_values: list[np.ndarray] = []
    top1_values: list[np.ndarray] = []
    margin_values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(query.shape[0]), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), int(query.shape[0]))
            batch_query = query[start:stop]
            batch_outcome = outcome[start:stop]
            batch_family = family_ids[start:stop]
            batch_params = params[start:stop]
            batch_env = env_ids[start:stop]
            true_loglik = model.log_likelihood(batch_params, batch_query, batch_family, batch_outcome)
            candidate_loglik = _candidate_loglik(
                model=model,
                query=batch_query,
                outcome=batch_outcome,
                family_ids=batch_family,
                true_params=batch_params,
                pool_params=params,
                negative_count=negative_count,
                batch_env_ids=batch_env,
                pool_env_ids=env_ids,
            )
            best_negative = torch.max(candidate_loglik[:, 1:], dim=1).values
            nll_values.append((-true_loglik).detach().cpu().numpy().astype(np.float32))
            top1_values.append((torch.argmax(candidate_loglik, dim=1) == 0).float().detach().cpu().numpy().astype(np.float32))
            margin_values.append((true_loglik - best_negative).detach().cpu().numpy().astype(np.float32))
    nll = np.concatenate(nll_values, axis=0) if nll_values else np.asarray([1.0], dtype=np.float32)
    top1 = np.concatenate(top1_values, axis=0) if top1_values else np.asarray([0.0], dtype=np.float32)
    margin = np.concatenate(margin_values, axis=0) if margin_values else np.asarray([0.0], dtype=np.float32)
    return {
        "nll": float(np.mean(nll)),
        "top1": float(np.mean(top1)),
        "margin": float(np.mean(margin)),
        "margin_median": float(np.median(margin)),
        "margin_positive_frac": float(np.mean(margin > 0.0)),
        "nll_values": nll,
    }


def train_probe_likelihood_model(
    windows: dict[str, np.ndarray],
    action_vocab_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    negative_count: int,
    hidden_dim: int = 128,
    seed: int = 0,
) -> SysIdTrainingResult:
    """Train a likelihood model that scores candidate mechanics for probe windows."""
    env_instance_id = np.asarray(windows["env_instance_id"], dtype=np.int64)
    train_ids, val_ids = _split_env_ids(env_instance_id, seed=seed)
    train_mask = np.isin(env_instance_id, train_ids)
    val_mask = np.isin(env_instance_id, val_ids)
    if not np.any(train_mask):
        train_mask = np.ones_like(env_instance_id, dtype=bool)
    if not np.any(val_mask):
        val_mask = train_mask.copy()
    features = build_probe_sysid_features(
        windows,
        action_vocab_size=action_vocab_size,
        fit_mask=train_mask,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed))
    model = ProbeLikelihoodModel(
        param_dim=features.env_params_norm.shape[1],
        query_dim=features.query_features.shape[1],
        outcome_dim=features.outcome_features.shape[1],
        num_families=len(features.stats.family_names),
        hidden_dim=hidden_dim,
    ).to(device)

    train_query = _tensor(features.query_features[train_mask], device)
    train_outcome = _tensor(features.outcome_features[train_mask], device)
    train_params = _tensor(features.env_params_norm[train_mask], device)
    train_family = _tensor(features.family_ids[train_mask], device, dtype=torch.long)
    train_env = _tensor(features.env_instance_id[train_mask], device, dtype=torch.long)
    val_query = _tensor(features.query_features[val_mask], device)
    val_outcome = _tensor(features.outcome_features[val_mask], device)
    val_params = _tensor(features.env_params_norm[val_mask], device)
    val_family = _tensor(features.family_ids[val_mask], device, dtype=torch.long)
    val_env = _tensor(features.env_instance_id[val_mask], device, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr), eps=1e-5)
    row_count = int(train_query.shape[0])
    for _epoch in range(max(0, int(epochs))):
        model.train()
        order = torch.randperm(row_count, device=device)
        for start in range(0, row_count, max(1, int(batch_size))):
            idx = order[start : start + max(1, int(batch_size))]
            query = train_query[idx]
            outcome = train_outcome[idx]
            params = train_params[idx]
            family_ids = train_family[idx]
            batch_env = train_env[idx]
            true_loglik = model.log_likelihood(params, query, family_ids, outcome)
            candidate_loglik = _candidate_loglik(
                model=model,
                query=query,
                outcome=outcome,
                family_ids=family_ids,
                true_params=params,
                pool_params=train_params,
                negative_count=negative_count,
                batch_env_ids=batch_env,
                pool_env_ids=train_env,
            )
            labels = torch.zeros((candidate_loglik.shape[0],), dtype=torch.long, device=device)
            nll_loss = -true_loglik.mean()
            rank_loss = F.cross_entropy(candidate_loglik / 0.25, labels)
            loss = nll_loss + 0.35 * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    train_eval = _evaluate(
        model=model,
        query=train_query,
        outcome=train_outcome,
        family_ids=train_family,
        params=train_params,
        env_ids=train_env,
        negative_count=negative_count,
        batch_size=batch_size,
    )
    val_eval = _evaluate(
        model=model,
        query=val_query,
        outcome=val_outcome,
        family_ids=val_family,
        params=val_params,
        env_ids=val_env,
        negative_count=negative_count,
        batch_size=batch_size,
    )
    nll_values = np.asarray(val_eval["nll_values"], dtype=np.float32)
    good_nll = float(np.quantile(nll_values, 0.25)) if nll_values.size else float(val_eval["nll"])
    bad_nll = float(np.quantile(nll_values, 0.75)) if nll_values.size else float(val_eval["nll"] + 1.0)
    if bad_nll <= good_nll + 1e-3:
        bad_nll = good_nll + 1.0
    stats = replace(
        features.stats,
        validation_nll_good=good_nll,
        validation_nll_bad=bad_nll,
    )
    metrics = {
        "train_nll": float(train_eval["nll"]),
        "validation_nll": float(val_eval["nll"]),
        "validation_top1": float(val_eval["top1"]),
        "validation_margin": float(val_eval["margin"]),
        "validation_margin_median": float(val_eval["margin_median"]),
        "validation_margin_positive_frac": float(val_eval["margin_positive_frac"]),
        "validation_nll_good": good_nll,
        "validation_nll_bad": bad_nll,
    }
    trusted = bool(metrics["validation_top1"] >= 0.45 and metrics["validation_margin"] >= 0.25)
    model.eval()
    return SysIdTrainingResult(model=model, stats=stats, metrics=metrics, trusted=trusted)
