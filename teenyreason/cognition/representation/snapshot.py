"""Snapshot construction and persistence helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...models.belief import ContrastiveProjector, WorldEncoder
from ...models.envbelief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ...crawler.probes.data import get_env_param_names
from .metrics import (
    compute_linear_env_fit,
    compute_split_retrieval_stats,
    quantize_vector,
)
from .window_encoding import encode_window_dataset


def compute_message_rate_distortion(
    *,
    env_mean: np.ndarray,
    env_split_mean_a: np.ndarray,
    env_split_mean_b: np.ndarray,
    env_uncertainty: np.ndarray,
    env_params: np.ndarray,
    belief_message_projector,
    device: torch.device,
    compression_bits: tuple[int, ...] = (0, 8, 4, 2),
) -> dict[str, np.ndarray]:
    """Evaluate how much predictive signal survives the belief message bottleneck."""
    env_mean_np = np.asarray(env_mean, dtype=np.float32)
    split_a_np = np.asarray(env_split_mean_a, dtype=np.float32)
    split_b_np = np.asarray(env_split_mean_b, dtype=np.float32)
    uncertainty_np = np.asarray(env_uncertainty, dtype=np.float32).reshape(-1, 1)
    with torch.no_grad():
        if belief_message_projector is not None:
            belief_message_projector.eval()
            message_t = belief_message_projector(
                torch.tensor(env_mean_np, dtype=torch.float32, device=device),
                torch.tensor(uncertainty_np, dtype=torch.float32, device=device),
            )
            split_message_a_t = belief_message_projector(
                torch.tensor(split_a_np, dtype=torch.float32, device=device),
                torch.tensor(uncertainty_np, dtype=torch.float32, device=device),
            )
            split_message_b_t = belief_message_projector(
                torch.tensor(split_b_np, dtype=torch.float32, device=device),
                torch.tensor(uncertainty_np, dtype=torch.float32, device=device),
            )
            message = np.asarray(message_t.cpu().numpy(), dtype=np.float32)
            split_message_a = np.asarray(split_message_a_t.cpu().numpy(), dtype=np.float32)
            split_message_b = np.asarray(split_message_b_t.cpu().numpy(), dtype=np.float32)
        else:
            message = env_mean_np
            split_message_a = split_a_np
            split_message_b = split_b_np

    mechanics_fit = []
    split_top1 = []
    split_mrr = []
    message_norm_mean = []
    for bits in compression_bits:
        quantized_message = np.stack([quantize_vector(row, int(bits)) for row in message], axis=0).astype(np.float32)
        quantized_split_a = np.stack([quantize_vector(row, int(bits)) for row in split_message_a], axis=0).astype(np.float32)
        quantized_split_b = np.stack([quantize_vector(row, int(bits)) for row in split_message_b], axis=0).astype(np.float32)
        top1, mrr = compute_split_retrieval_stats(quantized_split_a, quantized_split_b)
        mechanics_fit.append(compute_linear_env_fit(quantized_message, env_params))
        split_top1.append(top1)
        split_mrr.append(mrr)
        message_norm_mean.append(float(np.mean(np.linalg.norm(quantized_message, axis=1))))
    return {
        "compression_bits": np.asarray(compression_bits, dtype=np.int32),
        "compression_mechanics_fit_r2": np.asarray(mechanics_fit, dtype=np.float32),
        "compression_split_retrieval_top1": np.asarray(split_top1, dtype=np.float32),
        "compression_split_retrieval_mrr": np.asarray(split_mrr, dtype=np.float32),
        "compression_message_norm_mean": np.asarray(message_norm_mean, dtype=np.float32),
    }


def build_latent_snapshot(
    encoder: WorldEncoder,
    belief_aggregator: EnvBeliefAggregator,
    env_param_predictor: EnvParamPredictorEnsemble,
    env_future_predictor,
    env_metric_projector: ContrastiveProjector | None,
    device: torch.device,
    windows: dict[str, np.ndarray],
    env_name: str | None = None,
    benchmark_tag: str | None = None,
    support_size: int = 4,
    subset_count: int = 1,
    belief_message_projector=None,
    compression_bits: tuple[int, ...] = (0, 8, 4, 2),
    crawler_bundle=None,
) -> dict[str, np.ndarray]:
    """Build one dashboard-friendly env-belief snapshot from recorded probe windows."""
    from .analysis import build_env_belief_dataset

    window_mean, window_logvar = encode_window_dataset(encoder=encoder, device=device, windows=windows)
    env_dataset = build_env_belief_dataset(
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        env_future_predictor=env_future_predictor,
        env_metric_projector=env_metric_projector,
        device=device,
        window_mean=window_mean,
        window_logvar=window_logvar,
        windows=windows,
        env_name=env_name,
        support_size=support_size,
        subset_count=subset_count,
    )
    reward_sum = np.sum(windows["rewards"], axis=1, dtype=np.float32)

    snapshot = {
        "window_env_instance_id": windows["env_instance_id"].astype(np.int32),
        "window_episode_id": windows["episode_id"].astype(np.int32),
        "window_end_step_idx": windows["end_step_idx"].astype(np.int32),
        "window_probe_mode": windows["probe_mode"].astype("U"),
        "window_reward_sum": reward_sum.astype(np.float32),
        "window_terminated": windows["terminated"].astype(np.int8),
        "window_truncated": windows["truncated"].astype(np.int8),
        "window_latent_mean": window_mean.astype(np.float32),
        "window_latent_logvar": window_logvar.astype(np.float32),
        "env_param_names": np.asarray(
            get_env_param_names(env_name, env_dataset["env_params"].shape[1]),
            dtype="U",
        ),
        **env_dataset,
    }
    snapshot.update(
        compute_message_rate_distortion(
            env_mean=env_dataset["env_belief_mean"],
            env_split_mean_a=env_dataset["env_split_mean_a"],
            env_split_mean_b=env_dataset["env_split_mean_b"],
            env_uncertainty=env_dataset["env_uncertainty"],
            env_params=env_dataset["env_params"],
            belief_message_projector=belief_message_projector,
            device=device,
            compression_bits=compression_bits,
        )
    )
    if crawler_bundle is not None and str(getattr(crawler_bundle, "belief_mode", "latent_pool")) == "particle_sysid":
        particle_mean = []
        particle_std = []
        particle_entropy = []
        particle_entropy_norm = []
        particle_ess = []
        particle_leaveout = []
        particle_subset = []
        env_ids = env_dataset["env_instance_id"].astype(np.int32)
        window_env_ids = windows["env_instance_id"].astype(np.int32)
        for env_id in env_ids.tolist():
            env_indices = np.nonzero(window_env_ids == int(env_id))[0][: max(1, int(support_size))]
            records = [
                {
                    "states": windows["states"][idx],
                    "actions": windows["actions"][idx],
                    "rewards": windows["rewards"][idx],
                    "terminated": bool(windows["terminated"][idx]),
                    "truncated": bool(windows["truncated"][idx]),
                    "probe_family": str(np.asarray(windows["probe_mode"][idx]).item()),
                }
                for idx in env_indices.tolist()
            ]
            if not records:
                continue
            _belief, payload = crawler_bundle.build_particle_env_belief(records)
            particle_mean.append(payload["particle_param_mean"])
            particle_std.append(payload["particle_param_std"])
            particle_entropy.append(float(payload["particle_entropy"].reshape(-1)[0]))
            particle_entropy_norm.append(float(payload["particle_entropy_norm"].reshape(-1)[0]))
            particle_ess.append(float(payload["particle_ess_ratio"].reshape(-1)[0]))
            particle_leaveout.append(float(payload["particle_leaveout_shift"].reshape(-1)[0]))
            particle_subset.append(float(payload["particle_subset_stability"].reshape(-1)[0]))
        if particle_mean:
            snapshot["particle_param_mean"] = np.stack(particle_mean, axis=0).astype(np.float32)
            snapshot["particle_param_std"] = np.stack(particle_std, axis=0).astype(np.float32)
            snapshot["particle_entropy"] = np.asarray(particle_entropy, dtype=np.float32)
            snapshot["particle_entropy_norm"] = np.asarray(particle_entropy_norm, dtype=np.float32)
            snapshot["particle_ess_ratio"] = np.asarray(particle_ess, dtype=np.float32)
            snapshot["particle_leaveout_shift"] = np.asarray(particle_leaveout, dtype=np.float32)
            snapshot["particle_subset_stability"] = np.asarray(particle_subset, dtype=np.float32)
        metrics = getattr(crawler_bundle, "sysid_validation_metrics", {}) or {}
        snapshot["sysid_validation_top1"] = np.asarray([float(metrics.get("validation_top1", 0.0))], dtype=np.float32)
        snapshot["sysid_validation_margin"] = np.asarray([float(metrics.get("validation_margin", 0.0))], dtype=np.float32)
        snapshot["sysid_validation_nll"] = np.asarray([float(metrics.get("validation_nll", 0.0))], dtype=np.float32)
        snapshot["sysid_trusted"] = np.asarray([1.0 if bool(getattr(crawler_bundle, "sysid_trusted", False)) else 0.0], dtype=np.float32)
        snapshot["belief_mode"] = np.asarray("particle_sysid")
        snapshot["belief_source"] = np.asarray("sysid")
    elif crawler_bundle is not None:
        snapshot["belief_source"] = np.asarray("learned")
    if env_name is not None:
        snapshot["env_name"] = np.asarray(env_name)
    if benchmark_tag is not None:
        snapshot["benchmark_tag"] = np.asarray(benchmark_tag)
    return snapshot


def save_latent_snapshot(path: str | Path, snapshot: dict[str, np.ndarray]) -> None:
    """Persist one latent snapshot artifact to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **snapshot)


def load_latent_snapshot(path: str | Path) -> dict[str, np.ndarray]:
    """Load a saved latent snapshot artifact."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def list_latent_snapshot_paths(artifact_dir: str | Path) -> list[Path]:
    """Find all saved latent snapshot artifacts in the artifact directory."""
    artifact_dir = Path(artifact_dir)
    return sorted(artifact_dir.glob("*_latent_snapshot.npz"))
