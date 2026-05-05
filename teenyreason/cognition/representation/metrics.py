"""Shared representation metrics and compression helpers."""

from __future__ import annotations

import numpy as np


def project_latents_2d(latent_means: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project latents to 2D with a tiny PCA implementation."""
    centered = latent_means.astype(np.float32) - latent_means.mean(axis=0, keepdims=True).astype(np.float32)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    try:
        _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].astype(np.float32)
        projection = centered @ components.T
        explained = np.square(singular_values[:2]) / max(np.square(singular_values).sum(), 1e-6)
    except np.linalg.LinAlgError:
        components = np.zeros((2, centered.shape[1]), dtype=np.float32)
        if centered.shape[1] >= 1:
            components[0, 0] = 1.0
        if centered.shape[1] >= 2:
            components[1, 1] = 1.0
        projection = centered[:, :2] if centered.shape[1] >= 2 else np.pad(
            centered,
            ((0, 0), (0, max(0, 2 - centered.shape[1]))),
        )
        projection = projection.astype(np.float32)
        explained = np.zeros((2,), dtype=np.float32)
    return projection.astype(np.float32), components, explained.astype(np.float32)


def compute_linear_env_fit(latent_mean: np.ndarray, env_params: np.ndarray) -> float:
    """Fit env params from latent coordinates as a quick mechanics-alignment score."""
    if latent_mean.shape[0] < 4 or env_params.size == 0:
        return 0.0
    design = np.concatenate(
        [
            latent_mean.astype(np.float32),
            np.ones((latent_mean.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    coef, _residuals, _rank, _singular_values = np.linalg.lstsq(design, env_params, rcond=None)
    prediction = design @ coef
    ss_res = np.sum(np.square(env_params - prediction), axis=0)
    ss_tot = np.sum(np.square(env_params - env_params.mean(axis=0, keepdims=True)), axis=0)
    valid = ss_tot > 1e-6
    if not np.any(valid):
        return 0.0
    return float(np.mean(1.0 - ss_res[valid] / ss_tot[valid]))


def compute_split_retrieval_stats(
    split_mean_a: np.ndarray,
    split_mean_b: np.ndarray,
) -> tuple[float, float]:
    """Return top-1 and MRR retrieval stats for disjoint support halves."""
    if split_mean_a.shape[0] < 2 or split_mean_b.shape[0] != split_mean_a.shape[0]:
        return 0.0, 0.0
    a = split_mean_a.astype(np.float32)
    b = split_mean_b.astype(np.float32)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-6, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-6, None)
    similarity = a @ b.T
    labels = np.arange(a.shape[0], dtype=np.int32)
    ranked_idx = np.argsort(-similarity, axis=1)
    rank_positions = np.argmax(ranked_idx == labels[:, None], axis=1) + 1
    top1 = float(np.mean(ranked_idx[:, 0] == labels))
    mrr = float(np.mean(1.0 / rank_positions))
    return top1, mrr


def deterministic_rotation(dim: int, seed: int = 17) -> np.ndarray:
    """Build one deterministic rotation matrix for message compression studies."""
    if dim <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    rng = np.random.default_rng(seed + dim * 997)
    gaussian = rng.normal(size=(dim, dim)).astype(np.float32)
    q, r = np.linalg.qr(gaussian)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return (q * signs[None, :]).astype(np.float32)


def quantize_vector(values: np.ndarray, bits_per_dim: int) -> np.ndarray:
    """Quantize one vector at a fixed scalar bitrate."""
    vector = np.nan_to_num(
        np.asarray(values, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).reshape(-1)
    if bits_per_dim <= 0 or vector.size == 0:
        return vector
    rotation = deterministic_rotation(vector.size, seed=bits_per_dim)
    rotated = rotation @ vector
    levels = max(2, 2 ** int(bits_per_dim))
    scale = float(np.max(np.abs(rotated)))
    if not np.isfinite(scale) or scale <= 1e-6:
        return vector
    normalized = np.clip(rotated / scale, -1.0, 1.0)
    grid = np.linspace(-1.0, 1.0, num=levels, dtype=np.float32)
    quantized_idx = np.abs(normalized[:, None] - grid[None, :]).argmin(axis=1)
    quantized_rotated = grid[quantized_idx] * scale
    return np.nan_to_num(
        rotation.T @ quantized_rotated,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)
