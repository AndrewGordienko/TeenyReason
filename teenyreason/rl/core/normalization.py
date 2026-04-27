"""Running normalization used by PPO collectors and evaluators.

The normalizer is intentionally tiny and serializable: benchmark reporting
snapshots its raw mean/variance/count fields so policy checkpoints can be
evaluated with the same scale they saw during training.
"""

import numpy as np


class RunningNormalizer:
    """Online mean/variance tracker for observations or rewards."""
    def __init__(self, shape, clip: float = 5.0, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.clip = float(clip)

    def update(self, values):
        values_np = np.asarray(values, dtype=np.float64)
        if values_np.ndim == 1:
            values_np = values_np[None, :]
        if values_np.shape[0] == 0:
            return

        # Update running moments online so rollouts can be normalized incrementally.
        batch_mean = values_np.mean(axis=0)
        batch_var = values_np.var(axis=0)
        batch_count = float(values_np.shape[0])

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        correction = np.square(delta) * self.count * batch_count / total_count
        new_var = (mean_a + mean_b + correction) / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count

    def normalize(self, values):
        values_np = np.asarray(values, dtype=np.float32)
        normalized = (values_np - self.mean.astype(np.float32)) / np.sqrt(
            self.var.astype(np.float32) + 1e-8
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def scale_only(self, values):
        values_np = np.asarray(values, dtype=np.float32)
        scaled = values_np / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)
