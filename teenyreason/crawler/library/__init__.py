"""RL-facing crawler library facade.

This package contains the trained crawler bundle used by the legacy PPO and
probe-policy stack. The generic crawler contract lives one level up in
``teenyreason.crawler.core``.
"""

from .bundle import CrawlerModelBundle, aggregate_env_belief, encode_window_posterior
from .checkpoint import load_crawler_bundle_from_checkpoint
from .helpers import (
    belief_source_from_mode,
    build_evidence_batch,
    estimate_probe_family_cost,
    mean_pairwise_distance,
    quantize_vector,
    sanitize_array,
)
from .training import train_crawler_library

__all__ = [
    "CrawlerModelBundle",
    "aggregate_env_belief",
    "belief_source_from_mode",
    "build_evidence_batch",
    "encode_window_posterior",
    "estimate_probe_family_cost",
    "load_crawler_bundle_from_checkpoint",
    "mean_pairwise_distance",
    "quantize_vector",
    "sanitize_array",
    "train_crawler_library",
]
