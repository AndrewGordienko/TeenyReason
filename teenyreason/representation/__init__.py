"""Representation-learning entrypoints.

This package exists so the repo can talk about the latent environment-belief
system as one coherent subsystem instead of scattering the idea across
`models/` and `probe/`.
"""

from ..models.belief_world_model import (
    DeltaPredictorEnsemble,
    WorldEncoder,
    train_encoder_predictor,
)
from .analysis import (
    build_latent_snapshot,
    list_latent_snapshot_paths,
    load_latent_snapshot,
    save_latent_snapshot,
)

__all__ = [
    "DeltaPredictorEnsemble",
    "WorldEncoder",
    "build_latent_snapshot",
    "list_latent_snapshot_paths",
    "load_latent_snapshot",
    "save_latent_snapshot",
    "train_encoder_predictor",
]
