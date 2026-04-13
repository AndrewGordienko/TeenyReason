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
from ..models.env_belief import (
    EnvBeliefAggregator,
    EnvParamPredictorEnsemble,
    aggregate_env_posteriors,
    build_env_group_tensors,
)
from .analysis import (
    build_latent_snapshot,
    list_latent_snapshot_paths,
    load_latent_snapshot,
    save_latent_snapshot,
)

__all__ = [
    "DeltaPredictorEnsemble",
    "EnvBeliefAggregator",
    "EnvParamPredictorEnsemble",
    "WorldEncoder",
    "aggregate_env_posteriors",
    "build_latent_snapshot",
    "build_env_group_tensors",
    "list_latent_snapshot_paths",
    "load_latent_snapshot",
    "save_latent_snapshot",
    "train_encoder_predictor",
]
