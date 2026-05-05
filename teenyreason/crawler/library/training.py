"""Training entrypoint for the RL-facing crawler library."""

from __future__ import annotations

import numpy as np

from ...models.sysid import build_latin_hypercube_particles, train_probe_likelihood_model
from ...cognition.representation import train_encoder_predictor
from .bundle import CrawlerModelBundle
from .helpers import sanitize_array

def train_crawler_library(
    *,
    windows,
    z_dim: int,
    window_size: int,
    action_vocab_size: int,
    belief_mode: str = "latent_pool",
    sysid_epochs: int = 0,
    sysid_batch_size: int = 256,
    sysid_lr: float = 3e-4,
    sysid_negative_count: int = 15,
    sysid_particle_count: int = 128,
    sysid_likelihood_scale: float = 0.35,
    progress_callback=None,
    **train_kwargs,
) -> CrawlerModelBundle:
    """Train the crawler-side representation stack and return it as one bundle."""
    (
        encoder,
        predictor,
        belief_aggregator,
        env_param_predictor,
        env_future_predictor,
        env_family_future_predictor,
        family_value_predictor,
        env_metric_projector,
        belief_message_projector,
        controller_context_projector,
        oracle_context_projector,
        controller_trust_predictor,
        env_param_normalizer,
        device,
    ) = train_encoder_predictor(
        windows=windows,
        z_dim=z_dim,
        action_vocab_size=action_vocab_size,
        progress_callback=progress_callback,
        **train_kwargs,
    )
    family_names = tuple(sorted({str(mode) for mode in np.asarray(windows["probe_mode"], dtype="U").tolist()}))
    sysid_model = None
    sysid_stats = None
    sysid_particles_raw = None
    sysid_trusted = False
    sysid_validation_metrics: dict[str, float] | None = None
    if str(belief_mode) == "particle_sysid" and int(sysid_epochs) > 0:
        sysid_result = train_probe_likelihood_model(
            windows=windows,
            action_vocab_size=action_vocab_size,
            epochs=int(sysid_epochs),
            batch_size=int(sysid_batch_size),
            lr=float(sysid_lr),
            negative_count=int(sysid_negative_count),
            hidden_dim=128,
            seed=0,
        )
        sysid_model = sysid_result.model
        sysid_stats = sysid_result.stats
        sysid_particles_raw = build_latin_hypercube_particles(
            sysid_stats,
            count=int(sysid_particle_count),
            seed=173,
        )
        sysid_trusted = bool(sysid_result.trusted)
        sysid_validation_metrics = dict(sysid_result.metrics)
        print(
            "sysid validation | "
            f"trusted={sysid_trusted} | "
            f"top1={sysid_validation_metrics.get('validation_top1', 0.0):.3f} | "
            f"margin={sysid_validation_metrics.get('validation_margin', 0.0):.3f} | "
            f"nll={sysid_validation_metrics.get('validation_nll', 0.0):.3f}"
        )

    belief_message_dim = (
        int(belief_message_projector.output_dim)
        if belief_message_projector is not None and hasattr(belief_message_projector, "output_dim")
        else int(z_dim)
    )
    controller_context_dim = int(2 * z_dim + 2)
    return CrawlerModelBundle(
        encoder=encoder,
        predictor=predictor,
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        env_future_predictor=env_future_predictor,
        env_family_future_predictor=env_family_future_predictor,
        family_value_predictor=family_value_predictor,
        env_metric_projector=env_metric_projector,
        belief_message_projector=belief_message_projector,
        controller_context_projector=controller_context_projector,
        oracle_context_projector=oracle_context_projector,
        controller_trust_predictor=controller_trust_predictor,
        device=device,
        z_dim=z_dim,
        window_size=window_size,
        action_vocab_size=action_vocab_size,
        belief_message_dim=belief_message_dim,
        controller_context_dim=controller_context_dim,
        family_names=family_names,
        env_param_normalizer_mean=sanitize_array(env_param_normalizer["mean"]),
        env_param_normalizer_std=sanitize_array(env_param_normalizer["std"]),
        belief_mode=str(belief_mode),
        sysid_model=sysid_model,
        sysid_stats=sysid_stats,
        sysid_particles_raw=None if sysid_particles_raw is None else sanitize_array(sysid_particles_raw),
        sysid_trusted=sysid_trusted,
        sysid_validation_metrics=sysid_validation_metrics,
        sysid_likelihood_scale=float(sysid_likelihood_scale),
    )
