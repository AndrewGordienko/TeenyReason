"""Library-facing crawler bundle and training/loading helpers."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..models.belief_world_model import ContrastiveProjector, OutcomePredictor
from ..models.env_belief import EnvBeliefAggregator, EnvParamPredictorEnsemble
from ..probe.probe_latent import aggregate_env_belief, encode_window_posterior
from ..representation import DeltaPredictorEnsemble, WorldEncoder, train_encoder_predictor


@dataclass
class CrawlerModelBundle:
    """All crawler-side models needed to build env beliefs from probe evidence."""

    encoder: WorldEncoder
    predictor: DeltaPredictorEnsemble
    belief_aggregator: EnvBeliefAggregator
    env_param_predictor: EnvParamPredictorEnsemble
    env_future_predictor: nn.Module | None
    env_metric_projector: nn.Module | None
    device: torch.device
    z_dim: int
    window_size: int
    action_vocab_size: int

    def encode_probe_window(
        self,
        window_states,
        window_actions,
        window_rewards=None,
    ):
        """Return the posterior mean/logvar for one probe window."""
        return encode_window_posterior(
            encoder=self.encoder,
            device=self.device,
            window_states=window_states,
            window_actions=window_actions,
            window_rewards=window_rewards,
        )

    def build_env_belief(
        self,
        posterior_views,
    ):
        """Aggregate a set of probe-window posteriors into one env belief."""
        return aggregate_env_belief(
            belief_aggregator=self.belief_aggregator,
            env_param_predictor=self.env_param_predictor,
            device=self.device,
            posterior_views=posterior_views,
        )


def train_crawler_library(
    *,
    windows,
    z_dim: int,
    window_size: int,
    action_vocab_size: int,
    **train_kwargs,
) -> CrawlerModelBundle:
    """Train the crawler-side representation stack and return it as one bundle."""
    (
        encoder,
        predictor,
        belief_aggregator,
        env_param_predictor,
        env_future_predictor,
        env_metric_projector,
        device,
    ) = train_encoder_predictor(
        windows=windows,
        z_dim=z_dim,
        action_vocab_size=action_vocab_size,
        **train_kwargs,
    )
    return CrawlerModelBundle(
        encoder=encoder,
        predictor=predictor,
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        env_future_predictor=env_future_predictor,
        env_metric_projector=env_metric_projector,
        device=device,
        z_dim=z_dim,
        window_size=window_size,
        action_vocab_size=action_vocab_size,
    )


def load_crawler_bundle_from_checkpoint(
    *,
    checkpoint: dict,
    state_dim: int,
    action_vocab_size: int,
    device: torch.device,
) -> CrawlerModelBundle:
    """Rebuild a saved crawler bundle from one probe-policy checkpoint."""
    window_size = int(checkpoint["window_size"])
    z_dim = int(checkpoint["z_dim"])
    encoder = WorldEncoder(
        state_dim=state_dim,
        window_size=window_size,
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    predictor_ensemble_size = int(checkpoint.get("predictor_ensemble_size", 0))
    predictor = DeltaPredictorEnsemble(
        ensemble_size=predictor_ensemble_size,
        state_dim=state_dim,
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    predictor.eval()

    belief_aggregator = EnvBeliefAggregator(window_z_dim=z_dim).to(device)
    belief_aggregator.load_state_dict(checkpoint["belief_aggregator_state_dict"])
    belief_aggregator.eval()

    env_param_predictor = EnvParamPredictorEnsemble(
        ensemble_size=int(checkpoint["env_param_predictor_ensemble_size"]),
        input_dim=z_dim,
        output_dim=int(checkpoint["env_param_dim"]),
    ).to(device)
    env_param_predictor.load_state_dict(checkpoint["env_param_predictor_state_dict"])
    env_param_predictor.eval()

    env_future_predictor = None
    if checkpoint.get("env_future_predictor_state_dict") is not None:
        env_future_predictor = OutcomePredictor(
            input_dim=z_dim,
            output_dim=int(checkpoint["env_future_summary_dim"]),
        ).to(device)
        env_future_predictor.load_state_dict(checkpoint["env_future_predictor_state_dict"])
        env_future_predictor.eval()

    env_metric_projector = None
    if checkpoint.get("env_metric_projector_state_dict") is not None:
        env_metric_projector = ContrastiveProjector(
            input_dim=z_dim,
            output_dim=int(checkpoint.get("env_metric_dim", z_dim)),
        ).to(device)
        env_metric_projector.load_state_dict(checkpoint["env_metric_projector_state_dict"])
        env_metric_projector.eval()

    return CrawlerModelBundle(
        encoder=encoder,
        predictor=predictor,
        belief_aggregator=belief_aggregator,
        env_param_predictor=env_param_predictor,
        env_future_predictor=env_future_predictor,
        env_metric_projector=env_metric_projector,
        device=device,
        z_dim=z_dim,
        window_size=window_size,
        action_vocab_size=action_vocab_size,
    )
