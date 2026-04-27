"""Public PPO core surface.

Training code should import from `teenyreason.rl.core` or one of the concrete
modules below. The old monolithic `ppo_core.py` was split so data containers,
numerics, models, rollout packing, and optimizer mechanics each have one place
to debug.
"""

from .batches import (
    build_episode_batch,
    concat_episode_batches,
    corrupt_controller_context_sequences,
    prepare_recurrent_minibatch,
)
from .models import (
    BeliefNativeActorCritic,
    MatchedBeliefActorCritic,
    PlainGaussianActorCritic,
    ProbeConditionedGaussianActorCritic,
)
from .normalization import RunningNormalizer
from .numerics import (
    action_scale_bias,
    atanh,
    build_tanh_normal,
    compute_gae,
    evaluate_continuous_actions,
    evaluate_continuous_actions_with_scale_bias,
    init_linear,
    mean_to_continuous_action,
    sample_continuous_action,
    sanitize_numpy,
    sanitize_tensor,
    validate_continuous_env,
)
from .optim import set_optimizer_lr, update_ppo_policy
from .types import EpisodeBatch

__all__ = [
    "BeliefNativeActorCritic",
    "EpisodeBatch",
    "MatchedBeliefActorCritic",
    "PlainGaussianActorCritic",
    "ProbeConditionedGaussianActorCritic",
    "RunningNormalizer",
    "action_scale_bias",
    "atanh",
    "build_episode_batch",
    "build_tanh_normal",
    "compute_gae",
    "concat_episode_batches",
    "corrupt_controller_context_sequences",
    "evaluate_continuous_actions",
    "evaluate_continuous_actions_with_scale_bias",
    "init_linear",
    "mean_to_continuous_action",
    "prepare_recurrent_minibatch",
    "sample_continuous_action",
    "sanitize_numpy",
    "sanitize_tensor",
    "set_optimizer_lr",
    "update_ppo_policy",
    "validate_continuous_env",
]
