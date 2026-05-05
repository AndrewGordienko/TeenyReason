"""Controller-context transforms shared by probe evaluation and ablations."""

import numpy as np
import torch
import torch.nn as nn

from ....core import BeliefNativeActorCritic, sanitize_numpy


def transform_controller_context_input(
    context_input: np.ndarray,
    *,
    disable_controller_context: bool = False,
    shuffle_controller_context: bool = False,
    rng: np.random.Generator | None = None,
    stale_context_input: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one evaluation ablation to the full-system controller context."""
    base_context = sanitize_numpy(np.asarray(context_input, dtype=np.float32).reshape(-1))
    if stale_context_input is not None:
        return sanitize_numpy(np.asarray(stale_context_input, dtype=np.float32).reshape(-1))
    if disable_controller_context:
        return np.zeros_like(base_context, dtype=np.float32)
    if not shuffle_controller_context or base_context.size <= 2:
        return base_context
    rng = np.random.default_rng(0) if rng is None else rng
    shuffled = base_context.copy()
    code = shuffled[:-2].copy()
    rng.shuffle(code)
    shuffled[:-2] = code
    return sanitize_numpy(shuffled)


def policy_action_value_step(
    *,
    policy: nn.Module,
    state_t: torch.Tensor,
    context_t: torch.Tensor,
    hidden_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Run one policy step for either the matched or belief-native controller."""
    if isinstance(policy, BeliefNativeActorCritic):
        mean, value, next_hidden, aux = policy.forward_with_hidden(
            state_t,
            context_t,
            hidden_state=hidden_state,
        )
        return mean, value, next_hidden, aux
    mean, value = policy(state_t, context_t)
    return mean, value, hidden_state, {}
