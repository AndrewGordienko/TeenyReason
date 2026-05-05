"""Training loop orchestration for the belief world model stack."""

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim

from ..core.components import (
    BeliefMessageProjector,
    ContrastiveProjector,
    ControllerBeliefContextProjector,
    ControllerTrustPredictor,
    DeltaPredictorEnsemble,
    FamilyConditionedOutcomePredictor,
    FamilyConditionedValuePredictor,
    LatentTransitionModel,
    OracleControllerContextProjector,
    OutcomePredictor,
    ProbeModeAdversary,
    WorldEncoder,
)
from ..objectives.targets import build_training_tensors
from .common import modules_are_finite, sanitize_modules_
from .env import run_env_belief_update
from .window import run_window_training_epoch
from ...envbelief import EnvBeliefAggregator, EnvParamPredictorEnsemble, build_env_group_tensors


def train_encoder_predictor(
    windows: dict[str, np.ndarray],
    z_dim: int = 8,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    physics_loss_weight: float = 0.1,
    affordance_loss_weight: float = 1.0,
    decision_loss_weight: float = 1.0,
    return_loss_weight: float = 0.5,
    risk_loss_weight: float = 0.25,
    kl_loss_weight: float = 1e-3,
    contrastive_loss_weight: float = 0.25,
    env_consistency_loss_weight: float = 0.35,
    env_geometry_loss_weight: float = 0.20,
    mode_adversary_loss_weight: float = 0.10,
    latent_rollout_loss_weight: float = 0.15,
    latent_gaussian_loss_weight: float = 0.02,
    env_within_between_loss_weight: float = 0.30,
    env_retrieval_loss_weight: float = 0.30,
    belief_subset_count: int = 4,
    belief_subset_size: int = 6,
    contrastive_dim: int = 64,
    ensemble_size: int = 3,
    action_vocab_size: int | None = None,
    intervention_horizon: int = 12,
    analytic_affordances: bool = True,
    env_name: str | None = None,
    max_grad_norm: float = 1.0,
    representation_repair_mode: bool = False,
    progress_callback=None,
) -> tuple[
    WorldEncoder,
    DeltaPredictorEnsemble,
    EnvBeliefAggregator,
    EnvParamPredictorEnsemble,
    OutcomePredictor,
    FamilyConditionedOutcomePredictor,
    FamilyConditionedValuePredictor,
    ContrastiveProjector,
    BeliefMessageProjector,
    ControllerBeliefContextProjector,
    OracleControllerContextProjector,
    ControllerTrustPredictor,
    dict[str, np.ndarray],
    torch.device,
]:
    """Train the recurrent posterior encoder and its structured decoders jointly."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if action_vocab_size is None:
        action_vocab_size = int(np.max(windows["actions"])) + 1

    tensors = build_training_tensors(
        windows,
        action_vocab_size=action_vocab_size,
        intervention_horizon=intervention_horizon,
        analytic_affordances=analytic_affordances,
        env_name=env_name,
    )

    window_states = torch.tensor(tensors["window_states"], dtype=torch.float32, device=device)
    window_actions = torch.tensor(tensors["window_actions"], dtype=torch.long, device=device)
    window_rewards = torch.tensor(tensors["window_rewards"], dtype=torch.float32, device=device)
    prefix_states = torch.tensor(tensors["prefix_states"], dtype=torch.float32, device=device)
    prefix_actions = torch.tensor(tensors["prefix_actions"], dtype=torch.long, device=device)
    prefix_rewards = torch.tensor(tensors["prefix_rewards"], dtype=torch.float32, device=device)
    env_instance_id = torch.tensor(tensors["env_instance_id"], dtype=torch.long, device=device)
    probe_mode_idx = torch.tensor(tensors["probe_mode_idx"], dtype=torch.long, device=device)
    current_state = torch.tensor(tensors["current_state"], dtype=torch.float32, device=device)
    current_action = torch.tensor(tensors["current_action"], dtype=torch.long, device=device)
    target_delta = torch.tensor(tensors["target_delta"], dtype=torch.float32, device=device)
    target_env_params = torch.tensor(tensors["target_env_params"], dtype=torch.float32, device=device)
    target_affordances = torch.tensor(tensors["target_affordances"], dtype=torch.float32, device=device)
    target_decision = torch.tensor(tensors["target_decision"], dtype=torch.float32, device=device)
    target_return = torch.tensor(tensors["target_return"], dtype=torch.float32, device=device)
    target_risk = torch.tensor(tensors["target_risk"], dtype=torch.float32, device=device)
    target_future_summary = torch.tensor(tensors["target_future_summary"], dtype=torch.float32, device=device)

    encoder = WorldEncoder(
        state_dim=window_states.shape[-1],
        window_size=window_actions.shape[1],
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    predictor = DeltaPredictorEnsemble(
        ensemble_size=ensemble_size,
        state_dim=window_states.shape[-1],
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    belief_aggregator = EnvBeliefAggregator(
        window_z_dim=z_dim,
        param_dim=target_env_params.shape[-1],
        num_families=int(torch.max(probe_mode_idx).item()) + 1,
    ).to(device)
    env_group_tensors = build_env_group_tensors(
        window_mean=np.zeros((window_states.shape[0], z_dim), dtype=np.float32),
        window_logvar=np.zeros((window_states.shape[0], z_dim), dtype=np.float32),
        env_instance_id=tensors["env_instance_id"],
        env_params=tensors["target_env_params"],
    )
    env_target_env_params = torch.tensor(env_group_tensors["env_params"], dtype=torch.float32, device=device)
    env_param_predictor = EnvParamPredictorEnsemble(
        ensemble_size=ensemble_size,
        input_dim=z_dim,
        output_dim=env_target_env_params.shape[-1],
    ).to(device)
    env_future_predictor = OutcomePredictor(z_dim, target_future_summary.shape[-1]).to(device)
    env_family_future_predictor = FamilyConditionedOutcomePredictor(
        input_dim=z_dim,
        num_families=int(torch.max(probe_mode_idx).item()) + 1,
        output_dim=target_future_summary.shape[-1],
    ).to(device)
    family_value_predictor = FamilyConditionedValuePredictor(
        input_dim=z_dim + 4,
        num_families=int(torch.max(probe_mode_idx).item()) + 1,
        output_dim=3,
    ).to(device)
    belief_message_projector = BeliefMessageProjector(input_dim=z_dim, output_dim=z_dim).to(device)
    controller_context_projector = ControllerBeliefContextProjector(
        input_dim=z_dim,
        mechanics_dim=z_dim,
        affordance_dim=z_dim,
    ).to(device)
    oracle_context_projector = OracleControllerContextProjector(
        input_dim=env_target_env_params.shape[-1],
        mechanics_dim=z_dim,
        affordance_dim=z_dim,
    ).to(device)
    controller_trust_predictor = ControllerTrustPredictor(
        input_dim=z_dim,
    ).to(device)
    latent_transition_model = LatentTransitionModel(
        state_dim=window_states.shape[-1],
        action_vocab_size=action_vocab_size,
        z_dim=z_dim,
    ).to(device)
    affordance_predictor = OutcomePredictor(z_dim, target_affordances.shape[-1]).to(device)
    decision_predictor = OutcomePredictor(z_dim, target_decision.shape[-1]).to(device)
    return_predictor = OutcomePredictor(z_dim, target_return.shape[-1]).to(device)
    risk_predictor = OutcomePredictor(z_dim, target_risk.shape[-1]).to(device)
    contrastive_query = ContrastiveProjector(z_dim, output_dim=contrastive_dim).to(device)
    contrastive_key = ContrastiveProjector(target_future_summary.shape[-1], output_dim=contrastive_dim).to(device)
    env_projector = ContrastiveProjector(z_dim, output_dim=contrastive_dim).to(device)
    env_metric_projector = ContrastiveProjector(z_dim, output_dim=contrastive_dim).to(device)
    mode_adversary = ProbeModeAdversary(input_dim=z_dim, num_modes=int(torch.max(probe_mode_idx).item()) + 1).to(device)
    env_mode_adversary = ProbeModeAdversary(
        input_dim=z_dim,
        num_modes=int(torch.max(probe_mode_idx).item()) + 1,
    ).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(belief_aggregator.parameters())
        + list(env_param_predictor.parameters())
        + list(env_future_predictor.parameters())
        + list(env_family_future_predictor.parameters())
        + list(family_value_predictor.parameters())
        + list(belief_message_projector.parameters())
        + list(controller_context_projector.parameters())
        + list(oracle_context_projector.parameters())
        + list(controller_trust_predictor.parameters())
        + list(latent_transition_model.parameters())
        + list(affordance_predictor.parameters())
        + list(decision_predictor.parameters())
        + list(return_predictor.parameters())
        + list(risk_predictor.parameters())
        + list(contrastive_query.parameters())
        + list(contrastive_key.parameters())
        + list(env_projector.parameters())
        + list(env_metric_projector.parameters())
        + list(mode_adversary.parameters())
        + list(env_mode_adversary.parameters()),
        lr=lr,
        eps=1e-5,
    )
    train_modules = [
        encoder,
        predictor,
        belief_aggregator,
        env_param_predictor,
        env_future_predictor,
        env_family_future_predictor,
        family_value_predictor,
        belief_message_projector,
        controller_context_projector,
        oracle_context_projector,
        controller_trust_predictor,
        latent_transition_model,
        affordance_predictor,
        decision_predictor,
        return_predictor,
        risk_predictor,
        contrastive_query,
        contrastive_key,
        env_projector,
        env_metric_projector,
        mode_adversary,
        env_mode_adversary,
    ]

    for epoch in range(epochs):
        window_metrics = run_window_training_epoch(
            encoder=encoder,
            predictor=predictor,
            latent_transition_model=latent_transition_model,
            affordance_predictor=affordance_predictor,
            decision_predictor=decision_predictor,
            return_predictor=return_predictor,
            risk_predictor=risk_predictor,
            contrastive_query=contrastive_query,
            contrastive_key=contrastive_key,
            env_projector=env_projector,
            mode_adversary=mode_adversary,
            optimizer=optimizer,
            train_modules=train_modules,
            window_states=window_states,
            window_actions=window_actions,
            window_rewards=window_rewards,
            prefix_states=prefix_states,
            prefix_actions=prefix_actions,
            prefix_rewards=prefix_rewards,
            env_instance_id=env_instance_id,
            probe_mode_idx=probe_mode_idx,
            current_state=current_state,
            current_action=current_action,
            target_delta=target_delta,
            target_env_params=target_env_params,
            target_affordances=target_affordances,
            target_decision=target_decision,
            target_return=target_return,
            target_risk=target_risk,
            target_future_summary=target_future_summary,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            affordance_loss_weight=affordance_loss_weight,
            decision_loss_weight=decision_loss_weight,
            return_loss_weight=return_loss_weight,
            risk_loss_weight=risk_loss_weight,
            kl_loss_weight=kl_loss_weight,
            contrastive_loss_weight=contrastive_loss_weight,
            env_consistency_loss_weight=env_consistency_loss_weight,
            env_geometry_loss_weight=env_geometry_loss_weight,
            mode_adversary_loss_weight=mode_adversary_loss_weight,
            latent_rollout_loss_weight=latent_rollout_loss_weight,
            latent_gaussian_loss_weight=latent_gaussian_loss_weight,
        )
        env_metrics = run_env_belief_update(
            epoch_index=epoch,
            total_epochs=epochs,
            encoder=encoder,
            belief_aggregator=belief_aggregator,
            env_param_predictor=env_param_predictor,
            env_future_predictor=env_future_predictor,
            env_family_future_predictor=env_family_future_predictor,
            family_value_predictor=family_value_predictor,
            belief_message_projector=belief_message_projector,
            controller_context_projector=controller_context_projector,
            oracle_context_projector=oracle_context_projector,
            controller_trust_predictor=controller_trust_predictor,
            env_metric_projector=env_metric_projector,
            env_mode_adversary=env_mode_adversary,
            affordance_predictor=affordance_predictor,
            decision_predictor=decision_predictor,
            return_predictor=return_predictor,
            risk_predictor=risk_predictor,
            optimizer=optimizer,
            train_modules=train_modules,
            window_states=window_states,
            window_actions=window_actions,
            window_rewards=window_rewards,
            env_instance_id=env_instance_id,
            probe_mode_idx=probe_mode_idx,
            target_env_params=target_env_params,
            target_affordances=target_affordances,
            target_decision=target_decision,
            target_return=target_return,
            target_risk=target_risk,
            target_future_summary=target_future_summary,
            belief_subset_count=belief_subset_count,
            belief_subset_size=belief_subset_size,
            max_grad_norm=max_grad_norm,
            physics_loss_weight=physics_loss_weight,
            env_consistency_loss_weight=env_consistency_loss_weight,
            contrastive_loss_weight=contrastive_loss_weight,
            env_geometry_loss_weight=env_geometry_loss_weight,
            env_within_between_loss_weight=env_within_between_loss_weight,
            env_retrieval_loss_weight=env_retrieval_loss_weight,
            representation_repair_mode=representation_repair_mode,
        )
        print(
            f"encoder epoch {epoch + 1:02d} | "
            f"total loss = {window_metrics['loss']:.6f} | "
            f"contrastive = {window_metrics['contrastive_loss']:.6f} | "
            f"same-env = {window_metrics['env_consistency_loss']:.6f} | "
            f"geometry = {window_metrics['env_geometry_loss']:.6f} | "
            f"mode-adv = {window_metrics['mode_adversary_loss']:.6f} | "
            f"rollout = {window_metrics['latent_rollout_loss']:.6f} | "
            f"gaussian = {window_metrics['latent_gaussian_loss']:.6f} | "
            f"env-param = {env_metrics['env_param_loss']:.6f} | "
            f"env-split = {env_metrics['env_split_loss']:.6f} | "
            f"env-contrast = {env_metrics['env_split_contrastive_loss']:.6f} | "
            f"env-retrieval = {env_metrics['env_retrieval_loss']:.6f} | "
            f"raw-retrieval = {env_metrics['raw_env_retrieval_loss']:.6f} | "
            f"retrieval-term = {env_metrics['env_capped_retrieval_term']:.6f}/{env_metrics['env_uncapped_retrieval_term']:.6f} | "
            f"raw-retrieval-term = {env_metrics['env_capped_raw_retrieval_term']:.6f}/{env_metrics['env_uncapped_raw_retrieval_term']:.6f} | "
            f"spread-term = {env_metrics['env_capped_spread_term']:.6f}/{env_metrics['env_uncapped_spread_term']:.6f} | "
            f"env-future = {env_metrics['env_future_loss']:.6f} | "
            f"env-family = {env_metrics['env_family_future_loss']:.6f} | "
            f"env-leaveout = {env_metrics['env_leaveout_future_loss']:.6f} | "
            f"family-value = {env_metrics['family_value_loss']:.6f} | "
            f"family-shift = {env_metrics['family_belief_consistency_loss']:.6f} | "
            f"family-future = {env_metrics['family_future_consistency_loss']:.6f} | "
            f"env-expr = {env_metrics['env_expression_loss']:.6f} | "
            f"ctx-mech = {env_metrics['controller_mechanics_loss']:.6f} | "
            f"ctx-aff = {env_metrics['controller_affordance_loss']:.6f} | "
            f"ctx-succ = {env_metrics['controller_successor_loss']:.6f} | "
            f"ctx-score = {env_metrics['controller_score_loss']:.6f} | "
            f"ctx-rank = {env_metrics['controller_score_consistency_loss']:.6f} | "
            f"oracle-mech = {env_metrics['oracle_mechanics_loss']:.6f} | "
            f"oracle-aff = {env_metrics['oracle_affordance_loss']:.6f} | "
            f"ctx-teach = {env_metrics['controller_oracle_distill_loss']:.6f} | "
            f"ctx-trust = {env_metrics['controller_trust_loss']:.6f} | "
            f"env-gauss = {env_metrics['env_gaussian_loss']:.6f} | "
            f"env-param-support = {env_metrics['env_param_support_loss']:.6f} | "
            f"env-anchor = {env_metrics['env_param_anchor_loss']:.6f} | "
            f"gap-ratio = {env_metrics['env_gap_ratio_loss']:.6f} | "
            f"split-top1 = {env_metrics['env_split_retrieval_top1']:.3f} | "
            f"probe-leak = {env_metrics['env_probe_leakage']:.3f} | "
            f"retrieval-margin = {env_metrics['env_retrieval_margin_loss']:.6f} | "
            f"raw-margin = {env_metrics['raw_env_retrieval_margin_loss']:.6f} | "
            f"env-spread = {env_metrics['env_spread_loss']:.6f} | "
            f"uniformity = {env_metrics['env_uniformity_loss']:.6f} | "
            f"vicreg = {env_metrics['env_vicreg_loss']:.6f} | "
            f"env-metric = {env_metrics['env_within_between_loss']:.6f} | "
            f"env-mode-adv = {env_metrics['env_mode_adversary_loss']:.6f} | "
            f"uncert-cal = {env_metrics['uncertainty_calibration_loss']:.6f} | "
            f"env-total = {env_metrics['env_loss_total']:.6f} | "
            f"env-safe = {env_metrics['env_conservative_loss_total']:.6f} | "
            f"env-core = {env_metrics['env_core_loss_total']:.6f} | "
            f"env-phase = {env_metrics['env_phase_name']} | "
            f"repair = {int(env_metrics['env_representation_repair_mode'])} | "
            f"pred-scale = {env_metrics['env_predictive_phase_scale']:.3f} | "
            f"metric-scale = {env_metrics['env_metric_phase_scale']:.3f} | "
            f"metric-gate = {env_metrics['env_metric_gate_reason']} | "
            f"expr-scale = {env_metrics['env_expression_phase_scale']:.3f} | "
            f"ctrl-scale = {env_metrics['env_controller_phase_scale']:.3f} | "
            f"env-grad = {env_metrics['env_grad_norm']:.6f} | "
            f"env-dom = {env_metrics['env_dominant_term_name']}:{env_metrics['env_dominant_term_value']:.6f} | "
            f"env-step = {env_metrics['env_step_mode']} | "
            f"skipped-window = {int(window_metrics['skipped_window_batches'])} | "
            f"skipped-env = {int(env_metrics['skipped_env_updates'])} | "
            f"env-skip-reason = {env_metrics['env_skip_reason']} | "
            f"env-skip-detail = {env_metrics['env_skip_detail']} | "
            f"env-bad-grads = {int(env_metrics['env_nonfinite_grad_count'])} | "
            f"env-bad-params = {int(env_metrics['env_nonfinite_param_count'])}"
        )
        if progress_callback is not None:
            progress_callback(
                epoch=epoch + 1,
                total_epochs=epochs,
                window_metrics=window_metrics,
                env_metrics=env_metrics,
            )

    return (
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
        {
            "mean": np.asarray(tensors["target_env_params_mean"], dtype=np.float32),
            "std": np.asarray(tensors["target_env_params_std"], dtype=np.float32),
        },
        device,
    )
