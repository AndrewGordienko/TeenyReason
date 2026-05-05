import unittest
from types import SimpleNamespace

import numpy as np
import torch

from teenyreason.crawler import ControllerBeliefContext, EnvExpression
from teenyreason.rl.core import (
    BeliefNativeActorCritic,
    EpisodeBatch,
    ProbeConditionedGaussianActorCritic,
    update_ppo_policy,
)
from teenyreason.rl.probe_policy.handoff.audit import build_solver_expression_audit
from teenyreason.rl.probe_policy.handoff.message import (
    DEFAULT_DIAGNOSTIC_SCALE_CAP,
    DEFAULT_DIAGNOSTIC_SCALE_FLOOR,
    DEFAULT_FORCED_EVAL_EXPRESSION_SCALE,
    apply_solver_message_keep_scale,
    build_env_expression,
    build_solver_episode_expression,
    compute_message_mode,
    compute_env_expression_readiness_components,
    compute_strict_fair_diagnostic_scale,
    compute_solver_message_scale,
    compute_solver_training_dropout_prob,
    env_expression_is_ready,
    fair_env_expression_enabled,
    sample_solver_training_message_keep_scale,
    shadow_env_expression_diagnostics,
    shadow_env_expression_enabled,
)


class ProbeControllerTests(unittest.TestCase):
    def test_controller_belief_context_vector_is_stable(self):
        context = ControllerBeliefContext(
            mechanics_code=np.asarray([0.1, -0.2], dtype=np.float32),
            affordance_code=np.asarray([0.3, 0.4], dtype=np.float32),
            confidence=0.7,
            uncertainty_scalar=0.15,
            metadata={"source": "test"},
        )

        np.testing.assert_allclose(
            context.vector,
            np.asarray([0.1, -0.2, 0.3, 0.4, 0.7, 0.15], dtype=np.float32),
        )

    def test_probe_actor_uses_env_expression_when_confident(self):
        torch.manual_seed(0)
        model = ProbeConditionedGaussianActorCritic(
            state_dim=4,
            action_dim=1,
            belief_dim=6,
            hidden_dim=32,
        )
        state = torch.zeros((1, 4), dtype=torch.float32)
        expression_a = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.25]], dtype=torch.float32)
        expression_b = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.25]], dtype=torch.float32)

        mean_a, _value_a = model(state, expression_a)
        mean_b, _value_b = model(state, expression_b)

        self.assertFalse(torch.allclose(mean_a, mean_b))

    def test_probe_actor_accepts_custom_initial_action_noise(self):
        model = ProbeConditionedGaussianActorCritic(
            state_dim=8,
            action_dim=2,
            belief_dim=6,
            hidden_dim=32,
            initial_log_std=-0.5,
        )

        np.testing.assert_allclose(
            model.log_std.detach().cpu().numpy(),
            np.asarray([-0.5, -0.5], dtype=np.float32),
        )

    def test_low_confidence_expression_stays_close_to_state_only_path(self):
        torch.manual_seed(0)
        model = ProbeConditionedGaussianActorCritic(
            state_dim=4,
            action_dim=1,
            belief_dim=6,
            hidden_dim=32,
        )
        state = torch.zeros((1, 4), dtype=torch.float32)
        zero_expression = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.25]], dtype=torch.float32)
        low_conf_expression = torch.tensor([[1.0, -0.5, 0.5, 0.2, 0.05, 0.25]], dtype=torch.float32)
        high_conf_expression = torch.tensor([[1.0, -0.5, 0.5, 0.2, 1.0, 0.25]], dtype=torch.float32)

        mean_zero, value_zero = model(state, zero_expression)
        mean_low, value_low = model(state, low_conf_expression)
        mean_high, value_high = model(state, high_conf_expression)

        self.assertLess(torch.abs(mean_low - mean_zero).mean().item(), torch.abs(mean_high - mean_zero).mean().item())
        self.assertLess(torch.abs(value_low - value_zero).mean().item(), torch.abs(value_high - value_zero).mean().item())

    def test_belief_native_actor_uses_context_when_state_is_fixed(self):
        torch.manual_seed(0)
        model = BeliefNativeActorCritic(
            state_dim=4,
            action_dim=1,
            mechanics_dim=2,
            affordance_dim=2,
            hidden_dim=32,
        )
        state = torch.zeros((1, 4), dtype=torch.float32)
        context_a = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.8, 0.2]], dtype=torch.float32)
        context_b = torch.tensor([[0.9, -0.4, 0.6, 0.5, 0.8, 0.2]], dtype=torch.float32)

        mean_a, value_a, _hidden_a, aux_a = model.forward_with_hidden(state, context_a)
        mean_b, value_b, _hidden_b, aux_b = model.forward_with_hidden(state, context_b)

        self.assertFalse(torch.allclose(mean_a, mean_b))
        self.assertFalse(torch.allclose(value_a, value_b))
        self.assertIn("surprise", aux_a)
        self.assertIn("query_logit", aux_b)

    def test_belief_native_recurrent_state_is_initialized_from_context(self):
        torch.manual_seed(0)
        model = BeliefNativeActorCritic(
            state_dim=4,
            action_dim=1,
            mechanics_dim=2,
            affordance_dim=2,
            hidden_dim=32,
        )
        context_a = torch.tensor([[0.1, 0.0, 0.0, 0.2, 0.5, 0.2]], dtype=torch.float32)
        context_b = torch.tensor([[0.8, -0.6, 0.4, 0.9, 0.5, 0.2]], dtype=torch.float32)

        hidden_a = model.init_recurrent_state(context_a)
        hidden_b = model.init_recurrent_state(context_b)
        refreshed_hidden = model.refresh_recurrent_state(context_b, hidden_state=hidden_a, blend=0.5)

        self.assertFalse(torch.allclose(hidden_a, hidden_b))
        self.assertEqual(hidden_a.shape, hidden_b.shape)
        self.assertEqual(hidden_a.shape, refreshed_hidden.shape)
        self.assertFalse(torch.allclose(refreshed_hidden, hidden_a))

    def test_belief_native_context_gate_drops_with_low_trust(self):
        torch.manual_seed(0)
        model = BeliefNativeActorCritic(
            state_dim=4,
            action_dim=1,
            mechanics_dim=2,
            affordance_dim=2,
            hidden_dim=32,
        )
        state = torch.zeros((1, 4), dtype=torch.float32)
        high_trust_context = torch.tensor([[0.8, -0.6, 0.4, 0.9, 0.9, 0.05]], dtype=torch.float32)
        low_trust_context = torch.tensor([[0.8, -0.6, 0.4, 0.9, 0.1, 0.9]], dtype=torch.float32)

        _mean_a, _value_a, _hidden_a, aux_a = model.forward_with_hidden(state, high_trust_context)
        _mean_b, _value_b, _hidden_b, aux_b = model.forward_with_hidden(state, low_trust_context)

        self.assertGreater(float(aux_a["context_gate"].item()), float(aux_b["context_gate"].item()))

    def test_belief_native_forward_sequence_matches_time_major_shapes(self):
        torch.manual_seed(0)
        model = BeliefNativeActorCritic(
            state_dim=4,
            action_dim=1,
            mechanics_dim=2,
            affordance_dim=2,
            hidden_dim=32,
        )
        state = torch.zeros((2, 3, 4), dtype=torch.float32)
        belief = torch.tensor(
            [
                [[0.1, 0.0, 0.2, 0.1, 0.8, 0.2]] * 3,
                [[0.4, 0.3, -0.2, 0.6, 0.7, 0.1]] * 3,
            ],
            dtype=torch.float32,
        )
        init_hidden = model.init_recurrent_state(belief[:, 0, :])
        mean, value, next_hidden, aux = model.forward_sequence(state, belief, init_hidden)

        self.assertEqual(mean.shape, (2, 3, 1))
        self.assertEqual(value.shape, (2, 3))
        self.assertEqual(next_hidden.shape, (2, 32))
        self.assertEqual(aux["context_gate"].shape, (2, 3))

    def test_belief_native_recurrent_updater_uses_sequence_path(self):
        class SequenceOnlyBeliefActor(BeliefNativeActorCritic):
            def __init__(self):
                super().__init__(
                    state_dim=4,
                    action_dim=1,
                    mechanics_dim=2,
                    affordance_dim=2,
                    hidden_dim=16,
                )
                self.sequence_called = False

            def forward(self, state, belief):  # pragma: no cover - should never be used
                raise AssertionError("flat forward path should not be used for recurrent belief-native PPO")

            def forward_sequence(self, state, belief, hidden_state, mask=None):
                self.sequence_called = True
                return super().forward_sequence(state, belief, hidden_state, mask=mask)

        torch.manual_seed(0)
        model = SequenceOnlyBeliefActor()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = EpisodeBatch(
            states=np.zeros((4, 4), dtype=np.float32),
            actions=np.zeros((4, 1), dtype=np.float32),
            old_log_probs=np.zeros((4,), dtype=np.float32),
            old_values=np.zeros((4,), dtype=np.float32),
            returns=np.ones((4,), dtype=np.float32),
            advantages=np.ones((4,), dtype=np.float32),
            beliefs=np.tile(np.asarray([[0.1, 0.2, 0.3, 0.4, 0.8, 0.1]], dtype=np.float32), (4, 1)),
            recurrent_hidden_states=np.zeros((4, 16), dtype=np.float32),
            sequence_length=2,
        )

        update_ppo_policy(
            model=model,
            optimizer=optimizer,
            batch=batch,
            action_low=np.asarray([-1.0], dtype=np.float32),
            action_high=np.asarray([1.0], dtype=np.float32),
            clip_ratio=0.2,
            value_loss_weight=0.5,
            entropy_coef=0.0,
            ppo_epochs=1,
            minibatch_size=2,
            max_grad_norm=0.5,
            target_kl=0.02,
            controller_sequence_length=2,
        )

        self.assertTrue(model.sequence_called)

    def test_solver_message_scale_keeps_meaningful_early_trust(self):
        low_uncertainty_scale = compute_solver_message_scale(
            uncertainty_scalar=0.05,
            uncertainty_probe_threshold=0.10,
            current_episode=1,
            total_episodes=300,
        )
        high_uncertainty_scale = compute_solver_message_scale(
            uncertainty_scalar=0.30,
            uncertainty_probe_threshold=0.10,
            current_episode=1,
            total_episodes=300,
        )

        self.assertGreater(low_uncertainty_scale, 0.30)
        self.assertLess(high_uncertainty_scale, low_uncertainty_scale)

    def test_solver_message_scale_drops_when_belief_reliability_is_weak(self):
        reliable_scale = compute_solver_message_scale(
            uncertainty_scalar=0.10,
            uncertainty_probe_threshold=0.10,
            current_episode=12,
            total_episodes=300,
            future_probe_error=0.45,
            support_diversity_ratio=0.95,
            support_count=3,
            posterior_entropy=3.4,
        )
        weak_scale = compute_solver_message_scale(
            uncertainty_scalar=0.10,
            uncertainty_probe_threshold=0.10,
            current_episode=12,
            total_episodes=300,
            future_probe_error=1.10,
            support_diversity_ratio=0.45,
            support_count=1,
            posterior_entropy=5.0,
        )

        self.assertGreater(reliable_scale, weak_scale)

    def test_solver_message_scale_drops_when_local_geometry_is_bad(self):
        good_geometry_scale = compute_solver_message_scale(
            uncertainty_scalar=0.10,
            uncertainty_probe_threshold=0.10,
            current_episode=12,
            total_episodes=300,
            future_probe_error=0.55,
            support_diversity_ratio=0.90,
            support_count=4,
            posterior_entropy=3.5,
            gap_ratio=0.20,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            expression_ready=True,
        )
        bad_geometry_scale = compute_solver_message_scale(
            uncertainty_scalar=0.10,
            uncertainty_probe_threshold=0.10,
            current_episode=12,
            total_episodes=300,
            future_probe_error=0.55,
            support_diversity_ratio=0.90,
            support_count=4,
            posterior_entropy=3.5,
            gap_ratio=6.0,
            split_retrieval_margin_deficit=0.18,
            split_latent_disagreement=0.06,
            expression_ready=False,
        )

        self.assertGreater(good_geometry_scale, bad_geometry_scale)

    def test_env_expression_ready_requires_stable_geometry(self):
        ready = env_expression_is_ready(
            confidence=0.72,
            uncertainty_scalar=0.08,
            future_probe_error=0.40,
            support_count=4,
            support_diversity_ratio=0.90,
            posterior_entropy=3.5,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            uncertainty_probe_threshold=0.10,
        )
        not_ready = env_expression_is_ready(
            confidence=0.72,
            uncertainty_scalar=0.08,
            future_probe_error=0.40,
            support_count=4,
            support_diversity_ratio=0.90,
            posterior_entropy=3.5,
            gap_ratio=0.36,
            split_retrieval_margin_deficit=0.14,
            split_latent_disagreement=0.04,
            uncertainty_probe_threshold=0.10,
        )

        self.assertTrue(ready)
        self.assertFalse(not_ready)

    def test_env_expression_ready_fails_when_leaveout_axis_is_weak(self):
        ready = env_expression_is_ready(
            confidence=0.72,
            uncertainty_scalar=0.08,
            future_probe_error=0.40,
            support_count=4,
            support_diversity_ratio=0.90,
            posterior_entropy=3.5,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            leaveout_shift=0.04,
            leaveout_param_std_mean=0.02,
            uncertainty_probe_threshold=0.10,
        )
        not_ready = env_expression_is_ready(
            confidence=0.72,
            uncertainty_scalar=0.08,
            future_probe_error=0.40,
            support_count=4,
            support_diversity_ratio=0.90,
            posterior_entropy=3.5,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            leaveout_shift=0.22,
            leaveout_param_std_mean=0.18,
            uncertainty_probe_threshold=0.10,
        )

        self.assertTrue(ready)
        self.assertFalse(not_ready)

    def test_env_expression_ready_rejects_near_ready_two_support_case(self):
        ready = env_expression_is_ready(
            confidence=0.34,
            uncertainty_scalar=0.08,
            future_probe_error=0.62,
            heldout_family_future_error=0.58,
            support_size_matched_future_error=0.72,
            support_count=2,
            support_diversity_ratio=0.86,
            posterior_entropy=4.4,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            leaveout_shift=0.03,
            leaveout_param_std_mean=0.02,
            uncertainty_probe_threshold=0.10,
        )

        self.assertFalse(ready)

    def test_env_expression_ready_accepts_strong_two_support_case(self):
        ready = env_expression_is_ready(
            confidence=0.55,
            uncertainty_scalar=0.08,
            future_probe_error=0.62,
            heldout_family_future_error=0.60,
            support_size_matched_future_error=0.58,
            support_count=2,
            support_diversity_ratio=0.92,
            posterior_entropy=4.2,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            leaveout_shift=0.03,
            leaveout_param_std_mean=0.02,
            uncertainty_probe_threshold=0.10,
        )

        self.assertTrue(ready)

    def test_near_ready_thresholds_do_not_override_weak_geometry(self):
        not_ready = env_expression_is_ready(
            confidence=0.34,
            uncertainty_scalar=0.08,
            future_probe_error=0.62,
            heldout_family_future_error=0.58,
            support_size_matched_future_error=0.72,
            support_count=4,
            support_diversity_ratio=0.86,
            posterior_entropy=4.4,
            gap_ratio=0.36,
            split_retrieval_margin_deficit=0.14,
            split_latent_disagreement=0.04,
            leaveout_shift=0.03,
            leaveout_param_std_mean=0.02,
            uncertainty_probe_threshold=0.10,
        )
        weak_geometry_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.36,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "on",
                "readiness_score": 0.57,
                "future_probe_quality": 0.57,
                "online_subset_stability": 0.30,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )

        self.assertFalse(not_ready)
        self.assertFalse(fair_env_expression_enabled(env_expression=weak_geometry_expression))

    def test_env_expression_ready_blocks_confidence_below_new_floor(self):
        not_ready = env_expression_is_ready(
            confidence=0.31,
            uncertainty_scalar=0.08,
            future_probe_error=0.62,
            heldout_family_future_error=0.58,
            support_size_matched_future_error=0.72,
            support_count=2,
            support_diversity_ratio=0.86,
            posterior_entropy=4.4,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            leaveout_shift=0.03,
            leaveout_param_std_mean=0.02,
            uncertainty_probe_threshold=0.10,
        )

        self.assertFalse(not_ready)

    def test_readiness_future_quality_uses_support_size_matched_error_before_entropy(self):
        strong_predictive = compute_env_expression_readiness_components(
            future_probe_error=0.42,
            heldout_family_future_error=0.26,
            support_size_matched_future_error=0.20,
            support_diversity_ratio=0.90,
            posterior_entropy=4.9,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
        )
        weak_predictive = compute_env_expression_readiness_components(
            future_probe_error=0.42,
            heldout_family_future_error=0.74,
            support_size_matched_future_error=0.76,
            support_diversity_ratio=0.90,
            posterior_entropy=3.4,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
        )

        self.assertGreater(
            strong_predictive["future_probe_quality"],
            weak_predictive["future_probe_quality"],
        )
        self.assertGreater(strong_predictive["future_probe_quality"], 0.0)

    def test_message_mode_allows_strong_two_support_fair_handoff(self):
        components = compute_env_expression_readiness_components(
            future_probe_error=0.62,
            heldout_family_future_error=0.60,
            support_size_matched_future_error=0.58,
            support_count=2,
            support_diversity_ratio=0.92,
            posterior_entropy=4.2,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            leaveout_shift=0.03,
            leaveout_param_std_mean=0.02,
        )
        readiness_score = min(components.values())
        off_mode, off_blocker = compute_message_mode(
            support_count=0,
            confidence=0.55,
            readiness_components=components,
            readiness_score=readiness_score,
        )
        diag_mode, diag_blocker = compute_message_mode(
            support_count=2,
            confidence=0.55,
            readiness_components=components,
            readiness_score=readiness_score,
        )
        on_mode, on_blocker = compute_message_mode(
            support_count=4,
            confidence=0.55,
            readiness_components=components,
            readiness_score=readiness_score,
        )

        self.assertGreaterEqual(components["future_probe_quality"], 0.62)
        self.assertEqual(off_mode, "off")
        self.assertEqual(off_blocker, "support_count")
        self.assertEqual(diag_mode, "on")
        self.assertEqual(diag_blocker, "enabled")
        self.assertEqual(on_mode, "on")
        self.assertEqual(on_blocker, "enabled")

    def test_message_mode_keeps_near_ready_two_support_expression_diagnostic(self):
        components = compute_env_expression_readiness_components(
            future_probe_error=0.62,
            heldout_family_future_error=0.58,
            support_size_matched_future_error=0.72,
            support_count=2,
            support_diversity_ratio=0.86,
            posterior_entropy=4.4,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            leaveout_shift=0.03,
            leaveout_param_std_mean=0.02,
        )
        readiness_score = min(components.values())
        message_mode, blocker = compute_message_mode(
            support_count=2,
            confidence=0.34,
            readiness_components=components,
            readiness_score=readiness_score,
        )

        self.assertGreaterEqual(readiness_score, 0.55)
        self.assertLess(readiness_score, 0.62)
        self.assertLess(0.34, 0.40)
        self.assertGreaterEqual(0.34, 0.32)
        self.assertEqual(message_mode, "diag")
        self.assertEqual(blocker, "support_count")

    def test_low_confidence_message_enters_diag_instead_of_staying_off(self):
        components = compute_env_expression_readiness_components(
            future_probe_error=0.62,
            heldout_family_future_error=0.60,
            support_size_matched_future_error=0.58,
            support_count=2,
            support_diversity_ratio=0.92,
            posterior_entropy=4.2,
            gap_ratio=0.12,
            split_retrieval_margin_deficit=0.02,
            split_latent_disagreement=0.01,
            leaveout_shift=0.03,
            leaveout_param_std_mean=0.02,
        )
        readiness_score = min(components.values())

        diag_mode, diag_blocker = compute_message_mode(
            support_count=2,
            confidence=0.10,
            readiness_components=components,
            readiness_score=readiness_score,
        )

        self.assertEqual(diag_mode, "diag")
        self.assertEqual(diag_blocker, "support_count")

    def test_low_future_probe_quality_enters_diag_instead_of_turning_off(self):
        readiness_components = {
            "future_probe_quality": 0.25,
            "subset_stability": 0.75,
            "support_count": 1.0,
            "confidence": 0.70,
        }

        diag_mode, diag_blocker = compute_message_mode(
            support_count=2,
            confidence=0.70,
            readiness_components=readiness_components,
            readiness_score=0.25,
        )

        self.assertEqual(diag_mode, "diag")
        self.assertEqual(diag_blocker, "future_probe_quality")

    def test_marginal_subset_stability_enters_diag_instead_of_turning_off(self):
        readiness_components = {
            "future_probe_quality": 0.65,
            "subset_stability": 0.32,
            "leaveout_stability": 0.70,
            "support_diversity": 0.85,
        }

        diag_mode, diag_blocker = compute_message_mode(
            support_count=2,
            confidence=0.55,
            readiness_components=readiness_components,
            readiness_score=0.32,
        )

        self.assertEqual(diag_mode, "diag")
        self.assertEqual(diag_blocker, "subset_stability")

    def test_very_weak_subset_stability_stays_off(self):
        readiness_components = {
            "future_probe_quality": 0.65,
            "subset_stability": 0.12,
            "leaveout_stability": 0.70,
            "support_diversity": 0.85,
        }

        off_mode, off_blocker = compute_message_mode(
            support_count=2,
            confidence=0.55,
            readiness_components=readiness_components,
            readiness_score=0.12,
        )

        self.assertEqual(off_mode, "off")
        self.assertEqual(off_blocker, "subset_stability")

    def test_solver_message_keep_scale_preserves_scalar_slots(self):
        belief = apply_solver_message_keep_scale(
            torch.tensor([0.4, -0.2, 0.7, 0.9], dtype=torch.float32).numpy(),
            keep_scale=0.0,
        )

        self.assertAlmostEqual(float(belief[0]), 0.0)
        self.assertAlmostEqual(float(belief[1]), 0.0)
        self.assertAlmostEqual(float(belief[2]), 0.0)
        self.assertAlmostEqual(float(belief[3]), 0.9)

    def test_solver_message_dropout_probability_rises_for_weak_messages(self):
        strong_prob = compute_solver_training_dropout_prob(
            current_episode=18,
            total_episodes=300,
            message_scale=0.85,
            base_dropout_prob=0.08,
        )
        weak_prob = compute_solver_training_dropout_prob(
            current_episode=18,
            total_episodes=300,
            message_scale=0.15,
            base_dropout_prob=0.08,
        )

        self.assertGreater(weak_prob, strong_prob)

    def test_solver_message_keep_scale_uses_jitter_late_in_training(self):
        keep_scale = sample_solver_training_message_keep_scale(
            rng=np.random.default_rng(0),
            current_episode=90,
            total_episodes=120,
            message_scale=0.75,
            base_dropout_prob=0.25,
        )

        self.assertGreaterEqual(keep_scale, 0.5)
        self.assertLessEqual(keep_scale, 1.0)
        self.assertNotEqual(keep_scale, 0.0)

    def test_env_expression_metadata_includes_readiness_and_utility_fields(self):
        predictive_belief = SimpleNamespace(
            future_probe_error=0.42,
            support_count=3,
            support_diversity_ratio=0.88,
            metadata={
                "mechanics_posterior_entropy": 3.2,
                "split_latent_disagreement": 0.01,
                "leaveout_shift": 0.04,
                "leaveout_param_std_mean": 0.02,
                "online_subset_stability": 0.52,
                "online_geometry_complete": True,
                "online_split_latent_disagreement": 0.03,
                "online_split_retrieval_margin_deficit": 0.08,
                "online_leaveout_shift": 0.06,
            },
        )
        metric_belief = SimpleNamespace(
            gap_ratio=0.10,
            metadata={"split_retrieval_margin_deficit": 0.01},
        )
        uncertainty = SimpleNamespace(scalar=0.09)
        env_expression = build_env_expression(
            predictive_belief=predictive_belief,
            metric_belief=metric_belief,
            uncertainty=uncertainty,
            raw_expression=torch.tensor([0.2, -0.1], dtype=torch.float32).numpy(),
        )

        self.assertIn("readiness_score", env_expression.metadata)
        self.assertIn("readiness_reason", env_expression.metadata)
        self.assertIn("utility_forecast", env_expression.metadata)
        self.assertIn("message_mode", env_expression.metadata)
        self.assertIn("message_blocker", env_expression.metadata)
        self.assertIn("future_probe_quality", env_expression.metadata)
        self.assertIn("subset_stability", env_expression.metadata)
        self.assertIn("leaveout_stability", env_expression.metadata)
        self.assertIn("support_diversity", env_expression.metadata)
        self.assertIn("predictive_reuse_error", env_expression.metadata)
        self.assertIn("heldout_family_future_error", env_expression.metadata)
        self.assertIn("support_size_matched_future_quality", env_expression.metadata)
        self.assertIn("online_offline_gap", env_expression.metadata)
        self.assertIn("online_subset_stability", env_expression.metadata)
        self.assertIn("online_geometry_complete", env_expression.metadata)
        self.assertIn("online_split_latent_disagreement", env_expression.metadata)
        self.assertIn("online_split_retrieval_margin_deficit", env_expression.metadata)
        self.assertIn("online_leaveout_shift", env_expression.metadata)
        self.assertIn("online_leaveout_stability", env_expression.metadata)
        self.assertIn("fair_policy_enabled", env_expression.metadata)
        self.assertIn("fair_stop_ready", env_expression.metadata)
        self.assertIn("fair_stop_blocker", env_expression.metadata)

    def test_online_leaveout_stability_overrides_generic_leaveout_penalty(self):
        components = compute_env_expression_readiness_components(
            future_probe_error=0.42,
            heldout_family_future_error=0.42,
            support_size_matched_future_error=0.42,
            support_count=4,
            support_diversity_ratio=0.88,
            posterior_entropy=3.0,
            online_subset_stability=0.62,
            online_leaveout_stability=0.57,
            online_geometry_complete=True,
            leaveout_shift=0.30,
            leaveout_param_std_mean=0.02,
        )

        self.assertAlmostEqual(components["leaveout_stability"], 0.57, places=6)
        self.assertGreater(components["leaveout_stability"], 0.45)

    def test_fair_expression_enabled_requires_live_geometry_not_just_raw_ready(self):
        weak_live_geometry = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "readiness_score": 0.75,
                "future_probe_quality": 0.72,
                "subset_stability": 1.0,
                "online_subset_stability": 0.20,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )
        marginal_live_geometry = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "readiness_score": 0.75,
                "future_probe_quality": 0.72,
                "subset_stability": 1.0,
                "online_subset_stability": 0.55,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )
        strong_live_geometry = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "readiness_score": 0.75,
                "future_probe_quality": 0.72,
                "subset_stability": 1.0,
                "online_subset_stability": 0.68,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )

        self.assertFalse(fair_env_expression_enabled(env_expression=weak_live_geometry))
        self.assertFalse(fair_env_expression_enabled(env_expression=marginal_live_geometry))
        self.assertTrue(fair_env_expression_enabled(env_expression=strong_live_geometry))

    def test_solver_episode_expression_keeps_confidence_and_ready(self):
        env_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.9,
                "fair_policy_enabled": True,
                "online_subset_stability": 0.75,
                "readiness_score": 0.80,
            },
        )
        solver_expression, expression_scale = build_solver_episode_expression(
            env_expression=env_expression,
            current_episode=12,
            total_episodes=200,
            disable_env_expression=False,
            strict_fair_mode=True,
        )

        self.assertEqual(solver_expression.shape[0], 4)
        self.assertGreater(expression_scale, 0.0)
        self.assertGreater(float(solver_expression[-2]), 0.0)
        self.assertAlmostEqual(float(solver_expression[-1]), 0.1, places=5)

    def test_fair_expression_enabled_requires_ready_and_trust_floor(self):
        ready_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.9,
                "readiness_score": 0.80,
                "future_probe_quality": 0.75,
                "online_subset_stability": 0.70,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )
        low_trust_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.2,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.9,
                "readiness_score": 0.80,
                "future_probe_quality": 0.75,
                "online_subset_stability": 0.70,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )
        not_ready_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.9,
                "readiness_score": 0.80,
                "future_probe_quality": 0.75,
                "online_subset_stability": 0.70,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )

        self.assertTrue(fair_env_expression_enabled(env_expression=ready_expression))
        self.assertFalse(fair_env_expression_enabled(env_expression=low_trust_expression))
        self.assertFalse(fair_env_expression_enabled(env_expression=not_ready_expression))

    def test_fair_expression_enabled_rejects_near_ready_on_message(self):
        near_ready_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.36,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "on",
                "readiness_score": 0.57,
                "future_probe_quality": 0.57,
                "online_subset_stability": 0.55,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )

        self.assertLess(near_ready_expression.confidence, 0.45)
        self.assertLess(
            near_ready_expression.metadata["future_probe_quality"],
            0.58,
        )
        self.assertLess(
            near_ready_expression.metadata["readiness_score"],
            0.62,
        )
        self.assertFalse(fair_env_expression_enabled(env_expression=near_ready_expression))

    def test_strict_fair_expression_mutes_on_message_that_fails_fair_gate(self):
        near_ready_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.36,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "on",
                "readiness_score": 0.57,
                "future_probe_quality": 0.57,
                "online_subset_stability": 0.55,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )

        solver_expression, expression_scale = build_solver_episode_expression(
            env_expression=near_ready_expression,
            current_episode=24,
            total_episodes=200,
            disable_env_expression=False,
            strict_fair_mode=True,
        )

        self.assertEqual(expression_scale, 0.0)
        self.assertAlmostEqual(float(solver_expression[-2]), 0.0, places=6)

    def test_fair_expression_enabled_blocks_confidence_below_new_floor(self):
        low_conf_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.34,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "on",
                "readiness_score": 0.57,
                "future_probe_quality": 0.57,
                "online_subset_stability": 0.55,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )

        self.assertFalse(fair_env_expression_enabled(env_expression=low_conf_expression))

    def test_fair_expression_enabled_still_requires_message_mode_on(self):
        diag_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.80,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "diag",
                "readiness_score": 0.80,
                "future_probe_quality": 0.80,
                "online_subset_stability": 0.80,
                "online_geometry_complete": True,
                "online_offline_gap": 0.01,
            },
        )

        self.assertFalse(fair_env_expression_enabled(env_expression=diag_expression))

    def test_solver_episode_expression_mutes_not_ready_fair_handoff(self):
        ready_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.9,
                "fair_policy_enabled": True,
                "online_subset_stability": 0.70,
                "readiness_score": 0.80,
            },
        )
        not_ready_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.45,
                "readiness_score": 0.40,
                "future_probe_quality": 0.35,
                "online_subset_stability": 0.20,
                "online_geometry_complete": False,
                "online_offline_gap": 0.10,
            },
        )

        _ready_solver_expression, ready_scale = build_solver_episode_expression(
            env_expression=ready_expression,
            current_episode=12,
            total_episodes=200,
            disable_env_expression=False,
            strict_fair_mode=True,
        )
        not_ready_solver_expression, not_ready_scale = build_solver_episode_expression(
            env_expression=not_ready_expression,
            current_episode=12,
            total_episodes=200,
            disable_env_expression=False,
            strict_fair_mode=True,
        )

        self.assertGreater(ready_scale, not_ready_scale)
        self.assertEqual(not_ready_scale, 0.0)
        self.assertAlmostEqual(float(not_ready_solver_expression[0]), 0.0, places=6)
        self.assertAlmostEqual(float(not_ready_solver_expression[1]), 0.0, places=6)
        self.assertAlmostEqual(float(not_ready_solver_expression[-2]), 0.0, places=6)
        self.assertAlmostEqual(float(not_ready_solver_expression[-1]), 0.1, places=6)

    def test_solver_expression_audit_shows_exact_parity_when_fair_message_is_muted(self):
        not_ready_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.45,
                "readiness_score": 0.40,
                "future_probe_quality": 0.35,
                "online_subset_stability": 0.20,
                "online_geometry_complete": False,
                "online_offline_gap": 0.10,
            },
        )

        audit = build_solver_expression_audit(
            env_expression=not_ready_expression,
            current_episode=12,
            total_episodes=200,
            strict_fair_mode=True,
        )

        self.assertAlmostEqual(float(audit["enabled_scale"]), 0.0, places=6)
        self.assertAlmostEqual(float(audit["muted_scale"]), 0.0, places=6)
        self.assertAlmostEqual(float(audit["input_delta"]), 0.0, places=6)
        self.assertAlmostEqual(float(audit["message_delta"]), 0.0, places=6)
        self.assertFalse(bool(audit["enabled_has_message"]))
        self.assertFalse(bool(audit["muted_has_message"]))

    def test_solver_expression_scale_caps_marginal_but_enabled_fair_handoffs(self):
        marginal_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.9,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.95,
                "fair_policy_enabled": True,
                "readiness_score": 0.66,
                "online_subset_stability": 0.52,
            },
        )
        strong_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.9,
            ready=True,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.95,
                "fair_policy_enabled": True,
                "readiness_score": 0.82,
                "online_subset_stability": 0.72,
            },
        )

        _marginal_solver_expression, marginal_scale = build_solver_episode_expression(
            env_expression=marginal_expression,
            current_episode=30,
            total_episodes=120,
            disable_env_expression=False,
            strict_fair_mode=False,
        )
        _strong_solver_expression, strong_scale = build_solver_episode_expression(
            env_expression=strong_expression,
            current_episode=30,
            total_episodes=120,
            disable_env_expression=False,
            strict_fair_mode=False,
        )

        self.assertLessEqual(marginal_scale, 0.35)
        self.assertGreater(strong_scale, marginal_scale)

    def test_shadow_expression_gate_uses_exact_thresholds(self):
        shadow_ready_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.30,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.9,
                "future_probe_quality": 0.65,
                "subset_stability": 0.45,
                "leaveout_stability": 0.45,
                "support_diversity": 0.90,
                "readiness_score": 0.48,
                "utility_forecast": 0.58,
                "message_mode": "diag",
            },
        )
        shadow_muted_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.30,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "geometry_scale": 0.9,
                "future_probe_quality": 0.65,
                "subset_stability": 0.45,
                "leaveout_stability": 0.45,
                "support_diversity": 0.90,
                "readiness_score": 0.47,
                "utility_forecast": 0.58,
                "message_mode": "diag",
            },
        )

        shadow_diag = shadow_env_expression_diagnostics(
            env_expression=shadow_ready_expression
        )
        self.assertTrue(shadow_env_expression_enabled(env_expression=shadow_ready_expression))
        self.assertAlmostEqual(float(shadow_diag["shadow_score"]), 0.585, places=6)
        self.assertAlmostEqual(float(shadow_diag["scale_cap"]), 0.135, places=6)

        solver_expression, shadow_scale = build_solver_episode_expression(
            env_expression=shadow_ready_expression,
            current_episode=24,
            total_episodes=200,
            disable_env_expression=False,
            shadow_expression_mode=True,
        )
        muted_solver_expression, muted_shadow_scale = build_solver_episode_expression(
            env_expression=shadow_muted_expression,
            current_episode=24,
            total_episodes=200,
            disable_env_expression=False,
            shadow_expression_mode=True,
        )

        self.assertGreater(shadow_scale, 0.0)
        self.assertAlmostEqual(shadow_scale, 0.135, places=6)
        self.assertGreater(float(solver_expression[-2]), 0.0)
        self.assertFalse(shadow_env_expression_enabled(env_expression=shadow_muted_expression))
        self.assertEqual(
            shadow_env_expression_diagnostics(env_expression=shadow_muted_expression)["blocker"],
            "readiness_score",
        )
        self.assertEqual(muted_shadow_scale, 0.0)
        self.assertAlmostEqual(float(muted_solver_expression[-2]), 0.0, places=6)
        self.assertAlmostEqual(float(muted_solver_expression[-1]), 0.1, places=6)

    def test_solver_episode_expression_softens_diag_message_in_strict_fair_mode(self):
        diag_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "diag",
                "geometry_scale": 0.9,
                "readiness_score": 0.68,
                "utility_forecast": 0.71,
                "future_probe_quality": 0.72,
                "online_subset_stability": 0.74,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )

        solver_expression, expression_scale = build_solver_episode_expression(
            env_expression=diag_expression,
            current_episode=18,
            total_episodes=200,
            disable_env_expression=False,
            strict_fair_mode=True,
        )

        self.assertGreater(expression_scale, 0.0)
        self.assertLessEqual(expression_scale, DEFAULT_DIAGNOSTIC_SCALE_CAP)
        self.assertAlmostEqual(float(solver_expression[-2]), expression_scale, places=6)

    def test_non_strict_diag_scale_softens_weaker_messages(self):
        strong_diag_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "diag",
                "geometry_scale": 0.9,
                "readiness_score": 0.68,
                "utility_forecast": 0.71,
                "future_probe_quality": 0.72,
                "online_subset_stability": 0.74,
                "online_geometry_complete": True,
                "online_offline_gap": 0.02,
            },
        )
        weak_diag_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.8,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "diag",
                "geometry_scale": 0.9,
                "readiness_score": 0.40,
                "utility_forecast": 0.45,
                "future_probe_quality": 0.25,
                "online_subset_stability": 0.46,
                "online_geometry_complete": False,
                "online_offline_gap": 0.08,
            },
        )

        _strong_solver_expression, strong_scale = build_solver_episode_expression(
            env_expression=strong_diag_expression,
            current_episode=18,
            total_episodes=200,
            disable_env_expression=False,
            strict_fair_mode=False,
        )
        _weak_solver_expression, weak_scale = build_solver_episode_expression(
            env_expression=weak_diag_expression,
            current_episode=18,
            total_episodes=200,
            disable_env_expression=False,
            strict_fair_mode=False,
        )
        strong_strict_diag_scale = compute_strict_fair_diagnostic_scale(
            env_expression=strong_diag_expression,
            base_scale=0.0,
        )
        weak_strict_diag_scale = compute_strict_fair_diagnostic_scale(
            env_expression=weak_diag_expression,
            base_scale=0.0,
        )

        self.assertGreater(strong_scale, weak_scale)
        self.assertGreater(strong_strict_diag_scale, weak_strict_diag_scale)
        self.assertLess(weak_strict_diag_scale, DEFAULT_DIAGNOSTIC_SCALE_FLOOR)
        self.assertGreater(weak_strict_diag_scale, 0.0)

    def test_solver_episode_expression_can_force_small_eval_message_through_off_gate(self):
        off_expression = EnvExpression(
            vector=torch.tensor([0.25, -0.5], dtype=torch.float32).numpy(),
            confidence=0.18,
            ready=False,
            uncertainty_scalar=0.1,
            compressed=False,
            metadata={
                "message_mode": "off",
                "geometry_scale": 0.9,
                "readiness_score": 0.40,
                "utility_forecast": 0.45,
                "future_probe_quality": 0.30,
                "online_subset_stability": 0.40,
                "online_geometry_complete": False,
                "online_offline_gap": 0.08,
            },
        )

        muted_expression, muted_scale = build_solver_episode_expression(
            env_expression=off_expression,
            current_episode=30,
            total_episodes=120,
            strict_fair_mode=True,
        )
        forced_expression, forced_scale = build_solver_episode_expression(
            env_expression=off_expression,
            current_episode=30,
            total_episodes=120,
            strict_fair_mode=True,
            force_message_mode="diag",
            forced_expression_scale=DEFAULT_FORCED_EVAL_EXPRESSION_SCALE,
        )

        self.assertEqual(muted_scale, 0.0)
        self.assertAlmostEqual(float(muted_expression[-2]), 0.0, places=6)
        self.assertAlmostEqual(
            forced_scale,
            DEFAULT_FORCED_EVAL_EXPRESSION_SCALE,
            places=6,
        )
        self.assertAlmostEqual(
            float(forced_expression[-2]),
            DEFAULT_FORCED_EVAL_EXPRESSION_SCALE,
            places=6,
        )
        self.assertAlmostEqual(float(forced_expression[-1]), 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
