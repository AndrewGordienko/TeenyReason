import unittest

import gymnasium as gym
import numpy as np
import torch

from teenyreason.crawler.probes.data.probe_crawler import ProbeCrawler
from teenyreason.crawler.probes.data.probe_env import CartPolePhysics, StaticEnvPhysics, default_env_params
from teenyreason.crawler.probes.data.probe_policy import ProbePolicy
from teenyreason.crawler.probes.latent.probe_actions import collect_adaptive_probe_window, safe_choice_weights


class _StubEncoder:
    def eval(self):
        return self

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((1, batch_size, 1), dtype=torch.float32, device=device)

    def update_belief(
        self,
        *,
        prev_state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del next_state
        del action
        del reward
        batch_size = int(prev_state.shape[0])
        mean = torch.zeros((batch_size, 2), dtype=torch.float32, device=prev_state.device)
        logvar = torch.zeros((batch_size, 2), dtype=torch.float32, device=prev_state.device)
        return hidden, mean, logvar


class _ShortEpisodeEnv(gym.Env):
    metadata = {}

    def __init__(self, *, terminate_after_steps: int):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )
        self.terminate_after_steps = int(terminate_after_steps)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        next_state = np.asarray(
            [float(self.step_count), float(action), 0.0, 0.0],
            dtype=np.float32,
        )
        terminated = self.step_count >= self.terminate_after_steps
        return next_state, 1.0, bool(terminated), False, {}


class _TerminalWindowPlanner:
    support_goal_sequence = ("action_0",)
    active_step_cap = 8
    active_window_stride = 1
    max_windows_per_rollout = 1
    emit_partial_terminal_window = True
    partial_window_min_steps = 2
    rollout_goal = "action_0"

    def choose_rollout_goal(self, state):
        del state
        return "action_0"

    def begin_rollout(self, primary_goal=None):
        self.rollout_goal = primary_goal or "action_0"

    def ensure_goal(self, state):
        del state
        return self.rollout_goal

    def action_prior_scores(self, state, recent_actions):
        del state
        del recent_actions
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def observe_transition(self, **kwargs):
        del kwargs


class ProbeCollectionTests(unittest.TestCase):
    def test_default_probe_crawler_can_open_standard_gym_cartpole(self):
        crawler = ProbeCrawler(window_size=2, randomize_physics=False)
        try:
            self.assertEqual(crawler.env_name, "CartPole-v1")
            self.assertEqual(crawler.action_dim, 2)
        finally:
            crawler.close()

    def test_unknown_env_params_fall_back_to_static_fingerprint(self):
        env = gym.make("MountainCar-v0")
        try:
            params = default_env_params("MountainCar-v0", env)
        finally:
            env.close()

        self.assertIsInstance(params, StaticEnvPhysics)
        self.assertEqual(params.as_array().shape, (3,))

    def test_single_action_probe_policy_handles_anti_repeat(self):
        policy = ProbePolicy(action_space_n=1)
        action = policy.sample_action("anti_repeat", 3, np.random.default_rng(0))

        self.assertEqual(action, 0)

    def test_safe_choice_weights_rejects_empty_scores(self):
        with self.assertRaises(ValueError):
            safe_choice_weights(np.asarray([], dtype=np.float32), temperature=0.35)

    def test_collect_adaptive_probe_window_salvages_long_partial_rollout(self):
        env = _ShortEpisodeEnv(terminate_after_steps=2)
        encoder = _StubEncoder()

        window_states, window_actions, window_rewards, probe_failed, probe_steps_used = collect_adaptive_probe_window(
            env=env,
            encoder=encoder,
            predictor=None,
            device=torch.device("cpu"),
            rng=np.random.default_rng(0),
            window_size=4,
            episode_physics=CartPolePhysics(),
            action_values=None,
            max_probe_retries=3,
        )

        self.assertFalse(probe_failed)
        self.assertEqual(int(probe_steps_used), 6)
        self.assertEqual(window_states.shape, (5, 4))
        self.assertEqual(window_actions.shape, (4,))
        self.assertEqual(window_rewards.shape, (4,))
        np.testing.assert_allclose(window_rewards[:2], np.asarray([1.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(window_rewards[2:], np.asarray([0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(window_states[-1], window_states[-2])
        self.assertEqual(int(window_actions[-1]), int(window_actions[-2]))

    def test_active_probe_episode_saves_short_terminal_probe_window(self):
        crawler = ProbeCrawler(window_size=4, randomize_physics=False)
        try:
            crawler.env.close()
            crawler.env = _ShortEpisodeEnv(terminate_after_steps=2)
            crawler.action_values = None
            crawler.action_dim = 2
            crawler.run_active_probe_episode(
                env_instance_id=0,
                episode_id=0,
                episode_physics=CartPolePhysics(),
                max_steps=8,
                planner=_TerminalWindowPlanner(),
                active_step_cap=8,
                active_window_stride=1,
                max_windows_per_rollout=1,
                primary_goal="action_0",
            )

            self.assertEqual(len(crawler.windows), 1)
            window = crawler.windows[0]
            self.assertEqual(window.probe_mode, "action_0")
            self.assertTrue(window.terminated)
            self.assertEqual(window.states.shape, (5, 4))
            self.assertEqual(window.actions.shape, (4,))
            np.testing.assert_allclose(window.rewards[:2], np.asarray([1.0, 1.0], dtype=np.float32))
            np.testing.assert_allclose(window.rewards[2:], np.asarray([0.0, 0.0], dtype=np.float32))
            np.testing.assert_allclose(window.states[-1], window.states[-2])
        finally:
            crawler.close()

    def test_collect_adaptive_probe_window_keeps_failing_for_too_short_rollouts(self):
        env = _ShortEpisodeEnv(terminate_after_steps=1)
        encoder = _StubEncoder()

        window_states, window_actions, window_rewards, probe_failed, probe_steps_used = collect_adaptive_probe_window(
            env=env,
            encoder=encoder,
            predictor=None,
            device=torch.device("cpu"),
            rng=np.random.default_rng(1),
            window_size=5,
            episode_physics=CartPolePhysics(),
            action_values=None,
            max_probe_retries=2,
        )

        self.assertTrue(probe_failed)
        self.assertIsNone(window_states)
        self.assertIsNone(window_actions)
        self.assertIsNone(window_rewards)
        self.assertEqual(int(probe_steps_used), 2)


if __name__ == "__main__":
    unittest.main()
