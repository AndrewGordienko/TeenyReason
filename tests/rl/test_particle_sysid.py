import functools
import unittest

import numpy as np
import torch

from teenyreason.crawler.library import CrawlerModelBundle
from teenyreason.models.sysid import (
    ParticleBeliefState,
    ProbeLikelihoodModel,
    build_latin_hypercube_particles,
    build_probe_sysid_features,
    train_probe_likelihood_model,
)


def _linear_outcome(params: np.ndarray, family_id: int) -> np.ndarray:
    if family_id == 0:
        return np.asarray(
            [1.8 * params[0] - 0.3 * params[1], 0.7 * params[0] + 1.2 * params[1]],
            dtype=np.float32,
        )
    return np.asarray(
        [-0.9 * params[0] + 1.6 * params[1], 1.4 * params[0] - 0.4 * params[1]],
        dtype=np.float32,
    )


def _synthetic_windows(env_count: int = 24, repeats: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    families: list[str] = []
    env_ids: list[int] = []
    params_rows: list[np.ndarray] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    family_names = ("family_a", "family_b")
    for env_id in range(env_count):
        params = np.asarray(
            [
                -1.0 + 2.0 * env_id / max(env_count - 1, 1),
                np.sin(float(env_id) * 0.7),
            ],
            dtype=np.float32,
        )
        for family_id, family in enumerate(family_names):
            for repeat in range(repeats):
                start = np.asarray([0.1 * repeat, -0.05 * repeat], dtype=np.float32)
                end = start + _linear_outcome(params, family_id)
                path = np.stack(
                    [
                        start,
                        start + (end - start) / 3.0,
                        start + 2.0 * (end - start) / 3.0,
                        end,
                    ],
                    axis=0,
                ).astype(np.float32)
                action = np.full((3,), family_id + 1, dtype=np.int64)
                reward = np.full((3,), float(np.sum(end)), dtype=np.float32)
                states.append(path + rng.normal(0.0, 0.002, size=path.shape).astype(np.float32))
                actions.append(action)
                rewards.append(reward)
                families.append(family)
                env_ids.append(env_id)
                params_rows.append(params)
                terminated.append(False)
                truncated.append(False)
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "probe_mode": np.asarray(families, dtype="U"),
        "env_instance_id": np.asarray(env_ids, dtype=np.int64),
        "env_params": np.asarray(params_rows, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
    }


def _window_record(windows: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray | str | bool]:
    return {
        "states": windows["states"][index],
        "actions": windows["actions"][index],
        "rewards": windows["rewards"][index],
        "terminated": bool(windows["terminated"][index]),
        "truncated": bool(windows["truncated"][index]),
        "probe_family": str(windows["probe_mode"][index]),
    }


@functools.lru_cache(maxsize=1)
def _trained_synthetic():
    windows = _synthetic_windows()
    result = train_probe_likelihood_model(
        windows,
        action_vocab_size=3,
        epochs=90,
        batch_size=48,
        lr=3e-3,
        negative_count=7,
        hidden_dim=64,
        seed=7,
    )
    return windows, result


class _ParticleScoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def predict(self, params_norm, query_norm, family_ids):
        family = family_ids.reshape(-1, 1).float()
        signal = params_norm[:, :1]
        mean = torch.where(family < 0.5, signal, torch.zeros_like(signal))
        logvar = torch.full_like(mean, -3.0) + self.anchor * 0.0
        return mean, logvar


class ParticleSysIdTests(unittest.TestCase):
    def test_feature_extraction_shapes_and_family_ids(self):
        windows = _synthetic_windows(env_count=3, repeats=1)

        batch = build_probe_sysid_features(windows, action_vocab_size=3)

        self.assertEqual(batch.query_features.shape[0], 6)
        self.assertEqual(batch.outcome_features.shape[0], 6)
        self.assertEqual(batch.env_params_raw.shape, (6, 2))
        self.assertEqual(batch.env_params_norm.shape, (6, 2))
        self.assertTrue(np.isfinite(batch.query_features).all())
        self.assertTrue(np.isfinite(batch.outcome_features).all())
        self.assertEqual(batch.stats.family_names, ("family_a", "family_b"))
        np.testing.assert_array_equal(batch.family_ids[:2], np.asarray([0, 1], dtype=np.int64))

    def test_likelihood_model_learns_separable_params(self):
        _windows, result = _trained_synthetic()

        random_top1 = 1.0 / 8.0
        self.assertGreater(result.metrics["validation_top1"], 2.0 * random_top1)
        self.assertGreater(result.metrics["validation_margin"], 0.0)

    def test_particle_posterior_moves_toward_true_params(self):
        windows, result = _trained_synthetic()
        particles = build_latin_hypercube_particles(result.stats, count=64, seed=19)
        state = ParticleBeliefState.from_raw_particles(particles, result.stats)
        record_index = int(np.where(windows["env_instance_id"] == 0)[0][0])
        record = _window_record(windows, record_index)
        true_params = windows["env_params"][record_index]
        true_norm = (true_params - result.stats.env_param_mean) / result.stats.env_param_std
        prior_mean = state.summary()["particle_param_mean_norm"]

        updated = state.update_from_window(record, result.model, likelihood_scale=2.0)
        posterior_mean = updated.summary()["particle_param_mean_norm"]

        self.assertLess(
            float(np.linalg.norm(posterior_mean - true_norm)),
            float(np.linalg.norm(prior_mean - true_norm)),
        )

    def test_entropy_and_ess_drop_after_informative_evidence(self):
        windows, result = _trained_synthetic()
        particles = build_latin_hypercube_particles(result.stats, count=64, seed=23)
        state = ParticleBeliefState.from_raw_particles(particles, result.stats)
        before = state.summary()
        record = _window_record(windows, int(np.where(windows["env_instance_id"] == 1)[0][0]))

        after = state.update_from_window(record, result.model, likelihood_scale=2.0).summary()

        self.assertGreater(before["particle_entropy"], after["particle_entropy"])
        self.assertGreater(before["particle_ess_ratio"], after["particle_ess_ratio"])

    def test_family_scoring_prefers_informative_family(self):
        stats = build_probe_sysid_features(_synthetic_windows(env_count=4, repeats=1), action_vocab_size=3).stats
        particles_raw = np.asarray(
            [[-1.0, 0.0], [-0.3, 0.0], [0.4, 0.0], [1.0, 0.0]],
            dtype=np.float32,
        )
        particles_norm = (particles_raw - stats.env_param_mean) / stats.env_param_std
        weights = np.full((4,), 0.25, dtype=np.float32)
        bundle = CrawlerModelBundle(
            encoder=None,
            predictor=None,
            belief_aggregator=None,
            env_param_predictor=None,
            env_future_predictor=None,
            env_family_future_predictor=None,
            family_value_predictor=None,
            env_metric_projector=None,
            belief_message_projector=None,
            controller_context_projector=None,
            device=torch.device("cpu"),
            z_dim=16,
            window_size=3,
            action_vocab_size=3,
            belief_message_dim=16,
            controller_context_dim=34,
            family_names=("family_a", "family_b"),
            belief_mode="particle_sysid",
            sysid_model=_ParticleScoringModel(),
            sysid_stats=stats,
            sysid_particles_raw=particles_raw,
            sysid_trusted=True,
        )
        predictive = type(
            "PredictiveStub",
            (),
            {
                "future_probe_error": 0.0,
                "metadata": {
                    "belief_mode": "particle_sysid",
                    "particle_particles_norm": particles_norm,
                    "particle_weights": weights,
                }
            },
        )()

        scores = bundle.score_particle_probe_families(
            predictive,
            family_counts={},
            global_family_counts={},
        )

        self.assertGreater(
            scores["family_a"]["predicted_entropy_reduction"],
            scores["family_b"]["predicted_entropy_reduction"],
        )
        self.assertGreater(scores["family_a"]["selection_score"], scores["family_b"]["selection_score"])


if __name__ == "__main__":
    unittest.main()
