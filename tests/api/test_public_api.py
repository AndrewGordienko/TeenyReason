import unittest
from unittest.mock import patch

from teenyreason import (
    Belief,
    Crawler,
    Evidence,
    Message,
    Run,
    Step,
    bench,
    best_crawler,
    compare_ppo,
    crawler,
    crawler_for,
    load_crawler,
    make,
    ppo,
    probe_ppo,
    recipe,
    run,
)


class PublicApiTests(unittest.TestCase):
    def test_top_level_exports_cover_small_library_surface(self):
        self.assertTrue(callable(recipe))
        self.assertTrue(callable(ppo))
        self.assertTrue(callable(crawler))
        self.assertTrue(callable(best_crawler))
        self.assertTrue(callable(compare_ppo))
        self.assertTrue(callable(crawler_for))
        self.assertTrue(callable(load_crawler))
        self.assertTrue(callable(make))
        self.assertTrue(callable(bench))
        self.assertTrue(callable(run))
        self.assertTrue(callable(probe_ppo))
        self.assertTrue(Crawler)
        self.assertTrue(Evidence)
        self.assertTrue(Belief)
        self.assertTrue(Message)
        self.assertTrue(Step)
        self.assertTrue(Run)

    def test_recipe_dispatch_keeps_names_short(self):
        self.assertEqual(recipe("cartpole").name, "cartpole")
        self.assertEqual(recipe("mnist").name, "mnist")
        self.assertEqual(recipe("language").name, "language")
        self.assertEqual(recipe("continuous_lunar_lander").benchmark.kind, "ppo")

    def test_run_maps_public_profile_name_to_internal_benchmark_profile(self):
        with patch("teenyreason.algos.benchmarks.run_training_pipeline") as run_mock:
            run("ContinuousCartPole-v0", ppo(), seeds=2, profile="fast")

        self.assertTrue(run_mock.called)
        self.assertEqual(run_mock.call_args.kwargs["seeds"], [0, 1])
        self.assertEqual(run_mock.call_args.kwargs["config_override"]["benchmark_profile"], "fast")
        self.assertNotIn("profile", run_mock.call_args.kwargs["config_override"])

    def test_probe_ppo_runs_probe_conditioned_track_only(self):
        algo = probe_ppo(profile="fast")
        with patch("teenyreason.app.probe_ppo.run_probe_conditioned_pipeline") as run_mock:
            algo.train("ContinuousCartPole-v0", seeds=2)

        self.assertTrue(run_mock.called)
        self.assertEqual(run_mock.call_args.args[0], "ContinuousCartPole-v0")
        self.assertEqual(run_mock.call_args.kwargs["seeds"], [0, 1])
        self.assertEqual(run_mock.call_args.kwargs["config_override"]["benchmark_profile"], "fast")
        self.assertNotIn("profile", run_mock.call_args.kwargs["config_override"])

    def test_probe_ppo_defaults_to_one_training_run(self):
        algo = probe_ppo(profile="fast")
        with patch("teenyreason.app.probe_ppo.run_probe_conditioned_pipeline") as run_mock:
            algo.train("ContinuousCartPole-v0")

        self.assertEqual(run_mock.call_args.kwargs["seeds"], [0])

    def test_compare_ppo_accepts_one_env_name(self):
        with patch("teenyreason.app.ppo_comparison.run_ppo_comparison") as run_mock:
            compare_ppo("ContinuousCartPole-v0", seeds=1, profile="fast")

        self.assertEqual(run_mock.call_args.args[0], ("ContinuousCartPole-v0",))
        self.assertEqual(run_mock.call_args.kwargs["seeds"], [0])

    def test_crawler_and_probe_ppo_are_separate_components(self):
        crawler_setup = crawler("ContinuousCartPole-v0")
        algo = probe_ppo(profile="fast")

        self.assertEqual(crawler_setup.env_name, "ContinuousCartPole-v0")
        self.assertTrue(callable(algo.train))


if __name__ == "__main__":
    unittest.main()
