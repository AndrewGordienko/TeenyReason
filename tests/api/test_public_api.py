import unittest
from unittest.mock import patch

from teenyreason import Belief, Crawler, Evidence, Message, Run, Step, bench, ppo, recipe, run


class PublicApiTests(unittest.TestCase):
    def test_top_level_exports_cover_small_library_surface(self):
        self.assertTrue(callable(recipe))
        self.assertTrue(callable(ppo))
        self.assertTrue(callable(bench))
        self.assertTrue(callable(run))
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


if __name__ == "__main__":
    unittest.main()
