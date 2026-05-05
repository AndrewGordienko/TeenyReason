import unittest
from pathlib import Path

from teenyreason import bench, ppo, recipe, run
from teenyreason.envs import CONTINUOUS_CARTPOLE_NAME
from teenyreason.recipes import (
    build_benchmark_recipe,
    build_cartpole_recipe,
    build_language_recipe,
    build_mnist_recipe,
)
from teenyreason.recipes.evidence import STANDARD_EVIDENCE_FIELDS


class CrawlerRecipeTests(unittest.TestCase):
    def test_cartpole_recipe_builds_generic_crawler(self):
        recipe = build_cartpole_recipe()
        crawler = recipe.build_crawler()
        result = crawler.run(seed=3)

        self.assertIsNotNone(recipe.benchmark)
        self.assertEqual(recipe.benchmark.env_name, CONTINUOUS_CARTPOLE_NAME)
        self.assertEqual(result.final_belief_state.support_size, 2)
        self.assertGreater(len(result.final_message.vector), 0)

    def test_multidomain_recipes_share_same_crawler_api(self):
        recipes = [
            build_cartpole_recipe(),
            build_mnist_recipe(),
            build_language_recipe(),
        ]

        for recipe in recipes:
            crawler = recipe.build_crawler()
            result = crawler.run(seed=1)
            self.assertGreaterEqual(result.final_belief_state.support_size, 1)
            self.assertGreater(len(result.final_message.vector), 0)
            self.assertIn("modality", recipe.metadata)
            self.assertIn(
                result.final_message.metadata["belief_source"],
                {"sysid", "learned"},
            )

    def test_domain_recipes_emit_standardized_real_evidence(self):
        expected_modalities = {
            "cartpole": "control",
            "language": "language",
            "mnist": "image",
        }
        for recipe_builder in (build_cartpole_recipe, build_language_recipe, build_mnist_recipe):
            recipe = recipe_builder()
            result = recipe.build_crawler(max_steps=2).run(seed=5)
            first_step = result.steps[0]
            payload = first_step.evidence.payload

            for field in STANDARD_EVIDENCE_FIELDS:
                self.assertIn(field, payload)
            self.assertEqual(payload["modality"], expected_modalities[recipe.name])
            self.assertEqual(payload["query_family"], first_step.query_name)
            self.assertEqual(payload["source_id"], first_step.evidence.source_id)
            self.assertGreaterEqual(float(payload["intervention_cost"]), 0.0)
            self.assertGreater(len(payload["local_state"]), 0)
            self.assertGreater(len(payload["outcome"]), 0)
            self.assertGreater(len(payload["hidden_target"]), 0)
            self.assertGreater(len(payload["vector"]), 1)

            if recipe.name == "cartpole":
                self.assertIn("states", payload)
                self.assertIn("actions", payload)
                self.assertIn("rewards", payload)
            elif recipe.name == "language":
                self.assertIn("grammar", payload["hidden_target"])
            else:
                self.assertIn("image", payload)

    def test_benchmark_recipe_keeps_env_selection_out_of_main_loop(self):
        lunar_recipe = build_benchmark_recipe("continuous_lunar_lander")
        self.assertIsNotNone(lunar_recipe.benchmark)
        self.assertEqual(lunar_recipe.benchmark.env_name, "continuous_lunar_lander")
        self.assertEqual(lunar_recipe.benchmark.kind, "ppo")

    def test_top_level_public_api_is_small_and_composable(self):
        cartpole = recipe("cartpole")
        consumer = ppo(profile="fast")

        self.assertEqual(cartpole.name, "cartpole")
        self.assertEqual(cartpole.benchmark.kind, "ppo")
        self.assertEqual(
            consumer.default_config_override["benchmark_profile"],
            "fast",
        )
        self.assertTrue(callable(run))
        self.assertTrue(callable(recipe))
        self.assertTrue(callable(bench))

    def test_crawler_core_stays_free_of_env_name_branching_and_rl_imports(self):
        core_source = Path("teenyreason/crawler/core.py").read_text(encoding="utf-8")
        self.assertNotIn("env_name", core_source)
        self.assertNotIn("teenyreason.rl", core_source)
        self.assertNotIn("from ..rl", core_source)

    def test_public_examples_stop_teaching_internal_build_and_consumer_names(self):
        public_files = [
            Path("main.py"),
            Path("README.md"),
            Path("docs/crawler_library_api.md"),
        ]
        joined = "\n".join(
            path.read_text(encoding="utf-8") for path in public_files
        )
        self.assertNotIn("PPOBenchmarkConsumer", joined)
        self.assertNotIn("build_benchmark_recipe", joined)
        self.assertIn("run(", joined)
        self.assertIn("ppo(", joined)
        self.assertNotIn("recipe(", joined)
        self.assertNotIn("bench(", joined)

    def test_small_public_files_stay_small(self):
        line_budgets = {
            "main.py": 80,
            "teenyreason/__init__.py": 180,
            "teenyreason/crawler/core.py": 400,
            "teenyreason/crawler/types.py": 280,
            "teenyreason/app/benchmark/runner.py": 2125,
        }
        for path_str, limit in line_budgets.items():
            line_count = len(Path(path_str).read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(
                line_count,
                limit,
                f"{path_str} grew to {line_count} lines; split it before it bloats further.",
            )


if __name__ == "__main__":
    unittest.main()
