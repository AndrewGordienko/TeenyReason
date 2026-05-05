import importlib
import unittest


class ImportCompatibilityTests(unittest.TestCase):
    def test_probe_policy_package_exports_train_entrypoints(self):
        probe_policy = importlib.import_module("teenyreason.rl.probe_policy")
        training = importlib.import_module("teenyreason.rl.probe_policy.training")

        self.assertIs(probe_policy.train_plain_ppo, training.train_plain_ppo)
        self.assertIs(probe_policy.train_probe_conditioned_ppo, training.train_probe_conditioned_ppo)

    def test_core_package_exports_public_symbols(self):
        core = importlib.import_module("teenyreason.rl.core")
        optim = importlib.import_module("teenyreason.rl.core.optim")
        normalization = importlib.import_module("teenyreason.rl.core.normalization")

        self.assertIs(core.update_ppo_policy, optim.update_ppo_policy)
        self.assertIs(core.RunningNormalizer, normalization.RunningNormalizer)

    def test_crawler_probes_package_exports_probe_planner(self):
        probes = importlib.import_module("teenyreason.crawler.probes")
        explorer = importlib.import_module("teenyreason.crawler.probes.explorer")

        self.assertIs(probes.build_probe_planner, explorer.build_probe_planner)

    def test_legacy_shim_modules_are_removed(self):
        for module_name in (
            "teenyreason.probe",
            "teenyreason.rl.ppo_core",
            "teenyreason.rl.probe_ppo",
            "teenyreason.rl.core.ppo_core",
            "teenyreason.rl.belief_planner_train",
            "teenyreason.rl.belief_affordance_train",
            "teenyreason.rl.belief_planner",
            "teenyreason.rl.belief_affordance",
            "teenyreason.rl.simulator_fanout",
            "teenyreason.rl.full_system",
            "teenyreason.rl.probe_policy.audit",
            "teenyreason.rl.probe_policy.eval",
            "teenyreason.rl.probe_policy.handoff_diagnostics",
            "teenyreason.rl.probe_policy.logging",
            "teenyreason.rl.probe_policy.messages",
            "teenyreason.rl.probe_policy.reporting",
            "teenyreason.rl.probe_policy.train_plain",
            "teenyreason.rl.probe_policy.train_probe",
            "teenyreason.rl.probe_policy.training.eval",
            "teenyreason.rl.probe_policy.handoff.messages",
            "teenyreason.rl.probe_policy.budget",
            "teenyreason.models.belief.belief_common",
            "teenyreason.models.belief.belief_components",
            "teenyreason.models.belief.belief_losses",
            "teenyreason.models.belief.belief_targets",
            "teenyreason.models.belief.belief_training",
            "teenyreason.models.belief.belief_training_common",
            "teenyreason.models.belief.belief_training_env",
            "teenyreason.models.belief.belief_training_env_config",
            "teenyreason.models.belief.belief_training_window",
            "teenyreason.models.belief_world_model",
            "teenyreason.models.env_belief",
            "teenyreason.recipes.bipedal",
            "teenyreason.recipes.cartpole",
            "teenyreason.recipes.language",
            "teenyreason.recipes.mnist",
            "teenyreason.multidomain.adapter_bridges",
            "teenyreason.multidomain.board_benchmark",
            "teenyreason.multidomain.cartpole_handoff",
            "teenyreason.multidomain.cartpole_mechanics",
            "teenyreason.multidomain.crawler_adapter",
            "teenyreason.multidomain.decision_gate",
            "teenyreason.multidomain.decision_handoff",
            "teenyreason.multidomain.decision_local",
            "teenyreason.multidomain.evidence_schema",
            "teenyreason.multidomain.family_bridges",
            "teenyreason.multidomain.handoff",
            "teenyreason.multidomain.handoff_repair",
            "teenyreason.multidomain.image_benchmark",
            "teenyreason.multidomain.image_handoff",
            "teenyreason.multidomain.image_models",
            "teenyreason.multidomain.image_residual",
            "teenyreason.multidomain.language_benchmark",
            "teenyreason.multidomain.language_handoff",
            "teenyreason.multidomain.language_models",
            "teenyreason.multidomain.real_causal_adapters",
            "teenyreason.multidomain.suite_acceptance",
            "teenyreason.multidomain.suite_cartpole",
            "teenyreason.algos",
        ):
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module_name)

    def test_belief_package_resolves_nested_symbols(self):
        belief = importlib.import_module("teenyreason.models.belief")
        components = importlib.import_module("teenyreason.models.belief.core.components")
        losses = importlib.import_module("teenyreason.models.belief.objectives.losses")
        training = importlib.import_module("teenyreason.models.belief.training.loop")

        self.assertIs(belief.WorldEncoder, components.WorldEncoder)
        self.assertIs(belief.info_nce_loss, losses.info_nce_loss)
        self.assertIs(belief.train_encoder_predictor, training.train_encoder_predictor)


if __name__ == "__main__":
    unittest.main()
