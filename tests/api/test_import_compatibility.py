import importlib
import unittest


class ImportCompatibilityTests(unittest.TestCase):
    def test_probe_policy_package_exports_train_entrypoints(self):
        probe_policy = importlib.import_module("teenyreason.rl.probe_policy")
        new_plain = importlib.import_module("teenyreason.rl.probe_policy.train_plain")
        new_probe = importlib.import_module("teenyreason.rl.probe_policy.train_probe")

        self.assertIs(probe_policy.train_plain_ppo, new_plain.train_plain_ppo)
        self.assertIs(probe_policy.train_probe_conditioned_ppo, new_probe.train_probe_conditioned_ppo)

    def test_full_system_package_exports_public_entrypoints(self):
        full_system = importlib.import_module("teenyreason.rl.full_system")
        planner_train = importlib.import_module("teenyreason.rl.full_system.planner_train")
        affordance_train = importlib.import_module("teenyreason.rl.full_system.affordance_train")
        planner = importlib.import_module("teenyreason.rl.full_system.planner")
        affordance = importlib.import_module("teenyreason.rl.full_system.affordance")
        fanout = importlib.import_module("teenyreason.rl.full_system.simulator_fanout")

        self.assertIs(full_system.train_belief_planner, planner_train.train_belief_planner)
        self.assertIs(
            full_system.train_belief_affordance_controller,
            affordance_train.train_belief_affordance_controller,
        )
        self.assertIs(full_system.plan_cem_action, planner.plan_cem_action)
        self.assertIs(full_system.choose_affordance_action, affordance.choose_affordance_action)
        self.assertIs(full_system.SimulatorFanoutAdapter, fanout.SimulatorFanoutAdapter)

    def test_core_package_exports_public_symbols(self):
        core = importlib.import_module("teenyreason.rl.core")
        optim = importlib.import_module("teenyreason.rl.core.optim")
        normalization = importlib.import_module("teenyreason.rl.core.normalization")

        self.assertIs(core.update_ppo_policy, optim.update_ppo_policy)
        self.assertIs(core.RunningNormalizer, normalization.RunningNormalizer)

    def test_legacy_shim_modules_are_removed(self):
        for module_name in (
            "teenyreason.rl.ppo_core",
            "teenyreason.rl.probe_ppo",
            "teenyreason.rl.core.ppo_core",
            "teenyreason.rl.belief_planner_train",
            "teenyreason.rl.belief_affordance_train",
            "teenyreason.rl.belief_planner",
            "teenyreason.rl.belief_affordance",
            "teenyreason.rl.simulator_fanout",
        ):
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(module_name)

    def test_old_and_new_belief_module_paths_resolve_same_symbol(self):
        old_components = importlib.import_module("teenyreason.models.belief.belief_components")
        new_components = importlib.import_module("teenyreason.models.belief.core.components")
        old_losses = importlib.import_module("teenyreason.models.belief.belief_losses")
        new_losses = importlib.import_module("teenyreason.models.belief.objectives.losses")
        old_training = importlib.import_module("teenyreason.models.belief.belief_training")
        new_training = importlib.import_module("teenyreason.models.belief.training.loop")

        self.assertIs(old_components.WorldEncoder, new_components.WorldEncoder)
        self.assertIs(old_losses.info_nce_loss, new_losses.info_nce_loss)
        self.assertIs(old_training.train_encoder_predictor, new_training.train_encoder_predictor)


if __name__ == "__main__":
    unittest.main()
