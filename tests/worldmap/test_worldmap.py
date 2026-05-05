import unittest

import numpy as np

from teenyreason.multidomain.planning.gym_mpc import TransitionBatch
from teenyreason.cognition.worldmap import (
    EvidenceRef,
    action_sequence_graph_score,
    build_control_worldmap,
    graph_guided_sequences,
)
from teenyreason.cognition.worldmap.graph import WorldMap
from teenyreason.cognition.worldmap.schema import WorldNode


class WorldMapTests(unittest.TestCase):
    def test_edge_validation_updates_confidence_and_utility(self):
        world = WorldMap()
        world.add_node(WorldNode("action:0:positive", "action_factor"))
        world.add_node(WorldNode("state:0", "state_factor"))
        edge = world.add_edge(
            "action:0:positive",
            "causes",
            "state:0",
            effect_size=0.5,
            confidence=0.4,
            utility=1.0,
        )

        updated = world.validate_edge(edge.edge_id, EvidenceRef("env", "r0", accepted=True, utility=3.0, cost=2.0))

        self.assertIsNotNone(updated)
        self.assertGreater(updated.confidence, edge.confidence)
        self.assertGreater(updated.utility, edge.utility)
        self.assertEqual(world.summary()["worldmap_edge_count"], 1.0)

    def test_control_worldmap_learns_action_state_relation(self):
        actions = np.linspace(-1.0, 1.0, 40, dtype=np.float32).reshape(-1, 1)
        observations = np.zeros((40, 2), dtype=np.float32)
        observations[:, 0] = np.linspace(-0.5, 0.5, 40, dtype=np.float32)
        next_observations = observations.copy()
        next_observations[:, 0] = observations[:, 0] + 0.5 * actions[:, 0]
        rewards = next_observations[:, 0].copy()
        dones = np.zeros((40,), dtype=np.float32)
        batch = TransitionBatch(observations, actions, rewards, next_observations, dones, np.asarray([float(np.sum(rewards))], dtype=np.float32))

        world = build_control_worldmap(batch, np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32))
        high = action_sequence_graph_score(world, np.ones((4, 1), dtype=np.float32), np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32))
        low = action_sequence_graph_score(world, -np.ones((4, 1), dtype=np.float32), np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32))

        self.assertGreater(world.summary()["worldmap_edge_count"], 0.0)
        self.assertGreater(high.trust, low.trust)
        self.assertTrue(graph_guided_sequences(world, np.asarray([-1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32), horizon=3, count=1))


if __name__ == "__main__":
    unittest.main()
