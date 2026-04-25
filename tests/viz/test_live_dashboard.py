import tempfile
import unittest

import numpy as np

from teenyreason.app.live_trace import LiveTrainingTraceWriter
from teenyreason.viz.dashboard import create_dashboard_app


class LiveDashboardTests(unittest.TestCase):
    def test_live_endpoint_returns_idle_payload_without_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_dashboard_app(tmpdir)
            client = app.test_client()

            response = client.get("/api/live")
            payload = response.get_json()

            self.assertEqual(response.status_code, 200)
            self.assertFalse(payload["available"])
            self.assertEqual(payload["stage"]["id"], "idle")

    def test_live_endpoint_returns_written_trace_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LiveTrainingTraceWriter(tmpdir, enabled=True, min_write_interval=0.0)
            writer.reset_session(
                env_name="ContinuousCartPole-v0",
                benchmark_tag="toy_benchmark",
                seeds=[0],
                total_runs=1,
            )
            writer.begin_seed(run_index=1, total_runs=1, seed=0)
            writer.record_policy_step(
                phase="probe_control",
                variant="probe",
                state=np.asarray([0.1, -0.2, 0.05, 0.3], dtype=np.float32),
                action_value=0.4,
                reward=1.25,
                episode=7,
                step_idx=19,
                episode_return=13.5,
                probe_count=2,
                uncertainty=0.23,
                message_scale=0.61,
                focus_label="chirp",
                expression_muted_by_policy=True,
                expression_mute_reason="fair_not_ready",
            )
            writer.record_episode_summary(
                variant="sim-fanout",
                episode=8,
                episode_return=11.0,
                avg10=12.0,
                avg50=10.0,
                total_env_steps=64,
                expression_muted_by_policy=True,
                expression_mute_reason="benchmark_no_expression",
            )
            writer.finish(
                summary={
                    "probe_episode_solves": [42],
                    "probe_no_expression_episode_solves": [11],
                    "full_system_episode_solves": [17],
                    "full_system_state_only_episode_solves": [23],
                    "sim_fanout_episode_solves": [5],
                }
            )

            app = create_dashboard_app(tmpdir)
            client = app.test_client()
            response = client.get("/api/live")
            payload = response.get_json()

            self.assertEqual(response.status_code, 200)
            self.assertTrue(payload["available"])
            self.assertEqual(payload["env_name"], "ContinuousCartPole-v0")
            self.assertEqual(payload["focus"]["phase"], "sim-fanout_episode_summary")
            self.assertTrue(payload["focus"]["expression_muted_by_policy"])
            self.assertEqual(payload["focus"]["expression_mute_reason"], "benchmark_no_expression")
            self.assertEqual(payload["cartpole"]["focus_label"], "chirp")
            self.assertAlmostEqual(payload["cartpole"]["x"], 0.1, places=6)
            self.assertAlmostEqual(payload["cartpole"]["action_value"], 0.4, places=6)
            self.assertEqual(len(payload["histories"]["probe_noexpr_returns"]), 0)
            self.assertEqual(len(payload["histories"]["probe_returns"]), 1)
            self.assertEqual(len(payload["history_runs"]), 1)
            self.assertEqual(payload["history_runs"][0]["summary"]["probe_episode_solves"], [42])
            self.assertEqual(payload["history_runs"][0]["summary"]["full_system_episode_solves"], [17])
            self.assertEqual(payload["history_runs"][0]["summary"]["full_system_state_only_episode_solves"], [23])
            self.assertEqual(payload["history_runs"][0]["summary"]["sim_fanout_episode_solves"], [5])
            self.assertEqual(payload["history_runs"][0]["archive_solve_label"], "sim-fanout solves")
            self.assertEqual(payload["history_runs"][0]["archive_solve_values"], [5])


if __name__ == "__main__":
    unittest.main()
