import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teenyreason.app.ppo.comparison import (
    PPOComparisonEnvResult,
    PPOComparisonSeedResult,
    _env_summary,
)
from teenyreason.viz.live import LiveTrainingTraceWriter
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

    def test_suite_endpoints_return_idle_and_named_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_dashboard_app(tmpdir)
            client = app.test_client()

            idle = client.get("/api/suite/latest").get_json()
            self.assertFalse(idle["available"])

            suite_payload = {
                "schema_version": 1,
                "suite_name": "tri_domain_belief_suite",
                "run_id": "suite-test",
                "created_at": 1.0,
                "domains": {"cartpole": {}, "language": {}, "image": {}},
                "cross_domain": {"metric_rows": [], "acceptance": {}},
            }
            path = app.config["ARTIFACT_DIR"] + "/tri_domain_belief_suite_test.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(suite_payload, handle)

            index_payload = client.get("/api/suites").get_json()
            latest = client.get("/api/suite/latest").get_json()
            named = client.get("/api/suite/tri_domain_belief_suite_test.json").get_json()

            self.assertTrue(index_payload["available"])
            self.assertEqual(latest["run_id"], "suite-test")
            self.assertEqual(named["artifact_name"], "tri_domain_belief_suite_test.json")

    def test_suite_endpoints_include_nested_four_domain_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_dashboard_app(tmpdir)
            client = app.test_client()

            nested_dir = Path(app.config["ARTIFACT_DIR"]) / "four_domain_run"
            nested_dir.mkdir(parents=True)
            suite_payload = {
                "schema_version": 1,
                "suite_name": "four_domain_belief_suite",
                "run_id": "four-domain-test",
                "created_at": 1.0,
                "domains": {
                    "cartpole": {},
                    "language": {},
                    "image": {},
                    "board": {"title": "Tic-Tac-Toe Rules"},
                },
                "cross_domain": {"metric_rows": [], "acceptance": {}},
            }
            path = nested_dir / "four_domain_belief_suite_test.json"
            path.write_text(json.dumps(suite_payload), encoding="utf-8")

            index_payload = client.get("/api/suites").get_json()
            latest = client.get("/api/suite/latest").get_json()
            named = client.get("/api/suite/four_domain_run/four_domain_belief_suite_test.json").get_json()

            self.assertIn("four_domain_run/four_domain_belief_suite_test.json", index_payload["suite_runs"])
            self.assertEqual(index_payload["latest"], "four_domain_run/four_domain_belief_suite_test.json")
            self.assertIn("board", latest["domains"])
            self.assertEqual(named["domains"]["board"]["title"], "Tic-Tac-Toe Rules")

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

    def test_live_history_clear_endpoint_removes_archived_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LiveTrainingTraceWriter(tmpdir, enabled=True, min_write_interval=0.0)
            writer.reset_session(
                env_name="ContinuousCartPole-v0",
                benchmark_tag="toy_comparison",
                seeds=[0],
                total_runs=1,
                comparison_suite_id="suite-1",
            )
            writer.finish(summary={"probe_episode_solves": [12]})

            app = create_dashboard_app(tmpdir)
            client = app.test_client()

            before = client.get("/api/live").get_json()
            response = client.post("/api/live/history/clear")
            after = client.get("/api/live").get_json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {"ok": True})
            self.assertEqual(len(before["history_runs"]), 1)
            self.assertEqual(after["history_runs"], [])

    def test_live_summary_can_update_before_finish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LiveTrainingTraceWriter(tmpdir, enabled=True, min_write_interval=0.0)
            writer.reset_session(
                env_name="ContinuousCartPole-v0",
                benchmark_tag="toy_comparison",
                seeds=[0],
                total_runs=1,
                comparison_suite_id="suite-1",
            )
            writer.update_summary(
                {
                    "pipeline": "ppo_comparison",
                    "encoder_probe_steps": [1000],
                    "baseline_env_step_solves": [172434],
                    "baseline_episode_solves": [571],
                }
            )
            writer.record_episode_summary(
                variant="probe",
                episode=1,
                episode_return=500.0,
                avg10=500.0,
                avg50=500.0,
                total_env_steps=1200,
            )
            writer.record_episode_summary(
                variant="probe",
                episode=2,
                episode_return=100.0,
                avg10=300.0,
                avg50=300.0,
                total_env_steps=1300,
            )

            app = create_dashboard_app(tmpdir)
            client = app.test_client()
            response = client.get("/api/live")
            payload = response.get_json()

            self.assertEqual(response.status_code, 200)
            self.assertTrue(payload["active"])
            self.assertEqual(payload["summary"]["baseline_env_step_solves"], [172434])
            self.assertEqual(payload["summary"]["baseline_episode_solves"], [571])
            self.assertEqual(payload["summary"]["probe_best_returns"], [500.0])
            self.assertEqual(payload["summary"]["probe_peak_env_steps_with_encoder"], [2200])

    def test_completed_comparison_seed_summary_publishes_probe_solve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LiveTrainingTraceWriter(tmpdir, enabled=True, min_write_interval=0.0)
            writer.reset_session(
                env_name="ContinuousCartPole-v0",
                benchmark_tag="toy_comparison",
                seeds=[0, 1, 2],
                total_runs=3,
                comparison_suite_id="suite-1",
            )
            result = PPOComparisonEnvResult(
                env_name="ContinuousCartPole-v0",
                benchmark_tag="toy",
                profile="fast",
                solved_return=475.0,
                solve_avg_window=10,
                seed_results=(
                    PPOComparisonSeedResult(
                        seed=0,
                        encoder_probe_steps=180,
                        baseline_best_return=500.0,
                        baseline_best_episode=454,
                        baseline_best_env_steps=117000,
                        baseline_solved_episode=454,
                        baseline_solved_env_steps=117000,
                        baseline_total_env_steps=117000,
                        probe_best_return=500.0,
                        probe_best_episode=448,
                        probe_best_env_steps=121662,
                        probe_best_env_steps_with_encoder=121842,
                        probe_solved_episode=448,
                        probe_solved_env_steps=121662,
                        probe_solved_env_steps_with_encoder=121842,
                        probe_total_env_steps=121662,
                        probe_total_env_steps_with_encoder=121842,
                    ),
                ),
            )

            writer.update_summary(_env_summary(result))

            app = create_dashboard_app(tmpdir)
            payload = app.test_client().get("/api/live").get_json()

            self.assertEqual(payload["summary"]["probe_episode_solves"], [448])
            self.assertEqual(payload["summary"]["probe_env_step_solves_with_encoder"], [121842])
            self.assertEqual(payload["summary"]["baseline_episode_solves"], [454])


if __name__ == "__main__":
    unittest.main()
