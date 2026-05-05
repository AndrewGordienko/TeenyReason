import unittest

from teenyreason.crawler import (
    EvidenceSlice,
    VideoBeliefTargets,
    VideoBenchmarkSpec,
    VideoEvidenceSpec,
    default_video_benchmarks,
    default_video_belief_targets,
    video_evidence_slice,
)


class VideoCrawlerScaffoldTests(unittest.TestCase):
    def test_video_spec_builds_generic_evidence_slice(self):
        spec = VideoEvidenceSpec(
            source_id="clip-001",
            frame_count=96,
            frame_shape=(720, 1280, 3),
            fps=24.0,
            camera_motion="moving",
            metadata={"scene": "toy_block_stack"},
        )

        evidence = video_evidence_slice(spec)

        self.assertIsInstance(evidence, EvidenceSlice)
        self.assertEqual(evidence.query_name, "passive_video")
        self.assertEqual(evidence.source_id, "clip-001")
        self.assertEqual(evidence.payload["modality"], "video")
        self.assertEqual(evidence.payload["frame_count"], 96)
        self.assertEqual(evidence.payload["frame_shape"], (720, 1280, 3))
        self.assertEqual(evidence.payload["fps"], 24.0)
        self.assertIn("depth_consistency", evidence.payload["belief_targets"])
        self.assertIn("goal_inference", evidence.payload["belief_targets"])
        self.assertIn("frames_to_structure", evidence.payload["benchmark_names"])
        self.assertIn("goal_inference_accuracy", evidence.payload["benchmark_names"])
        self.assertEqual(evidence.metadata["scene"], "toy_block_stack")

    def test_video_targets_can_disable_specific_belief_questions(self):
        targets = VideoBeliefTargets(goal_inference=False, contact_dynamics=False)
        spec = VideoEvidenceSpec(
            source_id="clip-002",
            frame_count=12,
            targets=targets,
        )

        evidence = video_evidence_slice(spec)

        self.assertIn("object_persistence", evidence.payload["belief_targets"])
        self.assertNotIn("goal_inference", evidence.payload["belief_targets"])
        self.assertNotIn("contact_dynamics", evidence.payload["belief_targets"])

    def test_default_video_targets_cover_world_structure_and_goals(self):
        targets = default_video_belief_targets().query_families()

        self.assertIn("depth_consistency", targets)
        self.assertIn("occlusion_consistency", targets)
        self.assertIn("goal_inference", targets)

    def test_default_video_benchmarks_cover_structure_peak_and_goals(self):
        benchmarks = default_video_benchmarks()
        by_name = {item.name: item for item in benchmarks}

        self.assertTrue(all(isinstance(item, VideoBenchmarkSpec) for item in benchmarks))
        self.assertIn("frames_to_structure", by_name)
        self.assertIn("frames_to_peak_task_score", by_name)
        self.assertIn("future_state_prediction", by_name)
        self.assertIn("goal_inference_accuracy", by_name)
        self.assertTrue(by_name["frames_to_structure"].lower_is_better)
        self.assertFalse(by_name["goal_inference_accuracy"].lower_is_better)

    def test_video_spec_validation_rejects_empty_clips(self):
        spec = VideoEvidenceSpec(source_id="clip-003", frame_count=0)

        with self.assertRaises(ValueError):
            video_evidence_slice(spec)


if __name__ == "__main__":
    unittest.main()
