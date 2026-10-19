import unittest
import numpy as np
import cv2
from motion_tracker.detector import MotionDetector, DetectedObject
from motion_tracker.optical_flow import OpticalFlowTracker
from motion_tracker.trajectory import TrajectoryTracker
from motion_tracker.heatmap import MotionHeatmap
from motion_tracker.tripwire import TripwireEngine
from motion_tracker.simulator import SyntheticVideoGenerator
from motion_tracker.engine import MotionTrackingEngine
from motion_tracker.app import app

class TestMotionVisionTracker(unittest.TestCase):
    def setUp(self):
        self.sim = SyntheticVideoGenerator(320, 240, num_targets=2)
        self.engine = MotionTrackingEngine()

    def test_synthetic_generator(self):
        frame = self.sim.next_frame()
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_motion_detector(self):
        detector = MotionDetector(min_area=50)
        # Process multiple frames so background model learns
        for _ in range(5):
            f = self.sim.next_frame()
            objects, mask = detector.detect(f)
            
        self.assertIsInstance(objects, list)
        self.assertEqual(mask.shape, (240, 320))

    def test_optical_flow_tracker(self):
        tracker = OpticalFlowTracker()
        f1 = self.sim.next_frame()
        f2 = self.sim.next_frame()
        
        _, flow = tracker.compute_dense_flow(f1)
        blended, flow = tracker.compute_dense_flow(f2)
        
        self.assertEqual(flow.shape, (240, 320, 2))
        self.assertEqual(blended.shape, (240, 320, 3))

    def test_trajectory_tracker(self):
        tracker = TrajectoryTracker()
        # Feed sequence of simulated centroids moving across
        centroids_1 = [(50, 50), (120, 100)]
        centroids_2 = [(53, 52), (124, 102)]
        
        targets_1 = tracker.update(centroids_1)
        self.assertEqual(len(targets_1), 2)
        
        targets_2 = tracker.update(centroids_2)
        self.assertEqual(len(targets_2), 2)
        self.assertIn(1, targets_2)
        self.assertGreater(len(targets_2[1].history), 1)

    def test_motion_heatmap_accumulation(self):
        heatmap = MotionHeatmap()
        mask = np.zeros((240, 320), dtype=np.uint8)
        mask[100:150, 100:150] = 255
        
        heatmap.update(mask)
        stats = heatmap.get_activity_stats()
        self.assertGreater(stats["active_area_percent"], 0.0)
        
        base = np.zeros((240, 320, 3), dtype=np.uint8)
        rendered = heatmap.generate_heatmap(base)
        self.assertEqual(rendered.shape, (240, 320, 3))

    def test_tripwire_intrusion(self):
        tripwire = TripwireEngine()
        zone = tripwire.add_zone("TEST_ZONE", [(10, 10), (100, 10), (100, 100), (10, 100)])
        
        # Fake tracked targets inside zone
        from motion_tracker.trajectory import TrackedTarget
        target = TrackedTarget(target_id=1, centroid=(50, 50))
        alerts = tripwire.check_intrusions({1: target})
        
        self.assertTrue(zone.is_breached)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].zone_name, "TEST_ZONE")

    def test_full_engine_pipeline(self):
        f = self.sim.next_frame()
        res = self.engine.process_frame(f, show_boxes=True, show_flow=True, show_trails=True, show_heatmap=True)
        
        self.assertIn("composite_frame", res)
        self.assertIn("fps", res)
        self.assertIn("detected_count", res)
        self.assertIn("tracked_count", res)
        self.assertEqual(res["composite_frame"].shape, f.shape)

    def test_flask_stats_endpoint(self):
        client = app.test_client()
        res = client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("fps", data)
        self.assertIn("tracked_count", data)

if __name__ == "__main__":
    unittest.main()
