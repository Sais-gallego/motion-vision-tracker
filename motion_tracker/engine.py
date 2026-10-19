"""Unified Motion Vision Tracking Engine."""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from .detector import MotionDetector
from .optical_flow import OpticalFlowTracker
from .trajectory import TrajectoryTracker
from .heatmap import MotionHeatmap
from .tripwire import TripwireEngine

class MotionTrackingEngine:
    def __init__(
        self,
        enable_optical_flow: bool = True,
        enable_trails: bool = True,
        enable_heatmap: bool = True,
        enable_tripwires: bool = True
    ):
        self.detector = MotionDetector(min_area=400)
        self.flow_tracker = OpticalFlowTracker() if enable_optical_flow else None
        self.trajectory_tracker = TrajectoryTracker() if enable_trails else None
        self.heatmap = MotionHeatmap() if enable_heatmap else None
        self.tripwire_engine = TripwireEngine() if enable_tripwires else None
        
        # Performance tracking
        self.prev_time = time.time()
        self.fps = 0.0
        self.frame_count = 0

    def setup_default_zones(self, width: int = 640, height: int = 480):
        """Set up standard restricted perimeter zone for demonstrations."""
        if self.tripwire_engine:
            # Central restricted security polygon
            cx, cy = width // 2, height // 2
            hw, hh = 120, 100
            poly = [
                (cx - hw, cy - hh),
                (cx + hw, cy - hh),
                (cx + hw, cy + hh),
                (cx - hw, cy + hh)
            ]
            self.tripwire_engine.add_zone("VAULT_PERIMETER", poly)

    def process_frame(
        self,
        frame: np.ndarray,
        show_boxes: bool = True,
        show_flow: bool = False,
        show_trails: bool = True,
        show_heatmap: bool = False,
        show_zones: bool = True
    ) -> Dict[str, Any]:
        """Execute full tracking pipeline and return composite frame and telemetry."""
        now = time.time()
        dt = max(1e-4, now - self.prev_time)
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.prev_time = now
        self.frame_count += 1
        
        out_frame = frame.copy()
        
        # 1. Motion detection & segmentation
        detected_objects, fg_mask = self.detector.detect(frame)
        centroids = [obj.centroid for obj in detected_objects]
        
        # 2. Heatmap accumulation
        if self.heatmap is not None:
            self.heatmap.update(fg_mask)
            if show_heatmap:
                out_frame = self.heatmap.generate_heatmap(out_frame, alpha=0.65)
                
        # 3. Dense optical flow
        raw_flow = None
        if self.flow_tracker is not None and show_flow:
            out_frame, raw_flow = self.flow_tracker.compute_dense_flow(out_frame, blend_weight=0.5)

        # 4. Trajectory and target tracking
        tracked_targets = {}
        if self.trajectory_tracker is not None:
            tracked_targets = self.trajectory_tracker.update(centroids)
            if show_trails:
                out_frame = self.trajectory_tracker.draw_trails(out_frame)

        # 5. Security zone intrusion checks
        alerts = []
        if self.tripwire_engine is not None:
            alerts = self.tripwire_engine.check_intrusions(tracked_targets)
            if show_zones:
                out_frame = self.tripwire_engine.draw_zones(out_frame)

        # 6. Bounding boxes & labels
        if show_boxes:
            for obj in detected_objects:
                x, y, w, h = obj.box
                cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0, 255, 200), 2)
                cv2.putText(
                    out_frame, f"Area: {int(obj.area)}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA
                )

        # 7. Heads-Up Display (HUD) Telemetry
        self._render_hud(out_frame, len(detected_objects), len(tracked_targets), alerts)
        
        return {
            "composite_frame": out_frame,
            "fg_mask": fg_mask,
            "detected_count": len(detected_objects),
            "tracked_count": len(tracked_targets),
            "fps": round(self.fps, 1),
            "alerts": alerts,
            "heatmap_stats": self.heatmap.get_activity_stats() if self.heatmap else {}
        }

    def _render_hud(self, frame: np.ndarray, detected_count: int, tracked_count: int, alerts: list):
        """Render modern status telemetry bar at top of frame."""
        h, w = frame.shape[:2]
        # Dark HUD top overlay
        hud_bar = frame[0:36, :].copy()
        cv2.rectangle(hud_bar, (0, 0), (w, 36), (15, 15, 18), -1)
        cv2.addWeighted(hud_bar, 0.8, frame[0:36, :], 0.2, 0, frame[0:36, :])
        
        # Telemetry text
        hud_text = f"FPS: {self.fps:.1f} | Moving Objects: {detected_count} | Active Tracks: {tracked_count}"
        cv2.putText(frame, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        
        if alerts:
            alert_text = f"! INTRUSION ALERT: {alerts[-1].zone_name} !"
            cv2.putText(frame, alert_text, (w - 300, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 255), 2, cv2.LINE_AA)
