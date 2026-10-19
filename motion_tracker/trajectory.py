"""Multi-object trajectory tracing and decaying path visualization."""

import math
import cv2
import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class TrackedTarget:
    def __init__(self, target_id: int, centroid: Tuple[int, int], max_history: int = 30):
        self.target_id = target_id
        self.history: deque = deque(maxlen=max_history)
        self.history.append(centroid)
        self.disappeared_frames = 0
        self.color = self._generate_color(target_id)
        self.velocity = 0.0

    @staticmethod
    def _generate_color(seed: int) -> Tuple[int, int, int]:
        np.random.seed(seed)
        c = np.random.randint(50, 255, size=3)
        return (int(c[0]), int(c[1]), int(c[2]))

    def update(self, new_centroid: Tuple[int, int]):
        if len(self.history) > 0:
            last = self.history[-1]
            dist = math.hypot(new_centroid[0] - last[0], new_centroid[1] - last[1])
            self.velocity = 0.7 * self.velocity + 0.3 * dist
        self.history.append(new_centroid)
        self.disappeared_frames = 0

class TrajectoryTracker:
    def __init__(self, max_distance: float = 60.0, max_disappeared: int = 15, max_history: int = 30):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.max_history = max_history
        self.next_id = 1
        self.targets: Dict[int, TrackedTarget] = {}

    def update(self, detected_centroids: List[Tuple[int, int]]) -> Dict[int, TrackedTarget]:
        """Match detected centroids to existing targets or register new ones."""
        if len(detected_centroids) == 0:
            for tid in list(self.targets.keys()):
                self.targets[tid].disappeared_frames += 1
                if self.targets[tid].disappeared_frames > self.max_disappeared:
                    del self.targets[tid]
            return self.targets

        if len(self.targets) == 0:
            for pt in detected_centroids:
                self.targets[self.next_id] = TrackedTarget(self.next_id, pt, self.max_history)
                self.next_id += 1
            return self.targets

        # Distance matrix
        target_ids = list(self.targets.keys())
        target_pts = [self.targets[tid].history[-1] for tid in target_ids]
        
        # Greedy assignment based on smallest distance
        used_centroids = set()
        used_targets = set()
        
        distances = []
        for ti, t_pt in enumerate(target_pts):
            for ci, c_pt in enumerate(detected_centroids):
                d = math.hypot(t_pt[0] - c_pt[0], t_pt[1] - c_pt[1])
                distances.append((d, ti, ci))
                
        distances.sort(key=lambda x: x[0])
        
        for d, ti, ci in distances:
            if d > self.max_distance:
                break
            if ti in used_targets or ci in used_centroids:
                continue
            used_targets.add(ti)
            used_centroids.add(ci)
            tid = target_ids[ti]
            self.targets[tid].update(detected_centroids[ci])

        # Handle unmatched existing targets
        for ti, tid in enumerate(target_ids):
            if ti not in used_targets:
                self.targets[tid].disappeared_frames += 1
                if self.targets[tid].disappeared_frames > self.max_disappeared:
                    del self.targets[tid]

        # Register new targets for unmatched centroids
        for ci, c_pt in enumerate(detected_centroids):
            if ci not in used_centroids:
                self.targets[self.next_id] = TrackedTarget(self.next_id, c_pt, self.max_history)
                self.next_id += 1

        return self.targets

    def draw_trails(self, frame: np.ndarray) -> np.ndarray:
        """Render fading motion glow trails for all tracked objects."""
        out_frame = frame.copy()
        
        for tid, target in self.targets.items():
            pts = list(target.history)
            if len(pts) < 2:
                continue
                
            n = len(pts)
            for i in range(1, n):
                # Fade thickness and intensity
                alpha = float(i) / float(n)
                thickness = max(1, int(4 * alpha))
                color = [int(c * alpha) for c in target.color]
                cv2.line(out_frame, pts[i - 1], pts[i], color, thickness)
                
            # Current centroid
            curr = pts[-1]
            cv2.circle(out_frame, curr, 5, target.color, -1)
            cv2.putText(
                out_frame,
                f"ID {tid} ({target.velocity:.1f}px/f)",
                (curr[0] + 8, curr[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, target.color, 1, cv2.LINE_AA
            )
            
        return out_frame
