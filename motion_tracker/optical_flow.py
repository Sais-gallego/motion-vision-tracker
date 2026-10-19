"""Dense and sparse optical flow velocity vector analysis."""

import cv2
import numpy as np
from typing import Tuple, Optional

class OpticalFlowTracker:
    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15, iterations: int = 3):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.prev_gray: Optional[np.ndarray] = None
        self.hsv_mask: Optional[np.ndarray] = None

    def compute_dense_flow(self, frame: np.ndarray, blend_weight: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Farneback dense optical flow and return HSV visualization and raw flow vectors."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]
        
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.hsv_mask = np.zeros((h, w, 3), dtype=np.uint8)
            self.hsv_mask[..., 1] = 255
            empty_flow = np.zeros((h, w, 2), dtype=np.float32)
            return frame.copy(), empty_flow

        # Farneback optical flow calculation
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        self.prev_gray = gray

        # Compute magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Map to HSV color-wheel
        self.hsv_mask[..., 0] = ang * 180 / np.pi / 2
        self.hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        bgr_flow = cv2.cvtColor(self.hsv_mask, cv2.COLOR_HSV2BGR)
        
        # Blend with original frame for spatial awareness
        blended = cv2.addWeighted(frame, 1.0 - blend_weight, bgr_flow, blend_weight, 0)
        return blended, flow

    def draw_flow_vectors(self, frame: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
        """Render a sparse grid of motion direction arrows over the frame."""
        h, w = frame.shape[:2]
        out_frame = frame.copy()
        y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        
        # Filter insignificant motion
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        
        for (x1, y1), (x2, y2) in lines:
            length = np.hypot(x2 - x1, y2 - y1)
            if length > 2.0:
                cv2.arrowedLine(out_frame, (x1, y1), (x2, y2), (0, 255, 255), 1, tipLength=0.3)
                
        return out_frame
