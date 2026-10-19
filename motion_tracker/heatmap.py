"""Long-exposure motion accumulation and activity thermal heatmap."""

import cv2
import numpy as np
from typing import Optional, Tuple

class MotionHeatmap:
    def __init__(self, decay: float = 0.02, gain: float = 1.0, colormap: int = cv2.COLORMAP_TURBO):
        self.decay = decay
        self.gain = gain
        self.colormap = colormap
        self.accumulator: Optional[np.ndarray] = None

    def update(self, fg_mask: np.ndarray):
        """Update motion accumulation buffer with incoming binary motion mask."""
        if fg_mask is None or fg_mask.size == 0:
            return
            
        h, w = fg_mask.shape[:2]
        if self.accumulator is None or self.accumulator.shape != (h, w):
            self.accumulator = np.zeros((h, w), dtype=np.float32)
            
        # Decay previous activity
        self.accumulator *= (1.0 - self.decay)
        
        # Accumulate current motion
        norm_mask = (fg_mask > 0).astype(np.float32) * self.gain
        self.accumulator += norm_mask
        
        # Clip max
        np.clip(self.accumulator, 0.0, 50.0, out=self.accumulator)

    def generate_heatmap(self, base_frame: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """Render thermal colormap blended over the base video frame."""
        if self.accumulator is None or base_frame is None:
            return base_frame.copy() if base_frame is not None else np.zeros((10, 10, 3), dtype=np.uint8)
            
        h, w = base_frame.shape[:2]
        if self.accumulator.shape != (h, w):
            acc_resized = cv2.resize(self.accumulator, (w, h))
        else:
            acc_resized = self.accumulator
            
        # Normalize accumulator to 0-255
        norm_acc = cv2.normalize(acc_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Colorize
        color_heatmap = cv2.applyColorMap(norm_acc, self.colormap)
        
        # Only blend where there is recorded motion (norm_acc > 10)
        mask = norm_acc > 10
        blended = base_frame.copy()
        blended[mask] = cv2.addWeighted(base_frame, 1.0 - alpha, color_heatmap, alpha, 0)[mask]
        
        return blended

    def get_activity_stats(self, threshold: float = 0.5) -> dict:
        """Compute statistical summary of monitored area."""
        if self.accumulator is None:
            return {"active_area_percent": 0.0, "max_intensity": 0.0}
        total_pixels = self.accumulator.size
        active_pixels = int(np.sum(self.accumulator > threshold))
        return {
            "active_area_percent": round((active_pixels / max(1, total_pixels)) * 100.0, 2),
            "max_intensity": round(float(np.max(self.accumulator)), 2)
        }
