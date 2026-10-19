"""Motion detection and background subtraction module."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class DetectedObject:
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[int, int]       # (cx, cy)
    area: float
    contour: np.ndarray

class MotionDetector:
    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 25.0,
        detect_shadows: bool = True,
        min_area: int = 500,
        max_area: Optional[int] = None
    ):
        self.min_area = min_area
        self.max_area = max_area
        
        # Check if MOG2 is available, otherwise use frame difference
        self.subtractor = None
        if hasattr(cv2, "createBackgroundSubtractorMOG2"):
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )
        self.prev_gray = None

    def detect(self, frame: np.ndarray) -> Tuple[List[DetectedObject], np.ndarray]:
        """Process frame and return list of detected moving objects and binary motion mask."""
        if frame is None or frame.size == 0:
            return [], np.zeros((10, 10), dtype=np.uint8)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        if self.subtractor is not None:
            fg_mask = self.subtractor.apply(gray_blurred)
            # Threshold out shadows (value 127 in MOG2)
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            if self.prev_gray is None:
                self.prev_gray = gray_blurred
                return [], np.zeros_like(gray)
            diff = cv2.absdiff(self.prev_gray, gray_blurred)
            _, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            self.prev_gray = gray_blurred

        # Morphological noise filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Find contours of moving blobs
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_objects: List[DetectedObject] = []
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
                
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            
            detected_objects.append(DetectedObject(
                box=(x, y, w, h),
                centroid=(cx, cy),
                area=area,
                contour=c
            ))
            
        return detected_objects, fg_mask
