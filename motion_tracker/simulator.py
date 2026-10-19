"""Synthetic video stream generator with moving targets for testing and demos."""

import cv2
import numpy as np
from typing import List, Tuple

class SyntheticMovingTarget:
    def __init__(self, x: float, y: float, vx: float, vy: float, radius: int, color: Tuple[int, int, int]):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color

    def update(self, bounds_w: int, bounds_h: int):
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off boundaries
        if self.x - self.radius < 0 or self.x + self.radius >= bounds_w:
            self.vx = -self.vx
            self.x = max(self.radius, min(bounds_w - self.radius, self.x))
        if self.y - self.radius < 0 or self.y + self.radius >= bounds_h:
            self.vy = -self.vy
            self.y = max(self.radius, min(bounds_h - self.radius, self.y))

class SyntheticVideoGenerator:
    def __init__(self, width: int = 640, height: int = 480, num_targets: int = 3):
        self.width = width
        self.height = height
        self.targets: List[SyntheticMovingTarget] = []
        
        np.random.seed(42)
        colors = [
            (50, 120, 255),   # Orange-ish BGR
            (255, 100, 50),   # Cyan-ish BGR
            (100, 255, 100)   # Green-ish BGR
        ]
        
        for i in range(num_targets):
            x = float(np.random.randint(100, width - 100))
            y = float(np.random.randint(100, height - 100))
            vx = float(np.random.choice([-4, -3, 3, 4]))
            vy = float(np.random.choice([-3, -2, 2, 3]))
            radius = int(np.random.randint(20, 35))
            color = colors[i % len(colors)]
            self.targets.append(SyntheticMovingTarget(x, y, vx, vy, radius, color))

    def next_frame(self) -> np.ndarray:
        """Render next synthetic frame with moving agents."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (35, 30, 25)  # Dark room environment
        
        # Static background elements (e.g. wall pillars)
        cv2.rectangle(frame, (50, 50), (120, 430), (50, 45, 40), -1)
        cv2.rectangle(frame, (self.width - 120, 50), (self.width - 50, 430), (50, 45, 40), -1)
        
        # Render targets and update physics
        for target in self.targets:
            target.update(self.width, self.height)
            center = (int(target.x), int(target.y))
            cv2.circle(frame, center, target.radius, target.color, -1)
            # Core highlight
            cv2.circle(frame, center, target.radius // 3, (255, 255, 255), -1)
            
        # Add subtle camera sensor grain
        noise = np.random.normal(0, 2, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame
