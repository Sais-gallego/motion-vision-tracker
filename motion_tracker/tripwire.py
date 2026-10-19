"""Virtual tripwires and restricted zone intrusion detection."""

import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class SecurityZone:
    name: str
    polygon: np.ndarray  # Shape (N, 2)
    color_normal: Tuple[int, int, int] = (0, 255, 0)
    color_alarm: Tuple[int, int, int] = (0, 0, 255)
    is_breached: bool = False
    last_alarm_time: float = 0.0

@dataclass
class SecurityAlert:
    zone_name: str
    target_id: int
    centroid: Tuple[int, int]
    timestamp: float

class TripwireEngine:
    def __init__(self, alarm_cooldown: float = 1.5):
        self.zones: Dict[str, SecurityZone] = {}
        self.alarm_cooldown = alarm_cooldown
        self.active_alerts: List[SecurityAlert] = []

    def add_zone(self, name: str, polygon_points: List[Tuple[int, int]]) -> SecurityZone:
        """Add a polygonal security zone or tripwire boundary."""
        poly = np.array(polygon_points, dtype=np.int32)
        zone = SecurityZone(name=name, polygon=poly)
        self.zones[name] = zone
        return zone

    def check_intrusions(self, tracked_targets: dict) -> List[SecurityAlert]:
        """Check if any tracked centroids have penetrated security zones."""
        now = time.time()
        new_alerts: List[SecurityAlert] = []
        
        for name, zone in self.zones.items():
            breached = False
            for tid, target in tracked_targets.items():
                if len(target.history) == 0:
                    continue
                pt = target.history[-1]
                # pointPolygonTest returns >= 0 if inside or on contour
                inside = cv2.pointPolygonTest(zone.polygon, (float(pt[0]), float(pt[1])), False) >= 0
                if inside:
                    breached = True
                    if now - zone.last_alarm_time > self.alarm_cooldown:
                        zone.last_alarm_time = now
                        alert = SecurityAlert(
                            zone_name=name,
                            target_id=tid,
                            centroid=pt,
                            timestamp=now
                        )
                        new_alerts.append(alert)
                        self.active_alerts.append(alert)
            zone.is_breached = breached

        # Keep last 20 alerts
        self.active_alerts = self.active_alerts[-20:]
        return new_alerts

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Render security zones with visual alarm feedback."""
        out_frame = frame.copy()
        
        for name, zone in self.zones.items():
            color = zone.color_alarm if zone.is_breached else zone.color_normal
            thickness = 3 if zone.is_breached else 2
            
            # Draw polygon perimeter
            cv2.polylines(out_frame, [zone.polygon], isClosed=True, color=color, thickness=thickness)
            
            # Semi-transparent fill
            fill_overlay = out_frame.copy()
            cv2.fillPoly(fill_overlay, [zone.polygon], color)
            alpha = 0.25 if zone.is_breached else 0.08
            cv2.addWeighted(fill_overlay, alpha, out_frame, 1.0 - alpha, 0, out_frame)
            
            # Label
            first_pt = tuple(zone.polygon[0])
            status_text = f"ZONE: {name} [BREACH!]" if zone.is_breached else f"ZONE: {name} [SECURE]"
            cv2.putText(
                out_frame, status_text,
                (first_pt[0], max(20, first_pt[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA
            )
            
        return out_frame
