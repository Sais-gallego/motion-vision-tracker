"""Motion Vision Tracker package."""

from .detector import MotionDetector, DetectedObject
from .optical_flow import OpticalFlowTracker
from .trajectory import TrajectoryTracker, TrackedTarget
from .heatmap import MotionHeatmap
from .tripwire import TripwireEngine, SecurityZone, SecurityAlert
from .simulator import SyntheticVideoGenerator
from .engine import MotionTrackingEngine

__all__ = [
    "MotionDetector",
    "DetectedObject",
    "OpticalFlowTracker",
    "TrajectoryTracker",
    "TrackedTarget",
    "MotionHeatmap",
    "TripwireEngine",
    "SecurityZone",
    "SecurityAlert",
    "SyntheticVideoGenerator",
    "MotionTrackingEngine"
]
__version__ = "1.0.0"
