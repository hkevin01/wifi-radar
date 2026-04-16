"""
ID: WR-PKG-ANALYSIS-001
Requirement: Expose FallDetector, FallEvent, FallSeverity, GaitAnalyzer,
             GaitMetrics, and StepEvent for import by the processing pipeline.
Purpose: Group fall detection and gait analysis under a single importable namespace
         so the main processing thread has a clean import path.
Rationale: Separating analysis logic from model inference allows independent
           testing and replacement of detection algorithms.
Assumptions: numpy and scipy are installed; pose keypoints are COCO-17 format.
References: COCO 17-point skeleton; FallDetector, GaitAnalyzer in this package.

wifi_radar.analysis
~~~~~~~~~~~~~~~~~~~
Fall detection and gait analysis over tracked pose sequences.
"""
from .fall_detector import FallDetector, FallEvent, FallSeverity
from .gait_analyzer import GaitAnalyzer, GaitMetrics, StepEvent

__all__ = [
    "FallDetector", "FallEvent", "FallSeverity",
    "GaitAnalyzer", "GaitMetrics", "StepEvent",
]
