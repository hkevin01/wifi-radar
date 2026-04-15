"""
ID: WR-ANALYSIS-GAIT-001
Purpose: Extract quantitative gait metrics (cadence, stride length, step
         symmetry, walking speed) from time-series ankle keypoints estimated
         via WiFi-CSI pose inference.

Algorithm:
  1. Maintain a rolling window of left/right ankle z-coordinates.
  2. Detect step events as local minima in the ankle trajectory (foot-strike).
  3. Compute cadence, stride length proxy and left-right symmetry.
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

# COCO ankle indices
_LEFT_ANKLE  = 15
_RIGHT_ANKLE = 16
_LEFT_HIP    = 11
_RIGHT_HIP   = 12

logger = logging.getLogger(__name__)


@dataclass
class StepEvent:
    """A single detected foot-strike."""
    foot:      str            # "left" or "right"
    timestamp: float          # UNIX time of strike
    position:  Tuple[float, float, float]  # ankle (x,y,z) at strike
    height:    float          # ankle z at time of strike (lower = foot down)


@dataclass
class GaitMetrics:
    """Snapshot of current gait characteristics."""
    cadence_spm:    float   # steps per minute
    stride_length:  float   # normalised units — distance between ipsilateral strikes
    step_symmetry:  float   # ratio left_step_time / right_step_time  (1.0 = perfect)
    speed_est:      float   # estimated walking speed (normalised units/s)
    num_steps:      int     # total steps counted in the current window
    window_s:       float   # duration of measurement window in seconds


class GaitAnalyzer:
    """Accumulates pose frames and yields :class:`GaitMetrics` on demand.

    Args:
        history_seconds: Size of the rolling analysis window (default 10 s).
        fps:             Expected frame rate; used to set peak-detection spacing.
        min_steps:       Minimum step count required before metrics are returned.
        confidence_thr:  Minimum ankle keypoint confidence to include a sample.
    """

    def __init__(
        self,
        history_seconds: float = 10.0,
        fps: float = 20.0,
        min_steps: int = 4,
        confidence_thr: float = 0.25,
    ) -> None:
        self._history_s    = history_seconds
        self._fps          = fps
        self._min_steps    = min_steps
        self._conf_thr     = confidence_thr
        self._max_frames   = int(history_seconds * fps)

        # Per-ankle time-series: [(timestamp, x, y, z)]
        self._left_ankle:  Deque[Tuple[float, float, float, float]] = deque(maxlen=self._max_frames)
        self._right_ankle: Deque[Tuple[float, float, float, float]] = deque(maxlen=self._max_frames)

        # Hip midpoint for speed estimation
        self._hip_x: Deque[Tuple[float, float]] = deque(maxlen=self._max_frames)

        self._step_events: List[StepEvent] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> None:
        """Ingest one pose frame.

        Args:
            keypoints:  (17, 3) normalised 3-D coordinates.
            confidence: (17,) confidence scores.
            timestamp:  UNIX time; defaults to ``time.time()``.
        """
        ts = timestamp if timestamp is not None else time.time()

        if confidence[_LEFT_ANKLE] >= self._conf_thr:
            kp = keypoints[_LEFT_ANKLE]
            self._left_ankle.append((ts, float(kp[0]), float(kp[1]), float(kp[2])))

        if confidence[_RIGHT_ANKLE] >= self._conf_thr:
            kp = keypoints[_RIGHT_ANKLE]
            self._right_ankle.append((ts, float(kp[0]), float(kp[1]), float(kp[2])))

        # Hip midpoint for speed
        if confidence[_LEFT_HIP] >= self._conf_thr and confidence[_RIGHT_HIP] >= self._conf_thr:
            hip_x = (keypoints[_LEFT_HIP][0] + keypoints[_RIGHT_HIP][0]) / 2.0
            self._hip_x.append((ts, float(hip_x)))

    def get_metrics(self) -> Optional[GaitMetrics]:
        """Compute and return current gait metrics, or None if insufficient data.

        Returns:
            :class:`GaitMetrics` if at least ``min_steps`` have been detected,
            ``None`` otherwise.
        """
        left_steps  = self._detect_steps(list(self._left_ankle),  "left")
        right_steps = self._detect_steps(list(self._right_ankle), "right")

        all_steps = sorted(left_steps + right_steps, key=lambda e: e.timestamp)
        if len(all_steps) < self._min_steps:
            return None

        window_s = all_steps[-1].timestamp - all_steps[0].timestamp
        if window_s < 0.5:
            return None

        cadence_spm = (len(all_steps) / window_s) * 60.0

        stride_length = self._stride_length(left_steps)

        step_symmetry = self._step_symmetry(left_steps, right_steps)

        speed_est = self._walking_speed()

        return GaitMetrics(
            cadence_spm=round(cadence_spm, 1),
            stride_length=round(stride_length, 3),
            step_symmetry=round(step_symmetry, 3),
            speed_est=round(speed_est, 3),
            num_steps=len(all_steps),
            window_s=round(window_s, 1),
        )

    def reset(self) -> None:
        """Clear all accumulated history."""
        self._left_ankle.clear()
        self._right_ankle.clear()
        self._hip_x.clear()
        self._step_events.clear()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _detect_steps(
        self, samples: List[Tuple[float, float, float, float]], foot: str
    ) -> List[StepEvent]:
        """Detect foot-strikes as local minima in ankle z (lowest point = contact)."""
        if len(samples) < 6:
            return []

        z_arr = np.array([s[3] for s in samples], dtype=float)
        ts_arr = np.array([s[0] for s in samples], dtype=float)

        # Invert z so foot-strikes (lowest z) become peaks
        # min_distance: at least 0.3 s between strikes (~100 bpm max)
        min_dist = max(3, int(0.3 * self._fps))
        peaks, props = find_peaks(-z_arr, distance=min_dist, prominence=0.02)

        events: List[StepEvent] = []
        for p in peaks:
            s = samples[p]
            events.append(
                StepEvent(
                    foot=foot,
                    timestamp=s[0],
                    position=(s[1], s[2], s[3]),
                    height=s[3],
                )
            )
        return events

    def _stride_length(self, ipsilateral_steps: List[StepEvent]) -> float:
        """Estimate stride length as mean distance between consecutive same-foot strikes."""
        if len(ipsilateral_steps) < 2:
            return 0.0
        dists = []
        for i in range(1, len(ipsilateral_steps)):
            p1 = np.array(ipsilateral_steps[i - 1].position[:2])  # x, y only
            p2 = np.array(ipsilateral_steps[i].position[:2])
            dists.append(float(np.linalg.norm(p2 - p1)))
        return float(np.mean(dists)) if dists else 0.0

    def _step_symmetry(
        self,
        left_steps:  List[StepEvent],
        right_steps: List[StepEvent],
    ) -> float:
        """Ratio of mean left inter-step interval to mean right inter-step interval."""
        def mean_interval(steps: List[StepEvent]) -> Optional[float]:
            if len(steps) < 2:
                return None
            intervals = [steps[i].timestamp - steps[i - 1].timestamp
                         for i in range(1, len(steps))]
            pos = [x for x in intervals if x > 0]
            return float(np.mean(pos)) if pos else None

        li = mean_interval(left_steps)
        ri = mean_interval(right_steps)
        if li is None or ri is None or ri < 1e-6:
            return 1.0   # unknown → assume symmetric
        return round(min(li / ri, ri / li), 3)   # keep in (0, 1]: 1.0 = perfectly symmetric

    def _walking_speed(self) -> float:
        """Estimate speed as the rate of change of hip X-position."""
        if len(self._hip_x) < 2:
            return 0.0
        hip = list(self._hip_x)
        ts  = np.array([h[0] for h in hip])
        x   = np.array([h[1] for h in hip])
        dt  = ts[-1] - ts[0]
        if dt < 0.5:
            return 0.0
        # Fit a linear trend; slope ≈ walking speed
        if len(ts) > 3:
            coeffs = np.polyfit(ts - ts[0], x, 1)
            return float(abs(coeffs[0]))
        return float(abs(x[-1] - x[0]) / dt)
