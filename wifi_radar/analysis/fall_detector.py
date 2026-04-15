"""
ID: WR-ANALYSIS-FALL-001
Purpose: Detect falls from time-series 3-D pose keypoints using a velocity +
         body-angle state machine.  Suitable for real-time alerting at 20 Hz.

Keypoint convention (COCO 17-point):
    0  nose          5  left_shoulder   10 right_wrist   15 left_ankle
    1  left_eye      6  right_shoulder  11 left_hip      16 right_ankle
    2  right_eye     7  left_elbow      12 right_hip
    3  left_ear      8  right_elbow     13 left_knee
    4  right_ear     9  left_wrist      14 right_knee
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np

# Keypoint indices used for fall analysis
_LEFT_SHOULDER  = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP       = 11
_RIGHT_HIP      = 12

logger = logging.getLogger(__name__)


class FallSeverity(IntEnum):
    NORMAL         = 0   # No anomaly detected
    POSSIBLE_FALL  = 1   # Velocity + angle criteria partially met
    FALL_DETECTED  = 2   # Height drop confirmed
    ALERT          = 3   # No recovery after alert_timeout_s seconds


@dataclass
class FallEvent:
    """Immutable record of a detected fall event."""
    person_id:      int
    timestamp:      float            # UNIX timestamp of detection
    severity:       FallSeverity
    centroid_before: Tuple[float, float, float]   # (x, y, z) normalised
    centroid_after:  Tuple[float, float, float]
    body_angle_deg:  float           # angle of torso from vertical (0 = upright)
    message:        str = ""


class FallDetector:
    """Per-person fall detector operating on 3-D pose keypoints.

    State machine transitions::

        NORMAL ──[velocity↓ & angle↑]──► POSSIBLE_FALL
               ──[height drop confirmed]──► FALL_DETECTED
               ──[no recovery within alert_timeout_s]──► ALERT
               ──[recovery]──────────────────────────────► NORMAL

    Args:
        person_id:            Unique ID for the tracked person.
        history_window:       Number of frames to retain (default 60 = ~3 s at 20 Hz).
        velocity_threshold:   Minimum downward centroid velocity to trigger (negative, m/s normalised).
        angle_threshold_deg:  Body-from-vertical angle that triggers possible-fall.
        height_drop_frac:     Required fractional Z-drop relative to standing height.
        recovery_frames:      Frames the body must be upright to declare recovery.
        alert_timeout_s:      Seconds with no recovery before escalating to ALERT.
    """

    def __init__(
        self,
        person_id: int = 0,
        history_window: int = 60,
        velocity_threshold: float = -0.20,
        angle_threshold_deg: float = 40.0,
        height_drop_frac: float = 0.35,
        recovery_frames: int = 15,
        alert_timeout_s: float = 5.0,
    ) -> None:
        self.person_id = person_id
        self.velocity_threshold  = velocity_threshold
        self.angle_threshold_deg = angle_threshold_deg
        self.height_drop_frac    = height_drop_frac
        self.recovery_frames     = recovery_frames
        self.alert_timeout_s     = alert_timeout_s

        self._centroid_z_history: deque = deque(maxlen=history_window)
        self._angle_history:      deque = deque(maxlen=history_window)
        self._timestamps:         deque = deque(maxlen=history_window)

        self._state = FallSeverity.NORMAL
        self._fall_detected_time: Optional[float] = None
        self._standing_z: Optional[float] = None   # baseline standing centroid z
        self._standing_frames = 0
        self._pending_event: Optional[FallEvent] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self,
        keypoints: np.ndarray,
        confidence: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Optional[FallEvent]:
        """Process one frame and return a FallEvent if the state changes.

        Args:
            keypoints:  (17, 3) array of normalised 3-D keypoint coordinates.
            confidence: (17,)  array of per-keypoint confidence scores [0, 1].
            timestamp:  UNIX time.  Uses ``time.time()`` when None.

        Returns:
            A :class:`FallEvent` when the severity level changes, else ``None``.
        """
        ts = timestamp if timestamp is not None else time.time()

        centroid = self._weighted_centroid(keypoints, confidence)
        angle    = self._body_angle(keypoints, confidence)

        self._centroid_z_history.append(centroid[2])
        self._angle_history.append(angle)
        self._timestamps.append(ts)

        # Calibrate standing z on first N frames
        if self._standing_z is None:
            if len(self._centroid_z_history) >= 20:
                self._standing_z = float(np.median(list(self._centroid_z_history)))
            return None

        velocity = self._centroid_velocity()
        event: Optional[FallEvent] = None

        if self._state == FallSeverity.NORMAL:
            if velocity < self.velocity_threshold and angle > self.angle_threshold_deg:
                self._state = FallSeverity.POSSIBLE_FALL
                event = self._make_event(centroid, centroid, angle)
                logger.debug("Person %d: POSSIBLE_FALL (vel=%.3f, angle=%.1f°)",
                             self.person_id, velocity, angle)

        elif self._state == FallSeverity.POSSIBLE_FALL:
            height_drop = (self._standing_z - centroid[2]) / max(abs(self._standing_z), 0.01)
            if height_drop >= self.height_drop_frac:
                self._state = FallSeverity.FALL_DETECTED
                self._fall_detected_time = ts
                event = self._make_event(
                    (0.0, 0.0, self._standing_z), centroid, angle
                )
                logger.warning("Person %d: FALL_DETECTED (drop=%.0f%%)", self.person_id, height_drop * 100)
            elif velocity > abs(self.velocity_threshold) * 0.5 and angle < self.angle_threshold_deg * 0.6:
                # Quick recovery — false positive
                self._state = FallSeverity.NORMAL

        elif self._state == FallSeverity.FALL_DETECTED:
            elapsed = ts - (self._fall_detected_time or ts)
            if elapsed > self.alert_timeout_s:
                self._state = FallSeverity.ALERT
                event = self._make_event(
                    (0.0, 0.0, self._standing_z), centroid, angle
                )
                logger.error("Person %d: ALERT — no recovery after %.0f s",
                             self.person_id, elapsed)
            elif self._recovery_detected(centroid, angle):
                self._state = FallSeverity.NORMAL
                self._fall_detected_time = None
                logger.info("Person %d: recovered from fall", self.person_id)

        elif self._state == FallSeverity.ALERT:
            if self._recovery_detected(centroid, angle):
                self._state = FallSeverity.NORMAL
                self._fall_detected_time = None
                logger.info("Person %d: recovered from ALERT state", self.person_id)

        return event

    @property
    def state(self) -> FallSeverity:
        return self._state

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _weighted_centroid(self, keypoints: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """Confidence-weighted centroid of valid keypoints."""
        mask = confidence > 0.3
        if not np.any(mask):
            return np.zeros(3, dtype=np.float32)
        w = confidence[mask]
        return (keypoints[mask] * w[:, None]).sum(axis=0) / w.sum()

    def _body_angle(self, keypoints: np.ndarray, confidence: np.ndarray) -> float:
        """Angle (degrees) of spine vector from vertical [0 = upright, 90 = horizontal]."""
        idx_top = [_LEFT_SHOULDER, _RIGHT_SHOULDER]
        idx_bot = [_LEFT_HIP, _RIGHT_HIP]

        conf_top = confidence[idx_top]
        conf_bot = confidence[idx_bot]

        if conf_top.min() < 0.2 or conf_bot.min() < 0.2:
            # Not enough data — assume upright
            return 0.0

        top = keypoints[idx_top].mean(axis=0)
        bot = keypoints[idx_bot].mean(axis=0)
        spine = top - bot

        # Angle from Z-axis (vertical)
        spine_len = np.linalg.norm(spine)
        if spine_len < 1e-6:
            return 0.0

        cos_theta = abs(spine[2]) / spine_len
        cos_theta = np.clip(cos_theta, 0.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def _centroid_velocity(self) -> float:
        """Estimate instantaneous vertical velocity from last 5 centroid-z samples."""
        hist = list(self._centroid_z_history)
        times = list(self._timestamps)
        n = min(5, len(hist))
        if n < 2:
            return 0.0
        dz = hist[-1] - hist[-n]
        dt = times[-1] - times[-n]
        if dt < 1e-6:
            return 0.0
        return dz / dt

    def _recovery_detected(self, centroid: np.ndarray, angle: float) -> bool:
        """True when body has returned to near-standing height and angle for enough frames."""
        standing_z = self._standing_z or 0.0
        z_ok = centroid[2] > standing_z - abs(standing_z) * self.height_drop_frac * 0.5
        angle_ok = angle < self.angle_threshold_deg * 0.5
        if z_ok and angle_ok:
            self._standing_frames += 1
        else:
            self._standing_frames = 0
        return self._standing_frames >= self.recovery_frames

    def _make_event(
        self,
        centroid_before: np.ndarray,
        centroid_after: np.ndarray,
        angle: float,
    ) -> FallEvent:
        severity_labels = {
            FallSeverity.POSSIBLE_FALL: "Possible fall",
            FallSeverity.FALL_DETECTED: "Fall detected",
            FallSeverity.ALERT:         "ALERT — no recovery",
        }
        return FallEvent(
            person_id=self.person_id,
            timestamp=time.time(),
            severity=self._state,
            centroid_before=tuple(centroid_before),
            centroid_after=tuple(centroid_after),
            body_angle_deg=angle,
            message=severity_labels.get(self._state, ""),
        )
