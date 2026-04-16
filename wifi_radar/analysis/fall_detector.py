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
    """Ordered severity levels for the fall detection state machine.

    ID: WR-ANALYSIS-FALL-SEV-001
    Requirement: Enumerate the four mutually exclusive states of the fall
                 detector state machine as comparable integer values.
    Purpose: Allow downstream components to compare severity levels numerically
             (e.g., severity >= FALL_DETECTED) without string comparisons.
    Rationale: IntEnum provides both human-readable names and integer comparison;
               values are monotonically increasing with severity.
    Inputs:
        N/A — class-level enumeration.
    Outputs:
        N/A — enumeration constants used by FallDetector and FallEvent.
    References:
        FallDetector._state; FallEvent.severity; WR-ANALYSIS-FALL-001.
    """


@dataclass
class FallEvent:
    """Immutable record of a detected fall event.

    ID: WR-ANALYSIS-FALL-EVENT-001
    Requirement: Store all contextual information about a fall state change so
                 that the dashboard, alert system, and audit log can record
                 a complete event without retaining the full frame buffer.
    Purpose: Decouple the fall detector from downstream consumers by providing
             a self-contained, serialisable event object.
    Rationale: dataclass with typed fields provides both runtime validation and
               IDE autocomplete; all fields are primitive types for easy JSON
               serialisation.
    Inputs:
        person_id        — int: track ID of the affected person.
        timestamp        — float: UNIX epoch seconds of detection.
        severity         — FallSeverity: one of POSSIBLE_FALL/FALL_DETECTED/ALERT.
        centroid_before  — Tuple[float,float,float]: normalised position before.
        centroid_after   — Tuple[float,float,float]: normalised position after.
        body_angle_deg   — float: torso deviation from vertical at detection.
        message          — str: human-readable description.
    Outputs:
        N/A — data container only.
    Preconditions:
        Created by FallDetector._make_event() only.
    References:
        FallDetector._make_event; FallSeverity; WR-ANALYSIS-FALL-001.
    """
    person_id:      int
    timestamp:      float            # UNIX timestamp of detection
    severity:       FallSeverity
    centroid_before: Tuple[float, float, float]   # (x, y, z) normalised
    centroid_after:  Tuple[float, float, float]
    body_angle_deg:  float           # angle of torso from vertical (0 = upright)
    message:        str = ""


class FallDetector:
    """Per-person fall detector operating on 3-D pose keypoints via a state machine.

    ID: WR-ANALYSIS-FALL-CLASS-001
    Requirement: Maintain a four-state machine (NORMAL/POSSIBLE_FALL/
                 FALL_DETECTED/ALERT) per person and emit FallEvent on transitions,
                 operating at 20 Hz without blocking the inference pipeline.
    Purpose: Provide real-time fall alerting for a single person track using
             centroid velocity, torso angle, and height-drop heuristics.
    Rationale: A state machine with configurable thresholds is more robust than a
               single-threshold trigger; requiring both velocity and angle criteria
               reduces false positives from bending/sitting motions.
    Inputs:
        Pose keypoints (17,3) and confidence (17,) per frame via update().
    Outputs:
        Optional FallEvent on state transitions; self.state for polling.
    Preconditions:
        At least 20 frames required before standing_z baseline is calibrated.
    Assumptions:
        ~20 Hz call rate; COCO-17 keypoint layout; normalised [-1,1] coordinates.
    Constraints:
        One FallDetector instance per tracked person (not shared between tracks).
    References:
        FallSeverity; FallEvent; WR-ANALYSIS-FALL-001 module docstring.
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
        """Initialise state machine, rolling history buffers, and thresholds.

        ID: WR-ANALYSIS-FALL-INIT-001
        Requirement: Store all threshold parameters, initialise deque history
                     buffers, and set initial state to FallSeverity.NORMAL.
        Purpose: Prepare the per-person fall detector for the first update() call.
        Rationale: deque with maxlen automatically evicts old frames; initialising
                   standing_z to None defers baseline calibration to the first 20
                   frames of data.
        Inputs:
            person_id           — int: unique track identifier.
            history_window      — int: frame buffer depth (default 60 ~ 3 s at 20 Hz).
            velocity_threshold  — float < 0: downward centroid velocity trigger.
            angle_threshold_deg — float > 0: torso-from-vertical trigger angle.
            height_drop_frac    — float (0,1): fraction of standing Z required to drop.
            recovery_frames     — int: consecutive upright frames for recovery.
            alert_timeout_s     — float: seconds without recovery before ALERT.
        Outputs:
            None — initialises self.
        Preconditions:
            None.
        Postconditions:
            self._state == FallSeverity.NORMAL; all history deques empty.
        Assumptions:
            update() will be called at ~20 Hz.
        Side Effects:
            Allocates three deque objects.
        Failure Modes:
            None expected at construction time.
        Error Handling:
            None required.
        Constraints:
            standing_z calibration requires 20 consecutive update() calls.
        Verification:
            Unit test: construct; assert state == NORMAL and standing_z is None.
        References:
            collections.deque; FallSeverity.NORMAL; WR-ANALYSIS-FALL-CLASS-001.
        """
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

        ID: WR-ANALYSIS-FALL-UPDATE-001
        Requirement: Compute centroid and body angle for the current frame,
                     advance the state machine, and return a FallEvent whenever
                     the severity level changes.
        Purpose: Provide the single call-per-frame API consumed by the processing
                 thread without requiring it to understand state machine internals.
        Rationale: Centralising all state transitions in update() ensures the
                   machine always receives a consistent, ordered sequence of frames.
        Inputs:
            keypoints  — (17,3) float32: normalised 3-D keypoint coordinates.
            confidence — (17,) float32: per-keypoint confidence [0,1].
            timestamp  — Optional[float]: UNIX epoch; auto-fills with time.time() if None.
        Outputs:
            Optional[FallEvent] — non-None only on severity state transitions.
        Preconditions:
            keypoints.shape == (17,3); confidence.shape == (17,).
        Postconditions:
            History deques updated; state machine potentially advanced.
        Assumptions:
            Called at ~20 Hz; COCO-17 keypoint layout.
        Side Effects:
            Updates self._centroid_z_history, self._angle_history, self._timestamps.
            May update self._state, self._fall_detected_time, self._standing_z.
        Failure Modes:
            All-low confidence: centroid falls back to zero vector; may miss fall.
        Error Handling:
            Returns None if standing_z not yet calibrated (first 20 frames).
        Constraints:
            Not thread-safe; not idempotent.
        Verification:
            Unit test: simulate step-function Z drop; assert FALL_DETECTED event returned.
        References:
            _weighted_centroid; _body_angle; _centroid_velocity; _recovery_detected;
            _make_event; WR-ANALYSIS-FALL-001.
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
        """Current fall detection state.

        ID: WR-ANALYSIS-FALL-STATE-001
        Requirement: Return the current FallSeverity state without mutating it.
        Purpose: Allow polling consumers to read the state without calling update().
        Rationale: Property prevents accidental external mutation of _state.
        Inputs:
            None.
        Outputs:
            FallSeverity — current state (NORMAL/POSSIBLE_FALL/FALL_DETECTED/ALERT).
        Preconditions:
            None.
        Side Effects:
            None — read-only.
        Verification:
            Unit test: construct; assert state == FallSeverity.NORMAL.
        References:
            FallSeverity; WR-ANALYSIS-FALL-CLASS-001.
        """

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _weighted_centroid(self, keypoints: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """Confidence-weighted centroid of valid keypoints (threshold 0.3).

        ID: WR-ANALYSIS-FALL-CENTROID-001
        Requirement: Select keypoints with confidence > 0.3 and return the
                     weighted mean position as a (3,) float32 vector.
        Purpose: Compute a robust body centre for velocity and height tracking.
        Rationale: Confidence weighting suppresses noisy low-confidence keypoints;
                   zero-vector fallback prevents NaN propagation.
        Inputs:
            keypoints  — (17,3) float32: normalised keypoint coordinates.
            confidence — (17,) float32: per-keypoint confidence [0,1].
        Outputs:
            (3,) float32 centroid; zero vector if all confidence <= 0.3.
        Preconditions:
            keypoints.shape == (17,3); confidence.shape == (17,).
        Postconditions:
            Return value has no NaN when at least one valid keypoint exists.
        Side Effects:
            None — pure function.
        Failure Modes:
            All confidence <= 0.3: returns zero vector.
        Error Handling:
            `not np.any(mask)` guard prevents division by zero.
        Verification:
            Unit test: uniform confidence 0.5; assert centroid near mean of keypoints.
        References:
            np.linalg.norm; WR-MODEL-MPT-TRACKER-CENTROID-001 (analogous).
        """
        mask = confidence > 0.3
        if not np.any(mask):
            return np.zeros(3, dtype=np.float32)
        w = confidence[mask]
        return (keypoints[mask] * w[:, None]).sum(axis=0) / w.sum()

    def _body_angle(self, keypoints: np.ndarray, confidence: np.ndarray) -> float:
        """Compute torso-from-vertical angle in degrees (0 = upright, 90 = horizontal).

        ID: WR-ANALYSIS-FALL-ANGLE-001
        Requirement: Estimate the angle between the spine vector (shoulder midpoint
                     to hip midpoint) and the vertical (Z) axis in degrees.
        Purpose: Detect horizontal/prone body orientation as a fall indicator
                 complementing the centroid velocity criterion.
        Rationale: Using shoulder and hip midpoints rather than single keypoints
                   reduces noise from asymmetric pose estimates.
        Inputs:
            keypoints  — (17,3) float32: normalised 3-D keypoint coordinates.
            confidence — (17,) float32: per-keypoint confidence [0,1].
        Outputs:
            float — torso angle in degrees [0, 90]; returns 0.0 if data unavailable.
        Preconditions:
            COCO-17 keypoints: indices 5,6 = shoulders; 11,12 = hips.
        Postconditions:
            Return value in [0.0, 90.0] (arccos mapped from [0,pi/2]).
        Assumptions:
            Shoulder and hip keypoints have confidence > 0.2 for a valid estimate.
        Side Effects:
            None — pure function.
        Failure Modes:
            Low-confidence keypoints (< 0.2): returns 0.0 (assume upright).
            Near-zero spine length: returns 0.0 to avoid division by zero.
        Error Handling:
            Confidence guard; spine_len < 1e-6 guard; np.clip to [0,1].
        Constraints:
            Only uses 4 keypoints (shoulders + hips); robust to partial occlusion.
        Verification:
            Unit test: horizontal spine (spine=[0,0,0] z-component); assert angle==90.
        References:
            COCO-17 keypoint indices; np.arccos; WR-ANALYSIS-FALL-001.
        """
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
        """Estimate instantaneous vertical velocity from the last 5 centroid-z samples.

        ID: WR-ANALYSIS-FALL-VELOCITY-001
        Requirement: Compute the finite-difference dZ/dt over the last min(5, n)
                     history samples and return the velocity in normalised units/s.
        Purpose: Detect rapid downward motion as the primary fall trigger criterion.
        Rationale: Using 5-sample finite difference is more noise-robust than a
                   single-step delta while still being sensitive to fast drops.
        Inputs:
            None — reads self._centroid_z_history and self._timestamps.
        Outputs:
            float — vertical velocity (normalised units/s); negative = downward.
        Preconditions:
            At least 2 samples in self._centroid_z_history.
        Postconditions:
            Return value is 0.0 if fewer than 2 samples exist.
        Assumptions:
            History deques are appended synchronously with consistent timestamps.
        Side Effects:
            None — read-only.
        Failure Modes:
            dt < 1e-6 (duplicate timestamps): returns 0.0.
        Error Handling:
            n<2 guard; dt<1e-6 guard.
        Constraints:
            Uses at most 5 samples regardless of history depth.
        Verification:
            Unit test: inject 5 frames with linear Z drop; assert velocity < 0.
        References:
            Finite-difference velocity estimation; WR-ANALYSIS-FALL-001.
        """
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
        """Return True when the body has been upright for self.recovery_frames frames.

        ID: WR-ANALYSIS-FALL-RECOVERY-001
        Requirement: Increment self._standing_frames when centroid Z and body angle
                     satisfy near-standing criteria; reset on failure; return True
                     when standing_frames >= recovery_frames.
        Purpose: Prevent immediate state machine reset after a single upright frame,
                 requiring sustained recovery before declaring the person safe.
        Rationale: Requiring multiple consecutive upright frames avoids false
                   recovery detection from transient upright postures during a fall.
        Inputs:
            centroid — (3,) float32: current normalised body centroid.
            angle    — float: current torso-from-vertical angle in degrees.
        Outputs:
            bool — True only when standing_frames >= recovery_frames.
        Preconditions:
            self._standing_z must be calibrated (not None).
        Postconditions:
            self._standing_frames incremented or reset based on criteria.
        Assumptions:
            Called only while state is FALL_DETECTED or ALERT.
        Side Effects:
            Updates self._standing_frames.
        Failure Modes:
            standing_z is None: uses 0.0 as fallback; may give wrong z_ok result.
        Error Handling:
            None; assumes standing_z is set (caller guards with >= 20 frames).
        Constraints:
            recovery_frames must be > 0.
        Verification:
            Unit test: simulate N upright frames; assert True on frame N.
        References:
            WR-ANALYSIS-FALL-CLASS-001 state machine; self.recovery_frames.
        """
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
        """Construct and return a FallEvent for the current state transition.

        ID: WR-ANALYSIS-FALL-MKEVENT-001
        Requirement: Build a FallEvent from the current state, centroids, and angle
                     using self._state to select the severity and message.
        Purpose: Centralise FallEvent construction so all callers in update() use
                 a consistent format without duplicating field assignments.
        Rationale: A dedicated factory method isolates FallEvent construction details
                   from the state machine logic in update().
        Inputs:
            centroid_before — (3,) or tuple: position before the event.
            centroid_after  — (3,) or tuple: position at event detection.
            angle           — float: body angle at detection time (degrees).
        Outputs:
            FallEvent — fully populated event record with current timestamp.
        Preconditions:
            self._state must be one of POSSIBLE_FALL/FALL_DETECTED/ALERT.
        Postconditions:
            Returned FallEvent.severity matches self._state.
        Assumptions:
            centroid_before and centroid_after are 3-element sequences.
        Side Effects:
            Calls time.time() for the event timestamp.
        Failure Modes:
            Unknown severity: FallEvent.message defaults to empty string.
        Error Handling:
            dict.get() with default '' for unknown severity labels.
        Constraints:
            None.
        Verification:
            Unit test: set state=FALL_DETECTED; call _make_event; assert severity==FALL_DETECTED.
        References:
            FallEvent; FallSeverity; WR-ANALYSIS-FALL-001.
        """
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
