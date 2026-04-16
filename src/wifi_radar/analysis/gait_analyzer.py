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
    """A single detected foot-strike event.

    ID: WR-ANALYSIS-GAIT-STEP-001
    Requirement: Record the foot side, timestamp, 3-D ankle position, and
                 ankle height for a single detected foot-strike.
    Purpose: Provide downstream gait metrics (cadence, symmetry, stride)
             with a consistent, typed event record.
    Rationale: dataclass with typed fields allows easy sorting by timestamp
               and safe field access without dict key errors.
    Inputs:
        foot      — str: 'left' or 'right'.
        timestamp — float: UNIX epoch of the detected foot-strike.
        position  — Tuple[float,float,float]: ankle (x,y,z) at strike.
        height    — float: ankle z at strike (lower = foot on ground).
    Outputs:
        N/A — data container only.
    Preconditions:
        Created by GaitAnalyzer._detect_steps() only.
    References:
        GaitAnalyzer._detect_steps; COCO ankle indices 15,16.
    """
    foot:      str            # "left" or "right"
    timestamp: float          # UNIX time of strike
    position:  Tuple[float, float, float]  # ankle (x,y,z) at strike
    height:    float          # ankle z at time of strike (lower = foot down)


@dataclass
class GaitMetrics:
    """Snapshot of current gait characteristics.

    ID: WR-ANALYSIS-GAIT-METRICS-001
    Requirement: Store all computed gait parameters for a measurement window
                 as a self-contained, serialisable snapshot.
    Purpose: Provide a single return type for GaitAnalyzer.get_metrics() so
             callers do not need to know which internal buffers were used.
    Rationale: dataclass avoids OrderedDict fragility; float fields with
               controlled precision reduce downstream formatting burden.
    Inputs:
        cadence_spm   — float: steps per minute.
        stride_length — float: normalised distance between ipsilateral strikes.
        step_symmetry — float (0,1]: ratio of left/right step intervals.
        speed_est     — float: estimated walking speed (normalised units/s).
        num_steps     — int: total steps in the measurement window.
        window_s      — float: measurement window duration in seconds.
    Outputs:
        N/A — data container only.
    Preconditions:
        Created by GaitAnalyzer.get_metrics() only.
    References:
        GaitAnalyzer.get_metrics; WR-ANALYSIS-GAIT-001.
    """
    cadence_spm:    float   # steps per minute
    stride_length:  float   # normalised units — distance between ipsilateral strikes
    step_symmetry:  float   # ratio left_step_time / right_step_time  (1.0 = perfect)
    speed_est:      float   # estimated walking speed (normalised units/s)
    num_steps:      int     # total steps counted in the current window
    window_s:       float   # duration of measurement window in seconds


class GaitAnalyzer:
    """Accumulates pose frames and yields GaitMetrics on demand.

    ID: WR-ANALYSIS-GAIT-CLASS-001
    Requirement: Maintain rolling ankle and hip position buffers, detect
                 foot-strike step events, and return GaitMetrics when at least
                 min_steps have been detected in the rolling window.
    Purpose: Extract quantitative gait parameters from 20 Hz pose inference
             output for clinical and fall-risk assessment.
    Rationale: Rolling deques with maxlen automatically evict stale data;
               find_peaks on inverted Z detects foot-strike troughs robustly.
    Inputs:
        history_seconds — float: rolling analysis window length in seconds.
        fps             — float: expected update rate (used for peak spacing).
        min_steps       — int: minimum steps before metrics are computed.
        confidence_thr  — float [0,1]: ankle keypoint confidence floor.
    Outputs:
        Optional[GaitMetrics] from get_metrics(); None if insufficient data.
    Preconditions:
        scipy.signal.find_peaks must be available.
    Assumptions:
        COCO-17 keypoints at indices 15 (left ankle) and 16 (right ankle).
    Constraints:
        Not thread-safe; wrap update() in a lock if called from multiple threads.
    References:
        scipy.signal.find_peaks; StepEvent; GaitMetrics; WR-ANALYSIS-GAIT-001.
    """

    def __init__(
        self,
        history_seconds: float = 10.0,
        fps: float = 20.0,
        min_steps: int = 4,
        confidence_thr: float = 0.25,
    ) -> None:
        """Initialise rolling history buffers and analysis parameters.

        ID: WR-ANALYSIS-GAIT-INIT-001
        Requirement: Store analysis parameters and initialise deque history
                     buffers for left/right ankle positions and hip X-position.
        Purpose: Prepare the analyzer for streaming update() calls without
                 requiring any initial pose data.
        Rationale: deque with maxlen automatically evicts frames older than
                   history_seconds*fps; pre-computing max_frames avoids
                   repeated multiplication on every update() call.
        Inputs:
            history_seconds — float > 0: rolling window depth in seconds.
            fps             — float > 0: expected frame rate for buffer sizing.
            min_steps       — int > 0: minimum steps before get_metrics() returns data.
            confidence_thr  — float [0,1]: ankle keypoint confidence floor.
        Outputs:
            None — initialises self.
        Preconditions:
            None.
        Postconditions:
            All deques empty; _step_events == [].
        Assumptions:
            update() will be called at ~fps Hz.
        Side Effects:
            Allocates three deque objects and one list.
        Failure Modes:
            None expected at construction time.
        Error Handling:
            None required.
        Constraints:
            max_frames = int(history_seconds * fps).
        Verification:
            Unit test: construct; assert len(_left_ankle) == 0.
        References:
            collections.deque; scipy.signal.find_peaks; WR-ANALYSIS-GAIT-CLASS-001.
        """
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
        """Ingest one pose frame and append high-confidence ankle and hip positions.

        ID: WR-ANALYSIS-GAIT-UPDATE-001
        Requirement: Append ankle (x,y,z) and hip-x samples to rolling buffers
                     when the respective keypoint confidence exceeds confidence_thr.
        Purpose: Accumulate the time-series data needed by get_metrics() to
                 detect step events and compute gait parameters.
        Rationale: Gating on confidence prevents noisy low-quality keypoints from
                   distorting the step detection peak-finding.
        Inputs:
            keypoints  — (17,3) float32: normalised 3-D keypoint coordinates.
            confidence — (17,) float32: per-keypoint confidence [0,1].
            timestamp  — Optional[float]: UNIX epoch; auto-fills via time.time().
        Outputs:
            None — updates internal deques as side effects.
        Preconditions:
            keypoints.shape == (17,3); confidence.shape == (17,).
        Postconditions:
            Left/right ankle deques contain the most recent max_frames samples.
        Assumptions:
            COCO-17 indices: left_ankle=15, right_ankle=16, left_hip=11, right_hip=12.
        Side Effects:
            Appends to self._left_ankle, self._right_ankle, self._hip_x.
        Failure Modes:
            confidence < confidence_thr: keypoint silently skipped for that frame.
        Error Handling:
            Confidence guard prevents appending invalid samples.
        Constraints:
            Not thread-safe.
        Verification:
            Unit test: call with conf=1.0; assert deque lengths increase by 1.
        References:
            _LEFT_ANKLE, _RIGHT_ANKLE, _LEFT_HIP, _RIGHT_HIP constants;
            WR-ANALYSIS-GAIT-CLASS-001.
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
        """Compute and return current gait metrics, or None if data is insufficient.

        ID: WR-ANALYSIS-GAIT-GETMETRICS-001
        Requirement: Run step detection on both ankle histories, compute cadence,
                     stride length, step symmetry and walking speed, and return a
                     GaitMetrics snapshot; return None if fewer than min_steps found.
        Purpose: Provide a single snapshot of all gait parameters on demand
                 without requiring the caller to manage intermediate state.
        Rationale: Computing all metrics in one call ensures consistency across
                   metrics derived from the same step event list.
        Inputs:
            None — reads self._left_ankle, self._right_ankle, self._hip_x.
        Outputs:
            GaitMetrics with cadence_spm, stride_length, step_symmetry, speed_est,
            num_steps, window_s; or None if insufficient data.
        Preconditions:
            update() must have been called enough times to populate buffers.
        Postconditions:
            Internal buffers are not modified by this call.
        Assumptions:
            Step events are accurately detected by _detect_steps.
        Side Effects:
            Calls _detect_steps (may create lists from deques).
        Failure Modes:
            Fewer than min_steps: returns None.
            window_s < 0.5 s: returns None (cadence would be meaningless).
        Error Handling:
            Returns None rather than raising for insufficient data conditions.
        Constraints:
            None.
        Verification:
            Unit test: inject 10 synthetic frames with oscillating ankle Z; assert
            get_metrics() returns GaitMetrics with cadence > 0.
        References:
            _detect_steps; _stride_length; _step_symmetry; _walking_speed;
            WR-ANALYSIS-GAIT-001.
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
        """Clear all accumulated history and step event records.

        ID: WR-ANALYSIS-GAIT-RESET-001
        Requirement: Empty all deque buffers and the step event list so the
                     analyzer starts fresh for a new measurement session.
        Purpose: Allow the same GaitAnalyzer instance to be reused across
                 measurement sessions without creating a new object.
        Rationale: Reusing an existing instance avoids re-creating deques and
                   re-allocating memory between sessions.
        Inputs:
            None.
        Outputs:
            None.
        Preconditions:
            None.
        Postconditions:
            All deques and step event list are empty.
        Assumptions:
            Caller is responsible for thread safety if update() runs concurrently.
        Side Effects:
            Clears self._left_ankle, self._right_ankle, self._hip_x, self._step_events.
        Failure Modes:
            None.
        Error Handling:
            None required.
        Constraints:
            Not thread-safe.
        Verification:
            Unit test: update N frames; reset(); assert all deques empty.
        References:
            collections.deque.clear; WR-ANALYSIS-GAIT-CLASS-001.
        """
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
        """Detect foot-strikes as local minima in the ankle Z-position series.

        ID: WR-ANALYSIS-GAIT-DETECT-001
        Requirement: Apply scipy.signal.find_peaks on the inverted ankle Z series
                     to locate foot-strike troughs and return a StepEvent list.
        Purpose: Convert raw ankle Z time-series into discrete step timestamps
                 for cadence and symmetry computation.
        Rationale: Inverting Z turns ground-contact troughs into peaks, enabling
                   find_peaks with a min_distance constraint (~100 bpm max).
        Inputs:
            samples — List[Tuple[ts,x,y,z]]: ankle sample buffer from the deque.
            foot    — str: 'left' or 'right' label for the returned events.
        Outputs:
            List[StepEvent] — one event per detected foot-strike; empty if < 6 samples.
        Preconditions:
            len(samples) >= 6 (otherwise returns []).
        Postconditions:
            All events in the list have foot == the provided foot argument.
        Assumptions:
            Ankle Z decreases at foot-strike relative to the swing phase.
        Side Effects:
            None — pure function; creates a new list.
        Failure Modes:
            No peaks found (monotonic signal): returns empty list.
        Error Handling:
            len(samples) < 6 guard returns [] immediately.
        Constraints:
            min_distance enforces at most ~100 bpm cadence detection.
        Verification:
            Unit test: inject sinusoidal Z with 1 Hz frequency at 20 Hz; assert
            ~20 step events detected in a 20-second window.
        References:
            scipy.signal.find_peaks; StepEvent; WR-ANALYSIS-GAIT-001.
        """
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
        """Estimate stride length as the mean XY distance between same-foot strikes.

        ID: WR-ANALYSIS-GAIT-STRIDE-001
        Requirement: Compute the mean Euclidean XY distance between consecutive
                     same-foot (ipsilateral) step positions.
        Purpose: Provide a normalised stride length proxy for gait health assessment
                 without requiring absolute distance calibration.
        Rationale: Using XY distance (ignoring Z) avoids confounding stride length
                   with vertical ankle trajectory amplitude.
        Inputs:
            ipsilateral_steps — List[StepEvent]: same-foot steps sorted by timestamp.
        Outputs:
            float — mean stride length in normalised units; 0.0 if < 2 steps.
        Preconditions:
            All steps in the list are from the same foot.
        Postconditions:
            Return value >= 0.0.
        Assumptions:
            Keypoints are in normalised [-1,1] space.
        Side Effects:
            None — pure function.
        Failure Modes:
            < 2 steps: returns 0.0.
            Empty dists list: returns 0.0.
        Error Handling:
            len guard returns 0.0 immediately for < 2 steps.
        Constraints:
            None.
        Verification:
            Unit test: 3 steps at (0,0), (0.5,0), (1.0,0); assert stride == 0.5.
        References:
            np.linalg.norm; WR-ANALYSIS-GAIT-001.
        """
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
        """Compute the step symmetry ratio from mean inter-step intervals.

        ID: WR-ANALYSIS-GAIT-SYMMETRY-001
        Requirement: Return the ratio of mean left to mean right inter-step
                     intervals clamped to (0, 1]; 1.0 = perfectly symmetric.
        Purpose: Quantify left-right walking asymmetry as a potential indicator
                 of gait abnormality or injury.
        Rationale: Using min(li/ri, ri/li) ensures the metric is symmetric and
                   always <= 1.0 regardless of which side is slower.
        Inputs:
            left_steps  — List[StepEvent]: detected left foot-strikes.
            right_steps — List[StepEvent]: detected right foot-strikes.
        Outputs:
            float — symmetry ratio (0,1]; 1.0 if data is insufficient.
        Preconditions:
            None; returns 1.0 if fewer than 2 steps on either side.
        Postconditions:
            Return value in (0.0, 1.0].
        Assumptions:
            Step timestamps are monotonically increasing within each list.
        Side Effects:
            None — pure function.
        Failure Modes:
            < 2 steps on either side: returns 1.0 (unknown = symmetric).
            ri < 1e-6: returns 1.0 to avoid division by zero.
        Error Handling:
            None guards for insufficient data; None check for mean intervals.
        Constraints:
            None.
        Verification:
            Unit test: left interval 0.5 s, right interval 0.6 s;
            assert symmetry == round(0.5/0.6, 3).
        References:
            np.mean; WR-ANALYSIS-GAIT-001.
        """
        def mean_interval(steps: List[StepEvent]) -> Optional[float]:
            """Compute mean positive inter-step interval from a list of StepEvents.

            ID: WR-ANALYSIS-GAIT-MEANINTERVAL-001
            Requirement: Return the mean of all positive consecutive time deltas
                         between StepEvents, or None if fewer than 2 events.
            Purpose: Provide left/right mean step intervals for symmetry calculation
                     while filtering out any spurious zero or negative timestamps.
            Inputs:
                steps — List[StepEvent]: ordered list of foot-strike events.
            Outputs:
                Optional[float] — mean positive interval in seconds; None if < 2 steps.
            Side Effects: None.
            Failure Modes: All intervals <= 0 returns None.
            Verification: [0.5s, 0.5s] → 0.5; single step → None.
            References: WR-ANALYSIS-GAIT-001.
            """
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
        """Estimate walking speed from the linear trend of hip X-position.

        ID: WR-ANALYSIS-GAIT-SPEED-001
        Requirement: Fit a linear polynomial to the hip-X time series and return
                     the absolute value of the slope as the walking speed estimate.
        Purpose: Provide a speed estimate in normalised units/s without requiring
                 a calibrated room coordinate system.
        Rationale: A linear fit is more robust than single-frame finite differences
                   for slow-moving subjects where per-frame X change is small.
        Inputs:
            None — reads self._hip_x deque.
        Outputs:
            float — estimated walking speed (normalised units/s); 0.0 if insufficient.
        Preconditions:
            At least 2 hip_x samples in self._hip_x.
        Postconditions:
            Return value >= 0.0.
        Assumptions:
            Subject walks in an approximately straight line during the window.
        Side Effects:
            None — read-only; may create lists from deque.
        Failure Modes:
            < 2 samples: returns 0.0.
            dt < 0.5 s: returns 0.0 (too short for reliable fit).
        Error Handling:
            len guard and dt guard return 0.0.
        Constraints:
            Requires scipy-free numpy.polyfit (degree 1).
        Verification:
            Unit test: inject linear X increase 0->1 over 10 s at 20 Hz;
            assert speed estimate ~0.1 normalised units/s.
        References:
            np.polyfit; WR-ANALYSIS-GAIT-001.
        """
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
