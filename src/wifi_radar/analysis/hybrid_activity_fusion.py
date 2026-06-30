"""
ID: WR-ANALYSIS-HYBRID-001
Requirement: Fuse multi-window CSI motion cues with pose confidence, gait metrics,
             and optional geometry metadata to produce a robust activity label.
Purpose: Add a lightweight research-inspired hybrid stage on top of the existing
         pose pipeline so the system remains useful when either CSI or pose
         signals alone are noisy.
Rationale: Recent 2026 WiFi sensing work emphasizes multi-window temporal fusion,
           geometry awareness, and lightweight continuous inference. This module
           distills those ideas into a practical online heuristic for the current
           project without introducing a heavyweight retraining dependency.
Assumptions: Inputs use the project's normalised CSI and COCO-17 pose format.
Constraints: CPU-only, stream-safe, and dependency-light.
References: arXiv:2602.08661, arXiv:2601.12252, Pattern Recognition 172 (2026) 112512.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Deque, Dict, Iterable, Optional, Sequence

import numpy as np


class HybridActivityFusion:
    """Fuse CSI motion, gait, and pose confidence into a stable activity estimate.

    ID: WR-ANALYSIS-HYBRID-CLASS-001
    Requirement: Maintain rolling multi-window motion summaries and output a
                 unified activity label plus supporting confidence scores.
    Purpose: Improve robustness against brief signal dropouts and temporal
             misalignment in live CSI streams.
    Rationale: Multi-window fusion borrows from recent WiFi HAR work showing
               that combining short and long windows improves temporal robustness.
    Inputs:
        window_sizes — iterable of positive ints defining fusion windows.
        motion_threshold — low-motion cutoff for stationary behaviour.
        high_motion_threshold — high-motion cutoff for energetic movement.
    Outputs:
        Dict summaries via update().
    Preconditions:
        update() must receive amplitude and phase tensors of matching shape.
    References:
        WR-ANALYSIS-HYBRID-001 module docstring.
    """

    def __init__(
        self,
        window_sizes: Iterable[int] = (4, 8, 16),
        motion_threshold: float = 0.05,
        high_motion_threshold: float = 0.18,
        motion_ema_alpha: float = 0.08,
        motion_dev_alpha: float = 0.10,
        noise_floor_k: float = 2.0,
        min_motion_scale: float = 0.05,
        hysteresis_frames: int = 3,
        adaptive_motion_thresholds: bool = True,
        threshold_warmup_frames: int = 24,
        threshold_alpha: float = 0.08,
        threshold_dev_alpha: float = 0.10,
        threshold_k: float = 2.0,
        min_stationary_threshold: float = 0.02,
        max_stationary_threshold: float = 0.20,
        high_threshold_multiplier: float = 3.0,
        max_high_motion_threshold: float = 0.60,
        fall_risk_alpha_rise: float = 0.45,
        fall_risk_alpha_fall: float = 0.12,
    ) -> None:
        """Initialise rolling fusion buffers and thresholds.

        ID: WR-ANALYSIS-HYBRID-INIT-001
        Requirement: Create per-window rolling buffers and reset prior CSI frame state.
        Purpose: Prepare the fusion stage for online streaming updates.
        Rationale: Bounded deques preserve recent motion context with constant memory.
        Inputs:
            window_sizes — iterable of ints; each value becomes a rolling window.
            motion_threshold — float; low-motion threshold.
            high_motion_threshold — float; strong-motion threshold.
        Outputs:
            None — initialises self.
        Preconditions:
            All window sizes must be positive after integer conversion.
        Postconditions:
            self._motion_windows and self._label_history are ready for streaming use.
        References:
            collections.deque.
        """
        sizes = sorted({max(1, int(size)) for size in window_sizes})
        self.window_sizes = tuple(sizes)
        self.motion_threshold = float(motion_threshold)
        self.high_motion_threshold = float(high_motion_threshold)
        self.motion_ema_alpha = float(np.clip(motion_ema_alpha, 1e-4, 1.0))
        self.motion_dev_alpha = float(np.clip(motion_dev_alpha, 1e-4, 1.0))
        self.noise_floor_k = float(max(0.0, noise_floor_k))
        self.min_motion_scale = float(max(1e-4, min_motion_scale))
        self.hysteresis_frames = int(max(1, hysteresis_frames))
        self.adaptive_motion_thresholds = bool(adaptive_motion_thresholds)
        self.threshold_warmup_frames = int(max(1, threshold_warmup_frames))
        self.threshold_alpha = float(np.clip(threshold_alpha, 1e-4, 1.0))
        self.threshold_dev_alpha = float(np.clip(threshold_dev_alpha, 1e-4, 1.0))
        self.threshold_k = float(max(0.0, threshold_k))
        self.min_stationary_threshold = float(max(1e-4, min_stationary_threshold))
        self.max_stationary_threshold = float(max(self.min_stationary_threshold, max_stationary_threshold))
        self.high_threshold_multiplier = float(max(1.2, high_threshold_multiplier))
        self.max_high_motion_threshold = float(max(self.high_motion_threshold, max_high_motion_threshold))
        self.fall_risk_alpha_rise = float(np.clip(fall_risk_alpha_rise, 1e-4, 1.0))
        self.fall_risk_alpha_fall = float(np.clip(fall_risk_alpha_fall, 1e-4, 1.0))

        self._prev_amplitude: Optional[np.ndarray] = None
        self._prev_phase: Optional[np.ndarray] = None
        self._motion_windows: Dict[int, Deque[float]] = {
            size: deque(maxlen=size) for size in self.window_sizes
        }
        self._label_history: Deque[str] = deque(maxlen=max(3, len(self.window_sizes) * 2))
        self._motion_ema: Optional[float] = None
        self._motion_dev_ema: float = 0.0
        self._active_label: Optional[str] = None
        self._pending_label: Optional[str] = None
        self._pending_label_frames: int = 0
        self._threshold_frames: int = 0
        self._motion_threshold_ema: float = self.motion_threshold
        self._motion_threshold_dev_ema: float = 0.0
        self._adaptive_motion_threshold: float = self.motion_threshold
        self._adaptive_high_motion_threshold: float = self.high_motion_threshold
        self._fall_risk_ema: Optional[float] = None

    def reset(self) -> None:
        """Reset all internal state for a new session.

        ID: WR-ANALYSIS-HYBRID-RESET-001
        Requirement: Clear rolling buffers and forget previous CSI frames.
        Purpose: Allow reuse across sessions without stale carryover.
        Rationale: Explicit reset improves determinism in tests and replay runs.
        Inputs:
            None.
        Outputs:
            None.
        Postconditions:
            Motion buffers are empty and previous-frame state is cleared.
        """
        self._prev_amplitude = None
        self._prev_phase = None
        for window in self._motion_windows.values():
            window.clear()
        self._label_history.clear()
        self._motion_ema = None
        self._motion_dev_ema = 0.0
        self._active_label = None
        self._pending_label = None
        self._pending_label_frames = 0
        self._threshold_frames = 0
        self._motion_threshold_ema = self.motion_threshold
        self._motion_threshold_dev_ema = 0.0
        self._adaptive_motion_threshold = self.motion_threshold
        self._adaptive_high_motion_threshold = self.high_motion_threshold
        self._fall_risk_ema = None

    def update(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        pose_confidence: Optional[np.ndarray] = None,
        gait_metrics: Optional[Any] = None,
        fall_severity: int = 0,
        layout_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update the hybrid estimate with one new CSI and pose snapshot.

        ID: WR-ANALYSIS-HYBRID-UPDATE-001
        Requirement: Compute multi-window motion scores and emit a stable hybrid
                     activity label with supporting scores.
        Purpose: Provide a practical real-time hybrid inference layer for the app.
        Rationale: Combining CSI dynamics with pose/gait reliability reduces the
                   chance of overreacting to one noisy modality.
        Inputs:
            amplitude — np.ndarray: CSI amplitude tensor for one frame.
            phase — np.ndarray: CSI phase tensor for one frame.
            pose_confidence — Optional per-keypoint confidence array.
            gait_metrics — Optional GaitMetrics or dict.
            fall_severity — int fall severity hint from the fall detector.
            layout_metadata — Optional geometry description.
        Outputs:
            Dict containing activity_label, motion_score, fall_risk, and support metrics.
        Preconditions:
            amplitude and phase share the same shape.
        Postconditions:
            Previous CSI frame and rolling motion windows are updated.
        Failure Modes:
            Shape mismatch raises ValueError.
        Error Handling:
            Invalid shapes are rejected early with a descriptive exception.
        References:
            MKFi-style multi-window fusion; geometry-aware conditioning ideas.
        """
        amplitude = np.asarray(amplitude, dtype=np.float32)
        phase = np.asarray(phase, dtype=np.float32)
        if amplitude.shape != phase.shape:
            raise ValueError(f"Amplitude/phase shape mismatch: {amplitude.shape} != {phase.shape}")

        delta_score = self._compute_delta_score(amplitude, phase)
        for window in self._motion_windows.values():
            window.append(delta_score)

        fused_motion = self._fuse_motion_windows()
        motion_after_floor, adaptive_floor = self._apply_adaptive_floor(fused_motion)
        geometry_scale = self._estimate_geometry_scale(layout_metadata)
        motion_score = motion_after_floor * geometry_scale
        self._update_adaptive_motion_thresholds(motion_score=motion_score, fall_severity=fall_severity)
        active_motion_threshold, active_high_motion_threshold = self._current_motion_thresholds()

        pose_reliability = float(np.clip(np.nanmean(pose_confidence), 0.0, 1.0)) if pose_confidence is not None else 0.0
        gait_summary = self._normalize_gait_metrics(gait_metrics)
        gait_reliability = gait_summary["gait_reliability"]
        raw_fall_risk = self._compute_fall_risk(
            motion_score=motion_score,
            pose_reliability=pose_reliability,
            gait_summary=gait_summary,
            fall_severity=fall_severity,
            active_high_motion_threshold=active_high_motion_threshold,
        )
        fall_risk = self._smooth_fall_risk(raw_fall_risk=raw_fall_risk, fall_severity=fall_severity)

        label = self._classify(
            motion_score=motion_score,
            gait_summary=gait_summary,
            pose_reliability=pose_reliability,
            fall_risk=fall_risk,
            fall_severity=fall_severity,
            active_motion_threshold=active_motion_threshold,
            active_high_motion_threshold=active_high_motion_threshold,
        )
        self._label_history.append(label)
        voted_label = self._stable_vote(label)
        stable_label = self._apply_hysteresis(voted_label)

        return {
            "activity_label": stable_label,
            "motion_score": float(motion_score),
            "raw_motion_score": float(fused_motion),
            "adaptive_noise_floor": float(adaptive_floor),
            "motion_threshold_active": float(active_motion_threshold),
            "high_motion_threshold_active": float(active_high_motion_threshold),
            "pose_reliability": float(pose_reliability),
            "gait_reliability": float(gait_reliability),
            "fall_risk": float(fall_risk),
            "raw_fall_risk": float(raw_fall_risk),
            "fall_risk_ema": float(fall_risk),
            "geometry_scale": float(geometry_scale),
            "cadence_spm": float(gait_summary["cadence_spm"]),
            "speed_est": float(gait_summary["speed_est"]),
            "step_symmetry": float(gait_summary["step_symmetry"]),
        }

    def _compute_delta_score(self, amplitude: np.ndarray, phase: np.ndarray) -> float:
        if self._prev_amplitude is None or self._prev_phase is None:
            self._prev_amplitude = amplitude.copy()
            self._prev_phase = phase.copy()
            return 0.0

        amp_delta = float(np.mean(np.abs(amplitude - self._prev_amplitude)))
        phase_delta = float(np.mean(np.abs(phase - self._prev_phase)))
        amp_scale = float(max(np.std(amplitude), self.min_motion_scale))
        phase_scale = float(max(np.std(phase), self.min_motion_scale))

        self._prev_amplitude = amplitude.copy()
        self._prev_phase = phase.copy()
        raw_score = 0.5 * (amp_delta / amp_scale + phase_delta / phase_scale)
        return float(np.clip(raw_score, 0.0, 4.0))

    def _fuse_motion_windows(self) -> float:
        if not self._motion_windows:
            return 0.0
        summaries = []
        for size in self.window_sizes:
            values = self._motion_windows[size]
            summaries.append(float(np.mean(values)) if values else 0.0)
        return float(np.mean(summaries))

    def _apply_adaptive_floor(self, fused_motion: float) -> tuple[float, float]:
        # The first valid sample defines the initial baseline and no floor is applied.
        if self._motion_ema is None:
            self._motion_ema = float(max(0.0, fused_motion))
            self._motion_dev_ema = 0.0
            return float(max(0.0, fused_motion)), 0.0

        baseline = float(max(0.0, self._motion_ema))
        floor = float(max(0.0, baseline + self.noise_floor_k * self._motion_dev_ema))
        adjusted = float(max(0.0, fused_motion - floor))

        mean_alpha = self.motion_ema_alpha
        dev_alpha = self.motion_dev_alpha

        # Update the noise baseline using only the low-motion envelope.
        # This avoids learning true walking motion as background noise.
        low_motion_cap = baseline + self.motion_threshold
        reference = float(min(fused_motion, low_motion_cap))

        new_mean = float((1.0 - mean_alpha) * self._motion_ema + mean_alpha * reference)
        deviation = abs(reference - new_mean)
        new_dev = float((1.0 - dev_alpha) * self._motion_dev_ema + dev_alpha * deviation)

        self._motion_ema = new_mean
        self._motion_dev_ema = new_dev
        return adjusted, floor

    def _estimate_geometry_scale(self, layout_metadata: Optional[Dict[str, Any]]) -> float:
        if not layout_metadata:
            return 1.0

        confidence = float(
            np.clip(
                layout_metadata.get(
                    "geometry_confidence",
                    layout_metadata.get("layout_confidence", 1.0),
                ),
                0.0,
                1.0,
            )
        )

        tx_positions = np.asarray(layout_metadata.get("tx_positions", []), dtype=np.float32)
        rx_positions = np.asarray(layout_metadata.get("rx_positions", []), dtype=np.float32)
        dist_scale = 1.0
        if tx_positions.ndim == 2 and rx_positions.ndim == 2 and tx_positions.size and rx_positions.size:
            diffs = tx_positions[:, None, :] - rx_positions[None, :, :]
            mean_dist = float(np.mean(np.linalg.norm(diffs, axis=-1)))
            dist_scale = float(np.clip(0.85 + mean_dist / 10.0, 0.85, 1.15))

        confidence_scale = 0.9 + 0.2 * confidence
        blended_dist = 1.0 + confidence * (dist_scale - 1.0)
        return float(np.clip(blended_dist * confidence_scale, 0.80, 1.20))

    def _normalize_gait_metrics(self, gait_metrics: Optional[Any]) -> Dict[str, float]:
        if gait_metrics is None:
            return {
                "cadence_spm": 0.0,
                "stride_length": 0.0,
                "step_symmetry": 1.0,
                "speed_est": 0.0,
                "num_steps": 0.0,
                "gait_reliability": 0.0,
            }

        if hasattr(gait_metrics, "__dict__"):
            data = dict(gait_metrics.__dict__)
        elif isinstance(gait_metrics, dict):
            data = dict(gait_metrics)
        else:
            data = {}

        num_steps = float(data.get("num_steps", 0.0) or 0.0)
        cadence = float(data.get("cadence_spm", 0.0) or 0.0)
        stride = float(data.get("stride_length", 0.0) or 0.0)
        symmetry = float(data.get("step_symmetry", 1.0) or 1.0)
        speed = float(data.get("speed_est", 0.0) or 0.0)
        gait_reliability = float(np.clip(num_steps / 6.0, 0.0, 1.0))

        return {
            "cadence_spm": cadence,
            "stride_length": stride,
            "step_symmetry": symmetry,
            "speed_est": speed,
            "num_steps": num_steps,
            "gait_reliability": gait_reliability,
        }

    def _compute_fall_risk(
        self,
        motion_score: float,
        pose_reliability: float,
        gait_summary: Dict[str, float],
        fall_severity: int,
        active_high_motion_threshold: float,
    ) -> float:
        severity_term = float(np.clip(fall_severity / 2.0, 0.0, 1.0))
        symmetry_term = float(np.clip((0.7 - gait_summary["step_symmetry"]) / 0.7, 0.0, 1.0))
        pose_term = float(np.clip((0.6 - pose_reliability) / 0.6, 0.0, 1.0))
        motion_term = float(np.clip(motion_score / max(active_high_motion_threshold, 1e-6), 0.0, 1.0))
        risk = float(np.clip(0.7 * severity_term + 0.1 * symmetry_term + 0.1 * pose_term + 0.1 * motion_term, 0.0, 1.0))
        if fall_severity >= 2:
            risk = max(risk, 0.85)
        return risk

    def _smooth_fall_risk(self, raw_fall_risk: float, fall_severity: int) -> float:
        if self._fall_risk_ema is None:
            self._fall_risk_ema = float(np.clip(raw_fall_risk, 0.0, 1.0))
        else:
            alpha = self.fall_risk_alpha_rise if raw_fall_risk >= self._fall_risk_ema else self.fall_risk_alpha_fall
            self._fall_risk_ema = float(self._fall_risk_ema + alpha * (raw_fall_risk - self._fall_risk_ema))

        if fall_severity >= 2:
            self._fall_risk_ema = max(self._fall_risk_ema, 0.85)
        return float(np.clip(self._fall_risk_ema, 0.0, 1.0))

    def _classify(
        self,
        motion_score: float,
        gait_summary: Dict[str, float],
        pose_reliability: float,
        fall_risk: float,
        fall_severity: int,
        active_motion_threshold: float,
        active_high_motion_threshold: float,
    ) -> str:
        if fall_severity >= 2 or fall_risk >= 0.8:
            return "possible_fall"
        if gait_summary["speed_est"] >= 0.5 and gait_summary["cadence_spm"] >= 70 and gait_summary["step_symmetry"] >= 0.7:
            return "walking"
        if motion_score >= active_high_motion_threshold:
            return "high_motion"
        if motion_score < active_motion_threshold and pose_reliability >= 0.6:
            return "stationary"
        return "transition"

    def _update_adaptive_motion_thresholds(self, motion_score: float, fall_severity: int) -> None:
        if not self.adaptive_motion_thresholds:
            return
        if fall_severity >= 2:
            return

        self._threshold_frames += 1
        reference_cap = self._adaptive_motion_threshold + max(0.03, self._adaptive_motion_threshold * 1.5)
        reference = float(min(max(0.0, motion_score), reference_cap))

        mean_alpha = self.threshold_alpha
        dev_alpha = self.threshold_dev_alpha
        new_mean = float((1.0 - mean_alpha) * self._motion_threshold_ema + mean_alpha * reference)
        deviation = abs(reference - new_mean)
        new_dev = float((1.0 - dev_alpha) * self._motion_threshold_dev_ema + dev_alpha * deviation)

        self._motion_threshold_ema = new_mean
        self._motion_threshold_dev_ema = new_dev

        adaptive_motion = float(
            np.clip(
                new_mean + self.threshold_k * new_dev,
                self.min_stationary_threshold,
                self.max_stationary_threshold,
            )
        )
        adaptive_high = float(
            np.clip(
                max(adaptive_motion * self.high_threshold_multiplier, adaptive_motion + 0.06),
                max(self.high_motion_threshold * 0.6, adaptive_motion + 0.02),
                self.max_high_motion_threshold,
            )
        )

        self._adaptive_motion_threshold = adaptive_motion
        self._adaptive_high_motion_threshold = adaptive_high

    def _current_motion_thresholds(self) -> tuple[float, float]:
        if not self.adaptive_motion_thresholds:
            return self.motion_threshold, self.high_motion_threshold

        warmup_ratio = min(1.0, self._threshold_frames / float(self.threshold_warmup_frames))
        motion_thr = float((1.0 - warmup_ratio) * self.motion_threshold + warmup_ratio * self._adaptive_motion_threshold)
        high_thr = float((1.0 - warmup_ratio) * self.high_motion_threshold + warmup_ratio * self._adaptive_high_motion_threshold)
        high_thr = float(max(high_thr, motion_thr + 1e-3))
        return motion_thr, high_thr

    def _stable_vote(self, newest_label: str) -> str:
        if len(self._label_history) < 3:
            return newest_label
        counts = Counter(self._label_history)
        return counts.most_common(1)[0][0]

    def _apply_hysteresis(self, candidate_label: str) -> str:
        if self._active_label is None:
            self._active_label = candidate_label
            return candidate_label

        if candidate_label == self._active_label:
            self._pending_label = None
            self._pending_label_frames = 0
            return self._active_label

        if candidate_label != self._pending_label:
            self._pending_label = candidate_label
            self._pending_label_frames = 1
        else:
            self._pending_label_frames += 1

        needed = self.hysteresis_frames
        if candidate_label == "possible_fall":
            needed = 1
        elif self._active_label == "possible_fall" and candidate_label != "possible_fall":
            needed = max(2, self.hysteresis_frames)

        if self._pending_label_frames >= needed:
            self._active_label = candidate_label
            self._pending_label = None
            self._pending_label_frames = 0

        return self._active_label
