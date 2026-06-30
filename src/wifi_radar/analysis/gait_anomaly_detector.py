
"""
ID: WR-ANALYSIS-GAITANOM-001
Requirement: Detect unusual gait patterns from rolling GaitMetrics
             snapshots using robust z-scores and optional unsupervised
             outlier detection.
Purpose: Provide early warning for abnormal cadence, symmetry, stride,
         or speed changes that may indicate instability or decline.
Rationale: Combining explainable z-score thresholds with IsolationForest
           improves robustness over either method alone.
Assumptions: GaitMetrics come from GaitAnalyzer at roughly 20 Hz updates.
Constraints: Warm-up samples required before anomaly decisions are made.
References: sklearn IsolationForest; clinical gait variability literature.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Union

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - optional import path
    IsolationForest = None

from .gait_analyzer import GaitMetrics

logger = logging.getLogger(__name__)


@dataclass
class _ProfileState:
    history: Deque[np.ndarray]
    model: Optional[Any]
    model_ready: bool = False
    anomaly_ema: float = 0.0
    active_severity: str = "normal"
    pending_severity: Optional[str] = None
    pending_frames: int = 0


class GaitAnomalyDetector:
    """Flag unusual gait patterns from a stream of GaitMetrics.

    ID: WR-ANALYSIS-GAITANOM-CLASS-001
    Requirement: Accept GaitMetrics snapshots and return a structured
                 anomaly assessment containing score, severity, and reasons.
    Purpose: Give the processing pipeline a low-cost way to flag gait drift
             without retraining the pose-estimation network.
    Rationale: Rolling baselines adapt to the current subject while still
               exposing sudden deviations in symmetry and cadence.
    Inputs:
        history_size    — int: max rolling samples to keep.
        warmup_samples  — int: samples required before anomaly scoring.
        z_threshold     — float: absolute z-score threshold.
        contamination   — float: expected anomaly ratio for IsolationForest.
        random_state    — int: RNG seed for reproducibility.
    Outputs:
        update() returns a dict describing anomaly status.
    Preconditions:
        Each update() call must receive a valid GaitMetrics instance.
    Postconditions:
        Internal history buffer grows until history_size.
    Assumptions:
        cadence, stride, symmetry, and speed are numeric and finite.
    Side Effects:
        Refits an IsolationForest periodically when enough history exists.
    Failure Modes:
        sklearn unavailable: detector falls back to z-score mode only.
    Error Handling:
        Non-finite metrics are coerced to 0.0 before scoring.
    Constraints:
        Not thread-safe without external locking.
    Verification:
        Unit test: stable baselines stay normal; large deviations flag anomalous.
    References:
        GaitMetrics; sklearn.ensemble.IsolationForest.
    """

    def __init__(
        self,
        history_size: int = 128,
        warmup_samples: int = 20,
        z_threshold: float = 3.0,
        contamination: float = 0.1,
        random_state: int = 42,
        robust_weight: float = 0.65,
        clip_sigma: float = 4.0,
        min_scale: float = 1e-3,
        ema_alpha_rise: float = 0.45,
        ema_alpha_fall: float = 0.12,
        debounce_frames_moderate: int = 1,
        debounce_frames_high: int = 2,
        min_identity_persistence: float = 0.35,
        enable_unsupervised: bool = True,
    ) -> None:
        """Initialise the rolling baseline and optional anomaly model.

        ID: WR-ANALYSIS-GAITANOM-INIT-001
        Requirement: Allocate internal buffers and configure the anomaly model.
        Purpose: Prepare the detector for streaming updates.
        Rationale: deque(maxlen=N) automatically retains recent baseline history.
        Inputs:
            history_size   — int > 0.
            warmup_samples — int > 0.
            z_threshold    — float > 0.
            contamination  — float in (0, 0.5].
            random_state   — int.
        Outputs:
            None — initialises self.
        Preconditions:
            history_size and warmup_samples must be positive.
        Postconditions:
            History deque is empty and ready to accept metrics.
        Assumptions:
            warmup_samples <= history_size.
        Side Effects:
            May allocate an IsolationForest instance.
        Failure Modes:
            sklearn missing: self._model is None.
        Error Handling:
            Optional import fallback disables model-based scoring gracefully.
        Constraints:
            history_size bounds RAM usage.
        Verification:
            Unit test: detector starts with empty history.
        References:
            collections.deque; IsolationForest.
        """
        self._history_size = int(max(8, history_size))
        self._warmup_samples = warmup_samples
        self._z_threshold = z_threshold
        self._random_state = random_state
        self._robust_weight = float(np.clip(robust_weight, 0.0, 1.0))
        self._clip_sigma = float(max(1.0, clip_sigma))
        self._min_scale = float(max(1e-6, min_scale))
        self._ema_alpha_rise = float(np.clip(ema_alpha_rise, 1e-4, 1.0))
        self._ema_alpha_fall = float(np.clip(ema_alpha_fall, 1e-4, 1.0))
        self._debounce_frames_moderate = int(max(1, debounce_frames_moderate))
        self._debounce_frames_high = int(max(1, debounce_frames_high))
        self._min_identity_persistence = float(np.clip(min_identity_persistence, 0.0, 1.0))
        self._enable_unsupervised = bool(enable_unsupervised)
        self._contamination = float(contamination)
        self._profiles: Dict[Union[str, int], _ProfileState] = {}

    def update(
        self,
        metrics: GaitMetrics,
        person_id: Optional[Union[int, str]] = None,
        identity_persistence: float = 1.0,
    ) -> Dict[str, object]:
        """Score one gait snapshot against the rolling baseline.

        ID: WR-ANALYSIS-GAITANOM-UPDATE-001
        Requirement: Convert metrics to a feature vector, compare it with
                     the rolling history, and return anomaly metadata.
        Purpose: Give the caller a simple per-snapshot anomaly verdict.
        Rationale: Feature scaling is implicit in z-score computation, making
                   the method robust across gait metrics with different units.
        Inputs:
            metrics — GaitMetrics: cadence, stride, symmetry, speed snapshot.
        Outputs:
            Dict with keys: is_anomaly, severity, score, reasons, history_size.
        Preconditions:
            metrics fields must be numeric.
        Postconditions:
            Current sample is appended to the rolling history.
        Assumptions:
            The history represents the current subject's nominal gait pattern.
        Side Effects:
            Periodically refits the optional IsolationForest.
        Failure Modes:
            Short history: returns non-anomalous warm-up result.
        Error Handling:
            Extremely small std values are clamped to avoid divide-by-zero.
        Constraints:
            None.
        Verification:
            Unit test: outlier cadence and symmetry return is_anomaly=True.
        References:
            _vectorise; IsolationForest.decision_function.
        """
        current = self._vectorise(metrics)
        identity_key: Union[str, int] = person_id if person_id is not None else "__default__"
        profile = self._profile(identity_key)
        persistence = float(np.clip(identity_persistence, 0.0, 1.0))
        warmup_required = self._warmup_samples
        if persistence < self._min_identity_persistence:
            warmup_required = max(warmup_required, self._warmup_samples + 4)

        if len(profile.history) < warmup_required:
            profile.history.append(current)
            if profile.model is not None and len(profile.history) >= warmup_required:
                self._refit_model(profile)
            return {
                "is_anomaly": False,
                "severity": "normal",
                "score": 0.0,
                "temporal_score": round(profile.anomaly_ema, 3),
                "reasons": [],
                "history_size": len(profile.history),
            }

        baseline = np.vstack(profile.history)
        mean = baseline.mean(axis=0)
        median = np.median(baseline, axis=0)
        mad = np.median(np.abs(baseline - median), axis=0)
        scale_floor = np.maximum(self._min_scale, 0.03 * np.maximum(np.abs(median), 0.1))
        std = np.where(baseline.std(axis=0) < scale_floor, scale_floor, baseline.std(axis=0))
        robust_scale = np.where(1.4826 * mad < scale_floor, scale_floor, 1.4826 * mad)

        z_scores = np.abs((current - mean) / std)
        robust_z = np.abs((current - median) / robust_scale)
        blended_z = self._robust_weight * robust_z + (1.0 - self._robust_weight) * z_scores
        feature_weights = self._feature_weights(metrics, persistence)
        weighted_z = blended_z * feature_weights
        names = ["cadence", "stride_length", "step_symmetry", "speed_est"]
        reasons = [
            f"{name} deviated by {z:.1f}σ"
            for name, z in zip(names, weighted_z)
            if z >= self._z_threshold
        ]

        iso_score = 0.0
        if profile.model is not None and len(profile.history) >= max(8, warmup_required):
            if not profile.model_ready or len(profile.history) % 8 == 0:
                self._refit_model(profile)
            if profile.model_ready:
                iso_score = float(profile.model.decision_function(current.reshape(1, -1))[0])
                iso_pred = int(profile.model.predict(current.reshape(1, -1))[0])
                if iso_pred == -1 and float(weighted_z.max()) >= self._z_threshold * 0.6:
                    reasons.append("unsupervised model marked the gait pattern as abnormal")

        score = float(weighted_z.max())
        raw_is_anomaly = bool(reasons)
        if iso_score < 0:
            score = max(score, abs(iso_score) * 10.0)

        candidate = self._candidate_severity(score)
        temporal_severity = self._apply_temporal_consistency(profile, candidate)

        severity = temporal_severity
        is_anomaly = severity != "normal"

        history_sample = current
        if raw_is_anomaly:
            lo = median - self._clip_sigma * robust_scale
            hi = median + self._clip_sigma * robust_scale
            history_sample = np.clip(current, lo, hi).astype(np.float32)

        profile.history.append(history_sample)
        return {
            "is_anomaly": is_anomaly,
            "severity": severity,
            "score": round(score, 3),
            "temporal_score": round(profile.anomaly_ema, 3),
            "reasons": reasons,
            "history_size": len(profile.history),
            "robust_score": round(float(robust_z.max()), 3),
            "feature_weights": {
                "cadence": round(float(feature_weights[0]), 3),
                "stride_length": round(float(feature_weights[1]), 3),
                "step_symmetry": round(float(feature_weights[2]), 3),
                "speed_est": round(float(feature_weights[3]), 3),
            },
        }

    def reset(self) -> None:
        """Clear all stored baseline history.

        ID: WR-ANALYSIS-GAITANOM-RESET-001
        Requirement: Remove all stored metric vectors and reset model state.
        Purpose: Allow reuse of the detector for a new subject/session.
        Rationale: Historical baselines should not leak between subjects.
        Inputs:
            None.
        Outputs:
            None.
        Preconditions:
            None.
        Postconditions:
            History is empty and the model is marked not ready.
        Assumptions:
            Caller handles any required external synchronisation.
        Side Effects:
            Clears internal buffers.
        Failure Modes:
            None.
        Error Handling:
            None.
        Constraints:
            None.
        Verification:
            Unit test: after reset, history_size == 0 on next update.
        References:
            collections.deque.clear.
        """
        self._profiles.clear()

    def _profile(self, identity_key: Union[str, int]) -> _ProfileState:
        if identity_key not in self._profiles:
            model = None
            if self._enable_unsupervised and IsolationForest is not None:
                model = IsolationForest(
                    n_estimators=100,
                    contamination=self._contamination,
                    random_state=self._random_state,
                )
            self._profiles[identity_key] = _ProfileState(
                history=deque(maxlen=self._history_size),
                model=model,
            )
        return self._profiles[identity_key]

    def _refit_model(self, profile: _ProfileState) -> None:
        if profile.model is None or len(profile.history) < max(8, self._warmup_samples):
            return
        profile.model.fit(np.vstack(profile.history).astype(np.float32))
        profile.model_ready = True

    def _candidate_severity(self, score: float) -> str:
        if score >= self._z_threshold * 1.5:
            return "high"
        if score >= self._z_threshold:
            return "moderate"
        return "normal"

    def _apply_temporal_consistency(self, profile: _ProfileState, candidate: str) -> str:
        target = {"normal": 0.0, "moderate": 0.65, "high": 1.0}[candidate]
        alpha = self._ema_alpha_rise if target >= profile.anomaly_ema else self._ema_alpha_fall
        profile.anomaly_ema = float(profile.anomaly_ema + alpha * (target - profile.anomaly_ema))

        ema_candidate = "normal"
        if profile.anomaly_ema >= 0.85:
            ema_candidate = "high"
        elif profile.anomaly_ema >= 0.45:
            ema_candidate = "moderate"

        if ema_candidate == profile.active_severity:
            profile.pending_severity = None
            profile.pending_frames = 0
            return profile.active_severity

        if ema_candidate != profile.pending_severity:
            profile.pending_severity = ema_candidate
            profile.pending_frames = 1
        else:
            profile.pending_frames += 1

        needed = 1
        if ema_candidate == "moderate":
            needed = self._debounce_frames_moderate
        elif ema_candidate == "high":
            needed = self._debounce_frames_high

        if profile.pending_frames >= needed:
            profile.active_severity = ema_candidate
            profile.pending_severity = None
            profile.pending_frames = 0

        return profile.active_severity

    def _feature_weights(self, metrics: GaitMetrics, identity_persistence: float) -> np.ndarray:
        step_quality = float(np.clip((metrics.num_steps - 2.0) / 10.0, 0.0, 1.0))
        window_quality = float(np.clip(metrics.window_s / 8.0, 0.0, 1.0))
        persistence_quality = float(np.clip(identity_persistence, 0.0, 1.0))
        quality = 0.45 * step_quality + 0.35 * window_quality + 0.20 * persistence_quality

        cadence_w = 0.55 + 0.45 * quality
        stride_w = 0.45 + 0.35 * quality
        symmetry_w = 0.55 + 0.45 * quality
        speed_w = 0.50 + 0.45 * quality
        return np.asarray([cadence_w, stride_w, symmetry_w, speed_w], dtype=np.float32)

    @staticmethod
    def _vectorise(metrics: GaitMetrics) -> np.ndarray:
        return np.asarray(
            [
                float(np.nan_to_num(metrics.cadence_spm, nan=0.0)),
                float(np.nan_to_num(metrics.stride_length, nan=0.0)),
                float(np.nan_to_num(metrics.step_symmetry, nan=0.0)),
                float(np.nan_to_num(metrics.speed_est, nan=0.0)),
            ],
            dtype=np.float32,
        )
