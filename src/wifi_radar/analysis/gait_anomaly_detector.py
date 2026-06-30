
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
from typing import Deque, Dict, List, Optional

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - optional import path
    IsolationForest = None

from .gait_analyzer import GaitMetrics

logger = logging.getLogger(__name__)


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
        self._history: Deque[np.ndarray] = deque(maxlen=history_size)
        self._warmup_samples = warmup_samples
        self._z_threshold = z_threshold
        self._random_state = random_state
        self._robust_weight = float(np.clip(robust_weight, 0.0, 1.0))
        self._clip_sigma = float(max(1.0, clip_sigma))
        self._min_scale = float(max(1e-6, min_scale))
        self._model = (
            IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=random_state,
            )
            if IsolationForest is not None
            else None
        )
        self._model_ready = False

    def update(self, metrics: GaitMetrics) -> Dict[str, object]:
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
        if len(self._history) < self._warmup_samples:
            self._history.append(current)
            if self._model is not None and len(self._history) >= self._warmup_samples:
                self._refit_model()
            return {
                "is_anomaly": False,
                "severity": "normal",
                "score": 0.0,
                "reasons": [],
                "history_size": len(self._history),
            }

        baseline = np.vstack(self._history)
        mean = baseline.mean(axis=0)
        median = np.median(baseline, axis=0)
        mad = np.median(np.abs(baseline - median), axis=0)
        scale_floor = np.maximum(self._min_scale, 0.03 * np.maximum(np.abs(median), 0.1))
        std = np.where(baseline.std(axis=0) < scale_floor, scale_floor, baseline.std(axis=0))
        robust_scale = np.where(1.4826 * mad < scale_floor, scale_floor, 1.4826 * mad)

        z_scores = np.abs((current - mean) / std)
        robust_z = np.abs((current - median) / robust_scale)
        blended_z = self._robust_weight * robust_z + (1.0 - self._robust_weight) * z_scores
        names = ["cadence", "stride_length", "step_symmetry", "speed_est"]
        reasons = [
            f"{name} deviated by {z:.1f}σ"
            for name, z in zip(names, blended_z)
            if z >= self._z_threshold
        ]

        iso_score = 0.0
        if self._model is not None and len(self._history) >= max(8, self._warmup_samples):
            if not self._model_ready or len(self._history) % 8 == 0:
                self._refit_model()
            if self._model_ready:
                iso_score = float(self._model.decision_function(current.reshape(1, -1))[0])
                iso_pred = int(self._model.predict(current.reshape(1, -1))[0])
                if iso_pred == -1 and float(blended_z.max()) >= self._z_threshold * 0.6:
                    reasons.append("unsupervised model marked the gait pattern as abnormal")

        score = float(blended_z.max())
        is_anomaly = bool(reasons)
        if iso_score < 0:
            score = max(score, abs(iso_score) * 10.0)

        severity = "normal"
        if is_anomaly:
            severity = "high" if score >= self._z_threshold * 1.5 else "moderate"

        history_sample = current
        if is_anomaly:
            lo = median - self._clip_sigma * robust_scale
            hi = median + self._clip_sigma * robust_scale
            history_sample = np.clip(current, lo, hi).astype(np.float32)

        self._history.append(history_sample)
        return {
            "is_anomaly": is_anomaly,
            "severity": severity,
            "score": round(score, 3),
            "reasons": reasons,
            "history_size": len(self._history),
            "robust_score": round(float(robust_z.max()), 3),
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
        self._history.clear()
        self._model_ready = False

    def _refit_model(self) -> None:
        if self._model is None or len(self._history) < max(8, self._warmup_samples):
            return
        self._model.fit(np.vstack(self._history).astype(np.float32))
        self._model_ready = True

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
