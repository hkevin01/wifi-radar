import pytest

from wifi_radar.analysis.gait_analyzer import GaitMetrics
from wifi_radar.analysis.gait_anomaly_detector import GaitAnomalyDetector


def test_gait_anomaly_detector_flags_large_deviation():
    detector = GaitAnomalyDetector(warmup_samples=5, z_threshold=2.0, contamination=0.2)

    baseline = GaitMetrics(
        cadence_spm=100.0,
        stride_length=0.6,
        step_symmetry=0.98,
        speed_est=1.1,
        num_steps=10,
        window_s=6.0,
    )
    for _ in range(6):
        result = detector.update(baseline)
        assert result["is_anomaly"] is False

    outlier = GaitMetrics(
        cadence_spm=35.0,
        stride_length=0.15,
        step_symmetry=0.35,
        speed_est=0.15,
        num_steps=10,
        window_s=6.0,
    )
    result = detector.update(outlier)
    assert result["is_anomaly"] is True
    assert result["severity"] in {"moderate", "high"}
    assert result["reasons"]


def test_gait_anomaly_detector_outlier_does_not_poison_baseline():
    detector = GaitAnomalyDetector(warmup_samples=6, z_threshold=2.2, contamination=0.2)

    baseline = GaitMetrics(
        cadence_spm=98.0,
        stride_length=0.58,
        step_symmetry=0.96,
        speed_est=1.05,
        num_steps=10,
        window_s=6.0,
    )
    for _ in range(8):
        detector.update(baseline)

    anomaly = detector.update(
        GaitMetrics(
            cadence_spm=20.0,
            stride_length=0.10,
            step_symmetry=0.20,
            speed_est=0.05,
            num_steps=10,
            window_s=6.0,
        )
    )
    assert anomaly["is_anomaly"] is True

    recovery = detector.update(
        GaitMetrics(
            cadence_spm=99.0,
            stride_length=0.57,
            step_symmetry=0.95,
            speed_est=1.04,
            num_steps=10,
            window_s=6.0,
        )
    )
    assert recovery["is_anomaly"] is False


def test_gait_anomaly_detector_reports_robust_score():
    detector = GaitAnomalyDetector(warmup_samples=5, z_threshold=2.0, contamination=0.2)

    baseline = GaitMetrics(
        cadence_spm=101.0,
        stride_length=0.61,
        step_symmetry=0.97,
        speed_est=1.12,
        num_steps=10,
        window_s=6.0,
    )
    for _ in range(6):
        detector.update(baseline)

    result = detector.update(
        GaitMetrics(
            cadence_spm=72.0,
            stride_length=0.43,
            step_symmetry=0.72,
            speed_est=0.62,
            num_steps=10,
            window_s=6.0,
        )
    )
    assert "robust_score" in result
    assert isinstance(result["robust_score"], float)


def test_gait_anomaly_detector_temporal_consistency_blocks_single_window_escalation():
    detector = GaitAnomalyDetector(
        warmup_samples=6,
        z_threshold=2.0,
        contamination=0.2,
        enable_unsupervised=False,
        debounce_frames_moderate=2,
        debounce_frames_high=3,
    )

    baseline = GaitMetrics(
        cadence_spm=102.0,
        stride_length=0.60,
        step_symmetry=0.97,
        speed_est=1.10,
        num_steps=12,
        window_s=8.0,
    )
    for _ in range(8):
        detector.update(baseline)

    single_spike = detector.update(
        GaitMetrics(
            cadence_spm=42.0,
            stride_length=0.20,
            step_symmetry=0.35,
            speed_est=0.25,
            num_steps=12,
            window_s=8.0,
        )
    )
    assert single_spike["severity"] in {"normal", "moderate"}
    assert single_spike["severity"] != "high"

    steady = None
    for _ in range(3):
        steady = detector.update(
            GaitMetrics(
                cadence_spm=40.0,
                stride_length=0.18,
                step_symmetry=0.32,
                speed_est=0.22,
                num_steps=12,
                window_s=8.0,
            )
        )

    assert steady is not None
    assert steady["severity"] in {"moderate", "high"}


def test_gait_anomaly_detector_personalized_baseline_by_identity_persistence():
    detector = GaitAnomalyDetector(
        warmup_samples=5,
        z_threshold=2.1,
        contamination=0.2,
        enable_unsupervised=False,
    )

    person_a = GaitMetrics(
        cadence_spm=108.0,
        stride_length=0.64,
        step_symmetry=0.97,
        speed_est=1.20,
        num_steps=12,
        window_s=8.0,
    )
    person_b = GaitMetrics(
        cadence_spm=70.0,
        stride_length=0.42,
        step_symmetry=0.88,
        speed_est=0.65,
        num_steps=11,
        window_s=8.0,
    )

    for _ in range(7):
        detector.update(person_a, person_id=101, identity_persistence=1.0)
        detector.update(person_b, person_id=202, identity_persistence=1.0)

    stable_a = detector.update(person_a, person_id=101, identity_persistence=1.0)
    stable_b = detector.update(person_b, person_id=202, identity_persistence=1.0)
    assert stable_a["is_anomaly"] is False
    assert stable_b["is_anomaly"] is False

    switched = detector.update(person_b, person_id=101, identity_persistence=1.0)
    assert switched["is_anomaly"] is True


def test_gait_anomaly_detector_quality_weighting_reduces_low_quality_alerts():
    detector = GaitAnomalyDetector(
        warmup_samples=6,
        z_threshold=2.0,
        contamination=0.2,
        enable_unsupervised=False,
    )

    baseline = GaitMetrics(
        cadence_spm=100.0,
        stride_length=0.58,
        step_symmetry=0.95,
        speed_est=1.05,
        num_steps=12,
        window_s=8.0,
    )
    for _ in range(8):
        detector.update(baseline)

    low_quality = detector.update(
        GaitMetrics(
            cadence_spm=84.0,
            stride_length=0.49,
            step_symmetry=0.84,
            speed_est=0.86,
            num_steps=2,
            window_s=1.2,
        )
    )
    high_quality = detector.update(
        GaitMetrics(
            cadence_spm=84.0,
            stride_length=0.49,
            step_symmetry=0.84,
            speed_est=0.86,
            num_steps=14,
            window_s=9.0,
        )
    )

    assert low_quality["score"] <= high_quality["score"]


def test_gait_anomaly_detector_long_horizon_drift_vs_abrupt_anomaly_response():
    detector = GaitAnomalyDetector(
        warmup_samples=8,
        z_threshold=2.2,
        contamination=0.2,
        enable_unsupervised=False,
    )

    for _ in range(10):
        detector.update(
            GaitMetrics(
                cadence_spm=104.0,
                stride_length=0.60,
                step_symmetry=0.96,
                speed_est=1.08,
                num_steps=12,
                window_s=8.0,
            ),
            person_id=1,
            identity_persistence=1.0,
        )

    drift_anomalies = 0
    for i in range(30):
        out = detector.update(
            GaitMetrics(
                cadence_spm=104.0 - 0.9 * i,
                stride_length=0.60 - 0.005 * i,
                step_symmetry=0.96 - 0.008 * i,
                speed_est=1.08 - 0.018 * i,
                num_steps=12,
                window_s=8.0,
            ),
            person_id=1,
            identity_persistence=1.0,
        )
        drift_anomalies += int(out["is_anomaly"])

    abrupt_detector = GaitAnomalyDetector(
        warmup_samples=8,
        z_threshold=2.2,
        contamination=0.2,
        enable_unsupervised=False,
    )
    for _ in range(10):
        abrupt_detector.update(
            GaitMetrics(
                cadence_spm=104.0,
                stride_length=0.60,
                step_symmetry=0.96,
                speed_est=1.08,
                num_steps=12,
                window_s=8.0,
            )
        )

    abrupt_high = None
    for _ in range(3):
        abrupt_high = abrupt_detector.update(
            GaitMetrics(
                cadence_spm=38.0,
                stride_length=0.18,
                step_symmetry=0.30,
                speed_est=0.20,
                num_steps=12,
                window_s=8.0,
            )
        )

    assert drift_anomalies < 20
    assert abrupt_high is not None
    assert abrupt_high["severity"] in {"moderate", "high"}
