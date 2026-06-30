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
