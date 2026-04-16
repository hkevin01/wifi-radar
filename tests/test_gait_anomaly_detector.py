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
