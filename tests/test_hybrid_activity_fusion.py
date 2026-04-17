import numpy as np

from wifi_radar.analysis.gait_analyzer import GaitMetrics
from wifi_radar.analysis.hybrid_activity_fusion import HybridActivityFusion


def test_hybrid_activity_fusion_detects_stationary_then_walking():
    fusion = HybridActivityFusion(window_sizes=(4, 8))
    still_amp = np.ones((3, 3, 64), dtype=np.float32)
    still_phase = np.zeros((3, 3, 64), dtype=np.float32)

    for _ in range(6):
        result = fusion.update(
            amplitude=still_amp,
            phase=still_phase,
            pose_confidence=np.ones(17, dtype=np.float32) * 0.95,
            gait_metrics=None,
        )

    assert result["activity_label"] == "stationary"
    assert result["motion_score"] < 0.05

    moving = None
    for i in range(10):
        amp = still_amp + np.random.randn(3, 3, 64).astype(np.float32) * 0.25 + i * 0.02
        phase = np.random.randn(3, 3, 64).astype(np.float32) * 0.2
        moving = fusion.update(
            amplitude=amp,
            phase=phase,
            pose_confidence=np.ones(17, dtype=np.float32) * 0.9,
            gait_metrics=GaitMetrics(
                cadence_spm=100.0,
                stride_length=0.6,
                step_symmetry=0.95,
                speed_est=1.1,
                num_steps=8,
                window_s=10.0,
            ),
        )

    assert moving["activity_label"] in {"walking", "high_motion"}
    assert moving["motion_score"] > 0.05


def test_hybrid_activity_fusion_escalates_possible_fall():
    fusion = HybridActivityFusion(window_sizes=(4,))
    amp = np.random.randn(3, 3, 64).astype(np.float32)
    phase = np.random.randn(3, 3, 64).astype(np.float32)

    result = fusion.update(
        amplitude=amp,
        phase=phase,
        pose_confidence=np.ones(17, dtype=np.float32) * 0.4,
        gait_metrics=GaitMetrics(
            cadence_spm=35.0,
            stride_length=0.15,
            step_symmetry=0.4,
            speed_est=0.1,
            num_steps=3,
            window_s=10.0,
        ),
        fall_severity=2,
    )

    assert result["activity_label"] == "possible_fall"
    assert result["fall_risk"] >= 0.8
