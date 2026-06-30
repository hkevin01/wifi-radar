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


def test_hybrid_activity_fusion_adaptive_floor_suppresses_jitter():
    fusion = HybridActivityFusion(window_sizes=(4, 8), motion_threshold=0.05)
    base_amp = np.ones((3, 3, 64), dtype=np.float32)
    base_phase = np.zeros((3, 3, 64), dtype=np.float32)

    outputs = []
    for _ in range(24):
        amp = base_amp + np.random.randn(3, 3, 64).astype(np.float32) * 0.01
        phase = base_phase + np.random.randn(3, 3, 64).astype(np.float32) * 0.01
        outputs.append(
            fusion.update(
                amplitude=amp,
                phase=phase,
                pose_confidence=np.ones(17, dtype=np.float32) * 0.95,
                gait_metrics=None,
            )
        )

    final = outputs[-1]
    assert final["adaptive_noise_floor"] >= 0.0
    assert final["motion_score"] < 0.08
    assert final["activity_label"] in {"stationary", "transition"}


def test_hybrid_activity_fusion_returns_motion_debug_fields():
    fusion = HybridActivityFusion(window_sizes=(4,))
    amp = np.random.randn(3, 3, 64).astype(np.float32)
    phase = np.random.randn(3, 3, 64).astype(np.float32)
    out = fusion.update(amplitude=amp, phase=phase, pose_confidence=np.ones(17, dtype=np.float32))

    assert "raw_motion_score" in out
    assert "adaptive_noise_floor" in out
