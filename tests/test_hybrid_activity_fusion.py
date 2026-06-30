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


def test_hybrid_activity_fusion_hysteresis_reduces_one_frame_flicker():
    rng = np.random.default_rng(123)
    fusion = HybridActivityFusion(window_sizes=(4,), hysteresis_frames=3)
    still_amp = np.ones((3, 3, 64), dtype=np.float32)
    still_phase = np.zeros((3, 3, 64), dtype=np.float32)

    baseline = None
    for _ in range(8):
        baseline = fusion.update(
            amplitude=still_amp,
            phase=still_phase,
            pose_confidence=np.ones(17, dtype=np.float32) * 0.95,
            gait_metrics=None,
        )
    assert baseline["activity_label"] == "stationary"

    # One-frame walking evidence should not immediately flip the stable label.
    transient = fusion.update(
        amplitude=still_amp + rng.normal(0.0, 0.08, (3, 3, 64)).astype(np.float32),
        phase=still_phase + rng.normal(0.0, 0.08, (3, 3, 64)).astype(np.float32),
        pose_confidence=np.ones(17, dtype=np.float32) * 0.9,
        gait_metrics=GaitMetrics(
            cadence_spm=105.0,
            stride_length=0.62,
            step_symmetry=0.96,
            speed_est=1.2,
            num_steps=9,
            window_s=8.0,
        ),
    )
    assert transient["activity_label"] != "walking"

    # Sustained evidence should pass hysteresis and eventually flip.
    converged = None
    for _ in range(3):
        converged = fusion.update(
            amplitude=still_amp + rng.normal(0.0, 0.08, (3, 3, 64)).astype(np.float32),
            phase=still_phase + rng.normal(0.0, 0.08, (3, 3, 64)).astype(np.float32),
            pose_confidence=np.ones(17, dtype=np.float32) * 0.9,
            gait_metrics=GaitMetrics(
                cadence_spm=103.0,
                stride_length=0.60,
                step_symmetry=0.95,
                speed_est=1.1,
                num_steps=9,
                window_s=8.0,
            ),
        )
    assert converged["activity_label"] == "walking"


def test_hybrid_activity_fusion_long_stationary_jitter_bursts_no_false_fall():
    rng = np.random.default_rng(7)
    fusion = HybridActivityFusion(window_sizes=(4, 8, 16), hysteresis_frames=3)
    base_amp = np.ones((3, 3, 64), dtype=np.float32)
    base_phase = np.zeros((3, 3, 64), dtype=np.float32)

    labels = []
    max_risk = 0.0
    for i in range(320):
        burst = 0.09 if (i % 64 in (0, 1)) else 0.01
        amp = base_amp + rng.normal(0.0, burst, (3, 3, 64)).astype(np.float32)
        phase = base_phase + rng.normal(0.0, burst, (3, 3, 64)).astype(np.float32)
        out = fusion.update(
            amplitude=amp,
            phase=phase,
            pose_confidence=np.ones(17, dtype=np.float32) * 0.95,
            gait_metrics=None,
            fall_severity=0,
        )
        labels.append(out["activity_label"])
        max_risk = max(max_risk, float(out["fall_risk"]))

    assert "possible_fall" not in labels
    assert max_risk < 0.8


def test_hybrid_activity_fusion_geometry_confidence_weights_scale():
    rng = np.random.default_rng(99)
    amp0 = np.ones((3, 3, 64), dtype=np.float32)
    phase0 = np.zeros((3, 3, 64), dtype=np.float32)
    amp1 = amp0 + rng.normal(0.0, 0.15, (3, 3, 64)).astype(np.float32)
    phase1 = phase0 + rng.normal(0.0, 0.15, (3, 3, 64)).astype(np.float32)
    layout = {
        "tx_positions": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        "rx_positions": [[0.0, 4.0, 1.0], [1.0, 4.0, 1.0], [2.0, 4.0, 1.0]],
    }

    low = HybridActivityFusion(window_sizes=(4,))
    high = HybridActivityFusion(window_sizes=(4,))

    # Warm-up to align previous-frame state.
    low.update(amplitude=amp0, phase=phase0, pose_confidence=np.ones(17, dtype=np.float32))
    high.update(amplitude=amp0, phase=phase0, pose_confidence=np.ones(17, dtype=np.float32))

    out_low = low.update(
        amplitude=amp1,
        phase=phase1,
        pose_confidence=np.ones(17, dtype=np.float32),
        layout_metadata={**layout, "geometry_confidence": 0.1},
    )
    out_high = high.update(
        amplitude=amp1,
        phase=phase1,
        pose_confidence=np.ones(17, dtype=np.float32),
        layout_metadata={**layout, "geometry_confidence": 0.95},
    )

    assert out_high["geometry_scale"] > out_low["geometry_scale"]
