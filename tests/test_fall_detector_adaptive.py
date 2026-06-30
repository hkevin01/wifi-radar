import numpy as np

from wifi_radar.analysis.fall_detector import FallDetector


def _make_keypoints(tilt_x: float = 0.0, z_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    conf = np.ones(17, dtype=np.float32)

    # Build a simple torso with configurable X-tilt and Z offset.
    shoulder_z = 1.0 + z_offset
    hip_z = 0.2 + z_offset
    keypoints[5] = np.array([-0.15 + tilt_x, 0.0, shoulder_z], dtype=np.float32)
    keypoints[6] = np.array([0.15 + tilt_x, 0.0, shoulder_z], dtype=np.float32)
    keypoints[11] = np.array([-0.12, 0.0, hip_z], dtype=np.float32)
    keypoints[12] = np.array([0.12, 0.0, hip_z], dtype=np.float32)

    # Populate remaining points around the torso so centroid behaves realistically.
    for idx in range(17):
        if idx in (5, 6, 11, 12):
            continue
        keypoints[idx] = np.array([0.0, 0.0, 0.6 + z_offset], dtype=np.float32)

    return keypoints, conf


def test_fall_detector_adapts_thresholds_to_motion_and_posture():
    detector = FallDetector(
        person_id=2,
        velocity_threshold=-0.2,
        angle_threshold_deg=40.0,
        adaptive_calibration=True,
    )

    # Warm-up with mild vertical jitter and slight forward lean.
    for i in range(36):
        z_jitter = 0.02 * np.sin(i / 2.0)
        kp, conf = _make_keypoints(tilt_x=0.20, z_offset=z_jitter)
        detector.update(kp, conf, timestamp=i * 0.05)

    vel_thr, ang_thr = detector.adaptive_thresholds
    assert vel_thr <= -0.2
    assert 28.0 <= ang_thr <= 64.0


def test_fall_detector_non_adaptive_keeps_configured_thresholds():
    detector = FallDetector(
        person_id=3,
        velocity_threshold=-0.2,
        angle_threshold_deg=40.0,
        adaptive_calibration=False,
    )

    for i in range(36):
        kp, conf = _make_keypoints(tilt_x=0.25, z_offset=0.02 * np.sin(i))
        detector.update(kp, conf, timestamp=i * 0.05)

    vel_thr, ang_thr = detector.adaptive_thresholds
    assert np.isclose(vel_thr, -0.2)
    assert np.isclose(ang_thr, 40.0)
