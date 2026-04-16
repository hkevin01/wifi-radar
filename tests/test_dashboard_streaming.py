import numpy as np

from wifi_radar.streaming.rtmp_streamer import RTMPStreamer
from wifi_radar.visualization.dashboard import Dashboard


def _sample_pose():
    keypoints = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        keypoints[i, 0] = (i / 8.0) - 1.0
        keypoints[i, 1] = 0.0
        keypoints[i, 2] = 1.0 - i / 20.0
    conf = np.ones(17, dtype=np.float32)
    return {"keypoints": keypoints, "confidence": conf}


def test_dashboard_regression_flow(tmp_path):
    dashboard = Dashboard(config={"system": {"simulation_mode": True}}, config_path=str(tmp_path / "config.yaml"))
    pose = _sample_pose()
    csi = (
        np.random.randn(3, 3, 64).astype(np.float32),
        np.random.randn(3, 3, 64).astype(np.float32),
    )
    dashboard.update_data(
        pose_data=pose,
        confidence_data=pose["confidence"],
        csi_data=csi,
        tracked_people=[{"person_id": 1, **pose}],
    )
    dashboard.update_events(
        fall_events=[{"message": "fall alert", "severity": 2, "timestamp": 123.0}],
        gait_metrics={"cadence_spm": 96.0, "stride_length": 0.55, "step_symmetry": 0.97, "speed_est": 1.05},
    )

    dashboard.confidence_history.append(float(np.mean(pose["confidence"])))
    pose_fig = dashboard._update_pose_figure(pose)
    conf_fig = dashboard._update_confidence_figure()
    csi_fig = dashboard._update_csi_figure(csi)

    assert len(pose_fig.data) > 0
    assert len(conf_fig.data) > 0
    assert len(csi_fig.data) > 0


def test_rtmp_streamer_renders_frame():
    streamer = RTMPStreamer(width=320, height=240, fps=10)
    pose = _sample_pose()
    streamer.update_frame(pose_data=pose, confidence_data=pose["confidence"])
    assert streamer.latest_frame is not None
    assert streamer.latest_frame.shape == (240, 320, 3)
    assert streamer.latest_frame.sum() > 0
