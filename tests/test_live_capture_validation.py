import numpy as np

from wifi_radar.data.csi_collector import CSICollector
from wifi_radar.utils.live_capture_validation import validate_capture_file


def test_validate_capture_file_on_npz(tmp_path):
    amplitude = np.random.randn(4, 3, 3, 64).astype(np.float32)
    phase = np.random.randn(4, 3, 3, 64).astype(np.float32)
    path = tmp_path / "live_capture.npz"
    np.savez(path, amplitude=amplitude, phase=phase)

    summary = validate_capture_file(str(path))
    assert summary["frames"] == 4
    assert summary["shape"] == [3, 3, 64]
    assert summary["finite"] is True
    assert summary["processed_ok"] is True


def test_csi_collector_replays_capture_file(tmp_path):
    amplitude = np.random.randn(3, 3, 3, 64).astype(np.float32)
    phase = np.random.randn(3, 3, 3, 64).astype(np.float32)
    path = tmp_path / "replay_capture.npz"
    np.savez(path, amplitude=amplitude, phase=phase)

    collector = CSICollector(buffer_size=8)
    collector.start(simulation_mode=False, replay_file=str(path))
    try:
        frame = collector.get_csi_data(timeout=1.0)
    finally:
        collector.stop()

    assert frame is not None
    amp_frame, phase_frame = frame
    assert amp_frame.shape == (3, 3, 64)
    assert phase_frame.shape == (3, 3, 64)
