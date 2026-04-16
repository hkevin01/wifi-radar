import struct
import numpy as np

from wifi_radar.data.csi_collector import CSICollector


def test_parse_csi_data_from_csi0_packet():
    collector = CSICollector()
    amp = np.arange(3 * 3 * 64, dtype=np.float32).reshape(3, 3, 64) / 10.0
    phase = np.linspace(-3.14, 3.14, 3 * 3 * 64, dtype=np.float32).reshape(3, 3, 64)

    raw = b"CSI0" + struct.pack("<III", 3, 3, 64) + amp.tobytes() + phase.tobytes()
    parsed_amp, parsed_phase = collector._parse_csi_data(raw)

    assert parsed_amp.shape == (3, 3, 64)
    assert parsed_phase.shape == (3, 3, 64)
    assert np.allclose(parsed_amp, amp)
    assert np.allclose(parsed_phase, phase)
