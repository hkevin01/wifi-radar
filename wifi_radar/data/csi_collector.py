import logging
import socket
import struct
import threading
import time
from queue import Queue

import numpy as np


class CSICollector:
    """Collects Channel State Information (CSI) from WiFi router."""

    def __init__(self, router_ip="192.168.1.1", port=5500, buffer_size=100):
        self.router_ip = router_ip
        self.port = port
        self.buffer_size = buffer_size
        self.csi_data_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.logger = logging.getLogger("CSICollector")

        # 3x3 MIMO configuration
        self.num_tx = 3  # Transmitting antennas
        self.num_rx = 3  # Receiving antennas
        self.num_subcarriers = 64  # OFDM subcarriers

        # For simulation mode
        self.simulation_mode = True

    def start(self, simulation_mode=True):
        """Start CSI data collection.

        Args:
            simulation_mode: If True, generate simulated CSI data instead of connecting to router.
        """
        self.simulation_mode = simulation_mode
        self.running = True

        if simulation_mode:
            self.logger.info("Starting CSI collector in simulation mode")
            self.collection_thread = threading.Thread(target=self._simulate_csi_data)
        else:
            self.logger.info(
                f"Starting CSI collector connecting to {self.router_ip}:{self.port}"
            )
            self.collection_thread = threading.Thread(target=self._collect_csi_data)

        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop(self):
        """Stop CSI data collection."""
        self.running = False
        if hasattr(self, "collection_thread"):
            self.collection_thread.join(timeout=1.0)
        self.logger.info("CSI collector stopped")

    def get_csi_data(self, block=True, timeout=None):
        """Get CSI data from the queue.

        Returns:
            Tuple of (amplitude, phase) arrays with shape (num_tx, num_rx, num_subcarriers)
        """
        try:
            return self.csi_data_queue.get(block=block, timeout=timeout)
        except Exception as e:
            self.logger.error(f"Error getting CSI data: {e}")
            return None

    def _collect_csi_data(self):
        """Collect real CSI data from WiFi router."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.router_ip, self.port))

            while self.running:
                # This is a simplified example - actual CSI data format depends on router firmware
                raw_data = sock.recv(8192)
                if not raw_data:
                    continue

                # Parse the raw CSI data (implementation depends on router and protocol)
                amplitude, phase = self._parse_csi_data(raw_data)

                if not self.csi_data_queue.full():
                    self.csi_data_queue.put((amplitude, phase))

        except Exception as e:
            self.logger.error(f"Error in CSI collection: {e}")
        finally:
            sock.close()

    def _simulate_csi_data(self):
        """Generate simulated CSI data for testing (supports 1–4 people)."""
        person_count = getattr(self, "sim_num_people", 1)
        while self.running:
            amplitude = np.random.rayleigh(
                scale=1.0, size=(self.num_tx, self.num_rx, self.num_subcarriers)
            )
            phase = np.random.uniform(
                -np.pi, np.pi, size=(self.num_tx, self.num_rx, self.num_subcarriers)
            )

            t = time.time()
            # Generate per-person positions (independent Lissajous trajectories)
            people_positions = []
            for i in range(person_count):
                offset = i * 2.0  # phase offset so people move independently
                x = 0.5 + 0.3 * np.sin(t * 0.5 + offset)
                y = 0.5 + 0.2 * np.cos(t * 0.3 + offset * 1.3)
                people_positions.append((float(np.clip(x, 0.05, 0.95)),
                                         float(np.clip(y, 0.05, 0.95))))

            self._add_simulated_human_presence(amplitude, phase, people_positions)

            if not self.csi_data_queue.full():
                self.csi_data_queue.put((amplitude, phase))

            time.sleep(0.05)  # 20 Hz sampling rate

    def _parse_csi_data(self, raw_data):
        """Parse raw CSI data from router.

        This is a placeholder implementation. The actual parsing depends on the
        specific format used by the router firmware.
        """
        # Placeholder implementation
        amplitude = np.zeros((self.num_tx, self.num_rx, self.num_subcarriers))
        phase = np.zeros((self.num_tx, self.num_rx, self.num_subcarriers))

        # Parse the raw data into amplitude and phase
        # This would be specific to the router and protocol

        return amplitude, phase

    def _add_simulated_human_presence(self, amplitude, phase, people=None):
        """Add simulated human-presence effects to CSI data (vectorized, multi-person).

        Args:
            amplitude: CSI amplitude array to modify in-place.
            phase:     CSI phase array to modify in-place.
            people:    List of (x_pos, y_pos) tuples in [0, 1].  If None, one
                       person is placed using the internal time state.
        """
        if people is None:
            t = time.time()
            x_pos = 0.5 + 0.3 * np.sin(t * 0.5)
            y_pos = 0.5 + 0.2 * np.cos(t * 0.3)
            people = [(float(x_pos), float(y_pos))]

        tx_idx = np.arange(self.num_tx)[:, None, None] / self.num_tx      # (tx,1,1)
        rx_idx = np.arange(self.num_rx)[None, :, None] / self.num_rx      # (1,rx,1)
        sc_idx = np.arange(self.num_subcarriers)[None, None, :]            # (1,1,sc)
        sc_factor = np.sin(sc_idx / self.num_subcarriers * np.pi * 4)     # (1,1,sc)

        for x_pos, y_pos in people:
            effect_magnitude = 0.2 * np.exp(
                -((tx_idx - x_pos) ** 2 + (rx_idx - y_pos) ** 2) * 10
            )  # (tx, rx, 1) — broadcasts over subcarriers
            amplitude *= 1 + effect_magnitude * sc_factor * 0.5
            phase    += effect_magnitude * sc_factor * 0.8
