"""
ID: WR-DATA-CSI-001
Requirement: Provide a thread-safe queue of (amplitude, phase) CSI frames at
             approximately 20 Hz for downstream signal processing and inference.
Purpose: Abstracts the CSI data source — real router connection or synthetic
         simulation — behind a uniform queue interface so the rest of the system
         is source-agnostic.
Assumptions:
    - 3×3 MIMO, 64 OFDM subcarriers (configurable via constructor).
    - Simulation mode generates plausible multipath-perturbed Rayleigh fading
      with Lissajous-trajectory human presence effects.
    - Real router mode requires a custom firmware that streams raw CSI over TCP.
Constraints: Queue is bounded (buffer_size frames).  Frames dropped when full.
"""
import logging
import socket
import struct
import threading
import time
from queue import Queue

import numpy as np


class CSICollector:
    """Collects Channel State Information (CSI) from a WiFi router or simulation.

    The collector runs a background daemon thread that continuously pushes
    (amplitude, phase) array pairs into a bounded queue.  Callers consume frames
    via ``get_csi_data()`` without needing to manage the thread directly.
    """

    def __init__(self, router_ip="192.168.1.1", port=5500, buffer_size=100):
        """Initialise collector parameters and the internal frame queue.

        Args:
            router_ip:   IPv4 address of the router running the CSI firmware.
                         Ignored in simulation mode.
            port:        TCP port the router firmware listens on.
            buffer_size: Maximum number of unread frames to buffer.
                         Older frames are silently discarded when the queue is full.

        Side Effects:
            Creates a thread-safe Queue but does not start any thread.
            Call ``start()`` to begin collection.
        """
        self.router_ip = router_ip
        self.port = port
        self.buffer_size = buffer_size
        self.csi_data_queue = Queue(maxsize=buffer_size)
        self.running = False
        self.logger = logging.getLogger("CSICollector")

        # ── 3×3 MIMO configuration ────────────────────────────────────────
        self.num_tx = 3           # Transmitting antennas
        self.num_rx = 3           # Receiving antennas
        self.num_subcarriers = 64 # OFDM subcarriers per TX-RX link

        # Flag toggled by start(); simulation thread checks this to select CSI source.
        self.simulation_mode = True

    def start(self, simulation_mode=True):
        """Start CSI data collection in a background daemon thread.

        Args:
            simulation_mode: If True, generate synthetic CSI frames locally.
                             If False, connect to the router at ``router_ip:port``.

        Side Effects:
            Sets ``self.running = True``.
            Spawns a daemon thread (``self.collection_thread``) that writes to
            ``self.csi_data_queue`` until ``stop()`` is called.
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

        # Daemon=True ensures the thread is automatically killed when the main
        # process exits even if stop() was never explicitly called.
        self.collection_thread.daemon = True
        self.collection_thread.start()

    def stop(self):
        """Signal the collection thread to exit and wait up to 1 s for it to join.

        Side Effects:
            Sets ``self.running = False``.
            Joins ``self.collection_thread`` with a 1-second timeout.
        """
        self.running = False
        if hasattr(self, "collection_thread"):
            self.collection_thread.join(timeout=1.0)
        self.logger.info("CSI collector stopped")

    def get_csi_data(self, block=True, timeout=None):
        """Dequeue the next available CSI frame.

        Args:
            block:   If True, block until a frame is available or ``timeout`` expires.
            timeout: Maximum seconds to wait when ``block=True``.  None = wait forever.

        Returns:
            Tuple (amplitude, phase):
              amplitude — (num_tx, num_rx, num_subcarriers) float64 array.
              phase     — (num_tx, num_rx, num_subcarriers) float64 array.
            Returns ``None`` on queue timeout or error.

        Side Effects:
            Removes one item from ``self.csi_data_queue``.
        """
        try:
            return self.csi_data_queue.get(block=block, timeout=timeout)
        except Exception as e:
            self.logger.error(f"Error getting CSI data: {e}")
            return None

    def _collect_csi_data(self):
        """Connect to the router over TCP and stream real CSI frames into the queue.

        Runs in the background collection thread.  The actual frame-parsing logic
        in ``_parse_csi_data()`` is firmware-specific and must be implemented for
        the target router platform (e.g. Atheros/OpenWRT).

        Side Effects:
            Opens a TCP socket to ``(self.router_ip, self.port)``.
            Writes decoded frames to ``self.csi_data_queue``.
            Closes the socket on thread exit (normal or exception).
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.router_ip, self.port))

            while self.running:
                # Receive a raw packet; the maximum useful CSI payload for
                # 3×3 MIMO with 64 subcarriers is well under 8 KiB.
                raw_data = sock.recv(8192)
                if not raw_data:
                    continue

                amplitude, phase = self._parse_csi_data(raw_data)

                if not self.csi_data_queue.full():
                    self.csi_data_queue.put((amplitude, phase))

        except Exception as e:
            self.logger.error(f"Error in CSI collection: {e}")
        finally:
            sock.close()

    def _simulate_csi_data(self):
        """Generate synthetic CSI frames at 20 Hz and push them into the queue.

        Algorithm:
            1. Draw a background Rayleigh-fading amplitude and uniform-random phase.
            2. Compute independent Lissajous-trajectory positions for each simulated
               person so their movement patterns do not overlap.
            3. Delegate to ``_add_simulated_human_presence()`` to modulate the
               background CSI with realistic person-induced multipath perturbations.

        The Lissajous trajectories use different frequency ratios per person
        (0.5 Hz / 0.3 Hz x-axis, shifted by a per-person phase offset) so each
        person traces a distinct elliptical path through the measurement space.

        Side Effects:
            Writes (amplitude, phase) tuples to ``self.csi_data_queue`` at ~20 Hz.
            Reads ``self.sim_num_people`` attribute (set by main.py before start()).
        """
        person_count = getattr(self, "sim_num_people", 1)
        while self.running:
            # ── Background channel: Rayleigh fading + uniform phase ────────
            amplitude = np.random.rayleigh(
                scale=1.0, size=(self.num_tx, self.num_rx, self.num_subcarriers)
            )
            phase = np.random.uniform(
                -np.pi, np.pi, size=(self.num_tx, self.num_rx, self.num_subcarriers)
            )

            t = time.time()
            # ── Lissajous-trajectory positions per person ──────────────────
            # Each person gets a phase offset so their trajectories diverge
            # (offset=2.0 rad per person on the 0.5 Hz / 0.3 Hz figure).
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
        """Parse raw bytes received from the router firmware into amplitude/phase arrays.

        This is a **placeholder** implementation.  The exact byte layout depends on
        the router chipset and firmware (e.g. Atheros ath9k, Intel IWL5300, Nexmon).
        Replace this body with the appropriate struct unpacking once the target
        hardware is known.

        Args:
            raw_data: Raw bytes received from the router TCP socket.

        Returns:
            Tuple (amplitude, phase), each (num_tx, num_rx, num_subcarriers) float64.
        """
        # Return zero arrays as a safe no-op placeholder.
        amplitude = np.zeros((self.num_tx, self.num_rx, self.num_subcarriers))
        phase = np.zeros((self.num_tx, self.num_rx, self.num_subcarriers))
        return amplitude, phase

    def _add_simulated_human_presence(self, amplitude, phase, people=None):
        """Modulate background CSI arrays in-place to simulate human multipath effects.

        Physical model:
            A person at normalised position (x_pos, y_pos) attenuates and phase-shifts
            the signal on TX-RX antenna pairs whose spatial indices are close to the
            person's position.  The effect is modelled as a Gaussian blob in the TX × RX
            index space multiplied by a sinusoidal subcarrier factor that mimics
            frequency-selective fading.

        Vectorised implementation:
            Broadcasting over (tx, 1, 1), (1, rx, 1), and (1, 1, sc) index arrays
            allows the effect to be computed for all antenna pairs and subcarriers
            in a single pass rather than a triple nested loop.

        Args:
            amplitude: (num_tx, num_rx, num_subcarriers) array — modified in-place.
            phase:     (num_tx, num_rx, num_subcarriers) array — modified in-place.
            people:    List of (x_pos, y_pos) tuples in [0, 1].  If None, one
                       person is placed using the current wall-clock time.

        Side Effects:
            Both ``amplitude`` and ``phase`` are modified in-place.
        """
        if people is None:
            t = time.time()
            x_pos = 0.5 + 0.3 * np.sin(t * 0.5)
            y_pos = 0.5 + 0.2 * np.cos(t * 0.3)
            people = [(float(x_pos), float(y_pos))]

        # ── Pre-compute broadcastable index arrays ─────────────────────────
        # Each array has a singleton dimension on the axes it will broadcast over,
        # so NumPy expands them to (num_tx, num_rx, num_subcarriers) automatically.
        tx_idx = np.arange(self.num_tx)[:, None, None] / self.num_tx      # (tx, 1, 1)
        rx_idx = np.arange(self.num_rx)[None, :, None] / self.num_rx      # (1, rx, 1)
        sc_idx = np.arange(self.num_subcarriers)[None, None, :]            # (1, 1, sc)

        # Sinusoidal subcarrier modulation: 4 full cycles across the subcarrier axis
        # approximates the frequency-selective fading caused by multipath reflections.
        sc_factor = np.sin(sc_idx / self.num_subcarriers * np.pi * 4)     # (1, 1, sc)

        for x_pos, y_pos in people:
            # Gaussian proximity: links closest to the person are most affected.
            # Scale factor 10 gives ~60 % attenuation at a normalised distance of 0.3.
            effect_magnitude = 0.2 * np.exp(
                -((tx_idx - x_pos) ** 2 + (rx_idx - y_pos) ** 2) * 10
            )  # shape (tx, rx, 1) — broadcasts over subcarriers

            # Amplitude is multiplicatively scaled (realistic signal attenuation).
            amplitude *= 1 + effect_magnitude * sc_factor * 0.5

            # Phase is additively shifted (realistic time-of-flight change).
            phase    += effect_magnitude * sc_factor * 0.8
