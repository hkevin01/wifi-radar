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
from pathlib import Path
from queue import Queue

import numpy as np


class CSICollector:
    """Collects Channel State Information (CSI) from a WiFi router or simulation.

    ID: WR-DATA-CSI-CLASS-001
    Requirement: Provide a thread-safe queue of (amplitude, phase) CSI frames
                 at approximately 20 Hz regardless of whether the data source is
                 a real router or the built-in Rayleigh-fading simulator.
    Purpose: Abstract the CSI data source behind a uniform queue interface so
             downstream signal processing is source-agnostic.
    Rationale: A bounded queue decouples production rate from consumption rate;
               daemon thread ensures clean shutdown without explicit lifecycle calls.
    Inputs:
        router_ip   — str: IPv4 address of the router (simulation mode ignores this).
        port        — int: TCP port for CSI firmware streaming.
        buffer_size — int >= 1: maximum buffered frames before oldest are dropped.
    Outputs:
        (amplitude, phase) tuples via get_csi_data() at ~20 Hz.
    Preconditions:
        call start() before get_csi_data().
    Postconditions:
        After stop(), the collection thread has exited within 1 second.
    Assumptions:
        3x3 MIMO, 64 OFDM subcarriers (configurable via constructor attributes).
    Constraints:
        Queue is bounded; frames are silently dropped when full.
    References:
        Wi-Fi 802.11n CSI Tool; Rayleigh fading channel model.
    """

    def __init__(self, router_ip="192.168.1.1", port=5500, buffer_size=100):
        """Initialise collector parameters and the internal frame queue.

        ID: WR-DATA-CSI-INIT-001
        Requirement: Create all instance attributes and the bounded frame queue
                     without starting any background threads.
        Purpose: Allow the caller to configure the collector before starting
                 collection, separating construction from activation.
        Rationale: Deferred start() lets the main thread set sim_num_people and
                   other attributes before the background thread begins running.
        Inputs:
            router_ip:   str: IPv4 address of the router firmware (ignored in sim mode).
            port:        int in [1, 65535]: TCP port for CSI data streaming.
            buffer_size: int >= 1: maximum buffered (amplitude, phase) frames.
        Outputs:
            None — initialises self.
        Preconditions:
            None — no network connections are made.
        Postconditions:
            self.csi_data_queue is a Queue(maxsize=buffer_size).
            self.running == False; no threads started.
        Assumptions:
            Queue module and threading module are available in the standard library.
        Side Effects:
            Creates threading.Queue object.
        Failure Modes:
            ValueError if buffer_size < 1 (Queue raises).
        Error Handling:
            Queue constructor raises ValueError for invalid maxsize.
        Constraints:
            Buffer size should be tuned to the processing thread's consumption rate.
        Verification:
            Unit test: construct with defaults; assert queue.maxsize == 100.
        References:
            queue.Queue Python standard library; threading module.
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
        self.replay_file = None
        self.record_enabled = False
        self.record_output_dir = None
        self._recorded_amplitude = []
        self._recorded_phase = []

    def start(self, simulation_mode=True, replay_file=None):
        """Start CSI data collection in a background daemon thread.

        ID: WR-DATA-CSI-START-001
        Requirement: Set self.running to True and spawn a daemon thread that
                     writes (amplitude, phase) frames into self.csi_data_queue.
        Purpose: Begin producing CSI frames so downstream consumers can call
                 get_csi_data() and receive data immediately.
        Rationale: Daemon thread ensures automatic cleanup on main-process exit
                   even if stop() is never called.
        Inputs:
            simulation_mode — bool: True = synthetic Rayleigh frames; False = real router.
        Outputs:
            None.
        Preconditions:
            self.running must be False (not already started).
        Postconditions:
            self.running == True.
            self.collection_thread is a live daemon thread.
        Assumptions:
            If simulation_mode=False, the router is reachable at router_ip:port.
        Side Effects:
            Sets self.simulation_mode.
            Spawns self.collection_thread as a daemon thread.
        Failure Modes:
            Real-router mode: socket.connect() may fail if router is unreachable;
            the thread logs the error and exits.
        Error Handling:
            Network errors in the collection thread are caught and logged.
        Constraints:
            Only one collection thread should run at a time.
        Verification:
            Unit test (simulation mode): start(); sleep(0.2); assert queue.qsize() > 0.
        References:
            threading.Thread daemon flag; _simulate_csi_data(), _collect_csi_data().
        """
        self.simulation_mode = simulation_mode
        self.replay_file = replay_file
        self.running = True
        self._recorded_amplitude = []
        self._recorded_phase = []

        if replay_file:
            self.logger.info("Starting CSI collector replay from %s", replay_file)
            self.collection_thread = threading.Thread(
                target=self._replay_csi_data,
                args=(replay_file,),
            )
        elif simulation_mode:
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

        ID: WR-DATA-CSI-STOP-001
        Requirement: Set self.running to False and join the collection thread
                     with a 1-second timeout.
        Purpose: Provide a clean shutdown so the thread releases its socket
                 or simulation resources before the process exits.
        Rationale: A 1-second timeout prevents the main thread from blocking
                   indefinitely if the collection thread hangs.
        Inputs:
            None.
        Outputs:
            None.
        Preconditions:
            start() must have been called; self.collection_thread must exist.
        Postconditions:
            self.running == False.
            self.collection_thread is joined (or timed out).
        Assumptions:
            Collection thread checks self.running in its loop and exits promptly.
        Side Effects:
            Sets self.running = False.
            Calls self.collection_thread.join(timeout=1.0).
            Logs INFO message.
        Failure Modes:
            Thread may not exit within 1 second if blocked on a system call;
            join returns but thread may still be alive.
        Error Handling:
            No exception handling; hasattr guard prevents AttributeError if thread
            was never created.
        Constraints:
            Timeout is fixed at 1.0 second; adjust if the router has higher latency.
        Verification:
            Unit test: start(); stop(); assert self.running == False.
        References:
            threading.Thread.join(timeout); daemon thread cleanup.
        """
        self.running = False
        if hasattr(self, "collection_thread"):
            self.collection_thread.join(timeout=1.0)
        self._flush_recording()
        self.logger.info("CSI collector stopped")

    def get_csi_data(self, block=True, timeout=None):
        """Dequeue the next available CSI frame.

        ID: WR-DATA-CSI-GET-001
        Requirement: Return the next (amplitude, phase) tuple from the queue,
                     blocking or non-blocking based on parameters.
        Purpose: Provide the primary consumer interface for retrieving CSI frames
                 from the background collection thread.
        Rationale: Wrapping Queue.get() in a try/except returns None on timeout
                   instead of raising, simplifying consumer loop logic.
        Inputs:
            block   — bool: if True, block until a frame is available.
            timeout — Optional[float]: seconds to wait when block=True; None = forever.
        Outputs:
            Tuple (amplitude, phase) where each is (num_tx, num_rx, num_sub) float64,
            or None on timeout or exception.
        Preconditions:
            start() must have been called; queue must be populated.
        Postconditions:
            One item removed from self.csi_data_queue.
        Assumptions:
            Consumer thread calls this at approximately the same rate as production.
        Side Effects:
            Removes one item from self.csi_data_queue.
        Failure Modes:
            queue.Empty on timeout returns None (no exception propagated).
        Error Handling:
            All exceptions caught, logged, and None returned.
        Constraints:
            If timeout is too short, None may be returned during startup before
            the queue is populated.
        Verification:
            Unit test: start(sim=True); sleep(0.1); assert get_csi_data() is not None.
        References:
            queue.Queue.get() Python standard library.
        """
        try:
            return self.csi_data_queue.get(block=block, timeout=timeout)
        except Exception as e:
            self.logger.error(f"Error getting CSI data: {e}")
            return None

    def _collect_csi_data(self):
        """Connect to the router over TCP and stream real CSI frames into the queue.

        ID: WR-DATA-CSI-COLLECT-001
        Requirement: Open a TCP socket to (router_ip, port), receive raw packets,
                     parse them with _parse_csi_data(), and push frames to the queue.
        Purpose: Drive real-hardware CSI data collection in the background thread.
        Rationale: Running in a separate daemon thread isolates network blocking
                   calls from the inference pipeline.
        Inputs:
            None — reads self.router_ip, self.port, self.running.
        Outputs:
            None — side effect: writes to self.csi_data_queue.
        Preconditions:
            self.running == True; router is reachable at router_ip:port.
        Postconditions:
            Socket is closed in the finally block.
        Assumptions:
            Maximum useful CSI payload for 3x3 MIMO / 64 subcarriers < 8 KiB.
        Side Effects:
            Opens a TCP socket; writes decoded frames to self.csi_data_queue.
            Closes the socket on exit (normal or exception).
        Failure Modes:
            socket.connect() raises if router is unreachable — caught and logged.
            recv() returns empty bytes if connection is closed — loop skips frame.
        Error Handling:
            Top-level except catches all exceptions; logs error; finally closes socket.
        Constraints:
            Frame dropping: queue.full() check silently discards frames when buffer full.
        Verification:
            Integration test with a mock TCP server sending CSI packets.
        References:
            socket.SOCK_STREAM; _parse_csi_data() placeholder in this class.
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
                self._publish_frame(amplitude, phase)

        except Exception as e:
            self.logger.error(f"Error in CSI collection: {e}")
        finally:
            sock.close()

    def _simulate_csi_data(self):
        """Generate synthetic CSI frames at 20 Hz and push them into the queue.

        ID: WR-DATA-CSI-SIM-001
        Requirement: Produce plausible (amplitude, phase) arrays at ~20 Hz using
                     Rayleigh fading background with Lissajous-trajectory human effects.
        Purpose: Enable full system testing without real router hardware.
        Rationale: Lissajous trajectories with per-person phase offsets produce
                   independent, non-overlapping movement patterns for up to 4 people.
        Inputs:
            None — reads self.running, self.sim_num_people (default 1).
        Outputs:
            None — side effect: writes to self.csi_data_queue at ~20 Hz.
        Preconditions:
            self.running == True.
        Postconditions:
            Queue receives one frame per 50 ms (20 Hz) while running.
        Assumptions:
            sim_num_people attribute set on self before start() is called.
        Side Effects:
            Writes (amplitude, phase) tuples to self.csi_data_queue.
            Calls time.sleep(0.05) per iteration.
        Failure Modes:
            Queue full: frame is silently dropped (checked with queue.full()).
        Error Handling:
            No exception handling inside the loop; thread exits when self.running=False.
        Constraints:
            Simulation accuracy is sufficient for integration testing only.
        Verification:
            Unit test: start(sim=True, num_people=2); assert >10 frames in 1 second.
        References:
            Rayleigh fading: np.random.rayleigh; Lissajous figures.
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
            self._publish_frame(amplitude, phase)

            time.sleep(0.05)  # 20 Hz sampling rate

    def enable_recording(self, output_dir):
        """Enable capture recording to compressed session files.

        ID: WR-DATA-CSI-RECORD-001
        Requirement: Turn on frame recording and ensure the target output
                     directory exists before collection starts.
        Purpose: Preserve live or simulated CSI sessions for later replay,
                 transfer learning, and offline validation.
        Rationale: Writing recordings only when explicitly enabled avoids
                   unnecessary disk I/O during ordinary interactive runs.
        Inputs:
            output_dir — str or path-like: destination directory for .npz captures.
        Outputs:
            None — updates recorder state on self.
        Preconditions:
            Caller supplies a writable filesystem path.
        Postconditions:
            self.record_enabled == True and self.record_output_dir exists.
        Assumptions:
            The operator has sufficient disk space for the capture session.
        Side Effects:
            Creates the output directory if needed.
        Failure Modes:
            Filesystem permission errors propagate from mkdir().
        Error Handling:
            No local catch; caller sees the filesystem exception.
        Constraints:
            Recording is append-in-memory during runtime and flushed on stop().
        Verification:
            Integration test: enable_recording(); start replay; stop(); assert a file is created.
        References:
            pathlib.Path.mkdir; numpy.savez_compressed.
        """
        self.record_enabled = True
        self.record_output_dir = Path(output_dir).expanduser()
        self.record_output_dir.mkdir(parents=True, exist_ok=True)

    def _publish_frame(self, amplitude, phase):
        """Publish one CSI frame to the queue and optional recorder.

        ID: WR-DATA-CSI-PUBLISH-001
        Requirement: Normalise one frame pair to float32, optionally append it
                     to the recording buffers, and enqueue it when capacity allows.
        Purpose: Centralise queue publication so collection, simulation, and replay
                 all share identical output and recording behaviour.
        Rationale: One helper avoids duplication and ensures new ingestion paths
                   cannot forget to honour the recording flag.
        Inputs:
            amplitude — array-like: CSI amplitude tensor for one frame.
            phase     — array-like: CSI phase tensor for one frame.
        Outputs:
            None — writes to queue / recorder as side effects.
        Preconditions:
            amplitude and phase have matching CSI tensor shapes.
        Postconditions:
            Frame is buffered for consumers unless the queue is full.
        Assumptions:
            Frame rate is moderate enough for in-memory buffering.
        Side Effects:
            Appends to self._recorded_amplitude / self._recorded_phase.
            Pushes a tuple into self.csi_data_queue when space is available.
        Failure Modes:
            Queue full: frame is dropped rather than blocking the producer.
        Error Handling:
            No explicit catch; numpy conversion errors propagate.
        Constraints:
            Uses float32 to bound memory usage during long captures.
        Verification:
            Unit test: call _publish_frame(); assert get_csi_data() returns arrays.
        References:
            numpy.asarray; queue.Queue.put.
        """
        amplitude = np.asarray(amplitude, dtype=np.float32)
        phase = np.asarray(phase, dtype=np.float32)

        if self.record_enabled:
            self._recorded_amplitude.append(amplitude.copy())
            self._recorded_phase.append(phase.copy())

        if not self.csi_data_queue.full():
            self.csi_data_queue.put((amplitude, phase))

    def _flush_recording(self):
        """Persist any recorded frames to disk at shutdown.

        ID: WR-DATA-CSI-FLUSH-001
        Requirement: Write the buffered amplitude and phase frame history to a
                     compressed .npz file and clear the in-memory buffers.
        Purpose: Finalise an explicit recording session so captured data can be
                 reused by validation and training workflows.
        Rationale: Delayed flush avoids repeated per-frame disk writes while the
                   collector is running in real time.
        Inputs:
            None — reads recorder state from self.
        Outputs:
            None — writes a compressed capture file as a side effect.
        Preconditions:
            Recording must be enabled and frames must have been buffered.
        Postconditions:
            Buffered frames are saved to disk and the buffers are cleared.
        Assumptions:
            All recorded frames share the same tensor shape.
        Side Effects:
            Creates a timestamped .npz file in the output directory.
        Failure Modes:
            Disk write failure propagates from numpy.savez_compressed().
        Error Handling:
            No local catch; the caller sees the write failure.
        Constraints:
            Uses a timestamped filename to avoid clobbering earlier sessions.
        Verification:
            Integration test: record frames; stop(); open the saved .npz and verify shape.
        References:
            time.strftime; numpy.savez_compressed.
        """
        if not self.record_enabled or not self._recorded_amplitude:
            return

        output_dir = self.record_output_dir or Path("~/wifi_data").expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"csi_capture_{stamp}.npz"
        np.savez_compressed(
            output_path,
            amplitude=np.stack(self._recorded_amplitude, axis=0),
            phase=np.stack(self._recorded_phase, axis=0),
        )
        self.logger.info("Saved %d recorded CSI frames to %s", len(self._recorded_amplitude), output_path)
        self._recorded_amplitude.clear()
        self._recorded_phase.clear()

    def _replay_csi_data(self, replay_file):
        """Replay a previously recorded CSI capture into the live queue.

        ID: WR-DATA-CSI-REPLAY-001
        Requirement: Load a saved capture file, iterate frame-by-frame in time
                     order, and publish each frame into the live queue at ~20 Hz.
        Purpose: Support offline validation and reproducible regression testing
                 using previously recorded live-hardware sessions.
        Rationale: Replaying captures through the same queue interface exercises
                   the production pipeline without requiring router hardware.
        Inputs:
            replay_file — str or path-like: path to a supported capture file.
        Outputs:
            None — writes frames to the live queue as a side effect.
        Preconditions:
            self.running == True and replay_file points to a valid .npz/.npy capture.
        Postconditions:
            All replay frames are published or replay stops early if self.running becomes False.
        Assumptions:
            The capture file contains amplitude/phase tensors with a frame axis.
        Side Effects:
            Imports the validation loader and publishes frames into the queue.
        Failure Modes:
            File missing or malformed: exception caught and logged.
        Error Handling:
            Exceptions are logged and the replay loop exits cleanly.
        Constraints:
            Uses a fixed 50 ms cadence to emulate the live collector rate.
        Verification:
            Unit test: replay a temp .npz file; assert get_csi_data() returns one frame.
        References:
            wifi_radar.utils.live_capture_validation.load_capture_file.
        """
        try:
            from wifi_radar.utils.live_capture_validation import load_capture_file

            amplitude_frames, phase_frames = load_capture_file(str(replay_file))
            for amplitude, phase in zip(amplitude_frames, phase_frames):
                if not self.running:
                    break
                self._publish_frame(amplitude, phase)
                time.sleep(0.05)
        except Exception as exc:
            self.logger.error("Error replaying CSI capture %s: %s", replay_file, exc)
        finally:
            self.running = False

    def _parse_csi_data(self, raw_data):
        """Parse raw bytes received from the router firmware into amplitude/phase arrays.

        ID: WR-DATA-CSI-PARSE-001
        Requirement: Convert a raw byte buffer from the router TCP stream into
                     (amplitude, phase) float32 arrays of shape (num_tx, num_rx, num_sub).
        Purpose: Isolate firmware-specific byte-layout parsing from the collection loop.
        Rationale: Supporting a small set of well-defined binary layouts enables real
                   integration testing while still preserving a safe fallback path.
        Inputs:
            raw_data — bytes: raw packet received from the router TCP socket.
        Outputs:
            Tuple (amplitude, phase), each (num_tx, num_rx, num_sub) float32.
        Preconditions:
            raw_data is a non-empty bytes object.
        Postconditions:
            Returns arrays of the correct shape when the payload matches one of the
            supported formats; otherwise returns zeros and logs a warning.
        Assumptions:
            Supported formats are:
              1. "CSI0" + <III dims> + float32 amplitude + float32 phase.
              2. Raw complex64 tensor bytes of shape (num_tx, num_rx, num_sub).
              3. Raw float32 bytes containing amplitude then phase back-to-back.
        Side Effects:
            Logs a warning for malformed or unsupported packets.
        Failure Modes:
            Malformed packet or wrong byte count: safe zero fallback returned.
        Error Handling:
            All unpack/reshape errors are caught and converted into warnings.
        Constraints:
            Real chipset-specific formats may still require an adapter layer.
        Verification:
            Unit test: send a CSI0 packet and confirm exact round-trip values.
        References:
            struct module; Linux 802.11n CSI Tool; Nexmon CSI extraction.
        """
        expected = self.num_tx * self.num_rx * self.num_subcarriers
        try:
            if not raw_data:
                raise ValueError("empty CSI packet")

            if raw_data[:4] == b"CSI0":
                header_size = 4 + struct.calcsize("<III")
                tx, rx, sub = struct.unpack("<III", raw_data[4:header_size])
                count = tx * rx * sub
                float_data = np.frombuffer(raw_data[header_size:], dtype=np.float32)
                if float_data.size != count * 2:
                    raise ValueError("CSI0 packet does not contain amplitude+phase payload")
                amplitude = float_data[:count].reshape(tx, rx, sub)
                phase = float_data[count:].reshape(tx, rx, sub)
                return amplitude.astype(np.float32), phase.astype(np.float32)

            complex_data = np.frombuffer(raw_data, dtype=np.complex64)
            if complex_data.size == expected:
                complex_data = complex_data.reshape(self.num_tx, self.num_rx, self.num_subcarriers)
                return np.abs(complex_data).astype(np.float32), np.angle(complex_data).astype(np.float32)

            float_data = np.frombuffer(raw_data, dtype=np.float32)
            if float_data.size == expected * 2:
                amplitude = float_data[:expected].reshape(self.num_tx, self.num_rx, self.num_subcarriers)
                phase = float_data[expected:].reshape(self.num_tx, self.num_rx, self.num_subcarriers)
                return amplitude.astype(np.float32), phase.astype(np.float32)

            raise ValueError(f"unsupported CSI packet length: {len(raw_data)} bytes")
        except Exception as exc:
            self.logger.warning("Failed to parse CSI packet: %s", exc)
            amplitude = np.zeros((self.num_tx, self.num_rx, self.num_subcarriers), dtype=np.float32)
            phase = np.zeros((self.num_tx, self.num_rx, self.num_subcarriers), dtype=np.float32)
            return amplitude, phase

    def _add_simulated_human_presence(self, amplitude, phase, people=None):
        """Modulate background CSI arrays in-place to simulate human multipath effects.

        ID: WR-DATA-CSI-HUM-001
        Requirement: Modify amplitude and phase arrays in-place to add person-induced
                     multipath effects modelled as Gaussian spatial blobs.
        Purpose: Create realistic-looking synthetic CSI data for training and integration
                 testing without real hardware.
        Rationale: Gaussian proximity model captures signal attenuation near a person;
                   sinusoidal subcarrier factor mimics frequency-selective fading from
                   multipath reflections.
        Inputs:
            amplitude — (num_tx, num_rx, num_sub) float64 array: modified in-place.
            phase     — (num_tx, num_rx, num_sub) float64 array: modified in-place.
            people    — Optional[List[Tuple[float,float]]]: (x_pos, y_pos) in [0,1].
                        If None, one person placed at current time-based Lissajous pos.
        Outputs:
            None — modifies amplitude and phase in-place.
        Preconditions:
            amplitude and phase have shape (num_tx, num_rx, num_sub).
        Postconditions:
            amplitude and phase contain person-induced multipath modulation.
        Assumptions:
            x_pos and y_pos are clipped to [0.05, 0.95] for stability.
        Side Effects:
            Modifies amplitude and phase arrays in-place.
        Failure Modes:
            Out-of-range positions are handled by Gaussian decay naturally.
        Error Handling:
            No exception handling; numpy operations propagate errors.
        Constraints:
            Vectorised over all antenna pairs and subcarriers in a single pass.
        Verification:
            Unit test: apply with one person at (0.5, 0.5); assert amplitude differs
            from Rayleigh baseline near the centre antenna pairs.
        References:
            Rayleigh fading Gaussian model; sinusoidal subcarrier factor approximation.
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
