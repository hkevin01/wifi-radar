"""
ID: WR-STREAM-RTMP-001
Requirement: Encode rendered pose frames in real time and push them to an RTMP
             endpoint so downstream consumers (OBS, VLC, nginx-rtmp, media
             players) can receive a live H.264 stream of the detected poses.
Purpose: Decouples frame rendering from the inference pipeline.  The streamer
         runs in its own daemon thread, consuming the latest rendered frame at
         a fixed frame rate without blocking the data-processing loop.
Architecture:
    update_frame()  (called by processing thread)
          │  thread-safe frame_lock
          ▼
    self.latest_frame  (shared numpy BGR image)
          │
    _stream_loop()  (daemon thread, ~fps iterations/s)
          │
    FFmpeg subprocess  (rawvideo → libx264 → FLV/RTMP)
          │
    RTMP server  (nginx-rtmp, RTMP broker, etc.)
FFmpeg lifecycle:
    _initialize_ffmpeg() — spawns a new subprocess.Popen; old process is
                           terminated first if still alive.
    _stream_loop()       — writes BGR bytes to stdin; restarts FFmpeg on
                           write error or dead process detection.
    stop()               — closes stdin, waits up to 5 s for graceful exit,
                           then SIGKILL if still running.
Constraints:
    - Requires FFmpeg with libx264 support installed in PATH.
    - ``cv2`` (OpenCV) is used only for frame rendering (circle, line, text).
    - Frame writes are non-blocking from the inference thread's perspective;
      if the stream falls behind, the latest frame is re-sent until updated.
"""
import logging
import os
import subprocess
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def _clip_point(width, height, x_pos, y_pos):
    """Clamp a point to the frame bounds."""
    return (
        max(0, min(width - 1, int(x_pos))),
        max(0, min(height - 1, int(y_pos))),
    )


def _draw_circle(frame, center, radius, color, thickness=-1):
    """Draw a filled marker with OpenCV when available, otherwise via NumPy."""
    if cv2 is not None:
        cv2.circle(frame, center, radius, color, thickness)
        return

    x_pos, y_pos = _clip_point(frame.shape[1], frame.shape[0], *center)
    radius = max(1, int(radius))
    y0, y1 = max(0, y_pos - radius), min(frame.shape[0], y_pos + radius + 1)
    x0, x1 = max(0, x_pos - radius), min(frame.shape[1], x_pos + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - x_pos) ** 2 + (yy - y_pos) ** 2 <= radius ** 2
    frame[y0:y1, x0:x1][mask] = color


def _draw_line(frame, pt1, pt2, color, thickness=1):
    """Draw a line segment using OpenCV or a lightweight NumPy fallback."""
    if cv2 is not None:
        cv2.line(frame, pt1, pt2, color, thickness)
        return

    x1, y1 = _clip_point(frame.shape[1], frame.shape[0], *pt1)
    x2, y2 = _clip_point(frame.shape[1], frame.shape[0], *pt2)
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    xs = np.linspace(x1, x2, steps + 1).astype(int)
    ys = np.linspace(y1, y2, steps + 1).astype(int)
    half = max(1, int(thickness)) // 2
    for x_pos, y_pos in zip(xs, ys):
        y0, y1 = max(0, y_pos - half), min(frame.shape[0], y_pos + half + 1)
        x0, x1 = max(0, x_pos - half), min(frame.shape[1], x_pos + half + 1)
        frame[y0:y1, x0:x1] = color


def _draw_rectangle(frame, pt1, pt2, color, thickness=1):
    """Draw a rectangle around the frame bounds."""
    if cv2 is not None:
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        return

    x1, y1 = _clip_point(frame.shape[1], frame.shape[0], *pt1)
    x2, y2 = _clip_point(frame.shape[1], frame.shape[0], *pt2)
    frame[y1:y1 + thickness, x1:x2 + 1] = color
    frame[max(y1, y2 - thickness + 1):y2 + 1, x1:x2 + 1] = color
    frame[y1:y2 + 1, x1:x1 + thickness] = color
    frame[y1:y2 + 1, max(x1, x2 - thickness + 1):x2 + 1] = color


def _put_text(frame, text, origin, scale, color, thickness=1):
    """Render text with OpenCV, or a minimal marker when unavailable."""
    if cv2 is not None:
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )
        return

    x_pos, y_pos = _clip_point(frame.shape[1], frame.shape[0], *origin)
    width = min(frame.shape[1] - x_pos, max(12, int(len(text) * 4 * max(scale, 0.5))))
    height = min(frame.shape[0] - y_pos, max(6, int(8 * max(scale, 0.5))))
    if width > 0 and height > 0:
        frame[y_pos:y_pos + height, x_pos:x_pos + width] = color


class RTMPStreamer:
    """Renders pose skeletons to BGR frames and streams them via RTMP/FFmpeg.

    ID: WR-STREAM-RTMP-CLASS-001
    Requirement: Accept pose keypoint dicts from the inference thread and push
                 a continuous H.264/FLV stream to an RTMP endpoint at a fixed FPS.
    Purpose: Decouple frame rendering from inference so the processing pipeline
             is not blocked by FFmpeg I/O latency.
    Rationale: A shared latest_frame under frame_lock lets the inference thread
               publish new data without waiting for the stream thread to consume it.
    Inputs:
        rtmp_url — str: RTMP destination URL.
        width, height — int: output frame dimensions in pixels.
        fps — int: target stream frame rate.
    Outputs:
        Continuous RTMP H.264 stream to the configured endpoint.
    Preconditions:
        FFmpeg with libx264 must be installed and reachable via PATH.
    Postconditions:
        After start(), stream frames are published at self.fps until stop().
    Assumptions:
        OpenCV (cv2) is installed for frame rendering.
    Constraints:
        Frame writes to FFmpeg stdin are synchronous; high-res / high-FPS may drop.
    References:
        FFmpeg RTMP documentation; OpenCV BGR24 format.
    """

    def __init__(self, rtmp_url=None, width=640, height=480, fps=30):
        """Initialise the streamer without starting any threads or processes.

        ID: WR-STREAM-RTMP-INIT-001
        Requirement: Store configuration parameters and create thread-safety
                     primitives without spawning any threads or subprocesses.
        Purpose: Allow configuration validation and attribute setting before the
                 background thread and FFmpeg process are started.
        Rationale: Separating construction from activation (start()) lets the
                   main thread configure the streamer before the processing loop begins.
        Inputs:
            rtmp_url — Optional[str]: RTMP destination; defaults to
                        'rtmp://localhost/live/wifi_radar'.
            width    — int > 0: output frame width in pixels.
            height   — int > 0: output frame height in pixels.
            fps      — int > 0: target stream frame rate.
        Outputs:
            None — initialises self.
        Preconditions:
            None — no network or subprocess interaction.
        Postconditions:
            self.frame_lock is a threading.Lock.
            self.running == False; self.ffmpeg_process == None.
        Assumptions:
            Caller will call start() before update_frame().
        Side Effects:
            Creates self.frame_lock (threading.Lock).
        Failure Modes:
            None expected at construction time.
        Error Handling:
            None required.
        Constraints:
            width and height must match FFmpeg -s argument exactly.
        Verification:
            Unit test: construct; assert self.running == False and ffmpeg_process is None.
        References:
            threading.Lock; subprocess.Popen; FFmpeg stdin pipe.
        """
        self.logger = logging.getLogger("RTMPStreamer")

        # If no URL provided, use a default local URL
        self.rtmp_url = rtmp_url or "rtmp://localhost/live/wifi_radar"
        self.width = width
        self.height = height
        self.fps = fps

        # Frame generation and streaming
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.stream_thread = None

        # FFmpeg process
        self.ffmpeg_process = None

    def start(self):
        """Spawn the background streaming thread and begin RTMP output.

        ID: WR-STREAM-RTMP-START-001
        Requirement: Set self.running=True and spawn a daemon thread targeting
                     _stream_loop() to begin RTMP frame output.
        Purpose: Activate the streaming pipeline so the inference thread can
                 call update_frame() and have frames published continuously.
        Rationale: Daemon thread ensures automatic cleanup on process exit.
        Inputs:
            None.
        Outputs:
            None.
        Preconditions:
            self.running must be False; start() is idempotent with a warning.
        Postconditions:
            self.running == True; self.stream_thread is a live daemon thread.
        Assumptions:
            FFmpeg is installed and reachable via PATH.
        Side Effects:
            Sets self.running = True.
            Spawns self.stream_thread as a daemon thread.
            Logs INFO message with RTMP URL.
        Failure Modes:
            FFmpeg not in PATH: _initialize_ffmpeg() logs error; stream thread
            continues running and retries on each frame write.
        Error Handling:
            Duplicate start() call: logs WARNING and returns without spawning.
        Constraints:
            Only one stream thread should be active at a time.
        Verification:
            Integration test: start(); sleep(0.5); assert stream_thread.is_alive().
        References:
            threading.Thread daemon=True; _stream_loop(); _initialize_ffmpeg().
        """
        if self.running:
            self.logger.warning("Streaming is already running")
            return

        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()

        self.logger.info(f"Started RTMP streaming to {self.rtmp_url}")

    def stop(self):
        """Stop the streaming thread and terminate the FFmpeg subprocess.

        ID: WR-STREAM-RTMP-STOP-001
        Requirement: Set self.running=False, join the stream thread, close
                     FFmpeg stdin, and wait for the FFmpeg process to exit.
        Purpose: Provide a clean shutdown so FFmpeg finishes encoding the last
                 frames and releases the RTMP connection gracefully.
        Rationale: Closing stdin signals end-of-stream to FFmpeg; a 5-second wait
                   allows the encoder to flush before SIGKILL is used as a fallback.
        Inputs:
            None.
        Outputs:
            None.
        Preconditions:
            start() must have been called; self.stream_thread must exist.
        Postconditions:
            self.running == False; self.ffmpeg_process == None.
        Assumptions:
            FFmpeg responds to stdin close within 5 seconds under normal load.
        Side Effects:
            Sets self.running = False.
            Joins self.stream_thread (timeout=2.0 s).
            Closes ffmpeg_process.stdin and waits up to 5 s; kills if still alive.
            Sets self.ffmpeg_process = None.
        Failure Modes:
            FFmpeg hangs: SIGKILL is sent after the 5-second wait timeout.
        Error Handling:
            Exception on stdin.close()/wait(): caught, logged; process.kill() attempted.
        Constraints:
            Total cleanup time bounded by ~7 s (2 s thread join + 5 s process wait).
        Verification:
            Integration test: start(); stop(); assert ffmpeg_process is None.
        References:
            subprocess.Popen.kill(); threading.Thread.join(timeout).
        """
        self.running = False

        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)

        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=5.0)
            except Exception as e:
                self.logger.error(f"Error stopping FFmpeg process: {e}")
                try:
                    self.ffmpeg_process.kill()
                except Exception:
                    pass

            self.ffmpeg_process = None

        self.logger.info("Stopped RTMP streaming")

    def update_frame(self, pose_data, confidence_data=None, background_color=(0, 0, 0)):
        """Render pose skeleton to a BGR frame and store it for the stream thread.

        ID: WR-STREAM-RTMP-UPDATE-001
        Requirement: Render COCO-17 keypoints and skeleton edges onto a BGR canvas
                     and store the result as self.latest_frame under frame_lock.
        Purpose: Provide the inference thread with a non-blocking way to publish
                 new pose data without waiting for the stream thread.
        Rationale: Holding frame_lock only during rendering and assignment prevents
                   the stream thread from reading a partially-written frame.
        Inputs:
            pose_data        — Dict: {'keypoints': (17,3), 'confidence': (17,)}.
            confidence_data  — reserved; unused in current implementation.
            background_color — Tuple[int,int,int]: BGR background (default black).
        Outputs:
            None — updates self.latest_frame as a side effect.
        Preconditions:
            pose_data must contain 'keypoints' and 'confidence' keys.
        Postconditions:
            self.latest_frame contains the newly rendered BGR frame.
        Assumptions:
            Keypoint coordinates are normalised to [-1, 1] on each axis.
        Side Effects:
            Acquires self.frame_lock.
            Updates self.latest_frame with a new numpy array.
        Failure Modes:
            pose_data is None: returns immediately without updating frame.
            Invalid keypoint shapes: numpy/cv2 raises, propagates to caller.
        Error Handling:
            None return guard for pose_data=None only.
        Constraints:
            cv2 (OpenCV) must be installed for drawing primitives.
        Verification:
            Unit test: call with synthetic pose_data; assert latest_frame shape
            is (height, width, 3) with dtype uint8.
        References:
            cv2.circle, cv2.line, cv2.putText; COCO-17 skeleton edges.
        """
        if pose_data is None:
            return

        with self.frame_lock:
            # Create a blank frame
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :] = background_color

            # Draw pose skeleton
            keypoints = pose_data["keypoints"]
            confidence = pose_data["confidence"]

            # Filter low-confidence keypoints
            threshold = 0.3
            valid_mask = confidence > threshold

            # Scale 3D coordinates to 2D screen coordinates
            # Use only x and y, discard z (or use it for sizing)
            x = keypoints[:, 0]
            y = keypoints[:, 1]

            # Scale to frame dimensions
            x_scaled = ((x + 1) / 2 * self.width).astype(int)
            y_scaled = ((y + 1) / 2 * self.height).astype(int)

            # Define human skeleton connections (same as in dashboard.py)
            edges = [
                (0, 1),
                (1, 2),
                (2, 3),  # Right leg
                (0, 4),
                (4, 5),
                (5, 6),  # Left leg
                (0, 7),  # Spine
                (7, 8),
                (8, 9),  # Neck and head
                (7, 10),
                (10, 11),
                (11, 12),  # Right arm
                (7, 13),
                (13, 14),
                (14, 15),  # Left arm
            ]

            # Draw keypoints
            for i, (x_pos, y_pos) in enumerate(zip(x_scaled, y_scaled)):
                if valid_mask[i]:
                    # Color based on confidence
                    color_value = int(confidence[i] * 255)
                    color = (0, color_value, 255 - color_value)

                    # Draw circle for keypoint
                    _draw_circle(frame, (x_pos, y_pos), 5, color, -1)

                    # Draw keypoint index
                    _put_text(
                        frame,
                        str(i),
                        (x_pos + 5, y_pos - 5),
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # Draw skeleton lines
            for edge in edges:
                if valid_mask[edge[0]] and valid_mask[edge[1]]:
                    pt1 = (x_scaled[edge[0]], y_scaled[edge[0]])
                    pt2 = (x_scaled[edge[1]], y_scaled[edge[1]])
                    _draw_line(frame, pt1, pt2, (0, 255, 0), 2)

            # Add border
            _draw_rectangle(
                frame, (0, 0), (self.width - 1, self.height - 1), (255, 255, 255), 1
            )

            # Add title
            _put_text(
                frame,
                "WiFi-Radar: Human Pose Estimation",
                (10, 30),
                0.7,
                (255, 255, 255),
                2,
            )

            # Add confidence display
            avg_confidence = (
                np.mean(confidence[valid_mask]) if np.any(valid_mask) else 0
            )
            _put_text(
                frame,
                f"Confidence: {avg_confidence:.2f}",
                (10, self.height - 20),
                0.6,
                (200, 200, 0),
                1,
            )

            self.latest_frame = frame.copy()

    def _stream_loop(self):
        """Background daemon thread: push frames to FFmpeg at a fixed frame rate.

        ID: WR-STREAM-RTMP-LOOP-001
        Requirement: Call _initialize_ffmpeg(), then write the latest BGR frame
                     bytes to FFmpeg stdin at self.fps Hz until self.running=False.
        Purpose: Maintain a continuous RTMP stream by periodically pushing the
                 most recently rendered frame independent of the inference rate.
        Rationale: Re-sending the latest frame when no new data arrives ensures
                   the stream does not freeze when inference is slower than the FPS.
        Inputs:
            None — reads self.running, self.latest_frame, self.ffmpeg_process.
        Outputs:
            None — writes BGR bytes to FFmpeg stdin as a side effect.
        Preconditions:
            Called from start() as a daemon thread.
        Postconditions:
            On exit: self.running may be set False on unhandled exception.
        Assumptions:
            self.fps > 0; FFmpeg is reachable.
        Side Effects:
            Calls _initialize_ffmpeg() on entry and on write error.
            Writes frame bytes to self.ffmpeg_process.stdin.
            Sleeps between frames to cap throughput at self.fps.
        Failure Modes:
            Write error to FFmpeg stdin: caught; _initialize_ffmpeg() called to restart.
            Unhandled exception: caught; self.running set False.
        Error Handling:
            BrokenPipeError / IOError on stdin.write: caught; FFmpeg restarted.
            Top-level except: caught, logged; self.running=False.
        Constraints:
            Frame rate limited by FFmpeg encoding latency at 'ultrafast' preset.
        Verification:
            Integration test: start(); wait 2 s; assert stream_thread.is_alive().
        References:
            _initialize_ffmpeg(); FFmpeg rawvideo stdin pipe protocol.
        """
        try:
            # Initialize FFmpeg process
            self._initialize_ffmpeg()

            # Generate blank frame for initialization
            blank_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Stream frames
            while self.running:
                start_time = time.time()

                # Get the latest frame
                with self.frame_lock:
                    frame = (
                        self.latest_frame.copy()
                        if self.latest_frame is not None
                        else blank_frame.copy()
                    )

                # Write frame to FFmpeg
                if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    try:
                        self.ffmpeg_process.stdin.write(frame.tobytes())
                    except Exception as e:
                        self.logger.error(f"Error writing to FFmpeg: {e}")
                        self._initialize_ffmpeg()  # Try to reinitialize
                else:
                    self._initialize_ffmpeg()  # Reinitialize if process is dead

                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, 1.0 / self.fps - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            self.logger.error(f"Error in streaming thread: {e}")
            self.running = False

    def _initialize_ffmpeg(self):
        """Spawn (or restart) the FFmpeg subprocess for RTMP streaming.

        ID: WR-STREAM-RTMP-FFMPEG-001
        Requirement: Start a new FFmpeg subprocess reading rawvideo BGR24 from
                     stdin and pushing H.264/FLV to self.rtmp_url; terminate any
                     existing process first.
        Purpose: Provide a single restart point so the stream thread can recover
                 from FFmpeg crashes without a full stop/start cycle.
        Rationale: stdin=PIPE allows frame bytes to be written by the stream loop;
                   libx264 ultrafast preset minimises encode latency for live monitoring;
                   yuv420p is required for broad H.264 player compatibility.
        Inputs:
            None — reads self.rtmp_url, self.width, self.height, self.fps.
        Outputs:
            None — sets self.ffmpeg_process to new Popen handle.
        Preconditions:
            FFmpeg with libx264 must be installed and reachable via PATH.
        Postconditions:
            self.ffmpeg_process is a live Popen object on success, or None on failure.
        Assumptions:
            The RTMP server at self.rtmp_url is accepting connections.
        Side Effects:
            Terminates existing self.ffmpeg_process if still alive.
            Spawns a new subprocess.Popen with stdin=PIPE.
            Logs INFO on success, ERROR on exception.
        Failure Modes:
            FFmpeg not in PATH: FileNotFoundError caught; self.ffmpeg_process=None.
            RTMP server unreachable: FFmpeg exits quickly; stream loop retries.
        Error Handling:
            All exceptions caught; self.ffmpeg_process set to None on failure.
        Constraints:
            stdout/stderr redirected to DEVNULL to suppress FFmpeg console output.
        Verification:
            Integration test: call _initialize_ffmpeg(); assert poll() is None.
        References:
            subprocess.Popen; FFmpeg -f flv -c:v libx264 -preset ultrafast.
        """
        try:
            if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                # Process is still running, terminate it first
                try:
                    self.ffmpeg_process.stdin.close()
                    self.ffmpeg_process.terminate()
                    self.ffmpeg_process.wait(timeout=5.0)
                except:
                    pass

            # Build FFmpeg command
            command = [
                "ffmpeg",
                "-y",  # Overwrite output files
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "-",  # Input from pipe
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                "-f",
                "flv",
                self.rtmp_url,
            ]

            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self.logger.info("Initialized FFmpeg for RTMP streaming")

        except Exception as e:
            self.logger.error(f"Error initializing FFmpeg: {e}")
            self.ffmpeg_process = None
