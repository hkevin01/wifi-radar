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

import cv2
import numpy as np


class RTMPStreamer:
    """Renders pose skeletons to BGR frames and streams them via RTMP/FFmpeg.

    The streamer manages its own background daemon thread that writes frames
    to an FFmpeg subprocess at a fixed rate.  The inference pipeline calls
    ``update_frame()`` to publish new pose data; the stream thread reads the
    latest frame under a lock and pushes it to FFmpeg's stdin.

    Attributes:
        rtmp_url:        Destination RTMP URL (e.g. ``rtmp://localhost/live/wifi_radar``).
        width:           Output frame width in pixels.
        height:          Output frame height in pixels.
        fps:             Target streaming frame rate.
        latest_frame:    Most recently rendered BGR image (numpy array).
        frame_lock:      ``threading.Lock`` protecting ``latest_frame``.
        running:         True while the stream thread is active.
        ffmpeg_process:  Active ``subprocess.Popen`` handle, or None.
    """

    def __init__(self, rtmp_url=None, width=640, height=480, fps=30):
        """Initialise the streamer without starting any threads or processes.

        Args:
            rtmp_url: RTMP destination URL.  Defaults to
                      ``rtmp://localhost/live/wifi_radar`` when None.
            width:    Frame width in pixels (must match FFmpeg ``-s`` argument).
            height:   Frame height in pixels.
            fps:      Target output frame rate used for both FFmpeg and the
                      sleep interval in ``_stream_loop()``.

        Side Effects:
            Creates ``self.frame_lock`` (threading.Lock).  Does NOT start any
            thread or subprocess — call ``start()`` for that.
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

        The thread targets ``_stream_loop()``, which initialises FFmpeg and
        then writes frames at ``self.fps`` until ``stop()`` is called.

        Side Effects:
            Sets ``self.running = True``.
            Spawns ``self.stream_thread`` as a daemon thread so it is
            automatically killed when the main process exits.

        Failure Modes:
            If FFmpeg is not found in PATH, ``_initialize_ffmpeg()`` will log
            the error and set ``self.ffmpeg_process = None``.  The stream
            thread continues running; each write attempt will try to reinitialise.
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

        Lifecycle:
            1. Sets ``self.running = False`` so the stream loop exits cleanly.
            2. Joins the stream thread (timeout = 2 s) to wait for graceful exit.
            3. Closes FFmpeg's stdin pipe to signal end-of-stream, then waits
               up to 5 s for the process to finish encoding and exit.
            4. If the process is still alive after the wait, sends SIGKILL.

        Side Effects:
            Sets ``self.ffmpeg_process = None`` after cleanup.
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

        Rendering pipeline:
            1. Create a blank ``(height, width, 3)`` BGR canvas.
            2. Scale normalised 3-D keypoint coordinates (range [-1, 1]) to pixel
               positions: ``pixel = (norm + 1) / 2 \u00d7 dimension``.
            3. Draw colour-coded keypoint circles (green→red gradient by confidence).
            4. Draw skeleton edges between pairs of valid keypoints.
            5. Overlay a title bar and average-confidence text.

        Thread safety:
            Acquires ``self.frame_lock`` for the duration of rendering and the
            subsequent assignment to ``self.latest_frame``, preventing the stream
            thread from reading a partially-written frame.

        Args:
            pose_data:        Dict with keys ``keypoints`` (17, 3) and
                              ``confidence`` (17,) — normalised 3-D coords and scores.
            confidence_data:  Unused reserved parameter for future per-limb colouring.
            background_color: BGR background tuple (default black).

        Side Effects:
            Updates ``self.latest_frame`` in-place under ``self.frame_lock``.
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
                    cv2.circle(frame, (x_pos, y_pos), 5, color, -1)

                    # Draw keypoint index
                    cv2.putText(
                        frame,
                        str(i),
                        (x_pos + 5, y_pos - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # Draw skeleton lines
            for edge in edges:
                if valid_mask[edge[0]] and valid_mask[edge[1]]:
                    pt1 = (x_scaled[edge[0]], y_scaled[edge[0]])
                    pt2 = (x_scaled[edge[1]], y_scaled[edge[1]])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Add border
            cv2.rectangle(
                frame, (0, 0), (self.width - 1, self.height - 1), (255, 255, 255), 1
            )

            # Add title
            cv2.putText(
                frame,
                "WiFi-Radar: Human Pose Estimation",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Add confidence display
            avg_confidence = (
                np.mean(confidence[valid_mask]) if np.any(valid_mask) else 0
            )
            cv2.putText(
                frame,
                f"Confidence: {avg_confidence:.2f}",
                (10, self.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 0),
                1,
            )

            self.latest_frame = frame.copy()

    def _stream_loop(self):
        """Background daemon thread: push frames to FFmpeg at a fixed frame rate.

        Lifecycle:
            1. Calls ``_initialize_ffmpeg()`` to start the FFmpeg subprocess.
            2. On each iteration:
               a. Acquires ``frame_lock`` and copies the latest rendered frame
                  (or a blank frame if none is available yet).
               b. Writes the raw BGR bytes to ``ffmpeg_process.stdin``.
               c. Reinitialises FFmpeg if the write fails or the process exits.
               d. Sleeps for the remaining time to maintain the target frame rate.
            3. Exits when ``self.running`` is set to False by ``stop()``.

        Side Effects:
            May reinitialise ``self.ffmpeg_process`` on error via
            ``_initialize_ffmpeg()``.  Sets ``self.running = False`` on
            unhandled exception so the outer ``stop()`` sees a clean state.
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

        FFmpeg command rationale:
            ``-f rawvideo -vcodec rawvideo -pix_fmt bgr24``
                Read raw BGR24 bytes from stdin; matches OpenCV's native pixel format.
            ``-s WxH -r FPS``
                Declare the input frame size and rate so FFmpeg knows how to
                interpret the byte stream.
            ``-i -``
                Read from stdin (the pipe written by ``_stream_loop``).
            ``-c:v libx264 -pix_fmt yuv420p``
                Encode with H.264.  ``yuv420p`` is required for broad player
                compatibility (HTML5, VLC, mobile devices).
            ``-preset ultrafast``
                Minimise encoder latency at the cost of slightly larger files;
                acceptable for a live monitoring stream that is never archived.
            ``-f flv``
                FLV container is required by the RTMP protocol.
            stdout/stderr → DEVNULL:
                Suppress FFmpeg's console output to avoid polluting logs.

        Lifecycle:
            If a previous FFmpeg process is still running, its stdin is closed and
            the process is terminated with a 5-second wait before the new one starts.

        Side Effects:
            Sets ``self.ffmpeg_process`` to the new ``subprocess.Popen`` handle,
            or ``None`` if the subprocess could not be created.
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
