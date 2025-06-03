import cv2
import numpy as np
import ffmpeg
import threading
import time
import logging
import subprocess
import os
import tempfile

class RTMPStreamer:
    """Streams pose visualization via RTMP protocol."""
    
    def __init__(self, rtmp_url=None, width=640, height=480, fps=30):
        self.logger = logging.getLogger("RTMPStreamer")
        
        # If no URL provided, use a default local URL
        self.rtmp_url = rtmp_url or 'rtmp://localhost/live/wifi_radar'
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
        """Start the RTMP streaming thread."""
        if self.running:
            self.logger.warning("Streaming is already running")
            return
            
        self.running = True
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        self.logger.info(f"Started RTMP streaming to {self.rtmp_url}")
        
    def stop(self):
        """Stop the RTMP streaming."""
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
                except:
                    pass
            
            self.ffmpeg_process = None
            
        self.logger.info("Stopped RTMP streaming")
        
    def update_frame(self, pose_data, confidence_data=None, background_color=(0, 0, 0)):
        """Update the current frame with new pose data.
        
        Args:
            pose_data: Dictionary containing keypoints and confidence
            confidence_data: Optional array of confidence values
            background_color: RGB background color
        """
        if pose_data is None:
            return
            
        with self.frame_lock:
            # Create a blank frame
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8)
            frame[:, :] = background_color
            
            # Draw pose skeleton
            keypoints = pose_data['keypoints']
            confidence = pose_data['confidence']
            
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
                (0, 1), (1, 2), (2, 3),  # Right leg
                (0, 4), (4, 5), (5, 6),  # Left leg
                (0, 7),  # Spine
                (7, 8), (8, 9),  # Neck and head
                (7, 10), (10, 11), (11, 12),  # Right arm
                (7, 13), (13, 14), (14, 15)   # Left arm
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
                    cv2.putText(frame, str(i), (x_pos + 5, y_pos - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw skeleton lines
            for edge in edges:
                if valid_mask[edge[0]] and valid_mask[edge[1]]:
                    pt1 = (x_scaled[edge[0]], y_scaled[edge[0]])
                    pt2 = (x_scaled[edge[1]], y_scaled[edge[1]])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # Add border
            cv2.rectangle(frame, (0, 0), (self.width-1, self.height-1),
                         (255, 255, 255), 1)
            
            # Add title
            cv2.putText(frame, "WiFi-Radar: Human Pose Estimation",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
            
            # Add confidence display
            avg_confidence = np.mean(confidence[valid_mask]) if np.any(valid_mask) else 0
            cv2.putText(frame, f"Confidence: {avg_confidence:.2f}",
                       (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (200, 200, 0), 1)
            
            self.latest_frame = frame.copy()
    
    def _stream_loop(self):
        """Main streaming loop that runs in a separate thread."""
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
                    frame = self.latest_frame.copy() if self.latest_frame is not None else blank_frame.copy()
                
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
                sleep_time = max(0, 1.0/self.fps - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            self.logger.error(f"Error in streaming thread: {e}")
            self.running = False
            
    def _initialize_ffmpeg(self):
        """Initialize FFmpeg process for RTMP streaming."""
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
                'ffmpeg',
                '-y',  # Overwrite output files
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',  # Input from pipe
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                self.rtmp_url
            ]
            
            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            self.logger.info("Initialized FFmpeg for RTMP streaming")
            
        except Exception as e:
            self.logger.error(f"Error initializing FFmpeg: {e}")
            self.ffmpeg_process = None
