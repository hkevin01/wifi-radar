# Streaming

This directory contains modules for streaming WiFi-Radar output:

- `rtmp_streamer.py`: Streams pose visualization via RTMP protocol

## RTMP Streamer

The `RTMPStreamer` class converts pose data to video frames and streams them via RTMP:
- Real-time video generation from pose data
- FFmpeg integration for video encoding
- Support for custom RTMP destinations

## Usage

```python
from wifi_radar.streaming.rtmp_streamer import RTMPStreamer

# Initialize streamer
streamer = RTMPStreamer(rtmp_url='rtmp://localhost/live/wifi_radar')

# Start streaming
streamer.start()

# Update frames with new pose data
streamer.update_frame(pose_data, confidence_data)

# Stop streaming
streamer.stop()
```
