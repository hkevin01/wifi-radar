"""
ID: WR-PKG-STREAMING-001
Requirement: Expose RTMPStreamer for import by the main inference pipeline.
Purpose: Group RTMP/FFmpeg streaming modules under a single importable namespace.
Rationale: Isolating streaming from inference allows the output channel
           to be changed (e.g. WebRTC, HLS) without touching other modules.
Assumptions: FFmpeg with libx264 is installed and accessible via PATH.
References: RTMPStreamer; FFmpeg RTMP documentation.

Streaming modules for WiFi-Radar.

This package contains modules for streaming pose estimation
results via protocols like RTMP.
"""
