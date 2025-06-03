# System Overview

## Architecture

WiFi-Radar consists of several interconnected components that work together to transform WiFi signals into human pose estimations:

```
CSI Data Acquisition → Signal Processing → Neural Network Processing → Pose Estimation → Visualization/Streaming
```

### 1. CSI Data Acquisition

The system captures Channel State Information (CSI) from WiFi signals using a commodity router with 3×3 MIMO capability. CSI data contains:
- Amplitude information
- Phase information
- For each subcarrier in the WiFi channel
- For each transmitter-receiver antenna pair

### 2. Signal Processing

Raw CSI data undergoes several preprocessing steps:
- Phase sanitization and unwrapping
- Amplitude normalization
- Time and frequency domain filtering
- Dimension reduction

### 3. Neural Network Processing

A dual-branch encoder processes the prepared CSI data:
- Amplitude branch: Captures human body reflection patterns
- Phase branch: Captures fine-grained movement information
- Feature fusion: Combines both types of information

### 4. Pose Estimation

The processed features are fed into:
- A DensePose estimation network
- Person detection and tracking modules
- Confidence scoring mechanism

### 5. Visualization and Streaming

Results are displayed in real-time:
- Interactive dashboard showing human wireframes
- Heatmap overlays for confidence levels
- Multi-person tracking with unique identifiers
- RTMP streaming for external broadcasting

## Key Technical Components

### WiFi CSI Extraction

We use the Linux CSI Tool or similar to extract CSI data from the WiFi router. This requires:
- Modified router firmware
- Driver modifications for CSI extraction
- Network configuration for real-time data streaming

### Signal Processing Pipeline

Our signal processing is implemented in Python using:
- NumPy and SciPy for numerical processing
- Custom filters for noise reduction
- Specialized algorithms for phase unwrapping

### Neural Network Architecture

The pose estimation system uses:
- PyTorch for deep learning implementation
- Custom dual-branch encoder
- DensePose-inspired decoder architecture
- LSTM components for temporal consistency

### Visualization System

The dashboard is built with:
- Dash by Plotly for interactive web components
- Three.js for 3D visualization
- Custom WebSocket implementation for real-time updates

### RTMP Streaming

Video output is handled by:
- FFmpeg for video encoding
- RTMP protocol for streaming
- Custom frame generation from pose data
