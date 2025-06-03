# Neural Network Models

This directory contains the neural network models used in WiFi-Radar:

- `encoder.py`: Dual-branch encoder for processing amplitude and phase CSI data
- `pose_estimator.py`: Neural network for estimating human poses from encoded features

## Dual-Branch Encoder

The `DualBranchEncoder` processes WiFi CSI data through separate branches for amplitude and phase information, then fuses these features for downstream tasks.

## Pose Estimator

The `PoseEstimator` takes encoded CSI features and produces:
- 3D coordinates for human body keypoints
- Confidence scores for each keypoint
- Support for multi-person detection

## Usage

```python
import torch
from wifi_radar.models.encoder import DualBranchEncoder
from wifi_radar.models.pose_estimator import PoseEstimator

# Initialize models
encoder = DualBranchEncoder()
pose_estimator = PoseEstimator()

# Process data
encoded_features = encoder(amplitude_tensor, phase_tensor)
keypoints, confidence, hidden_state = pose_estimator(encoded_features)

# Detect people
people = pose_estimator.detect_people(keypoints, confidence)
```
