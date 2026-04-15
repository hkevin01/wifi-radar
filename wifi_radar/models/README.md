# Neural Network Models

This directory contains the neural network models used in WiFi-Radar:

- Dual-branch encoder for processing amplitude and phase information
- Pose estimation networks
- Multi-person tracking models
- Model training and evaluation utilities

## Key Components

- Convolutional neural networks for spatial feature extraction
- Recurrent neural networks for temporal consistency
- Pose regression networks
- Human detection networks

## Architecture

The WiFi-Radar system uses a dual-branch architecture:
1. Amplitude branch: Processes signal strength patterns
2. Phase branch: Processes phase shift information
3. Fusion module: Combines features from both branches
4. Pose estimator: Predicts human keypoints and poses
