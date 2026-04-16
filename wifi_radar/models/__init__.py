"""
ID: WR-PKG-MODELS-001
Requirement: Expose neural network model classes for import by inference,
             training, and export scripts.
Purpose: Group all deep learning models (encoder, pose estimator, tracker)
         under a single importable namespace.
Rationale: Isolating model code in its own subpackage decouples architecture
           changes from data collection and visualisation modules.
Assumptions: PyTorch >= 1.10 is installed; CUDA optional.
References: PEP 328; DualBranchEncoder, PoseEstimator, MultiPersonTracker.

Neural network models for WiFi-Radar.

This package contains deep learning models for processing
WiFi CSI data and estimating human poses.
"""
