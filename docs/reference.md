# References and Research Background

The WiFi-Radar system is based on research in the field of RF-based human sensing, particularly focusing on WiFi signals for pose estimation. This document provides key references that form the theoretical foundation of this project.

## Key Research Papers

1. **DensePose from WiFi**
   - Authors: T. Li, Z. Yang, C. Jun, and C. Xiang
   - Published in: ACM SIGCOMM, 2022
   - [Link to paper](https://dl.acm.org/doi/abs/10.1145/3487552.3487868)
   - Summary: Presents a system for human pose estimation using WiFi signals, without requiring cameras or specialized hardware.

2. **Through-Wall Human Pose Estimation Using Radio Signals**
   - Authors: M. Zhao, T. Li, et al.
   - Published in: IEEE CVPR, 2018
   - [Link to paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.pdf)
   - Summary: Demonstrates human pose estimation through walls using specialized RF devices.

3. **WiFi-Based Human Activity Recognition Using Deep Learning**
   - Authors: Y. Wang, J. Liu, et al.
   - Published in: IEEE Pervasive Computing, 2019
   - [Link to paper](https://ieeexplore.ieee.org/document/8713982)
   - Summary: Explores the use of deep learning for WiFi-based activity recognition.

4. **CSI-Based Human Activity Recognition Using Channel State Information**
   - Authors: S. Yousefi, H. Narui, et al.
   - Published in: ACM Transactions on Computing for Healthcare, 2020
   - [Link to paper](https://dl.acm.org/doi/10.1145/3377816)
   - Summary: Presents techniques for extracting and processing CSI data for human activity recognition.

## Technical Background

### Channel State Information (CSI)

CSI data represents the channel properties of a communication link, containing information about:

- Signal amplitude
- Signal phase
- Signal-to-noise ratio
- Channel frequency response

In 802.11n/ac WiFi systems, CSI is collected for each subcarrier and for each transmitter-receiver antenna pair in MIMO systems.

### WiFi-Based Sensing Principles

WiFi signals (2.4GHz and 5GHz) exhibit specific behaviors when interacting with the human body:

1. **Reflection**: WiFi signals reflect off human bodies
2. **Absorption**: Human bodies absorb some RF energy
3. **Diffraction**: Signals bend around human obstacles
4. **Multipath**: Multiple signal paths create complex patterns

These interactions create measurable changes in the CSI, which can be analyzed to infer human presence, movement, and posture.

### Deep Learning for Pose Estimation

Modern pose estimation from WiFi signals uses deep learning techniques including:

1. **Convolutional Neural Networks (CNNs)**: For spatial feature extraction
2. **Recurrent Neural Networks (RNNs/LSTMs)**: For temporal feature extraction
3. **Dual-branch encoders**: To separately process amplitude and phase data
4. **Dense regression**: To estimate keypoint positions

## Tools and Resources

1. [Linux CSI Tool](https://github.com/spanev/linux-80211n-csitool): Tool for collecting CSI measurements from WiFi devices
2. [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose): Popular framework for pose estimation
3. [DensePose](https://github.com/facebookresearch/DensePose): Research framework for dense human pose estimation

## Future Research Directions

1. **Multi-person tracking**: Improving the ability to track multiple people simultaneously
2. **Fine-grained gestures**: Detecting subtle hand and finger movements
3. **Health monitoring**: Using WiFi-based pose for fall detection and gait analysis
4. **Privacy-preserving monitoring**: Developing systems that can monitor health without cameras
5. **Fusion with other sensors**: Combining WiFi sensing with other modalities for improved accuracy
