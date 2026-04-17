# WiFi-Radar Research Watch — January to April 2026

## Recent papers worth tracking

| Date | Work | Main idea | Relevance to this repo |
|---|---|---|---|
| 2026-01-18 | PerceptAlign / geometry-aware cross-layout pose estimation | Explicitly conditions the model on transceiver geometry to reduce layout overfitting | Strong fit for real-hardware deployment and replay validation |
| 2026-02-09 | WiFlow | Lightweight spatio-temporal decoupling for continuous WiFi pose estimation | Strong fit for low-latency and edge inference |
| 2026-02-26 | WiPowerSys | ESP32-based CSI capture with skeleton supervision | Strong fit for commodity deployment workflows |
| 2026-04-01 | MKFi | Multi-window fusion for temporally robust WiFi activity recognition under limited data | Strong fit for fall and gait robustness |

## Project takeaway

The most practical hybrid direction for this repository is:

1. keep the existing CSI-to-pose backbone,
2. add geometry-aware normalization where layout metadata exists,
3. fuse short-window and long-window CSI motion evidence with pose confidence and gait metrics.

This repository now includes a lightweight implementation of that idea through the hybrid activity fusion stage in the analysis package.
