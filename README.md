<div align="center">

# 📡 WiFi-Radar

### Human Pose Estimation Through WiFi Signals

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-1.15%2B-005CED?logo=onnx)](https://onnx.ai)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](docker/docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Detect, track and analyse human poses through walls — no cameras required*

</div>

---

## Overview

WiFi-Radar is a Python research system that uses commodity **802.11n/ac WiFi
routers** to estimate human body poses in real time.  By analysing how WiFi
signals (Channel State Information — CSI) reflect off human bodies, the system
detects presence, estimates 17-keypoint 3-D poses, tracks multiple people with
stable identities, detects falls, and measures gait — **through walls**, with no
cameras and no specialist hardware.

The architecture follows
*[DensePose from WiFi](https://dl.acm.org/doi/abs/10.1145/3487552.3487868)*
(Li et al., ACM SIGCOMM 2022) and extends it with multi-person tracking,
health analytics, ONNX edge export and a full Docker deployment stack.

```
WiFi Router ──► CSI Collector ──► Signal Processor ──► Dual-Branch CNN Encoder
                                                                │
                                                     Multi-Person LSTM Decoder
                                                                │
                                                    MultiPersonTracker (greedy IDs)
                                                                │
                     ┌──────────────────┬─────────────────────┤
                     ▼                  ▼                      ▼
              FallDetector        GaitAnalyzer           House Visualizer
            (4-state FSM)     (step + cadence)           (pygame overlay)
                     │                  │
                     └────────┬─────────┘
                              ▼
                       Dash Dashboard            RTMP Stream → nginx-rtmp → HLS
                 Live Monitor | Events         (OBS / browser / VLC)
                  | Configuration UI
```

---

## Features

| Feature | Status |
|---|---|
| CSI data collection (3×3 MIMO, 64 sub-carriers) | ✅ |
| Simulation mode — no router required (1–4 virtual people) | ✅ |
| Butterworth low-pass + sub-carrier smoothing | ✅ |
| Dual-branch CNN encoder (amplitude + phase) | ✅ |
| LSTM-based temporal pose estimator (17 keypoints, 3-D) | ✅ |
| Pre-trained simulation-baseline weights | ✅ |
| Multi-person detection + stable ID tracking | ✅ |
| Fall detection — velocity + body-angle state machine | ✅ |
| Gait analysis — cadence, stride, symmetry, speed | ✅ |
| ONNX export for edge deployment (Jetson / RPi) | ✅ |
| Real-time Dash dashboard — 3 tabs | ✅ |
| Live Configuration UI — YAML persistence | ✅ |
| Events tab — fall alerts + gait metrics table | ✅ |
| RTMP video stream (FFmpeg h264) | ✅ |
| Docker stack — nginx-rtmp + HLS browser playback | ✅ |
| Optional pygame house visualizer | ✅ |

---

## Requirements

### Software

- Python **3.9 or newer**
- Docker + Docker Compose (optional — for the full stack deployment)
- FFmpeg on the host system (for RTMP push without Docker)
- CUDA-capable GPU recommended; CPU-only works fine in simulation

### Hardware (real-world mode)

- 802.11n/ac access point with CSI extraction firmware
  - Atheros routers running OpenWrt + `ath9k` CSI patch
  - Intel 5300 NIC with [linux-80211n-csitool](https://github.com/spanev/linux-80211n-csitool)
  - 3×3 MIMO configuration (3 TX × 3 RX antennas)
- Linux host with a compatible wireless adapter

> **Tip:** Start with `--simulation` — no router, no GPU needed.

---

## Quick Start

### Option A — Python directly

```bash
# 1. Clone
git clone https://github.com/hkevin01/wifi-radar.git
cd wifi-radar

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. (Optional) Train simulation-baseline weights  (~2 min on CPU)
python scripts/train_simulation_baseline.py

# 5. Run  — dashboard opens at http://localhost:8050
python main.py --simulation
```

### Option B — Docker (includes nginx-rtmp RTMP server)

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Port | Service |
|---|---|
| **8050** | Dash dashboard |
| **1935** | RTMP ingest |
| **8080** | HLS playback + nginx stats |

HLS stream: `http://localhost:8080/hls/wifi_radar.m3u8`

---

## Usage

```
usage: main.py [-h] [--simulation] [--router-ip ROUTER_IP]
               [--router-port ROUTER_PORT] [--dashboard-port DASHBOARD_PORT]
               [--rtmp-url RTMP_URL] [--weights WEIGHTS]
               [--num-people NUM_PEOPLE] [--export-onnx]
               [--house-visualization] [--record] [--output-dir OUTPUT_DIR]
               [--replay REPLAY] [--config CONFIG] [--debug]
```

### Flag reference

| Flag | Default | Description |
|---|---|---|
| `--simulation` | — | Use built-in CSI simulator (no router) |
| `--num-people N` | `1` | Simulated people count (1–4) |
| `--router-ip IP` | `192.168.1.1` | Real router address |
| `--router-port P` | `5500` | CSI TCP port |
| `--weights PATH` | `weights/simulation_baseline.pth` | Checkpoint to load |
| `--export-onnx` | — | Export models to ONNX then exit |
| `--dashboard-port P` | `8050` | Dashboard port |
| `--rtmp-url URL` | `rtmp://localhost/live/wifi_radar` | RTMP push destination |
| `--house-visualization` | — | Enable pygame overlay |
| `--record` | — | Save CSI frames to disk |
| `--output-dir DIR` | `~/wifi_data` | Recording output directory |
| `--replay FILE` | — | Replay a recorded session |
| `--config FILE` | `~/.wifi_radar/config.yaml` | YAML config file |
| `--debug` | — | Verbose logging |

### Common examples

```bash
# Simulation — 2 virtual people
python main.py --simulation --num-people 2

# Load pre-trained weights explicitly
python main.py --simulation --weights weights/simulation_baseline.pth

# Connect to a real router
python main.py --router-ip <YOUR_ROUTER_IP>

# Export models to ONNX for edge deployment
python main.py --export-onnx --weights weights/simulation_baseline.pth

# Full pipeline with RTMP stream
python main.py --simulation --rtmp-url rtmp://localhost/live/wifi_radar

# Record a session
python main.py --simulation --record --output-dir ~/wifi_data

# Replay and analyse recorded data
python main.py --replay ~/wifi_data/session_001.npy
```

### Configuration file

```yaml
# ~/.wifi_radar/config.yaml  (also editable live from the Configuration tab)
router:
  ip: <YOUR_ROUTER_IP>
  port: 5500
  interface: wlan0
  csi_format: atheros        # "atheros" | "intel"

system:
  simulation_mode: true
  debug: false
  max_people: 4
  confidence_threshold: 0.30
  data_dir: ~/.wifi_radar/data

dashboard:
  port: 8050
  update_interval_ms: 100
  max_history: 100

streaming:
  rtmp_url: rtmp://localhost/live/wifi_radar
  fps: 30
  bitrate: "1000k"

fall_detection:
  enabled: true
  velocity_threshold: -0.20   # normalised units / second  (negative = downward)
  angle_threshold_deg: 40.0   # body-from-vertical angle to trigger possible-fall
  alert_timeout_s: 5.0        # seconds without recovery before escalating to ALERT
```

---

## Pre-Trained Weights

A simulation-baseline checkpoint can be generated in ~2 minutes on CPU:

```bash
python scripts/train_simulation_baseline.py
# → writes weights/simulation_baseline.pth
```

Advanced options:

```bash
python scripts/train_simulation_baseline.py \
  --epochs 200 \
  --n-samples 20000 \
  --batch-size 128 \
  --lr 5e-4 \
  --output-dir weights
```

`main.py` loads `weights/simulation_baseline.pth` automatically on startup if it
exists.

---

## Multi-Person Tracking

The `MultiPersonTracker` keeps consistent person IDs across frames using
**greedy nearest-centroid matching**.

```python
from wifi_radar.models.multi_person_tracker import MultiPersonTracker, MultiPersonPoseEstimator

tracker = MultiPersonTracker(
    max_people=4,
    existence_threshold=0.40,
    max_match_distance=0.40,
    id_timeout_frames=10,
)

# tracked is a list of TrackedPerson objects with .person_id, .keypoints, .centroid
tracked = tracker.update(detections, frame_id=n)
```

`MultiPersonPoseEstimator` adds a **bidirectional LSTM backbone** + N independent
hypothesis heads (one per person slot) that each predict:
- `existence` score — is person slot occupied?
- `keypoints` — 17 × 3 coordinates
- `confidence` — per-keypoint score

---

## Fall Detection

The `FallDetector` is a per-person **4-state finite state machine**:

```
NORMAL ──[velocity↓ & angle↑]──► POSSIBLE_FALL
       ──[height drop confirmed]──► FALL_DETECTED
       ──[no recovery in 5 s]────► ALERT
       ──[body upright again]─────► NORMAL
```

```python
from wifi_radar.analysis.fall_detector import FallDetector, FallSeverity

fd = FallDetector(
    person_id=0,
    velocity_threshold=-0.20,    # normalised units/s downward
    angle_threshold_deg=40.0,    # degrees from vertical
    height_drop_frac=0.35,       # fractional Z-drop vs standing height
    alert_timeout_s=5.0,
)

event = fd.update(keypoints, confidence)
if event and event.severity >= FallSeverity.FALL_DETECTED:
    send_alert(event)
```

Fall events surface in the dashboard **Events** tab in real time.

---

## Gait Analysis

`GaitAnalyzer` extracts quantitative gait metrics using **`scipy.signal.find_peaks`**
on ankle keypoint z-trajectories (lowest point = foot-strike).

```python
from wifi_radar.analysis.gait_analyzer import GaitAnalyzer

ga = GaitAnalyzer(history_seconds=10.0, fps=20.0)
ga.update(keypoints, confidence)

metrics = ga.get_metrics()
if metrics:
    print(f"Cadence:   {metrics.cadence_spm:.1f} steps/min")
    print(f"Stride:    {metrics.stride_length:.3f} (norm.)")
    print(f"Symmetry:  {metrics.step_symmetry:.2f}  (1.0 = perfect)")
    print(f"Speed est: {metrics.speed_est:.3f} units/s")
```

Gait metrics appear in the dashboard **Events** tab, updated every 2 seconds.

---

## ONNX Export

Export both models for edge deployment with a single command:

```bash
# Export (requires onnx + onnxruntime)
python scripts/export_onnx.py --weights weights/simulation_baseline.pth

# Validates with onnxruntime automatically (max diff < 1e-4)
# → weights/encoder.onnx
# → weights/pose_estimator.onnx
```

Or via the main entry point:

```bash
python main.py --export-onnx --weights weights/simulation_baseline.pth
```

The exported models use **opset 17** and include **dynamic batch axes**, making
them suitable for Jetson Nano, Raspberry Pi 4 with ONNX Runtime, or any
ONNX-compatible inference engine.

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("weights/encoder.onnx", providers=["CPUExecutionProvider"])
features = sess.run(["features"], {"amplitude": amp_np, "phase": phase_np})[0]
```

---

## Docker Deployment

The full stack (WiFi-Radar + nginx-rtmp RTMP server) runs with one command:

```bash
docker compose -f docker/docker-compose.yml up --build
```

**Services:**

| Container | Image | Role |
|---|---|---|
| `wifi-radar-app` | built from `docker/Dockerfile` | Python app, port 8050 |
| `wifi-radar-rtmp` | `alfg/nginx-rtmp` | RTMP ingest 1935, HLS 8080 |

**Watch the stream in a browser:**
```
http://localhost:8080/hls/wifi_radar.m3u8
```

**Check nginx-rtmp stats:**
```
http://localhost:8080/stat
```

Weights and config are persisted in named Docker volumes
(`wifi_radar_weights`, `wifi_radar_config`).

---

## Dashboard

The Dash dashboard at **http://localhost:8050** has three tabs:

### 📊 Live Monitor
- Real-time 3-D skeleton (Plotly Scatter3d)
- Confidence history and people-count chart
- CSI amplitude + phase signal (TX0·RX0)
- System status indicator

### 🚨 Events
- Fall alert feed with severity badge and timestamp
- Gait metrics table (cadence, stride, symmetry, speed, step count)
- Updates every 2 seconds

### ⚙️ Configuration
- Live-editable settings: router IP, simulation toggle, confidence threshold,
  max people, RTMP URL, stream FPS, fall-detection thresholds
- **Save** writes `~/.wifi_radar/config.yaml` and notifies the running system
- No restart required for most settings

---

## Project Structure

```
wifi-radar/
├── main.py                          # Entry point — arg parsing, orchestration
├── wifi_radar/                      # Core Python package
│   ├── analysis/
│   │   ├── fall_detector.py         # 4-state fall FSM (velocity + angle + height)
│   │   └── gait_analyzer.py         # Step detection + cadence / stride / symmetry
│   ├── config/
│   ├── data/
│   │   └── csi_collector.py         # CSI capture + multi-person simulation
│   ├── models/
│   │   ├── encoder.py               # Dual-branch CNN (amplitude + phase, adaptive pool)
│   │   ├── multi_person_tracker.py  # BiLSTM + N heads + greedy ID tracker
│   │   └── pose_estimator.py        # Single-person LSTM (17 kp, 3-D)
│   ├── processing/
│   │   └── signal_processor.py      # Phase unwrap, Butterworth LP, sub-carrier smooth
│   ├── streaming/
│   │   └── rtmp_streamer.py         # FFmpeg subprocess RTMP push
│   ├── utils/
│   │   ├── code_quality.py          # Dev helper: lint / format
│   │   └── model_io.py              # save_checkpoint / load_checkpoint with metadata
│   └── visualization/
│       ├── dashboard.py             # 3-tab Dash dashboard
│       └── house_visualizer.py      # Optional pygame overlay
├── scripts/
│   ├── export_onnx.py               # Export encoder + pose_estimator → ONNX
│   ├── train_simulation_baseline.py # Train on 8K synthetic CSI/pose pairs
│   ├── setup_venv.sh
│   └── check_code.sh
├── docker/
│   ├── Dockerfile                   # Python 3.11-slim + ffmpeg + OpenCV
│   ├── docker-compose.yml           # App + nginx-rtmp stack
│   └── nginx-rtmp.conf              # RTMP ingest + HLS output config
├── tests/
├── weights/                         # Checkpoint files (gitignored except .gitkeep)
│   └── .gitkeep
├── docs/
│   ├── setup-guide.md
│   ├── system_overview.md
│   └── reference.md
├── requirements.txt                 # Runtime — numpy, torch, dash, onnx, …
├── requirements-dev.txt             # Dev — black, pytest, mypy, …
├── pyproject.toml                   # PEP 517/518 build + tool configs
└── setup.cfg
```

---

## Architecture Detail

### Signal Pipeline

```
Router firmware (ath9k / Intel 5300)
    └─► TCP socket :5500
            └─► CSICollector.get_csi_data()
                    └─► amplitude[3,3,64] + phase[3,3,64]
                            └─► SignalProcessor.process()
                                    ├── phase unwrap (δ-phase tracking)
                                    ├── amplitude z-score normalisation
                                    ├── Butterworth LP (order 4, fc=0.2)
                                    └── 3-tap sub-carrier smoothing
```

### Neural Network

```
amplitude[B,3,3,64] ──► Conv branch A (3×Conv2d + BN + ReLU)
                                                                 ─► Concat
phase[B,3,3,64]     ──► Conv branch P (3×Conv2d + BN + ReLU)   ─► Conv1×1
                                                                     │
                                                           AdaptiveAvgPool2d(3,3)
                                                                     │
                                                               FC(1728 → 256)
                                                                     │
                                                 ┌───────────────────┴──────────────────┐
                                                 ▼                                      ▼
                                     PoseEstimator                        MultiPersonPoseEstimator
                                  LSTM(256→512, 1L)                       BiLSTM(256→512, 2L)
                                          │                                      │
                               keypoints[B,17,3]                     N × (existence + kp[17,3] + conf[17])
                               + confidence[B,17]                     └─► MultiPersonTracker (greedy IDs)
```

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
./scripts/check_code.sh

# Run tests
pytest tests/ -v --cov=wifi_radar

# Format code
black . && isort .

# Train simulation weights (quick — 80 epochs, ~2 min CPU)
python scripts/train_simulation_baseline.py

# Export to ONNX
python scripts/export_onnx.py --weights weights/simulation_baseline.pth
```

---

## Router Setup (Real-World Mode)

See [docs/setup-guide.md](docs/%23%20WiFi-Radar%20Setup%20Guide.md) for:

- Flashing OpenWrt firmware with CSI extraction patches
- Configuring `ath9k` or Intel 5300 CSI tools
- Streaming CSI frames to the collection host
- Antenna placement guidelines

> **Security note:** The CSI streaming port (default 5500) and the RTMP port
> (1935) should be firewall-restricted to your local network only.

---

## Research Background

| Paper | Venue | Summary |
|---|---|---|
| [DensePose from WiFi](https://dl.acm.org/doi/abs/10.1145/3487552.3487868) | SIGCOMM 2022 | WiFi → dense human pose without cameras |
| [Through-Wall Pose Estimation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.pdf) | CVPR 2018 | RF-based through-wall tracking |
| [WiFi Activity Recognition](https://ieeexplore.ieee.org/document/8713982) | IEEE Pervasive 2019 | Deep learning on CSI for activity classification |
| [WiPose](https://dl.acm.org/doi/10.1145/3372224.3380894) | MobiSys 2020 | 3-D body pose via commodity WiFi |

Full bibliography: [docs/reference.md](docs/reference.md)

---

## Contributing

Pull requests are welcome! Please read
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) first.  For major changes,
open an issue to discuss what you'd like to change.

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for
details.

---

<div align="center">
Built with 📡 WiFi signals and 🧠 deep learning
</div>
