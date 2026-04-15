<div align="center">

# 📡 WiFi-Radar

### Human Pose Estimation Through WiFi Signals

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Detect and track human poses through walls — no cameras required*

</div>

---

## Overview

WiFi-Radar is a Python research system that uses commodity **802.11n/ac WiFi
routers** to estimate human body poses in real time.  By analysing how WiFi
signals (Channel State Information — CSI) reflect off human bodies, the system
can detect presence, estimate 17-keypoint 3-D poses, and track movement
**through walls**, with no cameras and no specialist hardware.

The architecture follows the approach described in
*[DensePose from WiFi](https://dl.acm.org/doi/abs/10.1145/3487552.3487868)*
(Li et al., ACM SIGCOMM 2022).

```
WiFi Router ──► CSI Collector ──► Signal Processor ──► Dual-Branch Encoder
                                                              │
                                                    Pose Estimator (LSTM)
                                                              │
                            ┌─────────────────────────────────┤
                            ▼                                 ▼
                     Dash Dashboard                   RTMP Stream
                  (live 3-D skeleton)            (OBS / media player)
```

---

## Features

| Feature | Status |
|---|---|
| CSI data collection (3×3 MIMO, 64 sub-carriers) | ✅ |
| Simulation mode (no router required) | ✅ |
| Butterworth low-pass + sub-carrier smoothing | ✅ |
| Dual-branch CNN encoder (amplitude + phase) | ✅ |
| LSTM-based temporal pose estimator (17 keypoints) | ✅ |
| Real-time Dash web dashboard | ✅ |
| RTMP video stream (FFmpeg h264) | ✅ |
| Optional pygame house visualizer | ✅ |
| Multi-person detection | 🔜 |
| Pre-trained model weights | 🔜 |

---

## Requirements

### Software

- Python **3.9 or newer**
- CUDA-capable GPU recommended (CPU mode works in simulation)
- FFmpeg installed on the system (for RTMP streaming)

### Hardware (real-world mode)

- 802.11n/ac WiFi access point with CSI extraction support
  - Atheros-based routers running OpenWrt + `ath9k` CSI patch
  - Intel 5300 NIC with [linux-80211n-csitool](https://github.com/spanev/linux-80211n-csitool)
  - 3×3 MIMO configuration is required (3 TX × 3 RX antennas)
- System running Linux with a compatible wireless adapter

> **Tip:** Start with `--simulation` — no router needed.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/hkevin01/wifi-radar.git
cd wifi-radar

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run in simulation mode (opens dashboard at http://localhost:8050)
python main.py --simulation
```

Open **http://localhost:8050** in your browser to see the live 3-D pose dashboard.

---

## Usage

```
usage: main.py [-h] [--simulation] [--router-ip ROUTER_IP]
               [--router-port ROUTER_PORT] [--dashboard-port DASHBOARD_PORT]
               [--rtmp-url RTMP_URL] [--debug] [--config CONFIG]
               [--house-visualization] [--record] [--output-dir OUTPUT_DIR]
               [--replay REPLAY]
```

### Common examples

```bash
# Simulation mode (no hardware required)
python main.py --simulation

# Production — connect to a real router
python main.py --router-ip <YOUR_ROUTER_IP> --router-port 5500

# Enable the pygame house visualizer overlay
python main.py --simulation --house-visualization

# Stream output to OBS / VLC over RTMP
python main.py --simulation --rtmp-url rtmp://localhost/live/wifi_radar

# Record raw CSI data for offline replay
python main.py --simulation --record --output-dir ~/wifi_data

# Replay saved data
python main.py --replay ~/wifi_data/session_001.npy

# Custom config file + debug logging
python main.py --config config.yaml --debug
```

### Configuration file

Copy and edit the default config:

```yaml
# config.yaml
router:
  ip: <YOUR_ROUTER_IP>     # replace with your router's address
  port: 5500
  interface: wlan0
  csi_format: atheros       # "atheros" | "intel"

system:
  simulation_mode: false
  debug: false
  data_dir: ~/.wifi_radar/data

dashboard:
  port: 8050
  update_interval_ms: 100

streaming:
  rtmp_url: rtmp://localhost/live/wifi_radar
  fps: 30
  bitrate: "1000k"
```

---

## Project Structure

```
wifi-radar/
├── main.py                         # Entry point — arg parsing, orchestration
├── wifi_radar/                     # Core Python package
│   ├── data/
│   │   └── csi_collector.py        # CSI capture + simulation (vectorised)
│   ├── models/
│   │   ├── encoder.py              # Dual-branch CNN (amplitude + phase)
│   │   └── pose_estimator.py       # LSTM pose decoder (17 keypoints, 3-D)
│   ├── processing/
│   │   └── signal_processor.py     # Phase unwrap, Butterworth filter, normalization
│   ├── streaming/
│   │   └── rtmp_streamer.py        # FFmpeg subprocess RTMP output
│   ├── utils/
│   │   └── code_quality.py         # Dev helper: lint / format runner
│   └── visualization/
│       ├── dashboard.py            # Dash / Plotly real-time dashboard
│       └── house_visualizer.py     # Optional pygame 3-D house overlay
├── scripts/
│   ├── setup_venv.sh               # One-shot venv + VS Code setup
│   └── check_code.sh               # Lint + type-check wrapper
├── tests/                          # Pytest suite
├── docs/
│   ├── setup-guide.md              # Hardware setup and router flashing guide
│   ├── system_overview.md          # Architecture deep-dive
│   └── reference.md                # Research papers and bibliography
├── requirements.txt                # Runtime dependencies
├── requirements-dev.txt            # Dev / QA dependencies
├── pyproject.toml                  # PEP 517/518 build config + tool settings
└── setup.cfg                       # Flake8 / isort legacy overrides
```

---

## Architecture Detail

### CSI Data Flow

```
Router firmware (ath9k / Intel 5300)
    └─► TCP socket (port 5500)
            └─► CSICollector.get_csi_data()
                    └─► amplitude[3,3,64] + phase[3,3,64]
                            └─► SignalProcessor.process()
                                    ├── phase unwrap (δ-phase tracking)
                                    ├── per-pair amplitude normalisation (z-score)
                                    ├── Butterworth LP filter (order 4, fc=0.2)
                                    └── sub-carrier smoothing (3-tap moving avg)
```

### Neural Network

```
amplitude[3,3,64] ──► Conv branch A (3×Conv2d + BN + ReLU + MaxPool)
                                                                        ─► Fusion
phase[3,3,64]     ──► Conv branch P (3×Conv2d + BN + ReLU + MaxPool)   ─► Conv1×1
                                                                            │
                                                                    FC(256→512)
                                                                            │
                                                                    LSTM(512, 1 layer)
                                                                            │
                                                          keypoints[17,3] + conf[17]
```

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
python wifi_radar/utils/code_quality.py

# Run tests
pytest tests/ -v --cov=wifi_radar

# Format code
black .
isort .
```

---

## Router Setup (Real-World Mode)

See [docs/setup-guide.md](docs/setup-guide.md) for full instructions on:

- Flashing OpenWrt firmware with CSI extraction patches
- Configuring `ath9k` or Intel 5300 NIC tools
- Streaming CSI frames to the collection host
- Antenna placement recommendations

> **Security note:** The router streaming port (default 5500) should be
> firewall-restricted to your local network only.  Never expose it to the
> internet.

---

## Research Background

| Paper | Venue | Summary |
|---|---|---|
| [DensePose from WiFi](https://dl.acm.org/doi/abs/10.1145/3487552.3487868) | SIGCOMM 2022 | WiFi → dense human pose without cameras |
| [Through-Wall Pose Estimation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.pdf) | CVPR 2018 | RF-based through-wall tracking |
| [WiFi Activity Recognition](https://ieeexplore.ieee.org/document/8713982) | IEEE Pervasive 2019 | Deep learning on CSI for activity classification |

Full bibliography: [docs/reference.md](docs/reference.md)

---

## Roadmap

- [ ] Pre-trained model weights (simulation baseline)
- [ ] Multi-person pose clustering
- [ ] Fall detection and gait analysis
- [ ] ONNX export for edge deployment
- [ ] Docker container with RTMP server included
- [ ] Web UI configuration panel

---

## Contributing

Pull requests are welcome!  Please read
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
