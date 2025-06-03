# WiFi-Radar: Human Pose Estimation through WiFi Signals

A Python implementation for detecting and tracking human poses through walls using WiFi signals.

## Overview

WiFi-Radar uses commodity WiFi routers to capture Channel State Information (CSI) data, which contains phase and amplitude information of WiFi signals. By analyzing how these signals reflect off human bodies, our system can detect human presence, estimate poses, and track movements - even through walls.

## Features

- Capture CSI data from WiFi routers (3×3 MIMO)
- Process phase and amplitude information
- Dual-branch neural network encoder
- Real-time human pose estimation
- Multi-person tracking
- Through-wall detection
- Real-time visualization dashboard
- RTMP streaming for sharing output

## Requirements

- Python 3.8+
- Commodity WiFi router with CSI extraction capability (e.g., Nighthawk mesh router)
- NVIDIA GPU (recommended for real-time performance)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wifi-radar.git
cd wifi-radar

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```bash
# Start the WiFi-Radar system
python scripts/start_wifi_radar.py

# Run in simulation mode (no hardware required)
python scripts/start_wifi_radar.py --simulation

# View the dashboard
# Open your browser to http://localhost:8050

# View the RTMP stream (requires RTMP server or player like VLC)
vlc rtmp://localhost/live/wifi_radar
```

The dashboard will be available at http://localhost:8050

## Project Structure

```
wifi-radar/
├── docs/                 # Documentation
├── scripts/              # Utility scripts
├── tests/                # Unit and integration tests
├── wifi_radar/           # Main package
│   ├── data/             # CSI data collection
│   ├── models/           # Neural network models
│   ├── processing/       # Signal processing
│   ├── streaming/        # RTMP streaming
│   └── visualization/    # Dashboard visualization
├── .github/              # GitHub workflows and templates
├── .gitignore            # Git ignore rules
├── LICENSE               # License file
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── setup.py              # Package installation
```

## Documentation

See the [docs](./docs) directory for detailed documentation:
- [System Overview](./docs/system_overview.md)
- [Setup Guide](./docs/setup_guide.md)
- [Perplexity Labs](./docs/perplexity_labs.md)

## Research Background

This project is based on research in RF-based human sensing, particularly DensePose estimation from WiFi signals. For more information, see the related papers in the [docs/references.md](./docs/references.md) file.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## License

MIT
