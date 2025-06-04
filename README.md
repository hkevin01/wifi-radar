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
- 3D house visualization with through-wall tracking
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
python main.py

# Run in simulation mode (no hardware required)
python main.py --simulation

# Run with house visualization
python main.py --simulation --house-visualization

# View the dashboard
# Open your browser to http://localhost:8050

# View the RTMP stream (requires RTMP server or player like VLC)
vlc rtmp://localhost/live/wifi_radar
```

The dashboard will be available at http://localhost:8050

## Code Quality

We use several tools to maintain code quality:

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run code quality checks
python scripts/code_quality.py

# Automatically fix code issues
python scripts/code_quality.py --directory wifi_radar/
```

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
│   └── visualization/    # Dashboard & house visualization
├── .github/              # GitHub workflows and templates
├── .gitignore            # Git ignore rules
├── LICENSE               # License file
├── main.py               # Main entry point
├── README.md             # This file
├── requirements.txt      # Python dependencies
└── setup.py              # Package installation
```

## Visualization

WiFi-Radar offers two main visualization options:

1. **Web Dashboard** - Interactive 3D visualization of pose data and CSI information
2. **House Visualization** - 3D visualization of a house floor plan showing people through walls

To use the house visualization:

```bash
python main.py --simulation --house-visualization
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
