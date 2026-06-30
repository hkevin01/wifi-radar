# Scripts

This directory contains utility scripts for running the WiFi-Radar system:

- `start_wifi_radar.py`: Main script to start the system
- `evaluate_hybrid_false_positives.py`: Monte Carlo false-positive evaluation under configurable jitter bursts

## Starting WiFi-Radar

The main entry point for running the system:

```bash
# Basic usage
python scripts/start_wifi_radar.py

# With simulation mode (no real hardware needed)
python scripts/start_wifi_radar.py --simulation

# With custom router settings
python scripts/start_wifi_radar.py --router-ip 192.168.1.1 --router-port 5500

# With custom dashboard port
python scripts/start_wifi_radar.py --dashboard-port 8080

# With custom RTMP URL
python scripts/start_wifi_radar.py --rtmp-url rtmp://streaming-server/live/wifi_radar
```

## Hybrid False-Positive Evaluation

Run reproducible stationary-session simulations with configurable RF jitter and burst patterns.

```bash
python scripts/evaluate_hybrid_false_positives.py \
  --runs 30 \
  --frames 600 \
  --persons 3 \
  --noise-std 0.01 \
  --burst-std 0.12 \
  --burst-every 80 \
  --burst-len 3 \
  --risk-alarm-threshold 0.8 \
  --output-json reports/hybrid_fpr.json
```
