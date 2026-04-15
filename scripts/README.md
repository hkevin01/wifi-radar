# Scripts

This directory contains utility scripts for running the WiFi-Radar system:

- `start_wifi_radar.py`: Main script to start the system

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
