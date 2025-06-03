# Data Collection and Processing

This directory contains modules for WiFi CSI data collection and processing:

- `csi_collector.py`: Collects Channel State Information from WiFi routers
- Data preprocessing utilities
- Data storage and management

## CSI Collection

The `CSICollector` class connects to WiFi routers to extract Channel State Information, 
which contains amplitude and phase data for WiFi signals across multiple antennas and subcarriers.

## Usage

```python
from wifi_radar.data.csi_collector import CSICollector

# Initialize collector
collector = CSICollector(router_ip='192.168.1.1', port=5500)

# Start collection
collector.start()

# Get CSI data
amplitude, phase = collector.get_csi_data()

# Stop collection
collector.stop()
```

## Simulation Mode

For development and testing without actual hardware, use simulation mode:

```python
collector = CSICollector()
collector.start(simulation_mode=True)
```
