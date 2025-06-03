# Visualization

This directory contains modules for visualizing WiFi-Radar results:

- `dashboard.py`: Real-time web dashboard for displaying pose estimation

## Dashboard

The `Dashboard` class provides a real-time web interface for visualizing:
- 3D human pose wireframes
- Confidence metrics
- CSI data visualization
- System status information

## Usage

```python
from wifi_radar.visualization.dashboard import Dashboard

# Initialize dashboard
dashboard = Dashboard()

# Update dashboard with new data
dashboard.update_data(pose_data, confidence_data, csi_data)

# Run dashboard server
dashboard.run(port=8050)
```
