# WiFi-Radar Setup Guide

This guide will help you set up and run the WiFi-Radar system for human pose estimation through WiFi signals.

## Hardware Requirements

- A WiFi router capable of providing CSI (Channel State Information) data
  - Recommended: Nighthawk mesh router or similar devices
  - The router should support 3×3 MIMO capabilities
- A computer with Python 3.8+ installed
- (Optional but recommended) NVIDIA GPU for faster processing

## Router Configuration

### Enabling CSI Collection

To collect CSI data from your router, you'll need to:

1. Install custom firmware on your router that enables CSI extraction
   - For Nighthawk routers: Follow the manufacturer's instructions for firmware updates
   - You may need to install OpenWrt or similar open-source firmware

2. Configure the router to stream CSI data
   ```bash
   # SSH into your router
   ssh admin@192.168.1.1
   
   # Enable CSI tool
   csi-tool enable
   
   # Configure CSI streaming
   csi-tool stream --port 5500 --format binary
   ```

3. Verify that CSI data is being streamed
   ```bash
   # On your computer
   nc -u 192.168.1.1 5500 | hexdump -C
   ```
   You should see continuous data streaming from the router.

## Software Setup

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/wifi-radar.git
   cd wifi-radar
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package
   ```bash
   pip install -e .
   ```

### Configuration

Create a configuration file at `~/.wifi_radar/config.yaml`:

```yaml
router:
  ip: 192.168.1.1
  port: 5500

system:
  simulation_mode: false  # Set to true to use simulated data
  debug: false

dashboard:
  port: 8050

streaming:
  rtmp_url: rtmp://localhost/live/wifi_radar
```

## Running the System

1. Start the WiFi-Radar system
   ```bash
   python scripts/start_wifi_radar.py
   ```

2. For simulation mode (no real WiFi router required)
   ```bash
   python scripts/start_wifi_radar.py --simulation
   ```

3. Access the dashboard
   Open your web browser and navigate to `http://localhost:8050`

4. View the RTMP stream
   You can use a media player like VLC:
   ```bash
   vlc rtmp://localhost/live/wifi_radar
   ```

## Troubleshooting

### No CSI Data Received

- Verify your router is correctly configured for CSI extraction
- Check network connectivity between your computer and router
- Ensure no firewall is blocking the connection
- Try running in simulation mode to verify the rest of the system works

### Performance Issues

- Consider using a GPU for processing
- Reduce the dashboard update frequency
- Lower the resolution of the RTMP stream
- Use a more powerful computer for processing

### Installation Problems

- Make sure you have the correct Python version (3.8+)
- Check that all dependencies are installed correctly
- On Linux, you might need to install additional system packages:
  ```bash
  sudo apt-get install libopencv-dev python3-dev
  ```
