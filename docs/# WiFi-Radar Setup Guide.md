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
   - For TP-Link Archer series: Install OpenWrt and the `atheros-csi-tool` package
   - For ASUS routers: Use the Merlin firmware with CSI extraction patches
   - You may need to install OpenWrt or similar open-source firmware

2. Configure the router to stream CSI data
   ```bash
   # SSH into your router
   ssh admin@192.168.1.1
   
   # Enable CSI tool (Nighthawk method)
   csi-tool enable
   
   # Configure CSI streaming
   csi-tool stream --port 5500 --format binary
   
   # Alternative for OpenWrt-based routers
   atheros-csi-tool -i wlan0 -s 1 -c 1 -p 5500 &
   
   # Alternative for ASUS routers
   asus-csi-extractor --iface eth1 --port 5500 --daemon
   ```

3. Verify that CSI data is being streamed
   ```bash
   # On your computer
   nc -u 192.168.1.1 5500 | hexdump -C
   
   # Or use the provided testing utility
   python scripts/test_csi_connection.py --router-ip 192.168.1.1 --port 5500
   ```
   You should see continuous data streaming from the router.

4. Troubleshooting CSI collection:
   - Ensure your router model supports CSI extraction (most 802.11n/ac routers with supported chipsets)
   - Check that your router is operating in a supported WiFi mode (typically 802.11n/ac)
   - Try different antenna configurations if available
   - Some routers require specific channel width settings (20MHz, 40MHz, or 80MHz)
   - Verify that no other applications are using the specified port

5. CSI data format considerations:
   - Different routers may provide CSI in different formats
   - The WiFi-Radar system expects a specific format, which may require conversion
   - See the documentation for your specific router model
   - Our tool supports common formats including Intel, Atheros, and Broadcom CSI formats

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
  interface: wlan0       # Network interface to use
  csi_format: atheros    # Router CSI format (atheros, intel, broadcom)

system:
  simulation_mode: false  # Set to true to use simulated data
  debug: false
  log_level: info        # Logging level (debug, info, warning, error)
  data_dir: ~/.wifi_radar/data  # Directory to store collected data

dashboard:
  port: 8050
  theme: darkly         # Dashboard theme (darkly, flatly, etc.)
  update_interval_ms: 100  # Dashboard update interval in milliseconds
  max_history: 100      # Number of data points to keep in history

streaming:
  rtmp_url: rtmp://localhost/live/wifi_radar
  width: 640           # Stream resolution width
  height: 480          # Stream resolution height
  fps: 30              # Frames per second
  bitrate: 1000k       # Stream bitrate
```

You can create the configuration directory if it doesn't exist:

```bash
mkdir -p ~/.wifi_radar
```

#### Configuration Templates

We provide several configuration templates for different use cases:

```bash
# Copy the default configuration
cp config/default.yaml ~/.wifi_radar/config.yaml

# For simulation mode
cp config/simulation.yaml ~/.wifi_radar/config.yaml

# For high-performance mode
cp config/performance.yaml ~/.wifi_radar/config.yaml
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

3. Additional command-line options:
   ```bash
   # Specify configuration file location
   python scripts/start_wifi_radar.py --config /path/to/config.yaml
   
   # Run with debug logging
   python scripts/start_wifi_radar.py --debug
   
   # Specify dashboard port
   python scripts/start_wifi_radar.py --dashboard-port 8080
   
   # Record CSI data to file
   python scripts/start_wifi_radar.py --record --output-dir ~/wifi_data
   
   # Replay previously recorded CSI data
   python scripts/start_wifi_radar.py --replay ~/wifi_data/session_2023-05-15.dat
   ```

4. Running as a service (Linux):
   ```bash
   # Copy the service file
   sudo cp config/wifi-radar.service /etc/systemd/system/
   
   # Edit the service file to match your installation path
   sudo nano /etc/systemd/system/wifi-radar.service
   
   # Enable and start the service
   sudo systemctl enable wifi-radar
   sudo systemctl start wifi-radar
   ```

5. Access the dashboard
   Open your web browser and navigate to `http://localhost:8050`

6. View the RTMP stream
   You can use a media player like VLC:
   ```bash
   vlc rtmp://localhost/live/wifi_radar
   
   # With additional VLC options
   vlc rtmp://localhost/live/wifi_radar --network-caching=100 --live-caching=100
   
   # Save the stream to a file
   vlc rtmp://localhost/live/wifi_radar --sout file/mp4:wifi_radar_recording.mp4
   ```

7. Controlling the system via API:
   ```bash
   # Get system status
   curl http://localhost:8050/api/status
   
   # Change detection parameters
   curl -X POST http://localhost:8050/api/config -d '{"sensitivity": 0.8}'
   
   # Restart data collection
   curl -X POST http://localhost:8050/api/restart
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
