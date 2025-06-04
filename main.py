#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from wifi_radar.data.csi_collector import CSICollector
from wifi_radar.models.encoder import DualBranchEncoder
from wifi_radar.models.pose_estimator import PoseEstimator
from wifi_radar.processing.signal_processor import SignalProcessor
from wifi_radar.streaming.rtmp_streamer import RTMPStreamer
from wifi_radar.visualization.dashboard import Dashboard
from wifi_radar.visualization.house_visualizer import HouseVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WiFi-Radar: Human Pose Estimation through WiFi Signals"
    )

    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode (no real CSI data)",
    )
    parser.add_argument(
        "--router-ip",
        type=str,
        default="192.168.1.1",
        help="IP address of the WiFi router",
    )
    parser.add_argument(
        "--router-port", type=int, default=5500, help="Port for CSI data collection"
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=8050, help="Port for the web dashboard"
    )
    parser.add_argument(
        "--rtmp-url",
        type=str,
        default="rtmp://localhost/live/wifi_radar",
        help="RTMP URL for streaming",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--house-visualization",
        action="store_true",
        help="Enable house visualization GUI",
    )
    parser.add_argument("--record", action="store_true", help="Record CSI data to file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/wifi_data",
        help="Directory to save recorded data",
    )
    parser.add_argument("--replay", type=str, help="Replay recorded CSI data from file")

    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration.

    Args:
        debug: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("wifi_radar.log")],
    )


def load_config(config_path=None):
    """Load configuration from file or use defaults."""
    import yaml

    # Default configuration
    config = {
        "router": {
            "ip": "192.168.1.1",
            "port": 5500,
            "interface": "wlan0",
            "csi_format": "atheros",
        },
        "system": {
            "simulation_mode": False,
            "debug": False,
            "log_level": "info",
            "data_dir": os.path.expanduser("~/.wifi_radar/data"),
        },
        "dashboard": {
            "port": 8050,
            "theme": "darkly",
            "update_interval_ms": 100,
            "max_history": 100,
        },
        "streaming": {
            "rtmp_url": "rtmp://localhost/live/wifi_radar",
            "width": 640,
            "height": 480,
            "fps": 30,
            "bitrate": "1000k",
        },
        "house_visualization": {
            "enabled": False,
            "width": 800,
            "height": 600,
            "fps": 30,
            "wall_transparency": 0.5,
        },
    }

    # If config path is provided, load and override defaults
    if config_path:
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)

            # Update config with user settings
            for section, settings in user_config.items():
                if section in config:
                    config[section].update(settings)
                else:
                    config[section] = settings

            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")

    return config


def main():
    """Main function to run the WiFi-Radar system."""
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger("WiFi-Radar")
    logger.info("Starting WiFi-Radar system")

    # Load configuration
    config_path = args.config or os.path.expanduser("~/.wifi_radar/config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = load_config()

    # Override config with command line arguments
    if args.simulation:
        config["system"]["simulation_mode"] = True
    if args.router_ip:
        config["router"]["ip"] = args.router_ip
    if args.router_port:
        config["router"]["port"] = args.router_port
    if args.dashboard_port:
        config["dashboard"]["port"] = args.dashboard_port
    if args.rtmp_url:
        config["streaming"]["rtmp_url"] = args.rtmp_url
    if args.house_visualization:
        config["house_visualization"]["enabled"] = True

    # Initialize data collection
    logger.info("Initializing CSI data collection")
    csi_collector = CSICollector(
        router_ip=config["router"]["ip"], port=config["router"]["port"]
    )

    # Initialize signal processing
    logger.info("Initializing signal processing")
    signal_processor = SignalProcessor()

    # Initialize neural network models
    logger.info("Initializing neural network models")
    encoder = DualBranchEncoder()
    pose_estimator = PoseEstimator()

    # Initialize visualization
    logger.info("Initializing visualization")
    dashboard = Dashboard(
        update_interval_ms=config["dashboard"]["update_interval_ms"],
        max_history=config["dashboard"]["max_history"],
    )

    # Initialize RTMP streaming
    logger.info("Initializing RTMP streaming")
    rtmp_streamer = RTMPStreamer(
        rtmp_url=config["streaming"]["rtmp_url"],
        width=config["streaming"]["width"],
        height=config["streaming"]["height"],
        fps=config["streaming"]["fps"],
    )

    # Initialize house visualization if enabled
    house_visualizer = None
    if config["house_visualization"]["enabled"]:
        logger.info("Initializing house visualization")
        house_visualizer = HouseVisualizer(
            width=config["house_visualization"]["width"],
            height=config["house_visualization"]["height"],
            fps=config["house_visualization"]["fps"],
            wall_transparency=config["house_visualization"]["wall_transparency"],
        )

    # Start components
    try:
        # Start data collection
        logger.info("Starting CSI data collection")
        csi_collector.start(simulation_mode=config["system"]["simulation_mode"])

        # Start RTMP streaming
        logger.info("Starting RTMP streaming")
        rtmp_streamer.start()

        # Start house visualization if enabled
        if house_visualizer:
            logger.info("Starting house visualization")
            house_visualizer.start()

        # Create data processing thread
        import threading

        import numpy as np
        import torch

        def processing_thread():
            logger.info("Starting data processing thread")

            # For temporal consistency
            hidden_state = None

            try:
                while True:
                    # Get CSI data
                    csi_data = csi_collector.get_csi_data(block=True, timeout=1.0)
                    if csi_data is None:
                        continue

                    amplitude, phase = csi_data

                    # Process signal
                    processed_amplitude, processed_phase = signal_processor.process(
                        amplitude, phase
                    )

                    # Convert to PyTorch tensors
                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    amplitude_tensor = (
                        torch.from_numpy(processed_amplitude)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )
                    phase_tensor = (
                        torch.from_numpy(processed_phase)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )

                    # Run through neural network
                    with torch.no_grad():
                        # Encode CSI data
                        encoded_features = encoder(amplitude_tensor, phase_tensor)

                        # Estimate pose
                        keypoints, confidence, hidden_state = pose_estimator(
                            encoded_features, hidden_state
                        )

                        # Detect people
                        people = pose_estimator.detect_people(keypoints, confidence)

                        # Update dashboard and RTMP stream with the first detected person
                        if people:
                            first_person = people[0]

                            # Update dashboard
                            dashboard.update_data(
                                pose_data=first_person,
                                confidence_data=first_person["confidence"],
                                csi_data=(amplitude, phase),
                            )

                            # Update RTMP stream
                            rtmp_streamer.update_frame(
                                pose_data=first_person,
                                confidence_data=first_person["confidence"],
                            )

                            # Update house visualization
                            if house_visualizer:
                                house_visualizer.update_people(people)

                    # Sleep briefly to prevent CPU overuse
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in processing thread: {e}")

        # Start processing thread
        proc_thread = threading.Thread(target=processing_thread)
        proc_thread.daemon = True
        proc_thread.start()

        # Start dashboard (this will block until the dashboard is closed)
        logger.info(f"Starting dashboard on port {config['dashboard']['port']}")
        dashboard.run(debug=config["system"]["debug"], port=config["dashboard"]["port"])

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error running WiFi-Radar: {e}")
    finally:
        # Clean up
        logger.info("Stopping system components")
        csi_collector.stop()
        rtmp_streamer.stop()
        if house_visualizer:
            house_visualizer.stop()

    logger.info("WiFi-Radar system stopped")


if __name__ == "__main__":
    main()
