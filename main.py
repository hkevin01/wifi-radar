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

from wifi_radar.analysis.fall_detector import FallDetector, FallSeverity
from wifi_radar.analysis.gait_analyzer import GaitAnalyzer
from wifi_radar.data.csi_collector import CSICollector
from wifi_radar.models.encoder import DualBranchEncoder
from wifi_radar.models.multi_person_tracker import MultiPersonTracker
from wifi_radar.models.pose_estimator import PoseEstimator
from wifi_radar.processing.signal_processor import SignalProcessor
from wifi_radar.streaming.rtmp_streamer import RTMPStreamer
from wifi_radar.utils.model_io import load_checkpoint
from wifi_radar.visualization.dashboard import Dashboard
from wifi_radar.visualization.house_visualizer import HouseVisualizer  # noqa: F401


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
        default="192.168.1.1",  # Replace with your router's IP address
        help="IP address of the WiFi router (see docs/setup-guide.md)",
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
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to .pth checkpoint (e.g. weights/simulation_baseline.pth)",
    )
    parser.add_argument(
        "--num-people",
        type=int,
        default=1,
        help="Number of simulated people (1-4; simulation mode only)",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export loaded models to ONNX then exit (requires onnx / onnxruntime)",
    )

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
    if args.num_people:
        config["system"]["num_people"] = args.num_people

    # ── ONNX export shortcut ──────────────────────────────────────────────
    if args.export_onnx:
        import subprocess
        cmd = [sys.executable, "scripts/export_onnx.py"]
        if args.weights:
            cmd += ["--weights", args.weights]
        subprocess.run(cmd, check=True)
        sys.exit(0)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder        = DualBranchEncoder().to(device)
    pose_estimator = PoseEstimator().to(device)
    encoder.initialize_weights()

    weights_path = args.weights or os.path.join("weights", "simulation_baseline.pth")
    if os.path.exists(weights_path):
        try:
            info = load_checkpoint(encoder, pose_estimator, weights_path, device=device)
            logger.info("Loaded weights: epoch=%d  val_loss=%.4f", info["epoch"], info["val_loss"])
        except Exception as exc:
            logger.warning("Could not load weights from %s: %s", weights_path, exc)
    else:
        logger.info("No weights file found at %s — using random initialisation", weights_path)
        logger.info("Run: python scripts/train_simulation_baseline.py  to generate baseline weights.")

    # Multi-person tracker
    mp_tracker = MultiPersonTracker(
        max_people=config["system"].get("max_people", 4),
        existence_threshold=0.0,   # legacy detect_people() doesn't output existence score
    )

    # Per-person fall detectors and gait analysers (created on demand)
    fall_detectors: dict = {}
    gait_analysers: dict  = {}

    # Initialize visualization
    logger.info("Initializing visualization")
    dashboard = Dashboard(
        update_interval_ms=config["dashboard"]["update_interval_ms"],
        max_history=config["dashboard"]["max_history"],
        config=config,
        config_path=os.path.expanduser("~/.wifi_radar/config.yaml"),
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
        csi_collector.sim_num_people = config["system"].get("num_people", 1)
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
            hidden_state = None
            frame_id = 0

            fall_cfg = config.get("fall_detection", {})
            fall_enabled = fall_cfg.get("enabled", True)

            try:
                while True:
                    csi_data = csi_collector.get_csi_data(block=True, timeout=1.0)
                    if csi_data is None:
                        continue

                    amplitude, phase = csi_data
                    processed_amplitude, processed_phase = signal_processor.process(
                        amplitude, phase
                    )

                    amplitude_tensor = (
                        torch.from_numpy(processed_amplitude).unsqueeze(0).float().to(device)
                    )
                    phase_tensor = (
                        torch.from_numpy(processed_phase).unsqueeze(0).float().to(device)
                    )

                    with torch.no_grad():
                        encoded_features = encoder(amplitude_tensor, phase_tensor)
                        keypoints, confidence, hidden_state = pose_estimator(
                            encoded_features, hidden_state
                        )
                        raw_people = pose_estimator.detect_people(keypoints, confidence)

                    # Multi-person tracking — assigns stable IDs
                    tracked = mp_tracker.update(raw_people, frame_id=frame_id)
                    frame_id += 1

                    # Fall detection + gait analysis per tracked person
                    new_fall_events = []
                    ts_now = time.time()
                    for person in tracked:
                        pid = person.person_id

                        # Lazily create per-person analysers
                        if pid not in fall_detectors:
                            fall_detectors[pid] = FallDetector(
                                person_id=pid,
                                velocity_threshold=fall_cfg.get("velocity_threshold", -0.20),
                                angle_threshold_deg=fall_cfg.get("angle_threshold_deg", 40.0),
                            )
                            gait_analysers[pid] = GaitAnalyzer()

                        if fall_enabled:
                            ev = fall_detectors[pid].update(
                                person.keypoints, person.confidence, timestamp=ts_now
                            )
                            if ev is not None:
                                new_fall_events.append({
                                    "person_id":     ev.person_id,
                                    "timestamp":     ev.timestamp,
                                    "severity":      int(ev.severity),
                                    "body_angle_deg": ev.body_angle_deg,
                                    "message":       ev.message,
                                })

                        gait_analysers[pid].update(
                            person.keypoints, person.confidence, timestamp=ts_now
                        )

                    # Collect gait metrics from first active person
                    gait_metrics_dict = None
                    if tracked:
                        gm = gait_analysers[tracked[0].person_id].get_metrics()
                        if gm is not None:
                            gait_metrics_dict = {
                                "cadence_spm":    gm.cadence_spm,
                                "stride_length":  gm.stride_length,
                                "step_symmetry":  gm.step_symmetry,
                                "speed_est":      gm.speed_est,
                                "num_steps":      gm.num_steps,
                                "window_s":       gm.window_s,
                            }

                    # Dashboard updates
                    first_person_dict = None
                    first_conf        = None
                    if tracked:
                        first_person_dict = {
                            "keypoints":  tracked[0].keypoints,
                            "confidence": tracked[0].confidence,
                        }
                        first_conf = tracked[0].confidence

                    dashboard.update_data(
                        pose_data=first_person_dict,
                        confidence_data=first_conf,
                        csi_data=(amplitude, phase),
                        tracked_people=[
                            {"keypoints": t.keypoints, "confidence": t.confidence,
                             "person_id": t.person_id}
                            for t in tracked
                        ],
                    )

                    if new_fall_events or gait_metrics_dict:
                        dashboard.update_events(
                            fall_events=new_fall_events or None,
                            gait_metrics=gait_metrics_dict,
                        )

                    if tracked:
                        rtmp_streamer.update_frame(
                            pose_data=first_person_dict,
                            confidence_data=first_conf,
                        )
                        if house_visualizer:
                            house_visualizer.update_people([
                                {"keypoints": t.keypoints, "confidence": t.confidence}
                                for t in tracked
                            ])

                    time.sleep(0.01)

            except KeyboardInterrupt:
                pass
            except Exception as e:
                logger.exception("Fatal error in processing thread: %s", e)

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
