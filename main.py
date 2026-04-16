#!/usr/bin/env python3
"""
ID: WR-MAIN-001
Requirement: Orchestrate all WiFi-Radar subsystems (CSI collection, signal
             processing, neural-network inference, multi-person tracking,
             fall detection, gait analysis, visualisation, and RTMP streaming)
             into a single runnable application process.
Purpose: Entry point for both interactive use and automated deployment.
         Parses CLI arguments, loads configuration, initialises every module,
         starts background threads, then blocks on the Dash web dashboard.
Architecture:
    main()
      ├─ parse_args()          — argparse CLI interface
      ├─ setup_logging()       — file + console handlers
      ├─ load_config()         — YAML config with CLI overrides
      ├─ CSICollector.start()  — daemon thread producing CSI frames
      ├─ RTMPStreamer.start()  — daemon thread pushing H.264 to RTMP
      ├─ HouseVisualizer.start() (optional)
      ├─ processing_thread()   — daemon thread: process → infer → track → alert
      └─ Dashboard.run()       — Dash server (blocking, main thread)
References: See docs/system_overview.md for the full pipeline diagram.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add the src/ package directory to the Python path when running from the repo.
project_root = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.join(project_root, "src")
if os.path.isdir(src_root) and src_root not in sys.path:
    sys.path.insert(0, src_root)
elif project_root not in sys.path:
    sys.path.insert(0, project_root)

# Heavy ML/runtime imports are intentionally deferred until main() so that
# `python main.py --help` works even before the full stack is installed.


def parse_args():
    """Parse and validate command-line arguments for the WiFi-Radar application.

    ID: WR-MAIN-PARSEARGS-001
    Requirement: Define and parse all CLI arguments; return an argparse.Namespace
                 with validated values and defaults.
    Purpose: Allow operator control of simulation mode, router address, dashboard
             port, RTMP URL, weights path, and optional features at launch time.
    Rationale: argparse provides automatic --help generation, type validation,
               and default values without manual sys.argv parsing.
    Inputs:
        sys.argv — CLI arguments from the shell.
    Outputs:
        argparse.Namespace — parsed arguments with all fields present.
    Preconditions:
        Called before logging is configured; no side effects on the filesystem.
    Postconditions:
        All argument fields are populated with CLI values or defaults.
    Assumptions:
        Called once at startup before any subsystem is initialised.
    Side Effects:
        May call sys.exit(2) on invalid arguments (argparse behaviour).
    Failure Modes:
        Invalid argument type: argparse prints error and calls sys.exit(2).
    Error Handling:
        argparse handles type/range errors; no additional validation needed here.
    Constraints:
        --num-people is clamped to [1,4] by argparse min/max (not enforced here).
    Verification:
        Unit test: call with ['--simulation', '--debug']; assert namespace.simulation.
    References:
        argparse.ArgumentParser; WR-MAIN-001 module docstring.
    """
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
    parser.add_argument("--api", action="store_true", help="Enable the REST API server")
    parser.add_argument("--api-host", type=str, default="0.0.0.0", help="REST API bind host")
    parser.add_argument("--api-port", type=int, default=8081, help="REST API port")
    parser.add_argument("--headless", action="store_true", help="Run without blocking on the Dash dashboard")

    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Configure root logger with console and rotating file handlers.

    ID: WR-MAIN-SETUPLOG-001
    Requirement: Configure the root logging.Logger with a StreamHandler and a
                 FileHandler ('wifi_radar.log'), at DEBUG or INFO level.
    Purpose: Ensure all subsystem loggers (which use getLogger(__name__)) inherit
             a consistent format and dual output without requiring individual setup.
    Rationale: basicConfig on the root logger is the simplest way to provide
               global formatting; FileHandler persists logs across sessions.
    Inputs:
        debug — bool: if True, set log level to DEBUG; otherwise INFO.
    Outputs:
        None — configures the root logger as a side effect.
    Preconditions:
        Called before any subsystem creates a logger.
    Postconditions:
        Root logger is configured; all subsequent getLogger() calls inherit it.
    Assumptions:
        Current working directory is writable for 'wifi_radar.log'.
    Side Effects:
        Creates or appends to 'wifi_radar.log' in the CWD.
        Modifies the global root logging.Logger.
    Failure Modes:
        CWD not writable: FileHandler raises PermissionError; StreamHandler still works.
    Error Handling:
        No explicit error handling; FileHandler failure propagates to caller.
    Constraints:
        Log format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
    Verification:
        Unit test: call setup_logging(debug=True); assert root logger level == DEBUG.
    References:
        logging.basicConfig; logging.StreamHandler; logging.FileHandler.
    """
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("wifi_radar.log")],
    )


def load_config(config_path=None):
    """Load application configuration, merging built-in defaults with an optional YAML file.

    ID: WR-MAIN-LOADCFG-001
    Requirement: Return a nested config dict with sections: router, system, dashboard,
                 streaming, house_visualization; override defaults with values from
                 config_path if provided and readable.
    Purpose: Provide a single source of truth for all subsystem parameters so
             operators can tune the system without modifying source code.
    Rationale: Merging per-section with dict.update() preserves defaults for keys
               not present in the YAML file, preventing KeyErrors at runtime.
    Inputs:
        config_path — Optional[str]: path to a YAML configuration file.
    Outputs:
        Dict — nested config with all sections populated.
    Preconditions:
        yaml (PyYAML) must be installed.
    Postconditions:
        Return value contains all required section keys.
    Assumptions:
        config_path YAML uses the same top-level section keys as the defaults.
    Side Effects:
        Reads the filesystem if config_path is provided.
        Logs INFO on success, ERROR on failure.
    Failure Modes:
        config_path unreadable or invalid YAML: exception caught; defaults returned.
    Error Handling:
        try/except around yaml.safe_load; error logged; defaults returned intact.
    Constraints:
        Extra top-level sections not in defaults are added verbatim.
    Verification:
        Unit test: create a YAML file overriding router.ip; assert config['router']['ip']
        matches the YAML value.
    References:
        yaml.safe_load; WR-MAIN-001 module docstring.
    """
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
        "api": {
            "enabled": False,
            "host": "0.0.0.0",
            "port": 8081,
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
    """Initialise and run the complete WiFi-Radar pipeline.

    ID: WR-MAIN-MAIN-001
    Requirement: Parse CLI arguments, load configuration, instantiate all
                 subsystems, start daemon threads, and block on Dashboard.run().
    Purpose: Serve as the single runnable entry point for the WiFi-Radar system
             so all subsystems are started and stopped in a defined order.
    Rationale: Daemon threads for processing, streaming, and visualisation are
               started before Dashboard.run() so they are ready to serve the
               first browser request immediately.
    Inputs:
        sys.argv — via parse_args().
        Filesystem: config YAML, weights .pth.
    Outputs:
        Running system until KeyboardInterrupt or process termination.
    Preconditions:
        None — entry point.
    Postconditions:
        On exit: csi_collector, rtmp_streamer, and house_visualizer are stopped.
    Assumptions:
        Weights file is optional; system runs with random initialisation otherwise.
    Side Effects:
        Spawns daemon threads; opens network sockets; binds HTTP and RTMP ports.
        Writes wifi_radar.log.
    Failure Modes:
        Dashboard port already in use: OSError.
        ONNX export: delegates to scripts/export_onnx.py via subprocess.
    Error Handling:
        KeyboardInterrupt: caught; all components stopped in finally block.
        General Exception: logged; components stopped in finally block.
    Constraints:
        Dashboard.run() is blocking; must be called from the main thread.
    Verification:
        Integration test: run with --simulation; assert dashboard responds on port 8050.
    References:
        parse_args; setup_logging; load_config; WR-MAIN-001 module docstring.
    """
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
    if args.api:
        config["api"]["enabled"] = True
    if args.api_host:
        config["api"]["host"] = args.api_host
    if args.api_port:
        config["api"]["port"] = args.api_port
    if args.headless:
        config["system"]["headless"] = True

    # ── ONNX export shortcut ──────────────────────────────────────────────
    if args.export_onnx:
        import subprocess
        cmd = [sys.executable, "scripts/export_onnx.py"]
        if args.weights:
            cmd += ["--weights", args.weights]
        subprocess.run(cmd, check=True)
        sys.exit(0)

    try:
        import torch

        from wifi_radar.analysis.fall_detector import FallDetector, FallSeverity
        from wifi_radar.analysis.gait_analyzer import GaitAnalyzer
        from wifi_radar.analysis.gait_anomaly_detector import GaitAnomalyDetector
        from wifi_radar.data.csi_collector import CSICollector
        from wifi_radar.models.encoder import DualBranchEncoder
        from wifi_radar.models.multi_person_tracker import MultiPersonTracker
        from wifi_radar.models.pose_estimator import PoseEstimator
        from wifi_radar.processing.signal_processor import SignalProcessor
        from wifi_radar.streaming.rtmp_streamer import RTMPStreamer
        from wifi_radar.utils.model_io import load_checkpoint
        from wifi_radar.visualization.dashboard import Dashboard
        from wifi_radar.visualization.house_visualizer import HouseVisualizer  # noqa: F401

        try:
            from wifi_radar.api import AppState, run_api_server
        except Exception:
            AppState = None
            run_api_server = None
    except ImportError as exc:
        logger.error("Missing runtime dependency: %s. Install requirements.txt or activate the project venv.", exc)
        raise

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
    gait_analysers: dict = {}
    gait_anomaly_detectors: dict = {}
    api_state = AppState(config=config) if AppState is not None else None

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

        # Optional headless REST API for embedded integration
        import threading

        if config.get("api", {}).get("enabled") and run_api_server is not None and api_state is not None:
            logger.info(
                "Starting REST API on %s:%s",
                config["api"]["host"],
                config["api"]["port"],
            )
            api_thread = threading.Thread(
                target=run_api_server,
                kwargs={
                    "host": config["api"]["host"],
                    "port": config["api"]["port"],
                    "state": api_state,
                },
                daemon=True,
            )
            api_thread.start()

        import numpy as np
        import torch

        def processing_thread():
            """Main inference loop: CSI -> signal process -> encode -> pose -> track -> alert.

            ID: WR-MAIN-PROCTHREAD-001
            Requirement: Continuously dequeue CSI frames, run the full inference
                         pipeline, and publish results to Dashboard, RTMPStreamer,
                         and HouseVisualizer at ~20 Hz.
            Purpose: Decouple time-sensitive inference from the Dash HTTP server
                     thread so the web UI remains responsive under full inference load.
            Rationale: Daemon thread with inner try/except allows clean termination
                       on KeyboardInterrupt without needing explicit stop signals.
            Inputs:
                Reads from csi_collector.get_csi_data() (blocking, 1 s timeout).
            Outputs:
                Pushes results to dashboard, rtmp_streamer, house_visualizer.
            Preconditions:
                All subsystems must be initialised before this thread starts.
            Postconditions:
                On exit: fall_detectors and gait_analysers reflect final state.
            Assumptions:
                csi_collector provides (amplitude, phase) tuples at ~20 Hz.
            Side Effects:
                Calls dashboard.update_data(), dashboard.update_events().
                Calls rtmp_streamer.update_frame().
                Calls house_visualizer.update_people() if enabled.
                Lazily populates fall_detectors and gait_analysers dicts.
            Failure Modes:
                csi_collector returns None (timeout): iteration skipped.
                Unhandled exception: logged; thread exits.
            Error Handling:
                KeyboardInterrupt caught; generic Exception caught and logged.
            Constraints:
                Not real-time; time.sleep(0.01) used to yield CPU between frames.
            Verification:
                Integration test: run with --simulation; assert dashboard updates.
            References:
                CSICollector.get_csi_data; SignalProcessor.process;
                DualBranchEncoder; PoseEstimator; MultiPersonTracker.update;
                FallDetector.update; GaitAnalyzer.update; WR-MAIN-MAIN-001.
            """
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
                            gait_anomaly_detectors[pid] = GaitAnomalyDetector()

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
                        gm_person = gait_analysers[pid].get_metrics()
                        if gm_person is not None:
                            anomaly = gait_anomaly_detectors[pid].update(gm_person)
                            if anomaly.get("is_anomaly"):
                                new_fall_events.append({
                                    "person_id": pid,
                                    "timestamp": ts_now,
                                    "severity": int(FallSeverity.POSSIBLE_FALL),
                                    "body_angle_deg": None,
                                    "message": "Gait anomaly detected: " + "; ".join(anomaly.get("reasons", [])[:2]),
                                })

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

                    if api_state is not None:
                        api_state.ingest({
                            "tracked_people": [
                                {"keypoints": t.keypoints.tolist() if hasattr(t.keypoints, "tolist") else t.keypoints,
                                 "confidence": t.confidence.tolist() if hasattr(t.confidence, "tolist") else t.confidence,
                                 "person_id": t.person_id}
                                for t in tracked
                            ],
                            "gait_metrics": gait_metrics_dict,
                            "csi_summary": {
                                "amplitude_mean": float(np.mean(amplitude)),
                                "phase_mean": float(np.mean(phase)),
                            },
                            "events": new_fall_events,
                        })

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

        # Start dashboard unless headless mode is requested
        if config["system"].get("headless", False):
            logger.info("Headless mode enabled; dashboard server not started")
            while True:
                time.sleep(1.0)
        else:
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
