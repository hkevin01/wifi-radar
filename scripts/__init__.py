"""
ID: WR-PKG-SCRIPTS-001
Requirement: Mark the scripts/ directory as a Python package for import resolution.
Purpose: Allow scripts (export_onnx.py, train_simulation_baseline.py) to import
         wifi_radar submodules via sys.path insertion without installation.
Rationale: Packaging scripts/ avoids the need for editable installs during
           development workflows where the project root is on sys.path.
Assumptions: Project root is on sys.path before scripts are executed.
References: scripts/export_onnx.py, scripts/train_simulation_baseline.py.

Scripts for WiFi-Radar.

This package contains scripts for running and testing the WiFi-Radar system.
"""
