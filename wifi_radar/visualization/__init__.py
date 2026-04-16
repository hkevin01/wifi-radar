"""
ID: WR-PKG-VIZ-001
Requirement: Expose Dashboard and HouseVisualizer for import by the main pipeline.
Purpose: Group real-time visualisation modules under a single importable namespace.
Rationale: Separating visualisation from inference allows the UI layer to be
           replaced (e.g. Streamlit, Qt) without modifying the processing pipeline.
Assumptions: Dash, Plotly, and dash-bootstrap-components are installed.
References: Dashboard (Dash); HouseVisualizer (pygame).

Visualization modules for WiFi-Radar.

This package contains modules for visualizing pose estimation
results in real-time dashboards.
"""
