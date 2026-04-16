"""
ID: WR-PKG-DATA-001
Requirement: Expose CSICollector for import by the main processing pipeline.
Purpose: Group CSI data collection modules under a single importable namespace.
Rationale: Isolating data collection in its own subpackage allows the source
           (real router vs. simulator) to be swapped without touching inference code.
Assumptions: Socket and threading are available in the standard library.
References: CSICollector; wifi_radar processing pipeline.

Data collection and processing modules for WiFi-Radar.

This package contains modules for collecting and processing
Channel State Information (CSI) from WiFi signals.
"""
