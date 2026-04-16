"""
ID: WR-PKG-PROCESSING-001
Requirement: Expose SignalProcessor for import by the main inference pipeline.
Purpose: Group CSI signal processing stages under a single importable namespace.
Rationale: Decoupling signal processing from data collection and neural inference
           enables independent testing of filter parameters.
Assumptions: numpy and scipy.signal are installed.
References: SignalProcessor; Butterworth IIR filter design.

Signal processing modules for WiFi-Radar.

This package contains modules for processing raw CSI data
to prepare it for neural network analysis.
"""
