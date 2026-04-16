"""
ID: WR-PKG-ROOT-001
Requirement: Expose the WiFi-Radar package version and top-level namespace.
Purpose: Mark the wifi_radar directory as a Python package and declare
         the public version string consumed by setup.cfg and runtime introspection.
Rationale: A single __version__ string in the package root avoids version drift
           between setup.cfg, the package, and deployment metadata.
Assumptions: Semantic versioning (MAJOR.MINOR.PATCH).
References: PEP 8 package structure; setup.cfg version field.

WiFi-Radar: Human Pose Estimation through WiFi Signals

A Python implementation for detecting and tracking human poses through walls
using WiFi signals.
"""

__version__ = "0.1.0"
