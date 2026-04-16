"""
ID: WR-PKG-UTILS-001
Requirement: Expose utility helpers (model I/O, code quality) for import
             by all other wifi_radar subpackages.
Purpose: Group reusable, cross-cutting utilities under a single namespace.
Rationale: A utils subpackage prevents circular imports by keeping helpers
           independent of domain-specific modules.
Assumptions: Utilities have no side effects on import.
References: PEP 328 relative imports; wifi_radar package structure.

Utility modules for WiFi-Radar.

This package contains helper functions and utilities
used across the WiFi-Radar system.
"""
