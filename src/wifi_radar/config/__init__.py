"""
ID: WR-PKG-CONFIG-001
Requirement: Provide a dedicated namespace for configuration management modules.
Purpose: Organise configuration loading, validation, and default-value management
         under a single importable subpackage.
Rationale: Separating configuration from runtime logic enables independent testing
           of config parsing and future extension with schema validation.
Assumptions: YAML is the primary configuration format (PyYAML installed).
References: load_config() in main.py; wifi_radar system configuration schema.

Configuration modules for WiFi-Radar.

This package contains configuration settings and
management for the WiFi-Radar system.
"""
