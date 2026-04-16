"""
ID: WR-PKG-TESTS-ROOT-001
Requirement: Mark the top-level tests/ directory as a Python package for pytest discovery.
Purpose: Enable pytest to find and execute integration and system-level tests
         that span multiple wifi_radar submodules.
Rationale: A top-level tests/ directory is the conventional location for integration
           tests that require the full installed package, distinct from unit tests.
Assumptions: pytest >= 7.0 is installed; tests follow the test_*.py naming convention.
References: pytest configuration in pyproject.toml / setup.cfg.

Test suite for WiFi-Radar.

This package contains unit and integration tests for the WiFi-Radar system.
"""
