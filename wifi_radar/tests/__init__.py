"""
ID: WR-PKG-TESTS-INNER-001
Requirement: Mark wifi_radar/tests/ as a Python package for test discovery.
Purpose: Enable pytest to discover unit tests for individual wifi_radar submodules
         placed inside the package tree.
Rationale: Co-locating tests with source code simplifies relative imports
           and keeps test coverage close to the code under test.
Assumptions: pytest is installed; tests follow the test_*.py naming convention.
References: pytest test discovery documentation.

Test modules for WiFi-Radar.

This package contains test cases and testing utilities
for the WiFi-Radar system.
"""
