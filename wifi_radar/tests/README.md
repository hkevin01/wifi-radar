# Tests

This directory contains test modules for the WiFi-Radar system:

- Unit tests for individual components
- Integration tests for system functionality
- Test fixtures and utilities
- Mock objects for testing

## Key Components

- Test cases for each module
- Test data fixtures
- Mock WiFi router for testing
- Performance benchmarks

## Running Tests

Tests can be run using pytest:

```bash
# Run all tests
pytest src/tests/

# Run specific test module
pytest src/tests/test_csi_collector.py

# Run with coverage report
pytest src/tests/ --cov=src
```
