# Tests

This directory contains tests for the WiFi-Radar system:

- Unit tests for individual components
- Integration tests for system functionality
- Test data and utilities

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=wifi_radar

# Run specific test file
pytest tests/test_csi_collector.py

# Run tests matching a pattern
pytest -k "collector"
```

## Test Organization

- `test_data/`: Test data files and fixtures
- `test_*.py`: Individual test files for each module
- `conftest.py`: Shared pytest fixtures
