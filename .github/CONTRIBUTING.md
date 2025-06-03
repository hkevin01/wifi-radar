# Contributing to WiFi-Radar

Thank you for your interest in contributing to WiFi-Radar! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Bugs

- Use the issue tracker to report bugs
- Describe the bug in detail
- Include steps to reproduce
- Include system information (OS, Python version, etc.)
- If possible, include screenshots or logs

### Suggesting Enhancements

- Use the issue tracker to suggest enhancements
- Clearly describe the enhancement
- Provide examples of how it would be used
- Explain why it would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. Clone your fork of the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Testing

Run tests with pytest:
```bash
pytest
```

With coverage report:
```bash
pytest --cov=wifi_radar
```

## Coding Standards

- Follow PEP 8 style guide
- Write docstrings for all functions, classes, and modules
- Add unit tests for new functionality
- Keep functions small and focused
- Use meaningful variable and function names

## Git Workflow

- Keep pull requests focused on a single feature or bug fix
- Rebase your branch before submitting a pull request
- Squash commits into logical units
- Write clear commit messages

## Documentation

- Update documentation when changing code
- Document new features or changes in behavior
- Keep README.md updated
- Add examples for new functionality

## Thank You!

Your contributions make this project better for everyone!
