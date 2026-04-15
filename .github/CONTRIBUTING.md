# Contributing to WiFi-Radar

Thank you for your interest in contributing to WiFi-Radar! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Code Quality Standards

We maintain high code quality standards using automated tools:

- **Code Formatting**: We use [Black](https://github.com/psf/black) for consistent code formatting
- **Import Sorting**: We use [isort](https://github.com/PyCQA/isort) to sort imports
- **Linting**: We use [pylint](https://github.com/PyCQA/pylint) and [flake8](https://github.com/PyCQA/flake8)
- **Type Checking**: We use [mypy](https://github.com/python/mypy) for optional type checking
- **Docstrings**: We follow Google docstring style, checked with [pydocstyle](https://github.com/PyCQA/pydocstyle)

Before submitting a pull request, please run:

```bash
# Install pre-commit hooks
pre-commit install

# Check code quality
python scripts/code_quality.py
```

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
5. Run code quality checks (`python scripts/code_quality.py`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

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
4. Install pre-commit hooks:
   ```bash
   pre-commit install
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
