# Contributing to Harakat

Thank you for your interest in contributing to Harakat! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Features](#suggesting-features)

## Code of Conduct

This project follows a simple code of conduct: be respectful, be constructive, and focus on the work. We welcome contributors of all backgrounds and experience levels.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your changes
5. Make your changes and test them
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/harakat.git
cd harakat

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# Check that the module imports correctly
python -c "from harakat import diacritize; print(diacritize('مرحبا'))"
```

## How to Contribute

### Types of Contributions

1. **Bug Fixes**: Fix issues in the existing code
2. **Documentation**: Improve or add documentation
3. **Tests**: Add or improve test coverage
4. **Features**: Add new functionality (please discuss first)
5. **Performance**: Optimize existing code

### Before You Start

- Check existing [issues](https://github.com/jeranaias/harakat/issues) to avoid duplicating work
- For large changes, open an issue first to discuss the approach
- Make sure you're working on the latest version of `main`

## Pull Request Process

1. **Create a branch**: Use a descriptive name like `fix-shadda-detection` or `add-batch-processing`

2. **Make your changes**: Keep commits focused and atomic

3. **Write tests**: All new code should have tests

4. **Update documentation**: If your change affects usage, update the README

5. **Run the test suite**: Ensure all tests pass
   ```bash
   pytest tests/ -v
   ```

6. **Submit the PR**:
   - Provide a clear description of the changes
   - Reference any related issues
   - Include before/after examples if applicable

7. **Address feedback**: Be responsive to review comments

## Style Guidelines

### Python Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and small

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Keep the first line under 50 characters
- Add details in the body if needed

Example:
```
Add batch processing for multiple texts

- Implement batch_diacritize() function
- Add tests for batch processing
- Update README with usage examples
```

### Code Comments

- Comment complex logic, not obvious code
- Use Arabic comments for Arabic-specific logic if helpful
- Keep comments up to date with code changes

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=harakat --cov-report=html

# Run specific test file
pytest tests/test_harakat.py -v

# Run specific test
pytest tests/test_harakat.py::TestQuranMode::test_basmala -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test both success and failure cases

Example:
```python
def test_basic_diacritization():
    """Test that basic Arabic text gets diacritized."""
    result = diacritize("مرحبا")
    assert any(c in result for c in HARAKAT)
```

## Reporting Bugs

When reporting a bug, please include:

1. **Description**: Clear explanation of the bug
2. **Steps to reproduce**: How to trigger the bug
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, Harakat version
6. **Example input**: The Arabic text that causes the issue

Use the [bug report template](https://github.com/jeranaias/harakat/issues/new?template=bug_report.md) if available.

## Suggesting Features

For feature suggestions:

1. Check if the feature already exists
2. Check if it's already been suggested
3. Open an issue with:
   - Clear description of the feature
   - Use case / motivation
   - Example of how it would work
   - Any implementation ideas

## Questions?

- Open an issue for questions about contributing
- Check existing documentation first
- Be patient - this is a solo-maintained project

---

Thank you for contributing to Harakat!
