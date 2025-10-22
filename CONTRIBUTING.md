# Contributing to ML CLI

Thank you for your interest in contributing to ML CLI! This document provides guidelines and instructions for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml_cli.git
   cd ml_cli
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

## 🧪 Testing

### Run all tests
```bash
pytest tests/ -v
```

### Run tests with coverage
```bash
pytest tests/ -v --cov=ml_cli --cov-report=html --cov-report=term
```

### Run specific test file
```bash
pytest tests/test_api.py -v
```

### Run linting
```bash
# Check for syntax errors
flake8 ml_cli --count --select=E9,F63,F7,F82 --show-source --statistics

# Check code style
flake8 ml_cli --count --max-complexity=10 --max-line-length=127 --statistics
```

### Format code
```bash
# Format with black
black ml_cli/

# Sort imports
isort ml_cli/
```

## 🔨 Testing CI Locally

### Manual CI Testing
Run the same steps that CI runs:

```bash
# 1. Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# 2. Lint code
flake8 ml_cli --count --select=E9,F63,F7,F82 --show-source --statistics

# 3. Run tests with coverage
pytest tests/ -v --cov=ml_cli --cov-report=xml --cov-report=term

# 4. Build package
python -m build

# 5. Check distribution
twine check dist/*
```

### Using Act (GitHub Actions locally)
Install `act` to run GitHub Actions workflows locally:

```bash
# Install act (macOS)
brew install act

# Run the CI workflow
act -j test
```

## 📝 Code Style Guidelines

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and small
- Write tests for new features
- Maximum line length: 127 characters

## 🎯 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new features
   - Ensure all tests pass

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Test changes
   - `refactor:` - Code refactoring
   - `chore:` - Maintenance tasks

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Go to GitHub and create a PR
   - Describe your changes clearly
   - Link any related issues
   - Wait for review

## 🐛 Reporting Bugs

Please use GitHub Issues to report bugs. Include:
- Python version
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## 💡 Feature Requests

We welcome feature requests! Please:
- Check if it already exists in Issues
- Describe the use case clearly
- Explain why it would be valuable

## 📚 Documentation

When adding new features:
- Update the README.md if needed
- Add docstrings to new functions
- Update relevant documentation

## ✅ Pre-commit Checklist

Before submitting a PR, ensure:
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is formatted: `black ml_cli/`
- [ ] Imports are sorted: `isort ml_cli/`
- [ ] No linting errors: `flake8 ml_cli/`
- [ ] Documentation is updated
- [ ] Commit messages follow conventions

## 🏗️ Project Structure

```
ml_cli/
├── ml_cli/           # Main package
│   ├── commands/     # CLI commands
│   ├── core/         # Core ML logic
│   ├── api/          # FastAPI server
│   ├── config/       # Configuration models
│   └── utils/        # Utility functions
├── tests/            # Test suite
│   ├── integration/  # Integration tests
│   └── *.py         # Unit tests
├── examples/         # Example datasets
└── docs/            # Documentation
```

## 📞 Getting Help

- Open an issue for questions
- Check existing issues and discussions
- Read the documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

Thank you for contributing! 🎉
