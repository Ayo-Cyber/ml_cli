# Publishing Guide for ML CLI

This guide walks you through publishing the ML CLI package to PyPI and TestPyPI.

## Prerequisites

1. **Create Accounts**
   - TestPyPI: https://test.pypi.org/account/register/
   - PyPI: https://pypi.org/account/register/

2. **Generate API Tokens**
   - TestPyPI: https://test.pypi.org/manage/account/#api-tokens
   - PyPI: https://pypi.org/manage/account/#api-tokens

3. **Install Tools**
   ```bash
   pip install build twine
   ```

## For Testing with Friends (TestPyPI)

### Step 1: Build the Package
```bash
# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build new distribution
python -m build
```

### Step 2: Check the Distribution
```bash
twine check dist/*
```

### Step 3: Upload to TestPyPI
```bash
twine upload --repository testpypi dist/*
# Enter your TestPyPI username and API token when prompted
```

### Step 4: Share with Friends
Your friends can install with:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ml-cli
```

**Note:** The `--extra-index-url` flag is needed because TestPyPI doesn't have all the dependencies.

### Step 5: Test Installation
```bash
# Create a new virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ml-cli

# Test it works
ml --help
ml init test_project

# Clean up
deactivate
rm -rf test_env test_project
```

## For Public Release (PyPI)

### Option 1: Automated via GitHub Actions (Recommended)

1. **Set up GitHub Secrets**
   - Go to: Repository → Settings → Secrets and variables → Actions
   - Add secret: `TEST_PYPI_API_TOKEN` (your TestPyPI token)
   - Add secret: `PYPI_API_TOKEN` (your PyPI token)

2. **Update Version**
   Edit `setup.py` and increment the version:
   ```python
   version="1.0.0",  # or whatever version you're releasing
   ```

3. **Commit and Tag**
   ```bash
   git add setup.py CHANGELOG.md
   git commit -m "Release v1.0.0"
   git tag v1.0.0
   git push origin main --tags
   ```

4. **Create GitHub Release**
   - Go to: Repository → Releases → Create new release
   - Choose tag: v1.0.0
   - Fill in release notes (can copy from CHANGELOG.md)
   - Click "Publish release"

5. **Monitor the Build**
   - Go to: Repository → Actions
   - Watch the CI/CD pipeline run
   - If successful, package will be published to PyPI automatically

### Option 2: Manual Upload

1. **Build the Package**
   ```bash
   rm -rf dist/ build/ *.egg-info
   python -m build
   ```

2. **Check the Distribution**
   ```bash
   twine check dist/*
   ```

3. **Upload to PyPI**
   ```bash
   twine upload dist/*
   # Enter your PyPI username and API token
   ```

4. **Verify Installation**
   ```bash
   pip install ml-cli
   ml --help
   ```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
  - **MAJOR**: Incompatible API changes
  - **MINOR**: New functionality (backwards compatible)
  - **PATCH**: Bug fixes (backwards compatible)

Examples:
- `1.0.0` - Initial release
- `1.0.1` - Bug fix release
- `1.1.0` - New features added
- `2.0.0` - Breaking changes

## Pre-Release Checklist

Before publishing, ensure:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is formatted: `black ml_cli/`
- [ ] No linting errors: `flake8 ml_cli/`
- [ ] Version updated in `setup.py`
- [ ] CHANGELOG.md updated with changes
- [ ] README.md is up to date
- [ ] All new features documented
- [ ] Dependencies in `setup.py` are correct
- [ ] Package builds without errors: `python -m build`
- [ ] Distribution passes checks: `twine check dist/*`

## Post-Release

After publishing:

1. **Announce the Release**
   - Share on social media
   - Post in relevant communities
   - Update documentation sites

2. **Monitor Issues**
   - Watch for bug reports on GitHub
   - Respond to user questions
   - Plan next release based on feedback

3. **Update Documentation**
   - Update any external documentation
   - Create tutorials if needed
   - Update examples

## Troubleshooting

### "Package already exists"
- You can't overwrite an existing version on PyPI
- Increment the version number and rebuild

### "Invalid distribution"
- Check package structure with `twine check dist/*`
- Ensure all required files are included
- Verify setup.py configuration

### "Authentication failed"
- Double-check your API token
- Ensure you're using the correct repository URL
- Token should start with `pypi-` for PyPI or `pypi-` for TestPyPI

### "Dependencies not found" (TestPyPI)
- Use both index URLs when installing from TestPyPI
- Some dependencies may not be on TestPyPI

## Useful Commands

```bash
# Build package
python -m build

# Check distribution
twine check dist/*

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ml-cli

# Install from PyPI
pip install ml-cli

# Install specific version
pip install ml-cli==1.0.0

# Install from GitHub (development)
pip install git+https://github.com/Ayo-Cyber/ml_cli.git

# Install in development mode
pip install -e ".[dev]"
```

## Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [TestPyPI](https://test.pypi.org/)
- [Semantic Versioning](https://semver.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
