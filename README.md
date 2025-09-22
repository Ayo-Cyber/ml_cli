# 🤖 ML CLI Pipeline

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/ml-cli/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A comprehensive command-line interface for end-to-end machine learning workflows**

[Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples) • [Contributing](#-contributing)

</div>

## 🚀 Overview

ML CLI Pipeline is a powerful, user-friendly command-line tool that streamlines the entire machine learning workflow. From data exploration to model deployment, it provides a unified interface for data scientists and ML engineers to build, train, and serve machine learning models with minimal setup.

Built with modern Python technologies including **FastAPI**, **TPOT AutoML**, and **Click**, this tool eliminates the repetitive scripting often required in ML projects and provides production-ready model serving capabilities.

### ✨ Why ML CLI Pipeline?

- **🎯 Zero Configuration**: Get started with a single command - no complex setup files
- **🔄 End-to-End Workflow**: Complete ML pipeline from EDA to production deployment
- **🤖 AutoML Integration**: Leverage TPOT for automated model selection and hyperparameter tuning
- **🚀 Production Ready**: Built-in FastAPI server for immediate model deployment
- **📊 Rich Visualizations**: Automated EDA reports with beautiful plots and statistics
- **🔧 Flexible Configuration**: Support for both YAML and JSON configuration formats
- **🧪 Thoroughly Tested**: Comprehensive test suite ensuring reliability

## ✨ Features

<table>
<tr>
<td width="50%">

### 🛠️ **Core Capabilities**
- **Interactive Project Setup** - Guided initialization with intelligent defaults
- **Automated EDA** - Comprehensive data analysis with visualizations
- **Smart Preprocessing** - Automatic categorical encoding and data cleaning
- **AutoML Training** - TPOT-powered model optimization
- **Flexible Prediction** - Easy inference on new data
- **Production Serving** - FastAPI-based REST API deployment

</td>
<td width="50%">

### 🎯 **Advanced Features**
- **Dynamic API Documentation** - Auto-generated examples from your data
- **Model Reloading** - Hot-reload models without server restart
- **Artifact Tracking** - Automatic cleanup of generated files
- **Multi-format Support** - CSV, JSON, and remote URL data sources
- **SSL Support** - Secure data fetching from HTTPS sources
- **Rich CLI Interface** - Beautiful terminal output with progress bars

</td>
</tr>
</table>

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Quick Install

```bash
# Clone the repository
git clone https://github.com/Ayo-Cyber/ml_cli.git
cd ml_cli

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
ml --help
```

You should see the ML CLI help menu with all available commands.

## 🚀 Quick Start

Get up and running in under 5 minutes:

### 1. Initialize Your Project

```bash
ml init
```

This interactive command will:
- Guide you through project setup
- Validate your data file
- Create optimized configuration
- Set up your workspace

### 2. Explore Your Data

```bash
ml eda
```

Generates comprehensive analysis including:
- 📊 Summary statistics (`summary_statistics.csv`)
- 📈 Missing value reports (`eda_report.csv`)
- 🔥 Correlation heatmaps (`correlation_matrix.png`)

### 3. Preprocess Your Data

```bash
ml preprocess
```

Automatically handles:
- One-hot encoding for categorical variables
- Data validation and cleaning
- Feature preparation for training

### 4. Train Your Model

```bash
ml train
```

Leverages TPOT AutoML to:
- Find optimal algorithms
- Tune hyperparameters
- Export production-ready models
- Generate performance metrics

### 5. Serve Your Model

```bash
ml serve
```

Instantly deploy your model with:
- 🌐 RESTful API endpoints
- 📚 Interactive documentation at `/docs`
- 🔄 Hot-reload capabilities
- 📝 Auto-generated examples

## 📖 Documentation

### Command Reference

<details>
<summary><b>🔧 ml init</b> - Initialize new ML project</summary>

```bash
ml init [OPTIONS]
```

**Options:**
- `--format [yaml|json]` - Configuration format (default: yaml)
- `--ssl-verify/--no-ssl-verify` - SSL verification for URLs (default: enabled)

**Features:**
- Interactive project setup wizard
- Data validation and target column detection
- Intelligent default suggestions
- Support for local files and URLs

</details>

<details>
<summary><b>📊 ml eda</b> - Exploratory Data Analysis</summary>

```bash
ml eda
```

**Generates:**
- Summary statistics for all features
- Missing value analysis
- Data type information
- Correlation matrix visualization

**Output Files:**
- `summary_statistics.csv`
- `eda_report.csv` 
- `correlation_matrix.png`

</details>

<details>
<summary><b>🧹 ml preprocess</b> - Data Preprocessing</summary>

```bash
ml preprocess [OPTIONS]
```

**Options:**
- `--config, -c PATH` - Configuration file path (default: config.yaml)

**Features:**
- Automatic categorical variable encoding
- Data validation and cleaning
- Preprocessed data export

</details>

<details>
<summary><b>🤖 ml train</b> - Model Training</summary>

```bash
ml train [OPTIONS]
```

**Options:**
- `--config, -c PATH` - Configuration file path (default: config.yaml)

**Outputs:**
- `fitted_pipeline.pkl` - Serialized model
- `best_model_pipeline.py` - Exportable Python script
- `feature_info.json` - Model metadata and statistics

</details>

<details>
<summary><b>🔮 ml predict</b> - Make Predictions</summary>

```bash
ml predict -i INPUT_FILE -o OUTPUT_FILE -m MODEL_DIR
```

**Required Options:**
- `-i, --input-path PATH` - Input CSV file
- `-o, --output-path PATH` - Output predictions file
- `-m, --model-path PATH` - Model directory

</details>

<details>
<summary><b>🚀 ml serve</b> - Model Serving</summary>

```bash
ml serve [OPTIONS]
```

**Options:**
- `--host TEXT` - Host address (default: 127.0.0.1)
- `--port INTEGER` - Port number (default: 8000)
- `--reload/--no-reload` - Auto-reload on changes (default: True)
- `--config, -c PATH` - Configuration file (default: config.yaml)

**API Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /model-info` - Model metadata
- `POST /predict` - Make predictions
- `GET /docs` - Interactive documentation

</details>

<details>
<summary><b>🧹 ml clean</b> - Cleanup Artifacts</summary>

```bash
ml clean
```

Removes all generated files tracked in `.artifacts.log`

</details>

### Configuration Format

The [`ml init`](ml_cli/commands/init.py) command creates a configuration file with the following structure:

```yaml
# config.yaml
data:
  data_path: 'data/your_dataset.csv'
  target_column: 'target'

task:
  type: 'classification'  # or 'regression', 'clustering'

output_dir: 'output'

tpot:
  generations: 4
  population_size: 20
  verbosity: 2

training:
  test_size: 0.2
  random_state: 42
```

## 💡 Examples

### Complete Workflow Example

```bash
# 1. Set up project
ml init
# Follow prompts to configure your project

# 2. Analyze your data
ml eda

# 3. Prepare data for training
ml preprocess

# 4. Train your model
ml train

# 5. Start API server
ml serve --port 8080

# 6. Test your API
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"feature1": 1.0, "feature2": 2.0}'
```

### Working with Remote Data

```bash
# Initialize with remote dataset
ml init
# Enter URL: https://example.com/dataset.csv

# The tool automatically downloads and validates the data
```

### Custom Configuration

```bash
# Use JSON configuration
ml init --format json

# Train with custom config
ml train --config my_config.yaml

# Serve with custom host/port
ml serve --host 0.0.0.0 --port 9000
```

## 🏗️ Architecture

<div align="center">

```mermaid
graph TB
    A[ml init] --> B[Configuration Setup]
    B --> C[ml eda]
    C --> D[Data Analysis]
    D --> E[ml preprocess]
    E --> F[Data Cleaning]
    F --> G[ml train]
    G --> H[TPOT AutoML]
    H --> I[Model Export]
    I --> J[ml serve]
    J --> K[FastAPI Server]
    
    style A fill:#e1f5fe
    style G fill:#f3e5f5
    style J fill:#e8f5e8
```

</div>

### Project Structure

```
ml_cli/
├── 📁 ml_cli/
│   ├── 🌐 api/              # FastAPI application
│   │   └── main.py          # API endpoints and model serving
│   ├── ⚡ commands/         # CLI command implementations
│   │   ├── init.py          # Project initialization
│   │   ├── eda.py           # Exploratory data analysis
│   │   ├── train.py         # Model training
│   │   ├── serve.py         # Model serving
│   │   └── ...
│   ├── 🧠 core/            # Core ML logic
│   │   ├── data.py          # Data loading and validation
│   │   └── train.py         # Training algorithms
│   ├── 🛠️ utils/           # Utility functions
│   │   └── utils.py         # Helper functions
│   └── 🚀 cli.py           # Main CLI entry point
├── 🧪 tests/               # Comprehensive test suite
├── 📊 examples/            # Example datasets
├── 📋 requirements.txt     # Dependencies
└── ⚙️ setup.py            # Package configuration
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=ml_cli

# Run specific test file
python -m pytest tests/test_cli.py -v
```

### Test the API

Use the included [test script](test_api.py):

```bash
# Start the server
ml serve

# Run API tests (in another terminal)
python test_api.py
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ml_cli.git
cd ml_cli

# Create development environment
python -m venv venv
source venv/bin/activate

# Install in development mode with test dependencies
pip install -e .
pip install pytest pytest-cov black flake8

# Run tests to ensure everything works
pytest
```

### Contributing Guidelines

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **✅ Test** your changes (`pytest`)
4. **💫 Format** your code (`black ml_cli/`)
5. **📝 Commit** your changes (`git commit -m 'feat: Add amazing feature'`)
6. **🚀 Push** to the branch (`git push origin feature/amazing-feature`)
7. **📬 Open** a Pull Request

### Code Standards

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://github.com/psf/black) for code formatting
- Write comprehensive tests for new features
- Update documentation for user-facing changes

## 📈 Roadmap

### Upcoming Features

- [ ] 🐳 **Docker Support** - Containerized deployment
- [ ] ☁️ **Cloud Integration** - AWS/GCP/Azure deployment
- [ ] 📱 **Web UI** - Browser-based interface
- [ ] 🔄 **Model Versioning** - Track model iterations
- [ ] 📊 **Advanced Metrics** - Detailed performance analytics
- [ ] 🛡️ **Model Monitoring** - Production model tracking
- [ ] 🔌 **Plugin System** - Custom algorithm integration

### Performance Improvements

- [ ] ⚡ **Parallel Processing** - Multi-core training
- [ ] 💾 **Caching** - Intelligent result caching
- [ ] 📦 **Model Compression** - Optimized model sizes

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help

- 📚 **Documentation**: Check this README and inline help (`ml --help`)
- 🐛 **Bug Reports**: [Open an issue](https://github.com/Ayo-Cyber/ml_cli/issues)
- 💡 **Feature Requests**: [Start a discussion](https://github.com/Ayo-Cyber/ml_cli/discussions)
- 📧 **Direct Contact**: atunraseayomide@gmail.com

### FAQ

<details>
<summary><b>Q: What data formats are supported?</b></summary>

A: Currently supports CSV, TXT, and JSON files. Both local files and remote URLs (HTTP/HTTPS) are supported.
</details>

<details>
<summary><b>Q: Can I use custom machine learning algorithms?</b></summary>

A: Currently, the tool uses TPOT for AutoML. Custom algorithms will be supported in future versions through the plugin system.
</details>

<details>
<summary><b>Q: How do I deploy models to production?</b></summary>

A: Use `ml serve` to create a production-ready FastAPI server. For advanced deployment, consider using Docker or cloud platforms.
</details>

<details>
<summary><b>Q: Is GPU training supported?</b></summary>

A: GPU support depends on the underlying TPOT and scikit-learn implementations. Future versions will include explicit GPU acceleration.
</details>

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by [Atunrase Ayomide](https://github.com/Ayo-Cyber)

[🏠 Home](https://github.com/Ayo-Cyber/ml_cli) • [📖 Docs](#-documentation) • [🐛 Issues](https://github.com/Ayo-Cyber/ml_cli/issues) • [💬 Discussions](https://github.com/Ayo-Cyber/ml_cli/discussions)

</div>