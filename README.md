# ML CLI Pipeline

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🚀 Overview

The ML CLI Pipeline is a command-line interface (CLI) tool for streamlined machine learning workflows. It helps manage ML projects from initialization to deployment.

Built with `click` and leveraging the robust `TPOT` library for automated machine learning (AutoML), this tool empowers data scientists and developers to effortlessly manage their ML projects from data preparation to model deployment.

With ML CLI, you can:
- **Initialize** new ML projects with ease, setting up configurations for data, tasks, and output.
- Perform **Exploratory Data Analysis (EDA)** to gain insights into your datasets.
- **Preprocess** your data, handling categorical features automatically.
- **Train** state-of-the-art machine learning models using AutoML (TPOT).
- Make **predictions** on new data with your trained models.
- **Serve** your models as a high-performance REST API using FastAPI, ready for integration into other applications.
- **Clean up** generated artifacts to maintain a tidy project environment.

Say goodbye to repetitive scripting and hello to a more efficient, automated ML development cycle!

## ✨ Features

*   **Project Initialization (`ml init`)**: Quickly set up a new ML project with an interactive wizard, defining data paths, target columns, task types (classification, regression, clustering), and TPOT parameters. Supports both YAML and JSON configuration formats.
*   **Exploratory Data Analysis (`ml eda`)**: Generate comprehensive summary statistics, missing value reports, and correlation matrix heatmaps for your dataset, saving them as CSV and PNG files.
*   **Data Preprocessing (`ml preprocess`)**: Automatically handle categorical features using one-hot encoding, preparing your data for model training. Saves the preprocessed data for future use.
*   **Model Training (`ml train`)**: Train robust ML models using TPOT's AutoML capabilities. The best-performing pipeline is exported as a Python script and a scikit-learn compatible `.pkl` file, along with feature metadata.
*   **Prediction (`ml predict`)**: Utilize your trained models to make predictions on new, unseen data. Specify input data, output path for predictions, and the model's output directory.
*   **Model Serving (`ml serve`)**: Deploy your trained model as a production-ready REST API using FastAPI. The API automatically adapts to your model's features and provides endpoints for predictions, model info, and health checks.
*   **Artifact Cleanup (`ml clean`)**: Remove all generated files (e.g., EDA reports, preprocessed data, trained models) that are tracked in the `.artifacts.log` file, ensuring a clean project directory.

## 🛠️ Installation

To get started with ML CLI, follow these simple steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/ml_cli.git
    cd ml_cli
    ```

2.  **Create and activate a virtual environment** (highly recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the package in editable mode**:

    ```bash
    pip install -e .
    ```
    This command installs the `ml_cli` package and its dependencies listed in `requirements.txt`.

## 🚀 Usage

All commands are accessed via the `ml` entry point.

### 1. Initialize Your Project (`ml init`)

Start by setting up your project configuration. This command will guide you through the process.

```bash
ml init
```

**Options:**
*   `--format [yaml|json]`: Format of the configuration file (default: `yaml`).
*   `--ssl-verify / --no-ssl-verify`: Enable or disable SSL verification for URL data paths (default: `True`).

**Example Walkthrough:**
```
$ ml init
Initializing configuration...
Please enter the data directory path: data/customer_churn_dataset-testing-master.csv
Please select the task type: (Use arrow keys)
 > Classification
   Regression
   Clustering
Please enter the target variable column: Churn
Please enter the output directory path (output): my_ml_output
Please enter the number of TPOT generations (4): 10
Configuration file created at: config.yaml
✅ Project initialized in current directory!
💡 You can now run commands like 'ml train'.
```

### 2. Perform Exploratory Data Analysis (`ml eda`)

Generate insights into your dataset. This command uses the `data_path` specified in your `config.yaml`.

```bash
ml eda
```

**Output Files (generated in your current directory):**
*   `summary_statistics.csv`: Descriptive statistics of your dataset.
*   `eda_report.csv`: Report on missing values and data types.
*   `correlation_matrix.png`: Heatmap visualizing feature correlations.

### 3. Preprocess Your Data (`ml preprocess`)

Prepare your data for model training by handling categorical features.

```bash
ml preprocess
```

**Options:**
*   `--config, -c PATH`: Path to the configuration file (default: `config.yaml`).

**Output File:**
*   `output/preprocessed_data.csv` (or `YOUR_OUTPUT_DIR/preprocessed_data.csv`): Your dataset with categorical columns one-hot encoded.

### 4. Train Your Model (`ml train`)

Train an ML model using TPOT based on your configuration.

```bash
ml train
```

**Options:**
*   `--config, -c PATH`: Path to the configuration file (default: `config.yaml`).

**Output Files (generated in your configured output directory, e.g., `output/`):**
*   `best_model_pipeline.py`: A Python script representing the optimized scikit-learn pipeline.
*   `fitted_pipeline.pkl`: The serialized scikit-learn pipeline object.
*   `feature_info.json`: Metadata about the features and target column, including the model's performance score.

### 5. Make Predictions (`ml predict`)

Use your trained model to predict outcomes on new data.

```bash
ml predict -i path/to/new_data.csv -o path/to/predictions.csv -m output/
```

**Options:**
*   `-i, --input-path PATH`: Path to the input data for predictions (e.g., `data/new_samples.csv`). **(Required)**
*   `-o, --output-path PATH`: Path to save the predictions (e.g., `predictions.csv`). **(Required)**
*   `-m, --model-path PATH`: Path to the output directory where the trained model is saved (e.g., `output/`). **(Required)**

### 6. Serve Your Model (`ml serve`)

Deploy your model as a local REST API.

```bash
ml serve
```

**Options:**
*   `-h, --host TEXT`: The host to bind the server to (default: `127.0.0.1`).
*   `-p, --port INTEGER`: The port to bind the server to (default: `8000`).
*   `--reload / --no-reload`: Enable or disable auto-reloading of the server on code changes (default: `True`).
*   `--config, -c PATH`: Path to the configuration file (default: `config.yaml`).

**API Endpoints:**
*   `GET /`: Welcome message and model status.
*   `GET /health`: Health check.
*   `GET /model-info`: Information about the loaded model and features.
*   `POST /predict`: Make predictions using the trained model.
*   `POST /reload-model`: Reload the model after retraining without restarting the server.

**Access API Documentation:**
*   Swagger UI: `http://127.0.0.1:8000/docs`
*   ReDoc: `http://127.0.0.1:8000/redoc`

### 7. Clean Up Artifacts (`ml clean`)

Remove all generated files from your project.

```bash
ml clean
```

This command reads the `.artifacts.log` file and deletes all paths listed within it, then removes the log file itself.

## ⚙️ Configuration (`config.yaml`)

The `ml init` command generates a `config.yaml` (or `config.json`) file. Here's an example of a comprehensive configuration:

```yaml
# config.yaml generated by 'ml init'
data:
  data_path: 'data/customer_churn_dataset-testing-master.csv' # Path to your raw data file or URL
  target_column: 'Churn'                                     # Name of the target column for prediction
task:
  type: 'classification'                                     # Task type: 'classification', 'regression', or 'clustering'
output_dir: 'output'                                         # Directory to save all generated artifacts (models, reports, preprocessed data)
tpot:
  generations: 4                                             # Number of generations for TPOT optimization
  # You can add more TPOT parameters here, e.g.:
  # population_size: 20
  # verbosity: 2
```

## 📂 Project Structure

```
ml_cli/
├── ml_cli/
│   ├── api/             # FastAPI application for model serving
│   ├── commands/        # CLI command implementations (init, eda, train, etc.)
│   ├── core/            # Core logic for data loading, training, prediction
│   ├── utils/           # Utility functions (logging, artifact tracking)
│   └── cli.py           # Main CLI entry point
├── tests/               # Unit and integration tests
├── examples/            # Example datasets
├── requirements.txt     # Python dependencies
├── setup.py             # Package setup file
└── README.md            # Project documentation
```

## 🤝 Contributing

Contributions are highly welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and ensure tests pass.
4.  Commit your changes (`git commit -m 'feat: Add new feature X'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

Please ensure your code adheres to the project's coding style and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ❓ Support and Contact

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/Ayo-Cyber/ml_cli/issues).
