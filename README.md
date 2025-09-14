# ML CLI Pipeline

This project provides a command-line interface (CLI) for running a machine learning pipeline using the TPOT library. It facilitates the automation of the machine learning workflow, including data loading, preprocessing, model training, and exporting the best-performing model.

## Features

- Automatically checks for a preprocessed CSV file.
- Loads data from either a preprocessed or raw CSV file based on availability.
- Supports classification and regression tasks using TPOT.
- Configurable through a YAML file.
- Exports the optimized machine learning pipeline to a Python file.
- **New:** Train the model separately using the `train` command.
- **New:** Make predictions on new data using the `predict` command.
- **New:** Serve the trained model as a REST API using the `serve` command.

## Requirements

- Python 3.x
- Required Python packages:
  - `click`
  - `pandas`
  - `scikit-learn`
  - `TPOT`
  - `PyYAML`
  
You can install the required packages using pip:

```bash
pip install click pandas scikit-learn tpot pyyaml
```

## Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your configuration file**: Create a `config.yaml` file in the project directory with the following structure:

   ```yaml
   data:
     path: 'path/to/your/raw_data.csv'  # Path to raw data file
     target_column: 'your_target_column'      # Name of the target column
   task:
     type: 'classification'                    # Task type: 'classification' or 'regression'
   ```

5. **Prepare your data**: If you have preprocessed data, ensure it is named `preprocessed_data.csv` and is located in the project directory.

## Usage



### Train the model

To train the model separately, use the `train` command:

```bash
ml train
```

### Make predictions

To make predictions on new data, use the `predict` command:

```bash
ml predict --input-path /path/to/new_data.csv --output-path /path/to/predictions.csv --model-path /path/to/best_model_pipeline.py
```

### Serve the model

To serve the trained model as a REST API, use the `serve` command:

```bash
ml serve
```

This will start a FastAPI server at `http://127.0.0.1:8000`.

## Error Handling

If an error occurs during data loading or processing, you will be prompted to run the preprocessing command to prepare your data.

## Logging

The pipeline includes logging functionality to provide insights into the data loading process, model training, and any errors encountered. Logs will be printed to the console.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Markdown Notes:
- Make sure to replace `yourusername/your-repo-name` with your actual GitHub repository URL.
- You can copy this directly into your `README.md` file. It should render correctly on GitHub or any Markdown viewer.