import os
import logging
import pandas as pd
from pathlib import Path
from ml_cli.utils.utils import log_artifact
import ssl

def load_data(config: dict) -> pd.DataFrame:
    """Load data from the specified path in the config."""
    output_dir = Path(config.get('output_dir', 'output'))
    preprocessed_csv_path = output_dir / 'preprocessed_data.csv'

    logging.info("Checking for preprocessed CSV file...")

    if preprocessed_csv_path.exists():
        print(f"Preprocessed CSV found: {preprocessed_csv_path}. Using this file.")
        data_path = preprocessed_csv_path
        log_artifact(str(preprocessed_csv_path))
    else:
        print("No preprocessed CSV found. Using unprocessed data.")
        # Handle both 'data_path' and 'path' keys for backward compatibility
        data_path = config['data'].get('data_path') or config['data'].get('path')
        if not data_path:
            raise ValueError("No data path specified in config. Use 'data_path' or 'path' key.")

    try:
        # Create an unverified SSL context
        ssl._create_default_https_context = ssl._create_unverified_context
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
