import os
import logging
import pandas as pd
from pathlib import Path
from ml_cli.utils.utils import log_artifact

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
        data_path = config['data']['data_path']

    try:
        data = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
