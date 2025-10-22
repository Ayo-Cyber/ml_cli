import os
import logging
import pandas as pd
from pathlib import Path
from ml_cli.utils.utils import log_artifact
from ml_cli.utils.exceptions import DataError
from ml_cli.config.models import MLConfig

def load_data(config: MLConfig) -> pd.DataFrame:
    """Load data from the specified path in the config."""
    output_dir = Path(config.output_dir)
    preprocessed_csv_path = output_dir / 'preprocessed_data.csv'

    logging.info("Checking for preprocessed CSV file...")

    if preprocessed_csv_path.exists():
        logging.info(f"Preprocessed CSV found: {preprocessed_csv_path}. Using this file.")
        data_path = preprocessed_csv_path
        log_artifact(str(preprocessed_csv_path))
    else:
<<<<<<< HEAD
        print("No preprocessed CSV found. Using unprocessed data.")
        data_path = config.data.data_path
=======
        logging.info("No preprocessed CSV found. Using unprocessed data.")
        data_path = config['data'].get('data_path')
>>>>>>> main
        if not data_path:
            raise DataError("No data path specified in config. Use 'data_path' key.")

    try:
        data = pd.read_csv(data_path)
        if data.empty:
            logging.warning(f"The data file at {data_path} is empty.")
        logging.info(f"Data loaded successfully from {data_path}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"Data file not found at {data_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"The data file at {data_path} is empty.")
        raise
    except pd.errors.ParserError:
        logging.error(f"Error parsing the data file at {data_path}. Please check the file format.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading data from {data_path}: {e}")
        raise
