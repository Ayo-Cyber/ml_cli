import pandas as pd
import os
import yaml
import logging
import click
import json


def log_artifact(file_path):
    """Log the generated artifact file path to `.artifacts.log`."""
    artifact_log_path = os.path.join(os.getcwd(), '.artifacts.log')
    with open(artifact_log_path, 'a') as log_file:
        log_file.write(file_path + '\n')

def load_data(data_path):
    """Load the dataset from a specified path."""
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            click.secho("The dataset is empty. Nothing to preprocess.", fg='yellow')
            return None
        logging.info("Data loaded successfully for preprocessing.")
        return df
    except FileNotFoundError:
        click.secho(f"Error: Data file not found at '{data_path}'.", fg='red')
        logging.error(f"Data file not found at '{data_path}'.")
        return None
    except pd.errors.EmptyDataError:
        click.secho("The data file is empty.", fg='red')
        logging.error("The data file is empty.")
        return None
    except pd.errors.ParserError:
        click.secho("Error parsing the data file. Please check the file format.", fg='red')
        logging.error("Error parsing the data file.")
        return None
    except Exception as e:
        click.secho(f"An unexpected error occurred while loading data for preprocessing: {e}", fg='red')
        logging.error(f"An unexpected error occurred while loading data for preprocessing: {e}")
        return None

def encode_categorical_columns(df):
    """One-hot encode categorical columns in the DataFrame."""
    try:
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            df = pd.get_dummies(df, columns=object_cols, drop_first=True)
            logging.info(f"One-hot encoded columns: {list(object_cols)}")
        return df
    except AttributeError:
        click.secho("Error: The dataset is not a valid DataFrame.", fg='red')
        logging.error("The dataset is not a valid DataFrame.")
        return None
    except Exception as e:
        click.secho(f"An unexpected error occurred during one-hot encoding: {e}", fg='red')
        logging.error(f"An unexpected error occurred during one-hot encoding: {e}")
        return None

def save_preprocessed_data(df, file_path):
    """Save the preprocessed DataFrame to a specified file path."""
    try:
        df.to_csv(file_path, index=False)
        click.secho(f"Preprocessed data saved to {file_path}", fg="green")
        logging.info(f"Preprocessed data saved at: {file_path}")
        log_artifact(file_path)
    except IOError as e:
        click.secho(f"Error saving preprocessed data to {file_path}: {e}", fg='red')
        logging.error(f"Error saving preprocessed data to {file_path}: {e}")
    except Exception as e:
        click.secho(f"An unexpected error occurred while saving preprocessed data: {e}", fg='red')
        logging.error(f"An unexpected error occurred while saving preprocessed data: {e}")

@click.command(help="""Preprocess the dataset specified in the configuration file. 

Usage example:
  ml preprocess --config config.yaml
  ml preprocess --config config.json
""")
@click.option('--config', '-c', 'config_file', default="config.yaml",
              help='The absolute or relative path to the configuration file (config.yaml or config.json) that specifies data paths and preprocessing steps.')
def preprocess(config_file):
    """Preprocess the dataset to handle non-numeric columns using OneHotEncoder."""
    click.secho("Preprocessing data...", fg="green")

    # Load config (JSON or YAML)
    try:
        if config_file.endswith(".json"):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
        else:  # default to YAML
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
    except FileNotFoundError:
        click.secho(f"Error: Configuration file '{config_file}' not found.", fg='red')
        logging.error(f"Configuration file not found: {config_file}")
        return
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        click.secho(f"Error parsing configuration file: {e}", fg='red')
        logging.error(f"Error parsing configuration file: {e}")
        return
    except Exception as e:
        click.secho(f"An unexpected error occurred while reading the configuration file: {e}", fg='red')
        logging.error(f"An unexpected error occurred while reading the configuration file: {e}")
        return

    # Extract dataset path
    data_path = config_data.get("data", {}).get("data_path")
    if not data_path:
        click.secho("No data path specified in config file.", fg='red')
        return

    # Load dataset
    df = load_data(data_path)
    if df is None:
        return

    # One-hot encode categorical columns
    df = encode_categorical_columns(df)
    if df is None:
        return

    # Save preprocessed data
    output_dir = config_data.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_file = os.path.join(output_dir, "preprocessed_data.csv")
    save_preprocessed_data(df, preprocessed_file)
