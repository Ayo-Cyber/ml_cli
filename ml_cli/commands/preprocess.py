import pandas as pd
import os
import yaml
import logging
import click
import json
from ml_cli.utils.utils import load_data, encode_categorical_columns, save_preprocessed_data

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
