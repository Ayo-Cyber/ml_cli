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

    try:
        # Load config (JSON or YAML)
        try:
            if config_file.endswith(".json"):
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
            else:  # default to YAML
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")

        # Extract dataset path
        data_path = config_data.get("data", {}).get("data_path")
        if not data_path:
            raise ConfigurationError("No data path specified in config file.")

        # Load dataset
        df = load_data(data_path)

        # One-hot encode categorical columns
        df = encode_categorical_columns(df)

        # Save preprocessed data
        output_dir = config_data.get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        preprocessed_file = os.path.join(output_dir, "preprocessed_data.csv")
        save_preprocessed_data(df, preprocessed_file)

    except (ConfigurationError, DataError) as e:
        click.secho(f"Error: {e}", fg='red')
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red')
