import os
import logging
import sys
import click
import yaml
import json
from ml_cli.core.data import load_data
from ml_cli.core.train import train_model
from ml_cli.utils.exceptions import DataError, ConfigurationError
from ml_cli.utils.utils import load_config

@click.command(help="""Train the ML model based on the configuration file.

Usage examples:
  ml train                # uses config.yaml by default
  ml train --config config.json
  ml train --config custom_config.yaml
""")
@click.option('--config', '-c', 'config_file', default="config.yaml",
              help="Path to the configuration file (YAML or JSON).")
@click.option('--config', '-c', 'config_file', default="config.yaml",
              help='The absolute or relative path to the configuration file (config.yaml or config.json) that defines the training parameters, data paths, and model settings.')
def train(config_file):
    """Train the ML model based on the configuration."""
    click.secho("Training ML model...", fg="green")

    try:
        # Load config
        config = load_config(config_file)

        # Load the data
        click.secho("Loading data...", fg="blue")
        data = load_data(config)

        # Check for categorical data
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            click.secho(f"Found categorical columns: {categorical_cols}", fg="yellow")
            logging.info(f"Found categorical columns: {categorical_cols}")
            click.secho("Automatically preprocessing categorical data for TPOT...", fg="blue")
            logging.info("Automatically preprocessing categorical data for TPOT...")

        # Train the model
        train_model(data, config)

    except FileNotFoundError:
        click.secho("Error: Data file not found. Please check the data path in your config file.", fg='red')
        logging.error("Error: Data file not found. Please check the data path in your config file.")
        sys.exit(1)
    except (DataError, ConfigurationError) as e:
        click.secho(f"Error: {e}", fg='red')
        logging.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red')
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
