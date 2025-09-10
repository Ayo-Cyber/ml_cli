import os
import logging
import sys
import click
import yaml
from ml_cli.core.data import load_data
from ml_cli.core.train import train_model

@click.command(help="""Train the ML model based on the configuration file (config.yaml).

Usage example:
  ml train
""")
def train():
    """Train the ML model based on the configuration."""
    click.secho("Training ML model...", fg="green")
    
    try:
        # Load the configuration file
        config_path = os.path.join(os.getcwd(), 'config.yaml')
        if not os.path.exists(config_path):
            click.secho("Error: Configuration file 'config.yaml' not found in the current directory.", fg='red')
            logging.error("Configuration file not found.")
            sys.exit(1)
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Load the data
        data = load_data(config)

        # Train the model
        train_model(data, config)

    except FileNotFoundError:
        click.secho("Error: Data file not found. Please check the data path in your config file.", fg='red')
        sys.exit(1)
    except ValueError as e:
        click.secho(f"Error: {e}", fg='red')
        sys.exit(1)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red')
        sys.exit(1)
