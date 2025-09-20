import click
import json
import yaml
import questionary
import sys
import os
import logging
import time
import requests
from ml_cli.utils.utils import (
    write_config,
    should_prompt_target_column,
    is_readable_file,
    is_target_in_file,
    get_target_directory,
    log_artifact,
    download_data,
    create_convenience_script
)

# Constants
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."
LOCAL_DATA_DIR = ".ml_cli"
LOCAL_DATA_FILENAME = "local_data.csv"


def create_convenience_script(target_directory):
    """Create a convenience script to help users navigate to the project directory."""
    script_name = "activate.sh"
    script_path = os.path.join(target_directory, script_name)
    
    script_content = f"""#!/bin/bash
# Activate ML project environment
# Usage: source {script_name}

cd "{target_directory}"
echo "✅ Activated ML project environment in: {target_directory}"
echo "💡 You can now run commands like 'ml train', 'ml serve', etc."
"""
    
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        log_artifact(script_path)
        return script_path
    except Exception as e:
        logging.warning(f"Could not create convenience script: {e}")
        return None

def download_data(data_path, ssl_verify, target_directory):
    """Download data from a URL and save it locally."""
    if not data_path.startswith(('http://', 'https://')):
        return data_path

    click.secho(f"Downloading data from {data_path}...", fg="blue")
    try:
        response = requests.get(data_path, verify=ssl_verify, stream=True)
        response.raise_for_status()

        # Create local data directory
        local_data_path = os.path.join(target_directory, LOCAL_DATA_DIR)
        os.makedirs(local_data_path, exist_ok=True)

        local_file_path = os.path.join(local_data_path, LOCAL_DATA_FILENAME)

        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        click.secho(f"Data downloaded and saved to {local_file_path}", fg="green")
        return local_file_path
    except requests.exceptions.RequestException as e:
        click.secho(f"Error downloading data: {e}", fg='red')
        logging.error(f"Error downloading data: {e}")
        sys.exit(1)

@click.command(help="""Initializes a new ML project by creating a configuration file (config.yaml or config.json).
This command guides you through setting up your project's data path, target column, task type (classification, regression, clustering),
output directory, and TPOT generations. It also creates an 'activate.sh' script for easy project navigation.

Examples:
  ml-cli init
  ml-cli init --format json
  ml-cli init --no-ssl-verify # To disable SSL verification for data downloaded from URLs
""")
@click.option('--format', default='yaml', type=click.Choice(['yaml', 'json']),
              help='Specify the format of the configuration file to be created (yaml or json). Default is yaml.')
@click.option('--ssl-verify/--no-ssl-verify', default=True,
              help='Enable or disable SSL verification for data paths that are URLs. Default is enabled.')
def init(format, ssl_verify):
    """Initialize a new configuration file (YAML or JSON)"""
    click.secho("Initializing configuration...", fg="green")

    start_time = time.time()  # Start timing

    # Store the original working directory before any changes
    original_dir = os.getcwd()

    # Determine the target directory based on user choice
    target_directory = get_target_directory()

    # Track if we created a new directory
    created_new_directory = target_directory != original_dir

    data_path_input = click.prompt('Please enter the data directory path', type=str)
    

    # Log the data path input
    logging.info(f"Data path provided: {data_path_input}")

    # Download data if it's a URL
    data_path = download_data(data_path_input, ssl_verify, target_directory)
    

    # Check if the file path is readable, passing the SSL verification flag
    if not is_readable_file(data_path, ssl_verify=ssl_verify):
        click.secho("Error: The file does not exist, is not readable, or has an unsupported format. Please provide a valid CSV, TXT, or JSON file.", fg='red')
        logging.error("Invalid data path provided.")
        sys.exit(1)

    task_type = questionary.select(
        "Please select the task type:",
        choices=[
            questionary.Choice(title="Classification", value="classification"),
            questionary.Choice(title="Regression", value="regression"),
            questionary.Choice(title="Clustering", value="clustering")
        ]
    ).ask(kbi_msg=KEYBOARD_INTERRUPT_MESSAGE)
    

    if task_type is None:
        sys.exit(1)

    # Log the task type selection
    logging.info(f"Task type selected: {task_type}")

    target_column = click.prompt('Please enter the target variable column', type=str) if should_prompt_target_column(task_type) else None

    target_found, corrected_target_column = is_target_in_file(data_path, target_column, ssl_verify=ssl_verify)

    if target_found:
        target_column = corrected_target_column  # Update with corrected column name
    else:
        click.secho(f"Error: The target column '{target_column}' is not present in the data file.", fg='red')
        logging.error(f"Target column '{target_column}' not found in the data file.")
        sys.exit(1)

    target_found, corrected_target_column = is_target_in_file(data_path, target_column, ssl_verify=ssl_verify)
    if target_found:
        target_column = corrected_target_column  # Update with corrected column name
    else:
        click.secho(f"Error: The target column '{target_column}' is not present in the data file.", fg='red')
        logging.error(f"Target column '{target_column}' not found in the data file.")
        sys.exit(1)

    output_dir = click.prompt('Please enter the output directory path', type=str, default='output')
    
    generations = click.prompt('Please enter the number of TPOT generations', type=int, default=4)
    

    config_data = {
        'data': {
            'data_path': data_path,
            'target_column': target_column,
        },
        'task': {
            'type': task_type,
        },
        'output_dir': output_dir,
        'tpot': {
            'generations': generations
        }
    }

    # Prepare configuration filename and log the action
    config_filename = os.path.join(target_directory, f'config.{format}')
    logging.info(f"Writing configuration to {config_filename}")

    write_config(config_data, format, config_filename)

    # Log the generated configuration file as an artifact
    log_artifact(config_filename)
    
    # Create a convenience script for easy navigation
    create_convenience_script(target_directory)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    click.secho(f"Configuration file created at: {config_filename}", fg="green")
    logging.info(f"Configuration file created! (Time taken: {elapsed_time:.2f}s)")
    
    # Get the current working directory
    current_dir = os.getcwd()
    
    # Provide clear instructions based on whether we created a new directory
    if created_new_directory:
        activate_script_path = os.path.join(target_directory, 'activate.sh')
        click.secho(f"\n✅ Project initialized in: {target_directory}", fg="green", bold=True)
        click.secho(f"⚠️  Your terminal is still in: {original_dir}", fg="yellow")
        click.secho(f"\n💡 To move to your project directory, run:", fg="yellow")
        click.secho(f"   cd {target_directory}", fg="cyan", bold=True)
        click.secho(f"   # OR source the activation script:", fg="blue")
        click.secho(f"   source {activate_script_path}", fg="cyan")
    else:
        click.secho(f"\n✅ Project initialized in current directory!", fg="green", bold=True)
        click.secho(f"💡 You can now run commands like 'ml train'.", fg="yellow")
    
    click.secho(f"\n📋 Available commands:", fg="blue")
    click.secho(f"   ml eda       - Perform exploratory data analysis", fg="white")
    click.secho(f"   ml train      - Train your model", fg="white")
    click.secho(f"   ml serve      - Serve your model as an API", fg="white")
    click.secho(f"   ml predict    - Make predictions", fg="white")
    click.secho(f"   ml preprocess - Preprocess your data", fg="white")
    
    logging.info(f"Original directory: {original_dir}")
    logging.info(f"Target directory: {target_directory}")
    logging.info(f"Created new directory: {created_new_directory}")
