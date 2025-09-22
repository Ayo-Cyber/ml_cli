import click
import json
import yaml
import questionary
import sys
import os
import logging
import time
import io
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

# Configure logging without timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',  # Removed the timestamp
    handlers=[logging.StreamHandler(sys.stdout)]
)

@click.command(help="""Initialize a new configuration file (YAML or JSON).

Usage examples:
  ml init
  ml init --format json
"""
)
@click.option('--format', default='yaml', type=click.Choice(['yaml', 'json']), help='Format of the configuration file (yaml or json)')
@click.option('--ssl-verify/--no-ssl-verify', default=True, help='Enable or disable SSL verification for URL data paths')
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
    click.echo(f"DEBUG: data_path_input = {data_path_input}")

    # Log the data path input
    logging.info(f"Data path provided: {data_path_input}")

    # Download data if it's a URL
    data_path = download_data(data_path_input, ssl_verify, target_directory)
    click.echo(f"DEBUG: data_path = {data_path}")

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
    click.echo(f"DEBUG: task_type = {task_type}")

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
        sys.exit(1)

    output_dir = click.prompt('Please enter the output directory path', type=str, default='output')
    click.echo(f"DEBUG: output_dir = {output_dir}")
    generations = click.prompt('Please enter the number of TPOT generations', type=int, default=4)
    click.echo(f"DEBUG: generations = {generations}")

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
