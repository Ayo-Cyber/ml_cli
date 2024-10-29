import click
import json
import yaml
import questionary
import sys
import os
import logging
import time
import io 
from ml_cli.utils.utils import (write_config,
    should_prompt_target_column,
    get_dependencies, 
    is_readable_file,
    is_target_in_file,
    get_target_directory)


# Constants
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."

# ANSI escape codes for coloring text
class ColorFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'RESET': '\033[0m',    # Reset to default
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        message = super().format(record)
        return f"{color}{message}{self.COLORS['RESET']}"

# Configure logging without timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',  # Removed the timestamp
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Change the default formatter to the color formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(ColorFormatter())

@click.command()
@click.option('--format', default='yaml', type=click.Choice(['yaml', 'json']), help='Format of the configuration file (yaml or json)')
@click.option('--ssl-verify/--no-ssl-verify', default=True, help='Enable or disable SSL verification for URL data paths')
def init(format, ssl_verify):
    """Initialize a new configuration file (YAML or JSON)"""
    
    start_time = time.time()  # Start timing

    # Determine the target directory based on user choice
    target_directory = get_target_directory()
    os.chdir(target_directory)

    data_path = click.prompt('Please enter the data directory path', type=str)
    
    # Log the data path input
    logging.info(f"Data path provided: {data_path}")
    
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

    if target_column and not is_target_in_file(data_path, target_column, ssl_verify=ssl_verify):
        click.secho(f"Error: The target column '{target_column}' is not present in the data file.", fg='red')
        logging.error(f"Target column '{target_column}' not found in the data file.")
        sys.exit(1)

    dependencies = get_dependencies(task_type)

    config_data = {
        'data': {
            'data_path': data_path,
            'target_column': target_column,
        },
        'task': {
            'type': task_type,
        },
        'dependencies': dependencies,
    }

    # Prepare configuration filename and log the action
    config_filename = os.path.join(target_directory, f'config.{format}')
    logging.info(f"Writing configuration to {config_filename}")

    write_config(config_data, format, config_filename)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    click.secho(f"Configuration file created at : {config_filename}", fg="green")
    logging.info(f"Configuration file created! (Time taken: {elapsed_time:.2f}s)")
    logging.info("Current Working Directory: " + os.getcwd())
