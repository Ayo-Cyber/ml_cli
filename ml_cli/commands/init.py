import click
import json
import yaml
import questionary
import sys
import os
import logging
import time
import io 
from ml_cli.utils.utils import (
    write_config,
    should_prompt_target_column, 
    is_readable_file,
    is_target_in_file,
    get_target_directory,
    log_artifact
)

# Constants
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."

# Configure logging without timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',  # Removed the timestamp
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_convenience_script(target_directory):
    """Create a convenience script to help users navigate to the project directory."""
    script_name = "start_ml_project.sh"
    script_path = os.path.join(target_directory, script_name)
    
    # Also create a simple alias script
    alias_script_name = "goto_project.sh"
    alias_script_path = os.path.join(target_directory, alias_script_name)
    
    script_content = f"""#!/bin/bash
# ML CLI Project Convenience Script
# This script helps you quickly navigate to your ML project and see available commands

echo "🚀 ML CLI Project Directory"
echo "=========================="
echo "📁 Project location: {target_directory}"
echo ""

# Change to project directory
cd "{target_directory}"

echo "✅ Changed to project directory!"
echo ""
echo "💡 Available commands:"
echo "   ml train      - Train your model"
echo "   ml serve      - Serve your model as an API"
echo "   ml predict    - Make predictions"
echo "   ml preprocess - Preprocess your data"
echo "   ml eda        - Exploratory data analysis"
echo "   ml clean      - Clean up artifacts"
echo ""
echo "🔍 To get started, run: ml train"
echo ""

# Start a new shell in the project directory
exec $SHELL
"""

    alias_content = f"""#!/bin/bash
# Simple navigation script
# Usage: source {alias_script_name}
cd "{target_directory}"
echo "✅ Navigated to ML project directory: {target_directory}"
"""
    
    try:
        # Create main convenience script
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        log_artifact(script_path)
        
        # Create simple alias script
        with open(alias_script_path, 'w') as f:
            f.write(alias_content)
        os.chmod(alias_script_path, 0o755)
        log_artifact(alias_script_path)
        
        return script_path, alias_script_path
    except Exception as e:
        logging.warning(f"Could not create convenience script: {e}")
        return None, None

@click.command(help="""Initialize a new configuration file (YAML or JSON).

Usage examples:
  ml init
  ml init --format json
""")
@click.option('--format', default='yaml', type=click.Choice(['yaml', 'json']), help='Format of the configuration file (yaml or json)')
@click.option('--ssl-verify/--no-ssl-verify', default=True, help='Enable or disable SSL verification for URL data paths')
def init(format, ssl_verify):
    """Initialize a new configuration file (YAML or JSON)"""
    click.secho("Initializing configuration...", fg="green")

    
    start_time = time.time()  # Start timing

    # Determine the target directory based on user choice
    target_directory = get_target_directory()

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
    
    # Get the current working directory (which may have been changed during init)
    current_dir = os.getcwd()
    
    # If we're not in the original directory, provide clear instructions
    if target_directory != os.path.dirname(config_filename) or target_directory != current_dir:
        click.secho(f"\n📁 Project initialized in: {target_directory}", fg="blue", bold=True)
        click.secho(f"💡 To start working with your project, choose one of:", fg="yellow")
        click.secho(f"", fg="white")
        click.secho(f"   Option 1 - Manual navigation:", fg="green")
        click.secho(f"   cd {target_directory}", fg="cyan", bold=True)
        click.secho(f"   ml train", fg="cyan")
        click.secho(f"", fg="white")
        click.secho(f"   Option 2 - Quick navigation (source to change current shell):", fg="green")
        click.secho(f"   source {os.path.join(target_directory, 'goto_project.sh')}", fg="cyan", bold=True)
        click.secho(f"", fg="white")
        click.secho(f"   Option 3 - Full project environment (opens new shell):", fg="green")
        click.secho(f"   {os.path.join(target_directory, 'start_ml_project.sh')}", fg="cyan", bold=True)
    else:
        click.secho(f"\n✅ Project initialized in current directory!", fg="green", bold=True)
        click.secho(f"💡 You can now run:", fg="yellow")
        click.secho(f"   ml train", fg="cyan", bold=True)
    
    click.secho(f"\n📋 Available commands:", fg="blue")
    click.secho(f"   ml train      - Train your model", fg="white")
    click.secho(f"   ml serve      - Serve your model as an API", fg="white")
    click.secho(f"   ml predict    - Make predictions", fg="white")
    click.secho(f"   ml preprocess - Preprocess your data", fg="white")
    
    logging.info("Current Working Directory: " + current_dir)
