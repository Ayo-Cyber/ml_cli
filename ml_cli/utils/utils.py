import os
import json
import yaml
import pandas as pd
import requests
import questionary
import click
import sys
import logging
import io
import difflib


# Constants for file extensions
VALID_EXTENSIONS = ('.csv', '.txt', '.json')
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."
# Python
LOCAL_DATA_DIR = "data"
LOCAL_DATA_FILENAME = "data.csv"


def write_config(config_data, format, config_filename):
    """Write configuration data to a file in the specified format (YAML or JSON)."""
    try:
        with open(config_filename, 'w') as config_file:
            if format == 'yaml':
                yaml.dump(config_data, config_file)
            elif format == 'json':
                json.dump(config_data, config_file, indent=4)
    except IOError as e:
        logging.error(f"Failed to write config file: {e}")
        click.secho("Error writing configuration file.", fg='red')
        sys.exit(1)


def should_prompt_target_column(task_type):
    """Check if the target column prompt is needed based on task type."""
    return task_type in ['classification', 'regression']



def is_readable_file(data_path, ssl_verify=True):
    """Check if the provided path is a readable file (local or URL) and has a supported format."""
    if data_path.startswith('http://') or data_path.startswith('https://'):
        return check_url_readability(data_path, ssl_verify)
    else:
        return check_local_file_readability(data_path)


def check_url_readability(data_path, ssl_verify=True):
    """Check if the URL is reachable and has a valid file extension."""
    try:
        response = requests.head(data_path, verify=ssl_verify)
        if response.status_code == 200 and data_path.endswith(VALID_EXTENSIONS):
            logging.info(f"URL {data_path} is reachable.")
            return True
        logging.warning(f"URL {data_path} is not reachable or unsupported format.")
        return False
    except requests.RequestException as e:
        logging.error(f"RequestException: {e}")
        return False


def check_local_file_readability(data_path):
    """Check if the local file exists and is readable."""
    if os.path.isfile(data_path) and os.access(data_path, os.R_OK) and data_path.endswith(VALID_EXTENSIONS):
        logging.info(f"Local file {data_path} is readable.")
        return True
    logging.warning(f"Local file {data_path} is not readable or unsupported format.")
    return False


def is_target_in_file(data_path, target_column, ssl_verify=True):
    """Check if the target column exists in the data file."""
    logging.info("Checking for target column in data.")
    
    try:
        if data_path.startswith('http://') or data_path.startswith('https://'):
            # Fetch data from URL
            response = requests.get(data_path, verify=ssl_verify)
            response.raise_for_status()  # Raise an exception for bad status codes
            df = pd.read_csv(io.StringIO(response.text))
        else:
            # Load data locally
            df = pd.read_csv(data_path)

        # Check if target column exists
        if target_column in df.columns:
            logging.info(f"Target column '{target_column}' found in data.")
            return True, target_column

        suggested_column = suggest_column_name(target_column, df.columns)
        if suggested_column:
            confirm = questionary.confirm(f"Did you mean '{suggested_column}'?").ask()
            if confirm:
                logging.info(f"User accepted suggested column: '{suggested_column}'.")
                return True, suggested_column
        
        logging.warning(f"Target column '{target_column}' not found in data. Did you mean '{suggested_column}'?")
        return False, None
    
    except FileNotFoundError:
        logging.error(f"File not found at {data_path}")
        return False, None
    except requests.RequestException as e:
        logging.error(f"Error fetching data from URL {data_path}: {e}")
        return False, None
    except pd.errors.ParserError:
        logging.error(f"Error parsing the data file at {data_path}. Please check the file format.")
        return False, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading file {data_path}: {e}")
        return False, None


def get_target_directory():
    """Determine the target directory based on user choice."""
    logging.info("Prompting user for project initialization directory.")
    directory_choice = questionary.select(
        "Where do you want to initialize the project?",
        choices=[
            questionary.Choice(title="Current directory", value="current"),
            questionary.Choice(title="Another directory", value="another"),
            questionary.Choice(title="Create a new directory", value="create"),
        ]
    ).ask(kbi_msg=KEYBOARD_INTERRUPT_MESSAGE)

    if directory_choice is None:
        logging.warning("User cancelled the operation.")
        sys.exit(1)

    return handle_directory_choice(directory_choice)


def handle_directory_choice(directory_choice):
    """Handle user's directory choice."""
    if directory_choice == "current":
        current_dir = os.getcwd()
        logging.info(f"User selected the current directory: {current_dir}")
        return current_dir

    elif directory_choice == "another":
        target_directory = click.prompt('Please enter the target directory path', type=str)
        validate_existing_directory(target_directory)
        os.chdir(target_directory)  # Change to the selected directory
        return target_directory

    else:  # Create a new directory
        new_directory_name = click.prompt('Please enter the new directory name', type=str)
        target_directory = os.path.join(os.getcwd(), new_directory_name)
        os.makedirs(target_directory, exist_ok=True)
        os.chdir(target_directory)  # Change to the new directory
        logging.info(f"Created and changed to new directory: {target_directory}")
        return target_directory


def validate_existing_directory(target_directory):
    """Validate that the specified directory exists."""
    if not os.path.exists(target_directory):
        logging.error(f"The specified directory does not exist: {target_directory}")
        click.secho("Error: The specified directory does not exist.", fg='red')
        sys.exit(1)

def log_artifact(file_path):
    """Log the generated artifact file path to `.artifacts.log`."""
    artifact_log_path = os.path.join(os.getcwd(), '.artifacts.log')
    try:
        with open(artifact_log_path, 'a') as log_file:
            log_file.write(file_path + '\n')
    except IOError as e:
        logging.warning(f"Could not write to artifact log file: {e}")



def suggest_column_name(user_input, columns):
    """
    Suggest the closest column name from the list of columns.
    Returns the best match or None if no close match is found.
    """
    matches = difflib.get_close_matches(user_input, columns, n=1, cutoff=0.6)
    return matches[0] if matches else None

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
    except IOError as e:
        click.secho(f"Error saving downloaded data to {local_file_path}: {e}", fg='red')
        logging.error(f"Error saving downloaded data to {local_file_path}: {e}")
        sys.exit(1)