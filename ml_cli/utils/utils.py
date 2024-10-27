import json
import yaml
import os
import pandas as pd
import requests
import questionary
import click
import sys

# Define your keyboard interrupt message
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."

def write_config(config_data, format):
    """Write configuration data to a file in the specified format (YAML or JSON)."""
    config_filename = f'config.{format}'
    if format == 'yaml':
        with open(config_filename, 'w') as config_file:
            yaml.dump(config_data, config_file)
    else:
        with open(config_filename, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)


def should_prompt_target_column(task_type):
    """Check if the target column prompt is needed based on task type."""
    return task_type in ['classification', 'regression']


def get_dependencies(task_type):
    """Return a list of dependencies based on the task type."""
    dependencies = ['pandas', 'numpy', 'scikit-learn', 'click', 'pyyaml']
    if task_type in ['classification', 'regression']:
        dependencies.append('matplotlib')  # Add libraries useful for classification/regression
    return dependencies


def is_readable_file(data_path):
    """Check if the provided path is a readable file (local or URL) and has a supported format."""
    
    # Define valid file extensions
    valid_extensions = ('.csv', '.txt', '.json')

    # Check if the data_path is a URL
    if data_path.startswith('http://') or data_path.startswith('https://'):
        # Check if the URL is reachable and has a valid extension
        try:
            response = requests.head(data_path)
            if response.status_code == 200:
                # Check if the URL ends with a valid extension
                return data_path.endswith(valid_extensions)
            return False
        except requests.RequestException:
            return False
    else:
        # For local files, check if it exists, is readable, and has a valid extension
        return (os.path.isfile(data_path) and 
                os.access(data_path, os.R_OK) and 
                data_path.endswith(valid_extensions))



def is_target_in_file(data_path, target_column):
    """Check if the target column exists in the data file."""
    try:
        # Load the file depending on its extension
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        elif data_path.endswith('.txt'):
            df = pd.read_csv(data_path, sep='\t')  # Assuming tab-separated for .txt
        else:
            return False  # Unsupported file format

        # Check if the target column exists
        return target_column in df.columns
    except Exception as e:
        return False  # Handle any errors that occur during file reading



def get_target_directory():
    """Determine the target directory based on user choice."""
    directory_choice = questionary.select(
        "Where do you want to initialize the project?",
        choices=[
            questionary.Choice(title="Current directory", value="current"),
            questionary.Choice(title="Another directory", value="another"),
            questionary.Choice(title="Create a new directory", value="create"),
        ]
    ).ask(kbi_msg=KEYBOARD_INTERRUPT_MESSAGE)

    if directory_choice is None:
        sys.exit(1)

    if directory_choice == "current":
        return os.getcwd()
    elif directory_choice == "another":
        target_directory = click.prompt('Please enter the target directory path', type=str)
        if not os.path.exists(target_directory):
            click.secho("Error: The specified directory does not exist.", fg='red')
            sys.exit(1)
        return target_directory
    else:  # Create a new directory
        new_directory_name = click.prompt('Please enter the new directory name', type=str)
        target_directory = os.path.join(os.getcwd(), new_directory_name)
        os.makedirs(target_directory, exist_ok=True)  # Create the new directory
        return target_directory