import os
import json
import yaml
import pandas as pd
import requests
import questionary
import click
import sys
import logging
import ssl
import h2o
from h2o.sklearn import H2OAutoMLClassifier, H2OAutoMLRegressor


# Constants for file extensions
VALID_EXTENSIONS = ('.csv', '.txt', '.json')
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."


def configure_logging():
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',  # Removed timestamp
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Change the default formatter to the color formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(ColorFormatter())  # Assuming ColorFormatter is defined


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


def get_dependencies(task_type):
    """Return a list of dependencies based on the task type."""
    dependencies = ['pandas', 'numpy', 'scikit-learn', 'click', 'pyyaml']
    if task_type in ['classification', 'regression']:
        dependencies.append('matplotlib')  # Add libraries useful for classification/regression
    return dependencies


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


# Ensure SSL verification is disabled globally if needed
ssl._create_default_https_context = ssl._create_unverified_context

def is_target_in_file(data_path, target_column, ssl_verify=True):
    """Check if the target column exists in the data file."""
    logging.info("Checking for target column in data.")
    
    try:
        if data_path.startswith('http://') or data_path.startswith('https://'):
            # Fetch data from URL
            df = pd.read_csv(data_path)  # Read directly as CSV
        else:
            # Load data locally
            df = pd.read_csv(data_path)

        # Check if target column exists
        if target_column in df.columns:
            logging.info(f"Target column '{target_column}' found in data.")
            return True
        else:
            logging.warning(f"Target column '{target_column}' not found in data.")
            return False
    
    except Exception as e:
        logging.error(f"Error reading file {data_path}: {e}")
        return False


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




# def init_h2o():
#     """Initialize the H2O cluster."""
#     h2o.init()


# def load_data(file_path):
#     """Load data into H2O from a specified file path."""
#     return h2o.import_file(file_path)


# def run_automl_classifier(train, target_column, max_models=20, seed=1):
#     """Run H2O AutoML for classification tasks."""
#     # Specify predictors and response
#     x = train.columns
#     x.remove(target_column)
    
#     # Convert target variable to factor for classification
#     train[target_column] = train[target_column].asfactor()

#     # Create and train the H2O AutoML Classifier
#     aml_classifier = H2OAutoMLClassifier(max_models=max_models, seed=seed)
#     aml_classifier.fit(X=x, y=target_column, training_frame=train)

#     return aml_classifier


# def run_automl_regressor(train, target_column, max_models=20, seed=1):
#     """Run H2O AutoML for regression tasks."""
#     # Specify predictors and response
#     x = train.columns
#     x.remove(target_column)

#     # Create and train the H2O AutoML Regressor
#     aml_regressor = H2OAutoMLRegressor(max_models=max_models, seed=seed)
#     aml_regressor.fit(X=x, y=target_column, training_frame=train)

#     return aml_regressor


# def print_leaderboard(automl_model):
#     """Print the leaderboard of models."""
#     lb = automl_model.leaderboard
#     print(lb)
