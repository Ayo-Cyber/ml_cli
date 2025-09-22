import os
import json
from pathlib import Path
import yaml
import pandas as pd
import requests
import questionary
import click
import sys
import logging
import io
import difflib
from fastapi import FastAPI, HTTPException, Body
from pydantic import create_model


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
        sys.exit(1)
    except IOError as e:
        click.secho(f"Error saving downloaded data to {local_file_path}: {e}", fg='red')
        sys.exit(1)

def generate_realistic_example_from_stats(feature_info: dict) -> dict[str, any]:
    """Generate realistic examples based on feature statistics from the actual data"""
    example = {}
    
    # Check if we have feature statistics
    if 'feature_statistics' in feature_info:
        stats = feature_info['feature_statistics']
        for feature in feature_info.get('feature_names', []):
            if feature in stats and isinstance(stats[feature], dict):
                feature_stats = stats[feature]
                
                # Use mean if available, otherwise median, otherwise midpoint of min/max
                if 'mean' in feature_stats:
                    value = feature_stats['mean']
                elif 'median' in feature_stats:
                    value = feature_stats['median']
                elif 'min' in feature_stats and 'max' in feature_stats:
                    value = (feature_stats['min'] + feature_stats['max']) / 2
                else:
                    value = 1.0
                
                # Round to reasonable decimal places
                if isinstance(value, float):
                    example[feature] = round(value, 2)
                else:
                    example[feature] = value
            else:
                example[feature] = 1.0
    else:
        # Fallback if no statistics available
        for feature in feature_info.get('feature_names', []):
            example[feature] = 1.0
    
    return example

def load_model(output_dir: str):
    global pipeline, feature_info, PredictionPayload, sample_input_for_docs
    try:
        pipeline_path = Path(output_dir) / "fitted_pipeline.pkl"
        feature_info_path = Path(output_dir) / "feature_info.json"

        if not pipeline_path.exists() or not feature_info_path.exists():
            logging.warning("Model files not found. API will start but predictions will not work.")
            return

        pipeline = joblib.load(pipeline_path)
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)

        # Debug: Print feature_info structure
        logging.info(f"Feature info keys: {feature_info.keys()}")
        logging.info(f"Feature names: {feature_info.get('feature_names', [])}")

        # Generate realistic example from actual feature statistics
        sample_input_for_docs = generate_realistic_example_from_stats(feature_info)
        
        # Create the dynamic Pydantic model
        fields = {}
        feature_names = feature_info.get('feature_names', [])
        feature_types = feature_info.get('feature_types', {})
        
        for feature in feature_names:
            # Default to float if type is not specified or unclear
            feature_type = feature_types.get(feature)
            
            if feature_type:
                # Handle different ways feature types might be stored
                if isinstance(feature_type, str):
                    if 'int' in feature_type.lower() or 'integer' in feature_type.lower():
                        fields[feature] = (int, ...)
                    elif 'float' in feature_type.lower() or 'number' in feature_type.lower():
                        fields[feature] = (float, ...)
                    else:
                        fields[feature] = (str, ...)
                else:
                    # Handle pandas dtype objects
                    try:
                        if pd.api.types.is_integer_dtype(feature_type):
                            fields[feature] = (int, ...)
                        elif pd.api.types.is_float_dtype(feature_type):
                            fields[feature] = (float, ...)
                        else:
                            fields[feature] = (str, ...)
                    except:
                        # Fallback to float for numeric features
                        fields[feature] = (float, ...)
            else:
                # Default to float for all features if type info is missing
                fields[feature] = (float, ...)
        
        # Debug: Print fields being created
        logging.info(f"Creating Pydantic model with fields: {list(fields.keys())}")
        logging.info(f"Generated example: {sample_input_for_docs}")
        
        if fields:
            PredictionPayload = create_model("PredictionPayload", **fields)
            logging.info(f"Model loaded successfully with {len(fields)} features.")
        else:
            logging.error("No fields created for Pydantic model")

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model files not found. Please train a model first.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")


def load_data(data_path):
    """Load the dataset from a specified path."""
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            click.secho("The dataset is empty. Nothing to preprocess.", fg='yellow')
            return None
        logging.info("Data loaded successfully for preprocessing.")
        return df
    except FileNotFoundError:
        click.secho(f"Error: Data file not found at '{data_path}'.", fg='red')
        logging.error(f"Data file not found at '{data_path}'.")
        return None
    except pd.errors.EmptyDataError:
        click.secho("The data file is empty.", fg='red')
        logging.error("The data file is empty.")
        return None
    except pd.errors.ParserError:
        click.secho("Error parsing the data file. Please check the file format.", fg='red')
        logging.error("Error parsing the data file.")
        return None
    except Exception as e:
        click.secho(f"An unexpected error occurred while loading data for preprocessing: {e}", fg='red')
        logging.error(f"An unexpected error occurred while loading data for preprocessing: {e}")
        return None

def encode_categorical_columns(df):
    """One-hot encode categorical columns in the DataFrame."""
    try:
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            df = pd.get_dummies(df, columns=object_cols, drop_first=True)
            logging.info(f"One-hot encoded columns: {list(object_cols)}")
        return df
    except AttributeError:
        click.secho("Error: The dataset is not a valid DataFrame.", fg='red')
        logging.error("The dataset is not a valid DataFrame.")
        return None
    except Exception as e:
        click.secho(f"An unexpected error occurred during one-hot encoding: {e}", fg='red')
        logging.error(f"An unexpected error occurred during one-hot encoding: {e}")
        return None

def load_config(config_file='config.yaml'):
    """Load configuration file to get the data path."""
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        data_path = config_data['data']['data_path']
        return data_path
    except FileNotFoundError:
        click.secho(f"Error: Configuration file '{config_file}' not found.", fg='red')
        logging.error(f"Configuration file not found: {config_file}")
        return None
    except yaml.YAMLError as e:
        click.secho(f"Error parsing YAML file: {e}", fg='red')
        logging.error(f"Error parsing YAML file: {e}")
        return None
    except KeyError:
        click.secho("Error: 'data_path' not found in the configuration file.", fg='red')
        logging.error("'data_path' not found in the configuration file.")
        return None
    except Exception as e:
        click.secho(f"An unexpected error occurred while reading the configuration file: {e}", fg='red')
        logging.error(f"An unexpected error occurred while reading the configuration file: {e}")
        return None
    
def save_preprocessed_data(df, file_path):
    """Save the preprocessed DataFrame to a specified file path."""
    try:
        df.to_csv(file_path, index=False)
        click.secho(f"Preprocessed data saved to {file_path}", fg="green")
        logging.info(f"Preprocessed data saved at: {file_path}")
        log_artifact(file_path)
    except IOError as e:
        click.secho(f"Error saving preprocessed data to {file_path}: {e}", fg='red')
        logging.error(f"Error saving preprocessed data to {file_path}: {e}")
    except Exception as e:
        click.secho(f"An unexpected error occurred while saving preprocessed data: {e}", fg='red')
        logging.error(f"An unexpected error occurred while saving preprocessed data: {e}")