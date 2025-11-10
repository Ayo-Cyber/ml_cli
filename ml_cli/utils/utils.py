import os
import numpy as np
import sys
import io
import json
import logging
import difflib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
import pandas as pd
import requests
import questionary
import click
from urllib.parse import urlparse
from fastapi import HTTPException  # FastAPI used indirectly via HTTPException
from pydantic import create_model
import joblib


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
VALID_EXTENSIONS: Tuple[str, ...] = (".csv", ".txt", ".json")  # lower-case
VALID_MIME_TYPES = {
    "text/csv",
    "text/plain",
    "application/json",
    # Some servers use generic types; allow these but rely on filename hint too
    "application/octet-stream",
}
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."
LOCAL_DATA_DIR = "data"
LOCAL_DATA_FILENAME = "data.csv"  # used by download_data()
DEFAULT_HTTP_TIMEOUT = 12  # seconds

# Globals used by load_model
pipeline = None
feature_info: Dict[str, Any] | None = None
PredictionPayload = None
sample_input_for_docs: Dict[str, Any] | None = None


# -----------------------------------------------------------------------------
# Internal helpers (no public API changes)
# -----------------------------------------------------------------------------


def _has_allowed_extension(path_like: str) -> bool:
    return Path(path_like).suffix.lower() in VALID_EXTENSIONS


def _disposition_has_allowed_ext(disposition: Optional[str]) -> bool:
    if not disposition:
        return False
    disp = disposition.lower()
    return any(ext in disp for ext in VALID_EXTENSIONS)


def _is_allowed_mimetype(ct: Optional[str]) -> bool:
    if not ct:
        return False
    # Ignore params (e.g., charset)
    ct = ct.split(";", 1)[0].strip().lower()
    return ct in VALID_MIME_TYPES


def _head_or_get(url: str, verify: bool, timeout: float = DEFAULT_HTTP_TIMEOUT) -> Optional[requests.Response]:
    """Try HEAD (follow redirects), fall back to GET. Return a response or None."""
    try:
        r = requests.head(url, verify=verify, timeout=timeout, allow_redirects=True)
        if 200 <= r.status_code < 300:
            return r
    except requests.RequestException:
        pass  # fall back to GET

    try:
        r = requests.get(url, verify=verify, timeout=timeout, stream=True, allow_redirects=True)
        if 200 <= r.status_code < 300:
            return r
    except requests.RequestException:
        return None

    return None


def _response_looks_like_allowed_file(url: str, resp: requests.Response) -> bool:
    """Accept if extension OR headers indicate an allowed data file."""
    ct_ok = _is_allowed_mimetype(resp.headers.get("Content-Type"))
    cd_ok = _disposition_has_allowed_ext(resp.headers.get("Content-Disposition"))
    ext_ok = _has_allowed_extension(url)
    return ct_ok or cd_ok or ext_ok


def _read_dataframe(data_path: str, ssl_verify: bool = True) -> pd.DataFrame:
    """Read CSV/TXT/JSON from local path or URL, with basic resilience.
    - For .csv/.txt: use pandas' engine='python' with sep=None to sniff.
    - For .json: use pd.read_json.
    Raises exceptions; callers should catch and convert to user messages.
    """
    suffix = Path(urlparse(data_path).path if data_path.startswith(("http://", "https://")) else data_path).suffix.lower()

    if data_path.startswith(("http://", "https://")):
        # Let pandas fetch directly when possible; otherwise fetch and pass a buffer
        if suffix in {".csv", ".txt"}:
            # Use requests to respect ssl_verify consistently
            r = requests.get(data_path, verify=ssl_verify, timeout=DEFAULT_HTTP_TIMEOUT)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), engine="python", sep=None, on_bad_lines="skip")
        elif suffix == ".json":
            r = requests.get(data_path, verify=ssl_verify, timeout=DEFAULT_HTTP_TIMEOUT)
            r.raise_for_status()
            return pd.read_json(io.StringIO(r.text))
        else:
            # Fallback: try CSV parser
            r = requests.get(data_path, verify=ssl_verify, timeout=DEFAULT_HTTP_TIMEOUT)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), engine="python", sep=None, on_bad_lines="skip")
    else:
        # Local path
        p = Path(data_path).expanduser().resolve()
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(p, engine="python", sep=None, on_bad_lines="skip")
        elif suffix == ".json":
            return pd.read_json(p)
        else:
            # Fallback: try CSV
            return pd.read_csv(p, engine="python", sep=None, on_bad_lines="skip")


# -----------------------------------------------------------------------------
# Public functions (names unchanged)
# -----------------------------------------------------------------------------


def write_config(config_data, format, config_filename):
    """Write configuration data to a file in the specified format (YAML or JSON).
    NOTE: no sys.exit here; raise on error so callers can handle.
    """
    try:
        logging.info(f"Attempting to write configuration to {config_filename} in {format} format.")
        with open(config_filename, "w", encoding="utf-8") as config_file:
            if format == "yaml":
                yaml.safe_dump(config_data, config_file, sort_keys=False)
            elif format == "json":
                json.dump(config_data, config_file, indent=4)
            else:
                raise ValueError("Unsupported config format. Use 'yaml' or 'json'.")
        logging.info(f"Configuration successfully written to {config_filename}.")
    except ValueError as ve:
        logging.error(f"Unsupported format error: {ve}")
        raise
    except IOError as ioe:
        logging.error(f"I/O error while writing to {config_filename}: {ioe}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while writing config file: {e}")
        raise


def save_configuration_safely(config_data, format, target_directory):
    """Save configuration with error handling."""
    try:
        config_filename = os.path.join(target_directory, f"config.{format}")
        logging.info(f"Writing configuration to {config_filename}")
        write_config(config_data, format, config_filename)
        return config_filename
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")
        click.secho(f"❌ Error saving configuration: {str(e)}", fg="red")
        return None


def should_prompt_target_column(task_type):
    """Check if the target column prompt is needed based on task type."""
    return task_type in ["classification", "regression"]


def is_readable_file(data_path, ssl_verify=True):
    """Check if the provided path is a readable file (local or URL) and has a supported format."""
    if data_path.startswith("http://") or data_path.startswith("https://"):
        return validate_and_check_url(data_path, ssl_verify)
    else:
        return check_local_file_readability(data_path)


def validate_and_check_url(url, ssl_verify=True):
    """Validate URL format & reachability. Accept if extension or headers indicate an allowed file.
    Returns True/False; does not raise.
    """
    try:
        parsed = urlparse(url)
        if not (parsed.scheme and parsed.netloc):
            logging.warning(f"Invalid URL format: {url}")
            return False
    except Exception as e:
        logging.error(f"URL validation failed: {e}")
        return False

    try:
        resp = _head_or_get(url, verify=ssl_verify, timeout=DEFAULT_HTTP_TIMEOUT)
        if resp is None:
            logging.warning(f"URL not reachable: {url}")
            return False
        if _response_looks_like_allowed_file(url, resp):
            logging.info(f"URL looks valid and reachable: {url}")
            return True
        logging.warning(
            "URL reachable but content-type/filename not recognized as allowed. "
            f"CT={resp.headers.get('Content-Type')}, CD={resp.headers.get('Content-Disposition')}"
        )
        return False
    except requests.exceptions.SSLError as e:
        logging.error(f"SSL error for URL {url}: {e}")
        return False
    except requests.RequestException as e:
        logging.error(f"RequestException for URL {url}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during URL validation: {e}")
        return False


def check_url_readability(data_path, ssl_verify=True):
    """Compatibility wrapper around validate_and_check_url (kept to avoid breaking imports)."""
    return validate_and_check_url(data_path, ssl_verify)


def check_local_file_readability(data_path):
    """Check if the local file exists, is a file, readable, and has an allowed extension."""
    try:
        p = Path(data_path).expanduser().resolve()
        if not (p.exists() and p.is_file()):
            logging.warning(f"Local path is not a readable file: {data_path}")
            return False
        if not _has_allowed_extension(str(p)):
            logging.warning(f"Unsupported local file extension: {p.suffix}")
            return False
        # Basic read access check
        try:
            with open(p, "rb"):
                pass
        except OSError as e:
            logging.warning(f"Cannot open file for reading: {e}")
            return False
        logging.info(f"Local file {p} is readable.")
        return True
    except Exception as e:
        logging.error(f"Error checking local file readability: {e}")
        return False


def is_target_in_file(data_path, target_column, ssl_verify=True):
    """Check if the target column exists in the data file.
    Returns (True, name) if found (including a suggested close match the user accepts), else (False, None).
    """
    logging.info("Checking for target column in data.")
    try:
        df = _read_dataframe(data_path, ssl_verify=ssl_verify)

        if target_column in df.columns:
            logging.info(f"Target column '{target_column}' found in data.")
            return True, target_column

        suggested_column = suggest_column_name(target_column, df.columns)
        if suggested_column:
            confirm = questionary.confirm(f"Did you mean '{suggested_column}'?").ask()
            if confirm:
                logging.info(f"User accepted suggested column: '{suggested_column}'.")
                return True, suggested_column

        logging.warning(f"Target column '{target_column}' not found in data. Suggested: '{suggested_column}'")
        return False, None

    except FileNotFoundError:
        logging.error(f"File not found at {data_path}")
        return False, None
    except requests.RequestException as e:
        logging.error(f"Error fetching data from URL {data_path}: {e}")
        return False, None
    except pd.errors.EmptyDataError:
        logging.error("The data file is empty.")
        return False, None
    except pd.errors.ParserError:
        logging.error("Error parsing the data file. Please check the file format.")
        return False, None
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading file {data_path}: {e}")
        return False, None


def get_target_directory():
    """Determine the target directory based on user choice. Returns a path or None on cancel."""
    logging.info("Prompting user for project initialization directory.")
    directory_choice = questionary.select(
        "Where do you want to initialize the project?",
        choices=[
            questionary.Choice(title="Current directory", value="current"),
            questionary.Choice(title="Another directory", value="another"),
            questionary.Choice(title="Create a new directory", value="create"),
        ],
    ).ask(kbi_msg=KEYBOARD_INTERRUPT_MESSAGE)

    if directory_choice is None:
        logging.warning("User cancelled the operation.")
        return None

    return handle_directory_choice(directory_choice)


def handle_directory_choice(directory_choice):
    """Handle user's directory choice. Returns the selected path (and chdir side-effect)."""
    if directory_choice == "current":
        current_dir = os.getcwd()
        logging.info(f"User selected the current directory: {current_dir}")
        return current_dir

    elif directory_choice == "another":
        target_directory = click.prompt("Please enter the target directory path", type=str)
        validate_existing_directory(target_directory)
        os.chdir(target_directory)  # Side-effect retained for backward compatibility
        return target_directory

    else:  # Create a new directory
        new_directory_name = click.prompt("Please enter the new directory name", type=str)
        target_directory = os.path.join(os.getcwd(), new_directory_name)
        os.makedirs(target_directory, exist_ok=True)
        os.chdir(target_directory)
        logging.info(f"Created and changed to new directory: {target_directory}")
        return target_directory


def validate_existing_directory(target_directory):
    """Validate that the specified directory exists. Raise ValueError on failure."""
    if not os.path.exists(target_directory):
        logging.error(f"The specified directory does not exist: {target_directory}")
        click.secho("Error: The specified directory does not exist.", fg="red")
        raise ValueError(f"Directory does not exist: {target_directory}")


def log_artifact(file_path):
    """Log the generated artifact file path to `.artifacts.log`."""
    artifact_log_path = os.path.join(os.getcwd(), ".artifacts.log")
    try:
        with open(artifact_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(file_path + "\n")
    except IOError as e:
        logging.warning(f"Could not write to artifact log file: {e}")


def suggest_column_name(user_input, columns):
    """
    Suggest the closest column name from the list of columns.
    Returns the best match or None if no close match is found.
    """
    matches = difflib.get_close_matches(user_input, list(columns), n=1, cutoff=0.6)
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
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        log_artifact(script_path)
        return script_path
    except Exception as e:
        logging.warning(f"Could not create convenience script: {e}")
        return None


def download_data(data_path, ssl_verify, target_directory):
    """Download data from a URL and save it locally. Return local path or None."""
    try:
        if not data_path.startswith(("http://", "https://")):
            return data_path

        click.secho(f"Downloading data from {data_path}...", fg="blue")
        try:
            response = requests.get(data_path, verify=ssl_verify, stream=True, timeout=DEFAULT_HTTP_TIMEOUT)
            response.raise_for_status()

            # Create local data directory
            local_data_path = os.path.join(target_directory, LOCAL_DATA_DIR)
            os.makedirs(local_data_path, exist_ok=True)

            local_file_path = os.path.join(local_data_path, LOCAL_DATA_FILENAME)

            with open(local_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)

            click.secho(f"Data downloaded and saved to {local_file_path}", fg="green")
            return local_file_path
        except requests.exceptions.RequestException as e:
            click.secho(f"Error downloading data: {e}", fg="red")
            logging.error(f"Error downloading data: {e}")
            return None
        except IOError as e:
            click.secho(f"Error saving downloaded data: {e}", fg="red")
            logging.error(f"Error saving downloaded data: {e}")
            return None
    except Exception as e:
        logging.error(f"Error downloading data: {e}")
        click.secho(f"❌ Error downloading data: {str(e)}", fg="red")
        return None


def generate_realistic_example_from_stats(feature_info: dict) -> dict[str, Any]:
    """Generate realistic examples based on feature statistics from the actual data"""
    example: Dict[str, Any] = {}

    # Check if we have feature statistics
    if "feature_statistics" in feature_info:
        stats = feature_info["feature_statistics"]
        for feature in feature_info.get("feature_names", []):
            if feature in stats and isinstance(stats[feature], dict):
                feature_stats = stats[feature]

                # Use mean if available, otherwise median, otherwise midpoint of min/max
                if "mean" in feature_stats:
                    value = feature_stats["mean"]
                elif "median" in feature_stats:
                    value = feature_stats["median"]
                elif "min" in feature_stats and "max" in feature_stats:
                    value = (feature_stats["min"] + feature_stats["max"]) / 2
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
        for feature in feature_info.get("feature_names", []):
            example[feature] = 1.0

    return example


def load_model(output_dir: str):
    """Load PyCaret model and return the objects instead of setting globals"""
    try:
        model_path = Path(output_dir) / "pycaret_model.pkl"
        feature_info_path = Path(output_dir) / "feature_info.json"

        if not model_path.exists() or not feature_info_path.exists():
            logging.warning("Model files not found. API will start but predictions will not work.")
            return None, None, None, None

        # Load feature info first to get task type
        with open(feature_info_path, "r", encoding="utf-8") as f:
            feature_info = json.load(f)

        task_type = feature_info.get("task_type", "classification")

        # Load PyCaret model using the core module
        from ml_cli.core.predict import load_pycaret_model

        pipeline = load_pycaret_model(str(output_dir), task_type)

        logging.info(f"Feature info keys: {feature_info.keys()}")
        logging.info(f"Feature names: {feature_info.get('feature_names', [])}")

        # Generate realistic example from actual feature statistics
        sample_input_for_docs = generate_realistic_example_from_stats(feature_info)

        # Create the dynamic Pydantic model
        fields: Dict[str, Tuple[type, Any]] = {}
        feature_names = feature_info.get("feature_names", [])
        feature_types = feature_info.get("feature_types", {})

        for feature in feature_names:
            feature_type = feature_types.get(feature)
            if feature_type:
                if isinstance(feature_type, str):
                    ft = feature_type.lower()
                    if "int" in ft:
                        fields[feature] = (int, ...)
                    elif "float" in ft or "number" in ft:
                        fields[feature] = (float, ...)
                    else:
                        fields[feature] = (str, ...)
                else:
                    try:
                        if pd.api.types.is_integer_dtype(feature_type):
                            fields[feature] = (int, ...)
                        elif pd.api.types.is_float_dtype(feature_type):
                            fields[feature] = (float, ...)
                        else:
                            fields[feature] = (str, ...)
                    except Exception:
                        fields[feature] = (float, ...)
            else:
                fields[feature] = (float, ...)

        logging.info(f"Creating Pydantic model with fields: {list(fields.keys())}")
        logging.info(f"Generated example: {sample_input_for_docs}")

        PredictionPayload = None
        if fields:
            PredictionPayload = create_model("PredictionPayload", **fields)
            logging.info(f"Model loaded successfully with {len(fields)} features.")
        else:
            logging.error("No fields created for Pydantic model")

        return pipeline, feature_info, PredictionPayload, sample_input_for_docs

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model files not found. Please train a model first.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")


def get_config_output_dir(config_path: str = "config.yaml") -> str:
    """Get output directory from config file"""
    output_dir = "output"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                output_dir = config.get("output_dir", "output")
        except yaml.YAMLError as exc:
            logging.error(f"Error loading config file: {exc}")
    return output_dir


def load_data(data_path):
    """Load the dataset from a specified path (local or URL)."""
    try:
        df = _read_dataframe(data_path, ssl_verify=True)
        if df.empty:
            click.secho("The dataset is empty. Nothing to preprocess.", fg="yellow")
            return None
        logging.info("Data loaded successfully for preprocessing.")
        return df
    except FileNotFoundError:
        click.secho(f"Error: Data file not found at '{data_path}'.", fg="red")
        logging.error(f"Data file not found at '{data_path}'.")
        return None
    except requests.RequestException as e:
        click.secho(f"Network error loading data: {e}", fg="red")
        logging.error(f"Network error loading data: {e}")
        return None
    except pd.errors.EmptyDataError:
        click.secho("The data file is empty.", fg="red")
        logging.error("The data file is empty.")
        return None
    except pd.errors.ParserError:
        click.secho("Error parsing the data file. Please check the file format.", fg="red")
        logging.error("Error parsing the data file.")
        return None
    except Exception as e:
        click.secho(f"An unexpected error occurred while loading data for preprocessing: {e}", fg="red")
        logging.error(f"An unexpected error occurred while loading data for preprocessing: {e}")
        return None


def encode_categorical_columns(df):
    """One-hot encode categorical columns in the DataFrame."""
    try:
        object_cols = df.select_dtypes(include=["object"]).columns
        if len(object_cols) > 0:
            df = pd.get_dummies(df, columns=list(object_cols), drop_first=True)
            logging.info(f"One-hot encoded columns: {list(object_cols)}")
        return df
    except AttributeError:
        click.secho("Error: The dataset is not a valid DataFrame.", fg="red")
        logging.error("The dataset is not a valid DataFrame.")
        return None
    except Exception as e:
        click.secho(f"An unexpected error occurred during one-hot encoding: {e}", fg="red")
        logging.error(f"An unexpected error occurred during one-hot encoding: {e}")
        return None


def load_config(config_file="config.yaml"):
    """Load configuration file to get the data path."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        data_path = config_data["data"]["data_path"]
        return data_path
    except FileNotFoundError:
        click.secho(f"Error: Configuration file '{config_file}' not found.", fg="red")
        logging.error(f"Configuration file not found: {config_file}")
        return None
    except yaml.YAMLError as e:
        click.secho(f"Error parsing YAML file: {e}", fg="red")
        logging.error(f"Error parsing YAML file: {e}")
        return None
    except KeyError:
        click.secho("Error: 'data_path' not found in the configuration file.", fg="red")
        logging.error("'data_path' not found in the configuration file.")
        return None
    except Exception as e:
        click.secho(f"An unexpected error occurred while reading the configuration file: {e}", fg="red")
        logging.error(f"An unexpected error occurred while reading the configuration file: {e}")
        return None


def save_preprocessed_data(df, file_path):
    """Save the preprocessed DataFrame to a specified file path."""
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        click.secho(f"Preprocessed data saved to {file_path}", fg="green")
        logging.info(f"Preprocessed data saved at: {file_path}")
        log_artifact(file_path)
    except IOError as e:
        click.secho(f"Error saving preprocessed data to {file_path}: {e}", fg="red")
        logging.error(f"Error saving preprocessed data to {file_path}: {e}")
    except Exception as e:
        click.secho(f"An unexpected error occurred while saving preprocessed data: {e}", fg="red")
        logging.error(f"An unexpected error occurred while saving preprocessed data: {e}")


def get_validated_data_path_input(ssl_verify):
    """Get and validate data path input with retry logic."""
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            data_path_input = click.prompt(
                f"📊 Please enter the data file path or URL (attempt {attempt}/{max_attempts})",
                type=str,
            ).strip()

            if not data_path_input:
                click.secho("⚠️  Data path cannot be empty.", fg="yellow")
                if attempt < max_attempts:
                    continue
                else:
                    return None

            if data_path_input.startswith(("http://", "https://")):
                if validate_and_check_url(data_path_input, ssl_verify):
                    click.secho("✅ URL is valid and reachable.", fg="green")
                    return data_path_input
                else:
                    click.secho("❌ URL is not reachable or unsupported type.", fg="red")
                    if attempt < max_attempts:
                        continue
                    else:
                        return None
            else:
                try:
                    path = Path(data_path_input).expanduser().resolve()
                    if path.exists() and path.is_file() and _has_allowed_extension(str(path)):
                        click.secho("✅ Local file found and has a supported extension.", fg="green")
                        return str(path)

                    if not path.exists():
                        click.secho(f"❌ File not found: {data_path_input}", fg="red")
                    elif not path.is_file():
                        click.secho(f"❌ Path exists but is not a file: {data_path_input}", fg="red")
                    else:
                        click.secho(
                            f"❌ Unsupported file extension (allowed: {', '.join(VALID_EXTENSIONS)})",
                            fg="red",
                        )

                    # Suggest similar files (best-effort)
                    suggest_similar_files(data_path_input)

                    if attempt < max_attempts:
                        continue
                    else:
                        return None
                except Exception as e:
                    click.secho(f"❌ Invalid file path: {str(e)}", fg="red")
                    if attempt < max_attempts:
                        continue
                    else:
                        return None

        except KeyboardInterrupt:
            click.secho("\n❌ Operation cancelled by user.", fg="yellow")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in get_validated_data_path_input: {e}")
            click.secho(f"❌ Unexpected error: {str(e)}", fg="red")
            if attempt < max_attempts:
                continue
            else:
                return None

    click.secho("❌ Failed to get valid data path after maximum attempts.", fg="red")
    return None


def suggest_similar_files(input_path):
    """Suggest similar files in the current directory (best-effort, silent on error)."""
    try:
        current_dir = Path.cwd()
        input_name = Path(input_path).name.lower()

        similar_files = []
        for file in current_dir.glob("*"):
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS:
                name_l = file.name.lower()
                if input_name in name_l or name_l in input_name:
                    similar_files.append(file.name)

        if similar_files:
            click.secho("💡 Similar files found:", fg="blue")
            for file in similar_files[:3]:
                click.secho(f"   • {file}", fg="blue")
    except Exception:
        pass


def get_validated_output_dir():
    """Get and validate output directory name (not path existence); return a string."""
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            output_dir = click.prompt(
                f"📁 Please enter the output directory path (attempt {attempt}/{max_attempts})",
                type=str,
                default="output",
            ).strip()

            if not output_dir:
                click.secho("⚠️  Output directory cannot be empty, using default 'output'.", fg="yellow")
                return "output"

            if is_valid_directory_name(os.path.basename(output_dir)):
                return output_dir
            else:
                click.secho(f"❌ Invalid directory name: '{output_dir}'", fg="red")
                click.secho('Directory names cannot contain: < > : " | ? * or control characters', fg="yellow")
                if attempt < max_attempts:
                    continue
                else:
                    return "output"

        except KeyboardInterrupt:
            click.secho("\n❌ Operation cancelled by user.", fg="yellow")
            return "output"
        except Exception as e:
            click.secho(f"❌ Error getting output directory: {str(e)}", fg="red")
            if attempt < max_attempts:
                continue
            else:
                return "output"

    return "output"


def is_valid_directory_name(name):
    """Check if directory name is valid (no reserved characters/control codes)."""
    if not name or name.isspace():
        return False

    invalid_chars = set('<>:"|?*')
    if any(char in invalid_chars for char in name):
        return False

    if any(ord(char) < 32 for char in name):
        return False

    return True


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


def safe_array_check(arr):
    """Safely check if array/list has elements"""
    try:
        return len(arr) > 0 if hasattr(arr, "__len__") else arr is not None
    except:
        return False


def format_prediction_response(prediction, feature_info, probabilities=None):
    """Format prediction response based on task type"""
    task_type = feature_info.get("task_type", "unknown").lower()

    # Safely get prediction value
    prediction_value = None
    if safe_array_check(prediction):
        try:
            prediction_value = convert_numpy_types(prediction[0])
        except (IndexError, TypeError):
            prediction_value = convert_numpy_types(prediction)

    result = {
        "prediction": prediction_value,
        "task_type": task_type,
    }

    # Add task-specific information
    if task_type == "classification":
        if probabilities is not None and safe_array_check(probabilities):
            prob_list = convert_numpy_types(probabilities)
            result["probabilities"] = prob_list
            result["confidence"] = float(max(prob_list)) if isinstance(prob_list, list) and prob_list else None

        # For classification, add class information if available
        target_column = feature_info.get("target_column")
        if target_column:
            result["predicted_class"] = prediction_value

    elif task_type == "regression":
        # For regression, the prediction is the actual value
        result["predicted_value"] = prediction_value

    elif task_type == "clustering":
        # For clustering, prediction is the cluster ID
        result["cluster_id"] = prediction_value
        result["cluster"] = f"Cluster_{prediction_value}" if prediction_value is not None else "Unknown"

    return result
