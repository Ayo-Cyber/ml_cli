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
import sys
import logging
import io


# Constants for file extensions
VALID_EXTENSIONS = ('.csv', '.txt', '.json')
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
        with open(config_filename, "w", encoding="utf-8") as config_file:
            if format == "yaml":
                yaml.safe_dump(config_data, config_file, sort_keys=False)
            elif format == "json":
                json.dump(config_data, config_file, indent=4)
            else:
                raise ValueError("Unsupported config format. Use 'yaml' or 'json'.")
    except Exception as e:
        logging.error(f"Failed to write config file: {e}")
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

        logging.warning(
            f"Target column '{target_column}' not found in data. Suggested: '{suggested_column}'"
        )
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
    artifact_log_path = os.path.join(os.getcwd(), '.artifacts.log')
    with open(artifact_log_path, 'a') as log_file:
        log_file.write(file_path + '\n')