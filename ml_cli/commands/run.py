import os
import logging
import sys
import click
import yaml
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from ml_cli.utils.utils import log_artifact

@click.command(help="Run the ML pipeline based on the configuration.")
def run():
    """Run the ML pipeline based on the configuration."""
    click.secho("Running ML pipeline...", fg="green")
    
    # Load the configuration file
    config_path = os.path.join(os.getcwd(), 'config.yaml')  # Adjust as necessary
    if not os.path.exists(config_path):
        click.secho("Error: Configuration file 'config.yaml' not found in the current directory.", fg='red')
        logging.error("Configuration file not found.")
        sys.exit(1)
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Log the config file artifact
    log_artifact(config_path)
    
    # Check for a preprocessed CSV file in the current directory
    data_dir = os.getcwd()  # Assuming data is in the current working directory
    preprocessed_csv_path = os.path.join(data_dir, 'preprocessed_data.csv')

    logging.info("Checking for preprocessed CSV file...")
    
    if os.path.exists(preprocessed_csv_path):
        click.secho(f"Preprocessed CSV found: {preprocessed_csv_path}. Using this file.", fg='green')
        data_path = preprocessed_csv_path
        target_column = config['data']['target_column']  # Get the target column from config
        
        # Log the preprocessed data file as an artifact
        log_artifact(preprocessed_csv_path)
    else:
        click.secho("No preprocessed CSV found. Using unprocessed data.", fg='yellow')
        data_path = config['data']['data_path']
        target_column = config['data']['target_column']

    # Load the dataset
    try:
        data = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        click.secho(f"Error loading data: {e}", fg='red')
        logging.error(f"Error loading data: {e}")
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        sys.exit(1)

    # Split the data
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        click.secho(f"Error processing data: {e}", fg='red')
        logging.error(f"Error processing data: {e}")
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        sys.exit(1)

    # Initialize the model based on the task type
    task_type = config['task']['type']
    if task_type == "classification":
        model = TPOTClassifier(generations=4, verbosity=2, random_state=42, n_jobs=-1)
    elif task_type == "regression":
        model = TPOTRegressor(generations=4, verbosity=2, random_state=42, n_jobs=-1)
    else:
        click.secho("Error: Unsupported task type.", fg='red')
        logging.error("Unsupported task type.")
        sys.exit(1)

    # Train the model
    try:
        logging.info("Starting TPOT optimization...")
        model.fit(X_train, y_train)

        # Save the optimized pipeline
        output_dir = os.path.dirname(config_path) if os.path.exists(config_path) else data_dir
        model_file_path = os.path.join(output_dir, 'best_model_pipeline.py')
        model.export(model_file_path)
        logging.info(f"Model pipeline exported to {model_file_path}")

        # Log the exported model file as an artifact
        log_artifact(model_file_path)

        # Log performance metrics
        score = model.score(X_test, y_test)
        click.secho(f"Model performance score: {score}", fg='green')

        click.secho("TPOT optimization completed.", fg='green')
    except Exception as e:
        click.secho(f"Error during model training or exporting: {e}", fg='red')
        logging.error(f"Error during model training or exporting: {e}")
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        sys.exit(1)
