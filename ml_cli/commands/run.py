import os
import logging
import sys
import click
import yaml
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

@click.command()
def run():
    """Run the ML pipeline based on the configuration."""
    
    # Load the configuration file
    config_path = os.path.join(os.getcwd(), 'config.yaml')  # Adjust as necessary
    if not os.path.exists(config_path):
        click.secho("Error: Configuration file 'config.yaml' not found in the current directory.", fg='red')
        logging.error("Configuration file not found.")
        sys.exit(1)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract parameters from the config
    data_path = config['data']['data_path']
    target_column = config['data']['target_column']
    task_type = config['task']['type']
    
    # Load the dataset
    try:
        data = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        click.secho(f"Error loading data: {e}", fg='red')
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

    # Split the data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == "classification":
        model = TPOTClassifier(generations=4, verbosity=2, random_state=42, n_jobs=-1)
    elif task_type == "regression":
        model = TPOTRegressor(generations=4, verbosity=2, random_state=42, n_jobs=-1)
    else:
        click.secho("Error: Unsupported task type.", fg='red')
        logging.error("Unsupported task type.")
        sys.exit(1)


    logging.info("Starting TPOT optimization...")
    model.fit(X_train, y_train)

    # Save the optimized pipeline
    output_dir = os.path.dirname(config_path)  # Get the directory from config
    model_file_path = os.path.join(output_dir, 'best_model_pipeline.py')
    model.export(model_file_path)
    logging.info(f"Model pipeline exported to {model_file_path}")

    # Log performance metrics
    score = model.score(X_test, y_test)
    click.secho(f"Model performance score: {score}", fg='green')

    click.secho("TPOT optimization completed.", fg='green')


