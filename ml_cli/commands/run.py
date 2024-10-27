import click
import yaml
import time
import pandas as pd
import pycaret.classification as py_clf
import pycaret.regression as py_reg
import pycaret.clustering as py_clust
import logging

@click.command()
@click.option('--config', default='config.yaml', help='Path to the configuration file.')
def run(config):
    """Run a machine learning task using PyCaret based on the provided configuration file."""
    start_time = time.time()
    
    # Load Configuration
    logging.info(f"Loading configuration from {config}")
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    except (IOError, yaml.YAMLError) as e:
        click.secho("Error reading configuration file.", fg='red')
        logging.error(f"Failed to load configuration: {e}")
        return

    # Extract configuration details
    data_path = config_data['data'].get('data_path')
    task_type = config_data['task'].get('type')
    target_column = config_data['data'].get('target_column')

    # Validate essential configuration fields
    if not data_path or not task_type or not target_column:
        click.secho("Missing essential configuration fields (data_path, task_type, target_column).", fg='red')
        return

    # Load Dataset
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        click.secho("Error loading data file.", fg='red')
        return

    # Execute the Task Based on Type
    if task_type == 'classification':
        perform_classification(df, target_column)
    elif task_type == 'regression':
        perform_regression(df, target_column)
    elif task_type == 'clustering':
        perform_clustering(df)
    else:
        click.secho("Unsupported task type specified in the configuration.", fg='red')
        return

    # Completion
    elapsed_time = time.time() - start_time
    click.secho(f"Task completed successfully in {elapsed_time:.2f} seconds.", fg='green')
    logging.info(f"Task completed in {elapsed_time:.2f}s")