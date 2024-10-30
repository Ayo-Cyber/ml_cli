import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import logging
import click
import matplotlib.font_manager as fm
from sklearn.preprocessing import OneHotEncoder

plt.rcParams['font.family'] = 'Arial'

@click.command()
def preprocess():
    """Preprocess the dataset to handle non-numeric columns using OneHotEncoder."""
    
    # Load the configuration file to get the data path
    config_file = 'config.yaml'
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    data_path = config_data['data']['data_path']
    
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        logging.info("Data loaded successfully for preprocessing.")
    except Exception as e:
        click.secho(f"Error loading data for preprocessing: {e}", fg='red')
        logging.error(f"Error loading data for preprocessing: {e}")
        return
    
    # One-hot encode object columns
    object_cols = df.select_dtypes(include=['object']).columns
    if object_cols.any():
        df = pd.get_dummies(df, columns=object_cols, drop_first=True)
        logging.info(f"One-hot encoded columns: {object_cols}")
    
    # Save preprocessed data to a new file
    preprocessed_file = os.path.join(os.getcwd(), "preprocessed_data.csv")
    df.to_csv(preprocessed_file, index=False)
    click.secho(f"Preprocessed data saved to {preprocessed_file}", fg="green")
    logging.info(f"Preprocessed data saved at: {preprocessed_file}")