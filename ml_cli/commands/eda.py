import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import logging
import click
import matplotlib.font_manager as fm
from ml_cli.utils.utils import log_artifact



@click.command(help="""Perform exploratory data analysis (EDA) on the dataset specified in the configuration file.
""")
def eda():
    """Perform exploratory data analysis on the dataset."""
    
    click.secho("Performing EDA ...", fg="green")
    # Load the configuration file to get the data path
    config_file = 'config.yaml'
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        data_path = config_data['data']['data_path']
    except Exception as e:
        click.secho(f"Error reading configuration file: {e}", fg='red')
        logging.error(f"Error reading configuration file: {e}")
        return
    
    # Define artifact file paths
    summary_file = os.path.join(os.getcwd(), "summary_statistics.csv")
    eda_report_file = os.path.join(os.getcwd(), "eda_report.csv")
    correlation_matrix_file = os.path.join(os.getcwd(), "correlation_matrix.png")
    artifact_files = [summary_file, eda_report_file, correlation_matrix_file]
    
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        click.secho(f"Error loading data: {e}", fg='red')
        logging.error(f"Error loading data: {e}")
        
        # Remove any created artifacts
        _cleanup_artifacts(artifact_files)
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        return
    
    # Generate summary statistics
    try:
        summary_statistics = df.describe(include='all').to_dict()
        summary_df = pd.DataFrame(summary_statistics)
        summary_df.to_csv(summary_file, index=True)
        
        click.secho(f"Summary statistics saved to {summary_file}", fg="green")
        logging.info(f"Summary statistics generated and saved at: {summary_file}")
        log_artifact(summary_file)
    except Exception as e:
        click.secho(f"Error generating summary statistics: {e}", fg='red')
        logging.error(f"Error generating summary statistics: {e}")
        
        _cleanup_artifacts(artifact_files)
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        return

    # Check for missing values and data types
    try:
        missing_values = df.isnull().sum().to_dict()
        data_types = df.dtypes.to_dict()

        eda_df = pd.DataFrame({
            "Feature": list(df.columns),
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Missing Values": [missing_values[col] for col in df.columns],
        })
        eda_df.to_csv(eda_report_file, index=False)
        
        click.secho(f"EDA report saved to {eda_report_file}", fg="green")
        logging.info(f"EDA report generated and saved at: {eda_report_file}")
        log_artifact(eda_report_file)
    except Exception as e:
        click.secho(f"Error generating EDA report: {e}", fg='red')
        logging.error(f"Error generating EDA report: {e}")
        
        _cleanup_artifacts(artifact_files)
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        return

    # Generate and save the correlation matrix as a heatmap
    try:
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        
        plt.title('Correlation Matrix')
        plt.savefig(correlation_matrix_file, bbox_inches='tight')
        plt.close()  # Close the plot to avoid display in interactive environments
        
        click.secho(f"Correlation matrix heatmap saved to {correlation_matrix_file}", fg="green")
        logging.info(f"Correlation matrix heatmap generated and saved at: {correlation_matrix_file}")
        log_artifact(correlation_matrix_file)
    except Exception as e:
        click.secho(f"Error generating correlation matrix: {e}", fg='red')
        logging.error(f"Error generating correlation matrix: {e}")
        
        _cleanup_artifacts(artifact_files)
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        return

def _cleanup_artifacts(artifact_files):
    """Helper function to delete any generated artifacts."""
    for file_path in artifact_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed artifact: {file_path}")
            click.secho(f"Removed artifact: {file_path}", fg="yellow")
