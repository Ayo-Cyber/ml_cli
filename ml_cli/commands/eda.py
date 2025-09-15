import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import logging
import click
import matplotlib.font_manager as fm
from ml_cli.utils.utils import log_artifact, load_config
from ml_cli.utils.exceptions import ConfigurationError, DataError




@click.command(help="""Perform exploratory data analysis (EDA) on the dataset specified in the configuration file.
""")
@click.option('--config', '-c', 'config_file', default="config.yaml",
              help="Path to the configuration file (YAML or JSON).")
def eda(config_file):
    """Perform exploratory data analysis on the dataset."""
    
    click.secho("Performing EDA ...", fg="green")
    # Load the configuration file to get the data path
    try:
        config = load_config()
        data_path = config.data.data_path
        click.echo(f"DEBUG: data_path from config = {data_path}")
    except Exception as e:
        click.secho(f"Error loading configuration: {e}", fg='red')
        return
    
    # Define artifact file paths
    summary_file = os.path.join(os.getcwd(), "summary_statistics.csv")
    eda_report_file = os.path.join(os.getcwd(), "eda_report.csv")
    correlation_matrix_file = os.path.join(os.getcwd(), "correlation_matrix.png")
    artifact_files = [summary_file, eda_report_file, correlation_matrix_file]

    try:
        # Load the dataset
        try:
            df = pd.read_csv(data_path)
            logging.info("Data loaded successfully.")
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            raise DataError(f"Error loading data: {e}")
        
        # Generate summary statistics
        summary_statistics = df.describe(include='all').to_dict()
        summary_df = pd.DataFrame(summary_statistics)
        summary_df.to_csv(summary_file, index=True)
        
        click.secho(f"Summary statistics saved to {summary_file}", fg="green")
        logging.info(f"Summary statistics generated and saved at: {summary_file}")
        log_artifact(summary_file)

        # Check for missing values and data types
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

        # Generate and save the correlation matrix as a heatmap
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

    except (DataError, Exception) as e:
        click.secho(f"An error occurred during EDA: {e}", fg='red')
        logging.error(f"An error occurred during EDA: {e}")
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
