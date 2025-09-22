import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import logging
import click
import matplotlib.font_manager as fm
from ml_cli.utils.utils import log_artifact, load_config


@click.command(help="""Perform exploratory data analysis (EDA) on the dataset specified in the configuration file.
""")
def eda():
    """Perform exploratory data analysis on the dataset."""
    
    click.secho("Performing EDA ...", fg="green")
    # Load the configuration file to get the data path
    data_path = load_config()
    if not data_path:
        return
    
    # Define artifact file paths
    summary_file = os.path.join(os.getcwd(), "summary_statistics.csv")
    eda_report_file = os.path.join(os.getcwd(), "eda_report.csv")
    correlation_matrix_file = os.path.join(os.getcwd(), "correlation_matrix.png")
    artifact_files = [summary_file, eda_report_file, correlation_matrix_file]
    
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            click.secho("The dataset is empty. Nothing to do.", fg='yellow')
            logging.warning("The dataset is empty.")
            return
        logging.info("Data loaded successfully.")
    except FileNotFoundError:
        click.secho(f"Error: Data file not found at '{data_path}'.", fg='red')
        logging.error(f"Data file not found at '{data_path}'.")
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        return
    except pd.errors.EmptyDataError:
        click.secho("The data file is empty.", fg='red')
        logging.error("The data file is empty.")
        return
    except pd.errors.ParserError:
        click.secho("Error parsing the data file. Please check the file format.", fg='red')
        logging.error("Error parsing the data file.")
        return
    except Exception as e:
        click.secho(f"An unexpected error occurred while loading the data: {e}", fg='red')
        logging.error(f"An unexpected error occurred while loading the data: {e}")
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
    except AttributeError:
        click.secho("Error: The dataset is not a valid DataFrame.", fg='red')
        logging.error("The dataset is not a valid DataFrame.")
        _cleanup_artifacts(artifact_files)
        return
    except Exception as e:
        click.secho(f"An unexpected error occurred while generating summary statistics: {e}", fg='red')
        logging.error(f"An unexpected error occurred while generating summary statistics: {e}")
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
    except KeyError as e:
        click.secho(f"Error: A column was not found during EDA report generation: {e}", fg='red')
        logging.error(f"KeyError during EDA report generation: {e}")
        _cleanup_artifacts(artifact_files)
        return
    except Exception as e:
        click.secho(f"An unexpected error occurred while generating the EDA report: {e}", fg='red')
        logging.error(f"An unexpected error occurred while generating the EDA report: {e}")
        _cleanup_artifacts(artifact_files)
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        return

    # Generate and save the correlation matrix as a heatmap
    try:
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            click.secho("No numeric columns found for correlation matrix.", fg='yellow')
            logging.warning("No numeric columns found for correlation matrix.")
            # Don't return here, as other artifacts might have been generated successfully
        else:
            correlation_matrix = numeric_df.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
            
            plt.title('Correlation Matrix')
            plt.savefig(correlation_matrix_file, bbox_inches='tight')
            plt.close()  # Close the plot to avoid display in interactive environments
            
            click.secho(f"Correlation matrix heatmap saved to {correlation_matrix_file}", fg="green")
            logging.info(f"Correlation matrix heatmap generated and saved at: {correlation_matrix_file}")
            log_artifact(correlation_matrix_file)

    except TypeError as e:
        click.secho(f"Error generating correlation matrix: {e}", fg='red')
        logging.error(f"TypeError during correlation matrix generation: {e}")
        _cleanup_artifacts(artifact_files)
    except Exception as e:
        click.secho(f"An unexpected error occurred while generating the correlation matrix: {e}", fg='red')
        logging.error(f"An unexpected error occurred while generating the correlation matrix: {e}")
        _cleanup_artifacts(artifact_files)
        click.secho("Please run the 'ml preprocess' command to prepare the data.", fg='yellow')
        return

def _cleanup_artifacts(artifact_files):
    """Helper function to delete any generated artifacts."""
    for file_path in artifact_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed artifact: {file_path}")
