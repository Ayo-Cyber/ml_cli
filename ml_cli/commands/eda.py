import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import logging
import click
import matplotlib.font_manager as fm

plt.rcParams['font.family'] = 'Arial'

@click.command()
def eda():
    """Perform exploratory data analysis on the dataset."""
    
    # Load the configuration file to get the data path
    config_file = 'config.yaml'  # or get this from your context
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    data_path = config_data['data']['data_path']
    
    # Load the dataset
    try:
        df = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
    except Exception as e:
        click.secho(f"Error loading data: {e}", fg='red')
        logging.error(f"Error loading data: {e}")
        return
    
    # Generate summary statistics
    summary_statistics = df.describe(include='all').to_dict()
    
    # Save summary statistics to CSV
    summary_file = os.path.join(os.getcwd(), "summary_statistics.csv")
    summary_df = pd.DataFrame(summary_statistics)
    summary_df.to_csv(summary_file, index=True)
    
    click.secho(f"Summary statistics saved to {summary_file}", fg="green")
    logging.info(f"Summary statistics generated and saved at: {summary_file}")

    # Check for missing values and data types
    missing_values = df.isnull().sum().to_dict()
    data_types = df.dtypes.to_dict()

    # Save missing values and data types to a CSV file
    eda_report_file = os.path.join(os.getcwd(), "eda_report.csv")
    eda_df = pd.DataFrame({
        "Feature": list(df.columns),
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": [missing_values[col] for col in df.columns],
    })
    eda_df.to_csv(eda_report_file, index=False)
    
    click.secho(f"EDA report saved to {eda_report_file}", fg="green")
    logging.info(f"EDA report generated and saved at: {eda_report_file}")

    # Generate and save the correlation matrix as a heatmap
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    
    # Save the heatmap
    correlation_matrix_file = os.path.join(os.getcwd(), "correlation_matrix.png")
    plt.title('Correlation Matrix')
    plt.savefig(correlation_matrix_file, bbox_inches='tight')
    plt.close()  # Close the plot to avoid display in interactive environments
    
    click.secho(f"Correlation matrix heatmap saved to {correlation_matrix_file}", fg="green")
    logging.info(f"Correlation matrix heatmap generated and saved at: {correlation_matrix_file}")
