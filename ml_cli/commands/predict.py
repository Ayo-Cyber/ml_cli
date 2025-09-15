import click
import pandas as pd
import joblib
import json
import os
from ml_cli.utils.exceptions import ModelError, DataError

@click.command(help="""Make predictions on new data using a trained model.

Usage example:
  ml predict -i data/new_samples.csv -o predictions.csv
""" )
@click.option('--input-path', '-i', type=click.Path(exists=True), required=True, help='Path to the input data for predictions (e.g., data/new_samples.csv).')
@click.option('--output-path', '-o', type=click.Path(), required=True, help='Path to save the predictions (e.g., predictions.csv).')
@click.option('--model-path', '-m', type=click.Path(exists=True), help='Path to the output directory where the trained model is saved (optional).')
@click.option('--config', '-c', 'config_file', default="config.yaml",
              help="Path to the configuration file (YAML or JSON).")
def predict(input_path, output_path, model_path, config_file):
    """Make predictions on new data using a trained model."""
    click.secho("Making predictions...", fg="green")

    try:
        # Load config to get model path if not provided
        if not model_path:
            config = load_config(config_file)
            model_path = config.output_dir

        # Load the new data
        try:
            new_data = pd.read_csv(input_path)
        except FileNotFoundError:
            raise DataError(f"Input data file not found at {input_path}")

        # Load the pipeline and feature_info
        try:
            pipeline = joblib.load(os.path.join(model_path, "fitted_pipeline.pkl"))
            with open(os.path.join(model_path, "feature_info.json"), 'r') as f:
                feature_info = json.load(f)
        except FileNotFoundError:
            raise ModelError(f"Model files not found in {model_path}. Please train a model first.")

        # Reorder columns of new_data to match the order of features in feature_info
        new_data = new_data[feature_info['feature_names']]

        # Make predictions
        predictions = pipeline.predict(new_data)

        # Save the predictions
        pd.DataFrame(predictions, columns=['predictions']).to_csv(output_path, index=False)

        click.secho(f"Predictions saved to {output_path}", fg="green")

    except (DataError, ModelError) as e:
        click.secho(f"Error: {e}", fg='red')
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red')
