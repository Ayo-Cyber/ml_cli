import click
import pandas as pd
import joblib
import json
import os
import logging

@click.command(help="""Make predictions on new data using a trained model.

Usage example:
  ml predict -i data/new_samples.csv -o predictions.csv -m output
""")

@click.option('--input-path', '-i', type=click.Path(exists=True), required=True,
              help='The absolute or relative path to the CSV file containing the new data for which predictions are to be made.')
@click.option('--output-path', '-o', type=click.Path(), required=True,
              help='The absolute or relative path where the generated predictions (as a CSV file) will be saved.')
@click.option('--model-path', '-m', type=click.Path(exists=True), required=True,
              help='The absolute or relative path to the directory containing the trained model (e.g., "fitted_pipeline.pkl") and feature information ("feature_info.json").')
def predict(input_path: str, output_path: str, model_path: str):
    """Make predictions on new data using a trained model."""
    click.secho("Making predictions...", fg="green")

    try:
        # Load the new data
        new_data = pd.read_csv(input_path)
        if new_data.empty:
            click.secho("The input data is empty. Nothing to predict.", fg='yellow')
            logging.warning("The input data is empty. Nothing to predict.")
            return

        # Load the pipeline and feature_info
        pipeline = joblib.load(os.path.join(model_path, "fitted_pipeline.pkl"))
        with open(os.path.join(model_path, "feature_info.json"), 'r') as f:
            feature_info = json.load(f)

        # Reorder columns of new_data to match the order of features in feature_info
        new_data = new_data[feature_info['feature_names']]

        # Make predictions
        predictions = pipeline.predict(new_data)

        # Save the predictions
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(predictions, columns=['predictions']).to_csv(output_path, index=False)

        click.secho(f"Predictions saved to {output_path}", fg="green")
        logging.info(f"Predictions saved to {output_path}")

    except FileNotFoundError as e:
        click.secho(f"Error: Model file or feature info not found. {e}", fg='red')
        logging.error(f"Error: Model file or feature info not found. {e}")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        click.secho(f"Error reading the input data: {e}", fg='red')
        logging.error(f"Error reading the input data: {e}")
    except KeyError as e:
        click.secho(f"Error: One or more columns required by the model are not present in the input data. Missing columns: {e}", fg='red')
        logging.error(f"Error: One or more columns required by the model are not present in the input data. Missing columns: {e}")
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red')
        logging.error(f"An unexpected error occurred: {e}")
