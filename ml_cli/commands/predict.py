import click
import pandas as pd
import joblib
import json
import os

@click.command(help="""Generates predictions on new, unseen data using a previously trained and saved machine learning model.
This command loads the specified model and preprocessing pipeline, applies them to the input data,
and saves the resulting predictions to a specified output file.

Examples:
  ml-cli predict -i data/new_samples.csv -o predictions.csv -m output
  ml-cli predict --input-path test_data.csv --output-path results.csv --model-path trained_models/my_model
""")
@click.option('--input-path', '-i', type=click.Path(exists=True), required=True,
              help='The absolute or relative path to the CSV file containing the new data for which predictions are to be made.')
@click.option('--output-path', '-o', type=click.Path(), required=True,
              help='The absolute or relative path where the generated predictions (as a CSV file) will be saved.')
@click.option('--model-path', '-m', type=click.Path(exists=True), required=True,
              help='The absolute or relative path to the directory containing the trained model (e.g., "fitted_pipeline.pkl") and feature information ("feature_info.json").')
def predict(input_path, output_path, model_path):
    """Make predictions on new data using a trained model."""
    click.secho("Making predictions...", fg="green")

    try:
        # Load the new data
        new_data = pd.read_csv(input_path)

        # Load the pipeline and feature_info
        pipeline = joblib.load(os.path.join(model_path, "fitted_pipeline.pkl"))
        with open(os.path.join(model_path, "feature_info.json"), 'r') as f:
            feature_info = json.load(f)

        # Reorder columns of new_data to match the order of features in feature_info
        new_data = new_data[feature_info['feature_names']]

        # Make predictions
        predictions = pipeline.predict(new_data)

        # Save the predictions
        pd.DataFrame(predictions, columns=['predictions']).to_csv(output_path, index=False)

        click.secho(f"Predictions saved to {output_path}", fg="green")

    except FileNotFoundError as e:
        click.secho(f"Error: {e}", fg='red')
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red')
