import click
import pandas as pd
import importlib.util
import sys
from sklearn.pipeline import Pipeline

@click.command(help="""Make predictions on new data using a trained model.

Usage example:
  ml predict -i data/new_samples.csv -o predictions.csv -m models/best_model_pipeline.py
""")
@click.option('--input-path', '-i', type=click.Path(exists=True), required=True, help='Path to the input data for predictions (e.g., data/new_samples.csv).')
@click.option('--output-path', '-o', type=click.Path(), required=True, help='Path to save the predictions (e.g., predictions.csv).')
@click.option('--model-path', '-m', type=click.Path(exists=True), required=True, help='Path to the trained model pipeline file (e.g., models/best_model_pipeline.py).')
def predict(input_path, output_path, model_path):
    """Make predictions on new data using a trained model."""
    click.secho("Making predictions...", fg="green")

    try:
        # Load the new data
        new_data = pd.read_csv(input_path)

        # Dynamically import the pipeline from the model file
        spec = importlib.util.spec_from_file_location("model_pipeline", model_path)
        model_pipeline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_pipeline_module)

        # The exported pipeline from TPOT is expected to be named 'tpot_pipeline'
        if not hasattr(model_pipeline_module, 'tpot_pipeline'):
            click.secho("Error: 'tpot_pipeline' not found in the model file.", fg='red')
            sys.exit(1)

        pipeline = model_pipeline_module.tpot_pipeline

        # Make predictions
        predictions = pipeline.predict(new_data)

        # Save the predictions
        pd.DataFrame(predictions, columns=['predictions']).to_csv(output_path, index=False)

        click.secho(f"Predictions saved to {output_path}", fg="green")

    except FileNotFoundError as e:
        click.secho(f"Error: {e}", fg='red')
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg='red')
