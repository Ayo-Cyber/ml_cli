import os
import logging
import warnings
import json
import pandas as pd
import click
from ml_cli.utils.utils import log_artifact

# Suppress warnings
warnings.filterwarnings("ignore")


def train_model(data: pd.DataFrame, config: dict, test_size: float = None):
    """Train the model using PyCaret."""
    try:
        target_column = config["data"]["target_column"]
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        if test_size is None:
            test_size = config.get("training", {}).get("test_size", 0.2)

        click.echo(f"📊 Using {test_size:.1%} of data for testing")

    except KeyError as e:
        logging.error(f"Missing key in config: {e}")
        raise
    except ValueError as e:
        logging.error(f"ValueError in data processing: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during data processing: {e}")
        raise

    task_type = config["task"]["type"]
    pycaret_config = config.get("pycaret", {})

    # Get PyCaret parameters
    normalize = pycaret_config.get("normalize", True)
    feature_selection = pycaret_config.get("feature_selection", True)
    remove_outliers = pycaret_config.get("remove_outliers", False)
    n_select = pycaret_config.get("n_select", 3)
    session_id = pycaret_config.get("session_id", 42)
    fold = pycaret_config.get("fold", 3)
    verbose = pycaret_config.get("verbose", False)

    click.echo(f"\n🔧 PyCaret Configuration:")
    click.echo(f"   Task: {task_type}")
    click.echo(f"   Normalize: {normalize}")
    click.echo(f"   Feature Selection: {feature_selection}")
    click.echo(f"   Remove Outliers: {remove_outliers}")
    click.echo(f"   Models to compare: {n_select}")
    click.echo(f"   CV Folds: {fold}")
    click.echo()

    try:
        logging.info("Starting PyCaret setup...")
        click.echo("� Setting up PyCaret environment...\n")

        # Import appropriate PyCaret module based on task type
        if task_type == "classification":
            from pycaret.classification import setup, compare_models, save_model, finalize_model, pull
            
            # Setup experiment
            exp = setup(
                data=data,
                target=target_column,
                train_size=1 - test_size,
                normalize=normalize,
                feature_selection=feature_selection,
                remove_outliers=remove_outliers,
                session_id=session_id,
                fold=fold,
                verbose=verbose,
                html=False,
                silent=True,
                n_jobs=1,
            )
            
        elif task_type == "regression":
            from pycaret.regression import setup, compare_models, save_model, finalize_model, pull
            
            # Setup experiment
            exp = setup(
                data=data,
                target=target_column,
                train_size=1 - test_size,
                normalize=normalize,
                feature_selection=feature_selection,
                remove_outliers=remove_outliers,
                session_id=session_id,
                fold=fold,
                verbose=verbose,
                html=False,
                silent=True,
                n_jobs=1,
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        click.echo("✅ PyCaret environment setup complete!\n")
        
        # Compare models
        logging.info("Comparing models...")
        click.echo(f"🤖 Comparing top {n_select} models...\n")
        
        best_models = compare_models(
            n_select=n_select,
            sort='Accuracy' if task_type == 'classification' else 'R2',
            verbose=False,
        )
        
        # Get comparison results
        comparison_df = pull()
        click.echo("\n📊 Model Comparison Results:")
        click.echo(comparison_df.to_string(index=False))
        click.echo()
        
        # Select best model
        best_model = best_models[0] if isinstance(best_models, list) else best_models
        model_name = type(best_model).__name__
        click.echo(f"\n✅ Best model selected: {model_name}")
        
        # Finalize model (train on full dataset)
        logging.info(f"Finalizing model: {model_name}")
        click.echo(f"� Training final {model_name} model on full dataset...\n")
        final_model = finalize_model(best_model)

        # Save model
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "pycaret_model")
        save_model(final_model, model_path)
        
        click.echo(f"\n💾 Model saved to {model_path}.pkl")
        logging.info(f"Model saved to {model_path}.pkl")
        log_artifact(f"{model_path}.pkl")

        # Save feature information
        feature_info = {
            "model_name": model_name,
            "target_column": target_column,
            "task_type": task_type,
            "pycaret_config": pycaret_config,
        }
        
        feature_info_path = os.path.join(output_dir, "feature_info.json")
        with open(feature_info_path, "w") as f:
            json.dump(feature_info, f, indent=2)
        logging.info(f"Feature info saved to {feature_info_path}")
        log_artifact(feature_info_path)

        logging.info("PyCaret training completed successfully!")
        click.echo("\n✅ Training completed successfully!\n")
        
        return final_model
        
    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        click.echo(f"\n❌ Error during training: {e}\n")
        raise
        raise
