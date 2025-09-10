import os
import logging
import warnings
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from ml_cli.utils.utils import log_artifact

# Suppress the torch warning from TPOT
warnings.filterwarnings("ignore", message="Warning: optional dependency `torch` is not available.*")

def train_model(data, config):
    """Train the model using TPOT."""
    try:
        target_column = config['data']['target_column']
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        raise

    task_type = config['task']['type']
    tpot_config = config.get('tpot', {})
    generations = tpot_config.get('generations', 4)

    if task_type == "classification":
        model = TPOTClassifier(generations=generations, verbosity=2, random_state=42, n_jobs=-1)
    elif task_type == "regression":
        model = TPOTRegressor(generations=generations, verbosity=2, random_state=42, n_jobs=-1)
    else:
        logging.error("Unsupported task type.")
        raise ValueError("Unsupported task type.")

    try:
        logging.info("Starting TPOT optimization...")
        model.fit(X_train, y_train)

        output_dir = config.get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        model_file_path = os.path.join(output_dir, 'best_model_pipeline.py')
        model.export(model_file_path)
        logging.info(f"Model pipeline exported to {model_file_path}")

        log_artifact(model_file_path)

        score = model.score(X_test, y_test)
        print(f"Model performance score: {score}")

        print("TPOT optimization completed.")
        return model
    except Exception as e:
        logging.error(f"Error during model training or exporting: {e}")
        raise
