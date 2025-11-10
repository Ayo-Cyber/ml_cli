import os
import logging
import pandas as pd
import click


def load_pycaret_model(model_dir: str, task_type: str):
    """Load a PyCaret model from the specified directory."""
    try:
        model_path = os.path.join(model_dir, "pycaret_model")
        
        if not os.path.exists(f"{model_path}.pkl"):
            raise FileNotFoundError(f"Model file not found: {model_path}.pkl")
        
        # Import appropriate PyCaret module based on task type
        if task_type == "classification":
            from pycaret.classification import load_model
        elif task_type == "regression":
            from pycaret.regression import load_model
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        model = load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}.pkl")
        
        return model
        
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        raise


def make_predictions(model, data: pd.DataFrame, task_type: str):
    """Make predictions using a PyCaret model."""
    try:
        # Import appropriate PyCaret module
        if task_type == "classification":
            from pycaret.classification import predict_model
        elif task_type == "regression":
            from pycaret.regression import predict_model
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Make predictions
        predictions_df = predict_model(model, data=data)
        
        # Extract predictions based on task type
        if task_type == "classification":
            # PyCaret uses 'prediction_label' for classification
            if 'prediction_label' in predictions_df.columns:
                predictions = predictions_df['prediction_label'].values
            elif 'Label' in predictions_df.columns:  # Fallback
                predictions = predictions_df['Label'].values
            else:
                raise ValueError("Could not find prediction column in output")
        else:
            # PyCaret uses 'prediction_label' for regression too
            if 'prediction_label' in predictions_df.columns:
                predictions = predictions_df['prediction_label'].values
            elif 'Label' in predictions_df.columns:  # Fallback
                predictions = predictions_df['Label'].values
            else:
                raise ValueError("Could not find prediction column in output")
        
        logging.info(f"Generated {len(predictions)} predictions")
        
        return predictions, predictions_df
        
    except Exception as e:
        logging.error(f"Error making predictions: {e}", exc_info=True)
        raise
