"""
Prediction module for LightAutoML models.
"""
import os
import joblib
import pandas as pd
import logging


def load_lightautoml_model(model_dir: str):
    """
    Load a LightAutoML model from disk.
    
    Args:
        model_dir: Directory containing the lightautoml_model.pkl file
        
    Returns:
        Loaded LightAutoML model
    """
    try:
        model_path = os.path.join(model_dir, "lightautoml_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = joblib.load(model_path)
        logging.info(f"Successfully loaded LightAutoML model from {model_path}")
        return model
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def make_predictions(model, data: pd.DataFrame, task_type: str = "classification"):
    """
    Make predictions using a LightAutoML model.
    
    Args:
        model: Loaded LightAutoML model (or sklearn model for testing)
        data: DataFrame with features to predict
        task_type: Type of task ("classification" or "regression")
        
    Returns:
        tuple: (predictions array, predictions DataFrame with all info)
    """
    try:
        # Make predictions using LightAutoML
        predictions = model.predict(data)
        
        # Extract prediction data
        if hasattr(predictions, 'data'):
            # LightAutoML format
            pred_data = predictions.data
        else:
            # Standard numpy array or other format (e.g., sklearn for testing)
            pred_data = predictions
            
        # Convert to numpy array if needed
        import numpy as np
        if not isinstance(pred_data, np.ndarray):
            pred_data = np.array(pred_data)
            
        # Handle different prediction formats
        if task_type == "classification":
            # For classification, might return probabilities
            if len(pred_data.shape) > 1 and pred_data.shape[1] > 1:
                # Multiclass - get class with highest probability
                predictions_array = pred_data.argmax(axis=1)
            else:
                # Binary or already class labels - ensure it's 1D
                if len(pred_data.shape) > 1:
                    pred_data = pred_data.ravel()
                # Check if it looks like probabilities (floats between 0 and 1)
                if pred_data.dtype in [np.float32, np.float64] and np.all((pred_data >= 0) & (pred_data <= 1)):
                    predictions_array = (pred_data > 0.5).astype(int)
                else:
                    # Already class labels
                    predictions_array = pred_data.astype(int)
        else:
            # Regression - just flatten
            predictions_array = pred_data.ravel()
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'predictions': predictions_array
        })
        
        logging.info(f"Made {len(predictions_array)} predictions successfully")
        return predictions_array, predictions_df
        
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise
