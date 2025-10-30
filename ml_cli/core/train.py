import os
import logging
import warnings
import json
import joblib
import pandas as pd
import click
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ml_cli.utils.utils import log_artifact

# Suppress the torch warning from TPOT
warnings.filterwarnings("ignore", message="Warning: optional dependency `torch` is not available.*")

def preprocess_categorical_data(data: pd.DataFrame, target_column: str):
    """Preprocess categorical data for TPOT training."""
    data_copy = data.copy()
    
    # Identify categorical columns (object type)
    categorical_columns = data_copy.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target column from categorical processing if it's categorical
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
        
        # Handle categorical target variable
        if data_copy[target_column].dtype == 'object':
            le = LabelEncoder()
            data_copy[target_column] = le.fit_transform(data_copy[target_column])
            logging.info(f"Label encoded target column '{target_column}': {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # One-hot encode categorical features
    if categorical_columns:
        logging.info(f"One-hot encoding categorical columns: {categorical_columns}")
        data_copy = pd.get_dummies(data_copy, columns=categorical_columns, drop_first=True)
        logging.info(f"Data shape after encoding: {data_copy.shape}")
    
    # Ensure all columns are numeric
    for col in data_copy.columns:
        if data_copy[col].dtype == 'object':
            logging.warning(f"Column '{col}' is still non-numeric, attempting conversion...")
            try:
                data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
            except:
                # If conversion fails, drop the column
                logging.warning(f"Dropping non-convertible column: {col}")
                data_copy = data_copy.drop(columns=[col])
    
    # Handle any NaN values that might have been introduced
    if data_copy.isnull().any().any():
        logging.warning("Found NaN values after preprocessing, filling with 0")
        data_copy = data_copy.fillna(0)
    return data_copy

def train_model(data: pd.DataFrame, config: dict, test_size: float = None):
    """Train the model using TPOT."""
    try:
        target_column = config['data']['target_column']
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        if test_size is None:
            test_size = config.get('training', {}).get('test_size', 0.2)

        click.echo(f"📊 Using {test_size:.1%} of data for testing")

        # Preprocess categorical variables automatically
        data_processed = preprocess_categorical_data(data, target_column)

        X = data_processed.drop(columns=[target_column])
        y = data_processed[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y if config['task']['type'] == 'classification' else None
        )

        # Save feature information for serving
        feature_info = {
            'feature_names': X.columns.tolist(),
            'feature_types': X.dtypes.astype(str).to_dict(),
            'target_column': target_column,
            'task_type': config['task']['type']
        }

    except KeyError as e:
        logging.error(f"Missing key in config: {e}")
        raise
    except ValueError as e:
        logging.error(f"ValueError in data processing: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during data processing: {e}")
        raise

    task_type = config['task']['type']
    tpot_config = config.get('tpot', {})
    generations = tpot_config.get('generations', 4)

    if task_type == "classification":
        model = TPOTClassifier(generations=generations, random_state=42)
    elif task_type == "regression":
        model = TPOTRegressor(generations=generations, random_state=42)
    else:
        raise ValueError("Unsupported task type.")

    try:
        logging.info("Starting TPOT optimization...")
        model.fit(X_train, y_train)

        output_dir = config.get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Save fitted pipeline
        fitted_pipeline = model.fitted_pipeline_
        if fitted_pipeline is not None:
            pipeline_pkl_path = os.path.join(output_dir, 'fitted_pipeline.pkl')
            try:
                joblib.dump(fitted_pipeline, pipeline_pkl_path)
                logging.info(f"Fitted pipeline saved to {pipeline_pkl_path}")
                log_artifact(pipeline_pkl_path)
            except Exception as e:
                logging.warning(f"Could not save fitted pipeline: {e}")

        # Save feature metadata and model score
        try:
            score = fitted_pipeline.score(X_test, y_test) if fitted_pipeline else None
            if score is not None:
                feature_info['model_score'] = float(score)
                logging.info(f"Model performance score: {score}")

            feature_info_path = os.path.join(output_dir, 'feature_info.json')
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=2)

            logging.info(f"Feature info saved to {feature_info_path}")
            log_artifact(feature_info_path)
        except IOError as e:
            logging.error(f"Error saving feature info: {e}")
        except Exception as e:
            logging.error(f"Unexpected error while saving feature info: {e}")

        logging.info("TPOT optimization completed.")
        return model
    except Exception as e:
        logging.error(f"Error during model training or exporting: {e}")
        raise
