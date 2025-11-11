import os
import logging
import warnings
import json
import joblib
import pandas as pd
import click
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from ml_cli.utils.utils import log_artifact

# Suppress warnings
warnings.filterwarnings("ignore")


def train_model(data: pd.DataFrame, config: dict, test_size: float = None):
    """Train the model using LightAutoML."""
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
    lama_config = config.get("lightautoml", {})

    # Get LightAutoML parameters
    timeout = lama_config.get("timeout", 300)
    cpu_limit = lama_config.get("cpu_limit", 4)
    gpu_ids = lama_config.get("gpu_ids", None)
    
    click.echo(f"\n🔧 LightAutoML Configuration:")
    click.echo(f"   Task: {task_type}")
    click.echo(f"   Timeout: {timeout}s")
    click.echo(f"   CPU Limit: {cpu_limit}")
    click.echo(f"   GPU IDs: {gpu_ids if gpu_ids else 'None (CPU only)'}")
    click.echo()

    # ===================================================================
    # CATEGORICAL ENCODING - Detect and encode categorical features
    # ===================================================================
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target column from categorical encoding if it's categorical
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    encoders = {}
    feature_encodings = {}
    
    if categorical_cols:
        click.echo(f"🔤 Encoding {len(categorical_cols)} categorical feature(s):")
        
        for col in categorical_cols:
            encoder = LabelEncoder()
            # Fit and transform the column
            data[col] = encoder.fit_transform(data[col].astype(str))
            encoders[col] = encoder
            
            # Create human-readable mapping
            feature_encodings[col] = {
                str(label): int(idx) for idx, label in enumerate(encoder.classes_)
            }
            
            click.echo(f"   ✓ {col}: {len(encoder.classes_)} unique values")
            logging.info(f"Encoded {col} with {len(encoder.classes_)} categories: {list(encoder.classes_)[:5]}...")
        
        click.echo()
    else:
        click.echo("ℹ️  No categorical features detected (all numeric)\n")
    
    # ===================================================================

    try:
        logging.info("Starting LightAutoML training...")
        click.echo("🚀 Setting up LightAutoML environment...\n")

        # Import LightAutoML
        from lightautoml.automl.presets.tabular_presets import TabularAutoML
        from lightautoml.tasks import Task
        
        # Prepare data - split into train and test
        train_data, test_data = train_test_split(
            data, 
            test_size=test_size, 
            random_state=42,
            stratify=data[target_column] if task_type == "classification" else None
        )
        
        click.echo(f"   Training samples: {len(train_data)}")
        click.echo(f"   Test samples: {len(test_data)}\n")
        
        # Create task
        if task_type == "classification":
            # Detect if binary or multiclass
            n_classes = data[target_column].nunique()
            if n_classes == 2:
                task = Task('binary')
                click.echo("   Detected: Binary Classification")
            else:
                task = Task('multiclass')
                click.echo(f"   Detected: Multiclass Classification ({n_classes} classes)")
        elif task_type == "regression":
            task = Task('reg')
            click.echo("   Detected: Regression")
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        click.echo()
        
        # Define roles (which column is target)
        roles = {'target': target_column}
        
        # Create AutoML with configuration
        automl = TabularAutoML(
            task=task,
            timeout=timeout,
            cpu_limit=cpu_limit,
            gpu_ids=gpu_ids,
        )
        
        click.echo(f"🤖 Training LightAutoML model (timeout: {timeout}s)...\n")
        logging.info("Training LightAutoML model...")
        
        # Train and get out-of-fold predictions
        oof_predictions = automl.fit_predict(
            train_data,
            roles=roles,
            verbose=1,
        )
        
        click.echo("\n✅ Training complete!")
        
        # Evaluate on test set
        test_predictions = automl.predict(test_data)
        
        # Calculate metrics
        y_test = test_data[target_column].values
        
        if task_type == "classification":
            # For classification, predictions might be probabilities
            if len(test_predictions.data.shape) > 1 and test_predictions.data.shape[1] > 1:
                y_pred = test_predictions.data.argmax(axis=1)
            else:
                y_pred = (test_predictions.data > 0.5).astype(int).ravel()
                
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            click.echo(f"\n📊 Test Set Performance:")
            click.echo(f"   Accuracy: {accuracy:.4f}")
            click.echo(f"   F1 Score: {f1:.4f}")
            
            model_score = accuracy
            
        else:  # regression
            y_pred = test_predictions.data.ravel()
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            click.echo(f"\n📊 Test Set Performance:")
            click.echo(f"   MSE: {mse:.4f}")
            click.echo(f"   R² Score: {r2:.4f}")
            
            model_score = r2

        # Save model
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "lightautoml_model.pkl")
        joblib.dump(automl, model_path)
        
        click.echo(f"\n💾 Model saved to {model_path}")
        logging.info(f"Model saved to {model_path}")
        log_artifact(model_path)

        # Save encoders if any categorical features were encoded
        if encoders:
            encoders_path = os.path.join(output_dir, "encoders.pkl")
            joblib.dump(encoders, encoders_path)
            click.echo(f"💾 Encoders saved to {encoders_path}")
            logging.info(f"Saved {len(encoders)} encoder(s) to {encoders_path}")
            log_artifact(encoders_path)
            
            # Save human-readable feature encodings (for documentation/API)
            encodings_json_path = os.path.join(output_dir, "feature_encodings.json")
            with open(encodings_json_path, "w") as f:
                json.dump(feature_encodings, f, indent=2)
            click.echo(f"📄 Feature encodings saved to {encodings_json_path}")
            logging.info(f"Feature encodings saved to {encodings_json_path}")
            log_artifact(encodings_json_path)

        # Save feature information
        feature_names = [col for col in data.columns if col != target_column]
        feature_types = {col: str(data[col].dtype) for col in feature_names}
        
        feature_info = {
            "model_name": "LightAutoML",
            "target_column": target_column,
            "task_type": task_type,
            "feature_names": feature_names,
            "feature_types": feature_types,
            "categorical_features": list(encoders.keys()) if encoders else [],
            "model_score": float(model_score),
            "lightautoml_config": lama_config,
            "n_samples_train": len(train_data),
            "n_samples_test": len(test_data),
        }
        
        feature_info_path = os.path.join(output_dir, "feature_info.json")
        with open(feature_info_path, "w") as f:
            json.dump(feature_info, f, indent=2)
        logging.info(f"Feature info saved to {feature_info_path}")
        log_artifact(feature_info_path)

        logging.info("LightAutoML training completed successfully!")
        click.echo("\n✅ Training completed successfully!\n")
        
        return automl
        
    except Exception as e:
        logging.error(f"Error during model training: {e}", exc_info=True)
        click.echo(f"\n❌ Error during training: {e}\n")
        raise
