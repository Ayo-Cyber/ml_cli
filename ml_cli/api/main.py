import os
import logging
import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException, Body
from fastapi.openapi.utils import get_openapi
from ml_cli.utils.utils import load_model, get_config_output_dir, format_prediction_response, convert_numpy_types
from ml_cli.core.predict import make_predictions

# Create the FastAPI app
app = FastAPI(title="ML-CLI API", description="API for ML model predictions with dynamic examples", version="1.0.0")

# Global variables for this module
pipeline = None
feature_info = None
PredictionPayload = None
sample_input_for_docs = None
encoders = None  # NEW: Store categorical encoders


@app.on_event("startup")
def startup_event():
    """Load model on startup"""
    global pipeline, feature_info, PredictionPayload, sample_input_for_docs, encoders

    config_path = os.getenv("ML_CLI_CONFIG", "config.yaml")
    output_dir = get_config_output_dir(config_path)

    try:
        # Load model using utils function
        result = load_model(output_dir)
        
        # Check if model was loaded successfully
        if result is None or result[0] is None:
            logging.warning("⚠️  No trained model found!")
            logging.warning(f"   Please run 'ml train' first to train a model.")
            logging.warning(f"   The API will start but predictions will not work until a model is available.")
            pipeline = None
            feature_info = None
            PredictionPayload = None
            sample_input_for_docs = None
            encoders = None
            return
        
        loaded_pipeline, loaded_feature_info, loaded_payload_model, loaded_sample_input = result

        # Set globals in this module
        pipeline = loaded_pipeline
        feature_info = loaded_feature_info
        PredictionPayload = loaded_payload_model
        sample_input_for_docs = loaded_sample_input

        # Load encoders if they exist (NEW)
        encoders_path = Path(output_dir) / "encoders.pkl"
        if encoders_path.exists():
            encoders = joblib.load(encoders_path)
            logging.info(f"✅ Loaded encoders for {len(encoders)} categorical features: {list(encoders.keys())}")
            
            # Update sample input with categorical values
            if sample_input_for_docs and feature_info and 'categorical_features' in feature_info:
                categorical_features = feature_info['categorical_features']
                for feature_name, encoder in encoders.items():
                    if feature_name in sample_input_for_docs and len(encoder.classes_) > 0:
                        # Use first category as example
                        sample_input_for_docs[feature_name] = encoder.classes_[0]
                logging.info(f"📝 Updated sample input with categorical values")
        else:
            encoders = None
            logging.info("ℹ️  No encoders file found - model expects numeric input for all features")

        logging.info("✅ Model startup completed successfully")

    except Exception as e:
        logging.error(f"❌ Error during model startup: {e}")
        import traceback
        logging.error(traceback.format_exc())
        # Don't fail startup, but log the error
        pipeline = None
        feature_info = None
        PredictionPayload = None
        sample_input_for_docs = None
        encoders = None


def apply_categorical_encoding(payload: dict, encoders: dict) -> dict:
    """Apply categorical encoding to payload using saved encoders.
    
    Args:
        payload: Dictionary with feature names and values
        encoders: Dictionary of LabelEncoders for categorical features
        
    Returns:
        Encoded payload dictionary
        
    Raises:
        HTTPException: If unknown categorical value is encountered
    """
    if not encoders:
        return payload  # No encoding needed
    
    encoded_payload = payload.copy()
    
    for feature_name, encoder in encoders.items():
        if feature_name in encoded_payload:
            original_value = encoded_payload[feature_name]
            
            # Check if value is in encoder's known classes
            if original_value not in encoder.classes_:
                valid_values = list(encoder.classes_)
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown value '{original_value}' for feature '{feature_name}'. "
                           f"Valid values are: {valid_values}"
                )
            
            # Encode the value
            encoded_payload[feature_name] = int(encoder.transform([original_value])[0])
            logging.debug(f"Encoded {feature_name}: '{original_value}' -> {encoded_payload[feature_name]}")
    
    return encoded_payload


@app.get("/")
def root():
    """Welcome message and API status"""
    return {
        "message": "Welcome to the ML-CLI API!",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "status": ("operational" if pipeline is not None else "model_not_loaded"),
        "name": "ML-CLI API",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}


@app.get("/model-info")
def get_model_info():
    if pipeline is None or feature_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    task_type = feature_info.get("task_type", "unknown")

    info = {
        "model_type": task_type,
        "feature_count": len(feature_info.get("feature_names", [])),
        "feature_names": feature_info.get("feature_names", []),
        "model_score": feature_info.get("model_score"),
    }

    # Only add target_column for supervised tasks
    if task_type.lower() in ["classification", "regression"]:
        info["target_column"] = feature_info.get("target_column")

    return info


@app.get("/predict/example")
def get_prediction_example():
    if sample_input_for_docs is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return sample_input_for_docs


@app.post("/predict")
def predict(payload: dict = Body(...)):
    if pipeline is None or feature_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Apply categorical encoding if encoders are available (NEW)
        if encoders:
            try:
                payload = apply_categorical_encoding(payload, encoders)
                logging.info("Applied categorical encoding to input payload")
            except HTTPException:
                raise  # Re-raise validation errors
            except Exception as e:
                logging.error(f"Error applying categorical encoding: {e}")
                raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")
        
        # Validate payload has all required features
        feature_names = feature_info.get("feature_names", [])
        missing_features = [f for f in feature_names if f not in payload]
        # Fixed: avoid ambiguous array truth value
        if len(missing_features) > 0:
            raise HTTPException(status_code=400, detail=f"Missing required features: {missing_features}")

        # Create DataFrame from payload with explicit dtype handling
        try:
            input_df = pd.DataFrame([payload])

            # Ensure columns are in the right order and handle missing
            # columns gracefully
            for col in feature_names:
                if col not in input_df.columns:
                    raise HTTPException(status_code=400, detail=f"Missing feature: {col}")

            input_df = input_df[feature_names]

            # Convert to numeric where possible to avoid type issues
            for col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col], errors="ignore")
                except Exception:
                    pass  # Keep original type if conversion fails

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error creating input DataFrame: {str(e)}")

        # Make prediction using the proper core function
        try:
            task_type = feature_info.get("task_type", "classification").lower()
            predictions_array, _, probabilities = make_predictions(pipeline, input_df, task_type)
            # Convert to native Python types
            prediction = convert_numpy_types(predictions_array)
            if probabilities is not None:
                # Get the probabilities for the first (and only) sample
                probabilities = convert_numpy_types(probabilities[0])
        except Exception as e:
            logging.error(f"Pipeline prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

        # Format response based on task type
        result = format_prediction_response(prediction, feature_info, probabilities)

        # Add input features for reference (convert any numpy types)
        result["input_features"] = convert_numpy_types(payload)

        # Final conversion to ensure everything is JSON-serializable
        return convert_numpy_types(result)

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/predict/batch", summary="Get batch prediction example")
def get_batch_prediction_example():
    """Get an example of batch prediction format"""
    if sample_input_for_docs is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {"examples": [sample_input_for_docs, sample_input_for_docs]}  # You could modify this to show variation


@app.post("/predict/batch")
def predict_batch(payload: dict = Body(...)):
    """Make predictions on multiple samples"""
    if pipeline is None or feature_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Expect payload to have "samples" key with list of feature
        # dictionaries
        if "samples" not in payload:
            raise HTTPException(
                status_code=400, detail="Payload must contain 'samples' key with list " "of feature dictionaries"
            )

        samples = payload["samples"]
        if not isinstance(samples, list) or len(samples) == 0:
            raise HTTPException(status_code=400, detail="'samples' must be a non-empty list")

        # Apply categorical encoding to all samples (NEW)
        if encoders:
            try:
                encoded_samples = []
                for i, sample in enumerate(samples):
                    encoded_sample = apply_categorical_encoding(sample, encoders)
                    encoded_samples.append(encoded_sample)
                samples = encoded_samples
                logging.info(f"Applied categorical encoding to {len(samples)} samples")
            except HTTPException:
                raise  # Re-raise validation errors
            except Exception as e:
                logging.error(f"Error applying categorical encoding to batch: {e}")
                raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")

        feature_names = feature_info.get("feature_names", [])

        # Validate all samples
        for i, sample in enumerate(samples):
            missing_features = [f for f in feature_names if f not in sample]
            # Fixed: avoid ambiguous array truth value
            if len(missing_features) > 0:
                raise HTTPException(status_code=400, detail=f"Sample {i} missing required features: " f"{missing_features}")

        # Create DataFrame from all samples
        try:
            input_df = pd.DataFrame(samples)
            input_df = input_df[feature_names]

            # Convert to numeric where possible
            for col in input_df.columns:
                try:
                    input_df[col] = pd.to_numeric(input_df[col], errors="ignore")
                except Exception:
                    pass

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error creating batch DataFrame: {str(e)}")

        # Make predictions using the proper core function
        try:
            task_type = feature_info.get("task_type", "classification").lower()
            predictions_array, _, probabilities = make_predictions(pipeline, input_df, task_type)
            # Convert to native Python types
            predictions = convert_numpy_types(predictions_array)
            if probabilities is not None:
                probabilities = convert_numpy_types(probabilities)
        except Exception as e:
            logging.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

        # Format results
        results = []
        for i, (sample, pred) in enumerate(zip(samples, predictions)):
            prob = probabilities[i] if probabilities is not None else None
            result = format_prediction_response([pred], feature_info, prob)
            result["input_features"] = convert_numpy_types(sample)
            result["sample_index"] = i
            results.append(convert_numpy_types(result))

        return convert_numpy_types({
            "predictions": results, 
            "total_samples": len(samples), 
            "task_type": task_type
        })

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="ML-CLI API",
        version="1.0.0",
        description="API for ML model predictions with dynamic examples",
        routes=app.routes,
    )

    # Add example to the predict endpoint if we have sample input
    if sample_input_for_docs:
        predict_path = openapi_schema["paths"].get("/predict")
        if predict_path and "post" in predict_path:
            predict_path["post"]["requestBody"] = {
                "content": {"application/json": {"schema": {"type": "object"}, "example": sample_input_for_docs}}
            }

        # Add example for batch prediction
        batch_predict_path = openapi_schema["paths"].get("/predict/batch")
        if batch_predict_path and "post" in batch_predict_path:
            batch_predict_path["post"]["requestBody"] = {
                "content": {
                    "application/json": {
                        "schema": {"type": "object"},
                        "example": {"samples": [sample_input_for_docs, sample_input_for_docs]},
                    }
                }
            }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
