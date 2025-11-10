import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from fastapi.openapi.utils import get_openapi
from ml_cli.utils.utils import load_model, get_config_output_dir, format_prediction_response

# Create the FastAPI app
app = FastAPI(title="ML-CLI API", description="API for ML model predictions with dynamic examples", version="1.0.0")

# Global variables for this module
pipeline = None
feature_info = None
PredictionPayload = None
sample_input_for_docs = None


@app.on_event("startup")
def startup_event():
    """Load model on startup"""
    global pipeline, feature_info, PredictionPayload, sample_input_for_docs

    config_path = os.getenv("ML_CLI_CONFIG", "config.yaml")
    output_dir = get_config_output_dir(config_path)

    try:
        # Load model using utils function
        result = load_model(output_dir)
        loaded_pipeline, loaded_feature_info, loaded_payload_model, loaded_sample_input = result

        # Set globals in this module
        pipeline = loaded_pipeline
        feature_info = loaded_feature_info
        PredictionPayload = loaded_payload_model
        sample_input_for_docs = loaded_sample_input

        logging.info("Model startup completed successfully")

    except Exception as e:
        logging.error(f"Error during model startup: {e}")
        # Don't fail startup, but log the error


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

        # Make prediction with error handling
        try:
            from ml_cli.core.predict import make_predictions

            task_type = feature_info.get("task_type", "classification")
            predictions, predictions_df = make_predictions(pipeline, input_df, task_type)
            prediction = predictions  # predictions is already a numpy array
        except Exception as e:
            logging.error(f"Pipeline prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

        # Get prediction probabilities from predictions_df if available (for classification)
        probabilities = None
        task_type_lower = feature_info.get("task_type", "").lower()

        if task_type_lower == "classification":
            try:
                # PyCaret's predict_model includes probability columns
                # They're named like prediction_label_0, prediction_label_1, etc.
                prob_cols = [col for col in predictions_df.columns if col.startswith("prediction_label_")]
                if prob_cols:
                    probabilities = predictions_df[prob_cols].values[0]  # Get first row
            except Exception as e:
                logging.warning(f"Could not get probabilities: {e}")

        # Format response based on task type
        result = format_prediction_response(prediction, feature_info, probabilities)

        # Add input features for reference
        result["input_features"] = payload

        return result

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

        # Make predictions
        try:
            from ml_cli.core.predict import make_predictions

            task_type = feature_info.get("task_type", "classification")
            predictions, predictions_df = make_predictions(pipeline, input_df, task_type)
        except Exception as e:
            logging.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

        # Get probabilities from predictions_df if available
        probabilities = None
        task_type_lower = feature_info.get("task_type", "").lower()

        if task_type_lower == "classification":
            try:
                prob_cols = [col for col in predictions_df.columns if col.startswith("prediction_label_")]
                if prob_cols:
                    probabilities = predictions_df[prob_cols].values  # All rows
            except Exception as e:
                logging.warning(f"Could not get prediction probabilities: {e}")

        # Format results
        results = []
        for i, (sample, pred) in enumerate(zip(samples, predictions)):
            prob = probabilities[i] if probabilities is not None else None
            result = format_prediction_response([pred], feature_info, prob)
            result["input_features"] = sample
            result["sample_index"] = i
            results.append(result)

        return {"predictions": results, "total_samples": len(samples), "task_type": task_type}

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
