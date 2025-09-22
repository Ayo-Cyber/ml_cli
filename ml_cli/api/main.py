import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import create_model
from typing import Any, Dict
import os
import json
import logging
from pathlib import Path

# Create the FastAPI app
app = FastAPI(
    title="ML-CLI API",
    description="API for ML model predictions",
    version="1.0.0"
)

pipeline = None
feature_info = None
PredictionPayload = None
sample_input_for_docs = None

def load_model(output_dir: str):
    global pipeline, feature_info, PredictionPayload, sample_input_for_docs
    try:
        pipeline_path = Path(output_dir) / "fitted_pipeline.pkl"
        feature_info_path = Path(output_dir) / "feature_info.json"

        if not pipeline_path.exists() or not feature_info_path.exists():
            logging.warning("Model files not found. API will start but predictions will not work.")
            return

        pipeline = joblib.load(pipeline_path)
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)

        # Debug: Print feature_info structure
        logging.info(f"Feature info keys: {feature_info.keys()}")
        logging.info(f"Feature names: {feature_info.get('feature_names', [])}")

        # Create the dynamic Pydantic model
        fields = {}
        sample_input_for_docs = {}
        
        # Use feature_names list to ensure proper order
        feature_names = feature_info.get('feature_names', [])
        feature_types = feature_info.get('feature_types', {})
        
        for feature in feature_names:
            # Default to float if type is not specified or unclear
            feature_type = feature_types.get(feature)
            
            if feature_type:
                # Handle different ways feature types might be stored
                if isinstance(feature_type, str):
                    if 'int' in feature_type.lower() or 'integer' in feature_type.lower():
                        fields[feature] = (int, ...)
                        sample_input_for_docs[feature] = 1
                    elif 'float' in feature_type.lower() or 'number' in feature_type.lower():
                        fields[feature] = (float, ...)
                        sample_input_for_docs[feature] = 1.0
                    else:
                        fields[feature] = (str, ...)
                        sample_input_for_docs[feature] = "example"
                else:
                    # Handle pandas dtype objects
                    try:
                        if pd.api.types.is_integer_dtype(feature_type):
                            fields[feature] = (int, ...)
                            sample_input_for_docs[feature] = 1
                        elif pd.api.types.is_float_dtype(feature_type):
                            fields[feature] = (float, ...)
                            sample_input_for_docs[feature] = 1.0
                        else:
                            fields[feature] = (str, ...)
                            sample_input_for_docs[feature] = "example"
                    except:
                        # Fallback to float for numeric features
                        fields[feature] = (float, ...)
                        sample_input_for_docs[feature] = 1.0
            else:
                # Default to float for all features if type info is missing
                fields[feature] = (float, ...)
                sample_input_for_docs[feature] = 1.0
        
        # Debug: Print fields being created
        logging.info(f"Creating Pydantic model with fields: {list(fields.keys())}")
        
        if fields:
            PredictionPayload = create_model("PredictionPayload", **fields)
            logging.info(f"Model loaded successfully with {len(fields)} features.")
        else:
            logging.error("No fields created for Pydantic model")

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model files not found. Please train a model first.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

@app.on_event("startup")
def startup_event():
    config_path = os.getenv("ML_CLI_CONFIG", "config.yaml")
    output_dir = "output"
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                output_dir = config.get('output_dir', 'output')
            except yaml.YAMLError as exc:
                raise HTTPException(status_code=500, detail=f"Error loading config file: {exc}")
    load_model(output_dir)

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the ML-CLI API!",
        "endpoints": {
            "predict": "/predict",
            "example": "/example",
            "model_info": "/model-info",
            "health": "/health"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "payload_model_created": PredictionPayload is not None
    }

@app.get("/example")
def get_example():
    """Get example input data for testing"""
    if not sample_input_for_docs:
        raise HTTPException(status_code=503, detail="Model not loaded or example data not available.")
    
    return {
        "example_input": sample_input_for_docs,
        "usage": "Use this as the request body for POST /predict",
        "curl_example": f"""curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps(sample_input_for_docs)}'"""
    }

@app.get("/model-info")
def model_info():
    """Get model information"""
    if not feature_info:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    return {
        "feature_info": feature_info,
        "sample_input": sample_input_for_docs,
        "model_loaded": pipeline is not None
    }

@app.post("/predict")
def predict(payload: Dict[str, Any] = Body(...)):
    """
    Make a prediction based on the input payload.
    Use GET /example to see the expected input format.
    """
    if not pipeline or not feature_info:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")

    try:
        # Validate that all required features are present
        required_features = feature_info['feature_names']
        missing_features = [f for f in required_features if f not in payload]
        
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {missing_features}"
            )
        
        # Convert the payload to a DataFrame
        df = pd.DataFrame([payload])

        # Reorder columns to match training and select only required features
        df = df[required_features]

        # Make a prediction
        prediction = pipeline.predict(df)

        return {
            "prediction": prediction.tolist(),
            "input_features": payload
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input values: {e}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/reload-model")
def reload_model():
    """Reload the model"""
    global pipeline, feature_info, PredictionPayload, sample_input_for_docs
    pipeline = None
    feature_info = None
    PredictionPayload = None
    sample_input_for_docs = None
    startup_event()
    return {
        "message": "Model reloaded successfully",
        "model_loaded": pipeline is not None
    }