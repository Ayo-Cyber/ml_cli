
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import create_model
from typing import Any
import os
import json
import logging
from pathlib import Path



# Create the FastAPI app
app = FastAPI()

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

        # Create the dynamic Pydantic model
        fields = {}
        sample_input_for_docs = {}
        for feature, feature_type in feature_info['feature_types'].items():
            if pd.api.types.is_integer_dtype(feature_type):
                fields[feature] = (int, ...)
                sample_input_for_docs[feature] = 0
            elif pd.api.types.is_float_dtype(feature_type):
                fields[feature] = (float, ...)
                sample_input_for_docs[feature] = 0.0
            else:
                fields[feature] = (str, ...)
                sample_input_for_docs[feature] = "string"
        
        PredictionPayload = create_model("PredictionPayload", **fields)
        logging.info("Model loaded successfully.")

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model files not found. Please train a model first.")
    except Exception as e:
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

@app.post("/predict")
def predict(payload: PredictionPayload = Body(..., example=sample_input_for_docs)):
    """
    Make a prediction based on the input payload.
    """
    if not pipeline or not feature_info:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")

    try:
        # Convert the payload to a DataFrame
        df = pd.DataFrame([payload.dict()])

        # Reorder columns to match training
        df = df[feature_info['feature_names']]

        # Make a prediction
        prediction = pipeline.predict(df)

        # Return the prediction
        return {"prediction": prediction.tolist()}
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/")
def root():
    return {"message": "Welcome to the ML-CLI API!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    if not feature_info:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    return feature_info

@app.post("/reload-model")
def reload_model():
    global pipeline, feature_info, PredictionPayload, sample_input_for_docs
    pipeline = None
    feature_info = None
    PredictionPayload = None
    sample_input_for_docs = None
    startup_event()
    return {"message": "Model reloaded successfully."}

