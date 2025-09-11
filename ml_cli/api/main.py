
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, create_model
from typing import Optional, Dict, Any, Union, List
import os
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Create the FastAPI app
app = FastAPI()

pipeline = None
feature_info = None
PredictionPayload = None

def load_model(output_dir: str):
    global pipeline, feature_info, PredictionPayload
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
        def create_pydantic_model(name, feature_info):
            fields = {}
            for feature, feature_type in feature_info['feature_types'].items():
                if pd.api.types.is_integer_dtype(feature_type):
                    fields[feature] = (int, ...)
                elif pd.api.types.is_float_dtype(feature_type):
                    fields[feature] = (float, ...)
                else:
                    fields[feature] = (str, ...)
            return create_model(name, **fields)

        PredictionPayload = create_pydantic_model("PredictionPayload", feature_info)
        logging.info("Model loaded successfully.")

    except Exception as e:
        logging.error(f"Error loading model: {e}")

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
                logging.error(exc)
    load_model(output_dir)

@app.post("/predict")
def predict(payload: BaseModel):
    """
    Make a prediction based on the input payload.
    """
    if not pipeline or not feature_info or not PredictionPayload:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")

    try:
        # Convert the payload to a DataFrame
        df = pd.DataFrame([payload.dict()])

        # Make a prediction
        prediction = pipeline.predict(df)

        # Return the prediction
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

@app.get("/sample-input")
def sample_input():
    if not feature_info:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")

    sample = {}
    for feature, feature_type in feature_info['feature_types'].items():
        if pd.api.types.is_integer_dtype(feature_type):
            sample[feature] = 0
        elif pd.api.types.is_float_dtype(feature_type):
            sample[feature] = 0.0
        else:
            sample[feature] = "string"
    return {"sample_input": sample}

@app.post("/reload-model")
def reload_model():
    startup_event()
    return {"message": "Model reloaded successfully."}

