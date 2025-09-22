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
    description="API for ML model predictions with dynamic examples",
    version="1.0.0"
)

pipeline = None
feature_info = None
PredictionPayload = None
sample_input_for_docs = None

def generate_realistic_example_from_stats(feature_info: Dict) -> Dict[str, Any]:
    """Generate realistic examples based on feature statistics from the actual data"""
    example = {}
    
    # Check if we have feature statistics
    if 'feature_statistics' in feature_info:
        stats = feature_info['feature_statistics']
        for feature in feature_info.get('feature_names', []):
            if feature in stats and isinstance(stats[feature], dict):
                feature_stats = stats[feature]
                
                # Use mean if available, otherwise median, otherwise midpoint of min/max
                if 'mean' in feature_stats:
                    value = feature_stats['mean']
                elif 'median' in feature_stats:
                    value = feature_stats['median']
                elif 'min' in feature_stats and 'max' in feature_stats:
                    value = (feature_stats['min'] + feature_stats['max']) / 2
                else:
                    value = 1.0
                
                # Round to reasonable decimal places
                if isinstance(value, float):
                    example[feature] = round(value, 2)
                else:
                    example[feature] = value
            else:
                example[feature] = 1.0
    else:
        # Fallback if no statistics available
        for feature in feature_info.get('feature_names', []):
            example[feature] = 1.0
    
    return example

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

        # Generate realistic example from actual feature statistics
        sample_input_for_docs = generate_realistic_example_from_stats(feature_info)
        
        # Create the dynamic Pydantic model
        fields = {}
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
                    elif 'float' in feature_type.lower() or 'number' in feature_type.lower():
                        fields[feature] = (float, ...)
                    else:
                        fields[feature] = (str, ...)
                else:
                    # Handle pandas dtype objects
                    try:
                        if pd.api.types.is_integer_dtype(feature_type):
                            fields[feature] = (int, ...)
                        elif pd.api.types.is_float_dtype(feature_type):
                            fields[feature] = (float, ...)
                        else:
                            fields[feature] = (str, ...)
                    except:
                        # Fallback to float for numeric features
                        fields[feature] = (float, ...)
            else:
                # Default to float for all features if type info is missing
                fields[feature] = (float, ...)
        
        # Debug: Print fields being created
        logging.info(f"Creating Pydantic model with fields: {list(fields.keys())}")
        logging.info(f"Generated example: {sample_input_for_docs}")
        
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

def get_dynamic_example():
    """Get the current dynamic example based on loaded model"""
    if sample_input_for_docs:
        return sample_input_for_docs
    elif feature_info:
        # Regenerate if needed
        return generate_realistic_example_from_stats(feature_info)
    else:
        return {}

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the ML-CLI API!",
        "endpoints": {
            "predict": "/predict - Make predictions (with dynamic examples!)",
            "example": "/example - Get current example input data", 
            "model_info": "/model-info - Get model information",
            "health": "/health - Health check",
            "docs": "/docs - Interactive API documentation"
        },
        "tip": "Visit /docs to see the predict endpoint with auto-generated examples!"
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "feature_info_loaded": feature_info is not None,
        "example_available": sample_input_for_docs is not None,
        "feature_count": len(feature_info.get('feature_names', [])) if feature_info else 0
    }

@app.get("/example")
def get_example():
    """Get dynamically generated example input data for testing"""
    current_example = get_dynamic_example()
    
    if not current_example:
        raise HTTPException(status_code=503, detail="Model not loaded or example data not available.")
    
    return {
        "example_input": current_example,
        "feature_count": len(current_example),
        "usage": "Use this as the request body for POST /predict",
        "curl_example": f"""curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{json.dumps(current_example)}'""",
        "note": "This example is generated from your actual model's feature statistics"
    }

@app.get("/model-info")
def model_info():
    """Get comprehensive model information"""
    if not feature_info:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    current_example = get_dynamic_example()
    
    return {
        "feature_info": feature_info,
        "sample_input": current_example,
        "model_loaded": pipeline is not None,
        "feature_count": len(feature_info.get('feature_names', [])),
        "features": feature_info.get('feature_names', []),
        "has_statistics": 'feature_statistics' in feature_info,
        "example_source": "feature_statistics" if 'feature_statistics' in feature_info else "default_values"
    }

@app.post("/predict")
def predict(
    payload: Dict[str, Any] = Body(
        ...,
        description="Input features for prediction. The example below is auto-generated from your model's actual feature statistics!"
    )
):
    """
    🔮 Make a prediction based on the input payload.
    
    **Dynamic Example**: The example in the request body is automatically generated from your model's 
    actual feature statistics (mean, median, or reasonable defaults), so it will work for any project!
    
    💡 **How it works**:
    - If your model has feature statistics, it uses mean/median values
    - Otherwise, it uses sensible default values
    - The example updates automatically when you reload your model
    
    🚀 **Usage**: Click "Try it out" and the example will be pre-filled with realistic values!
    """
    if not pipeline or not feature_info:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")

    # Get current dynamic example for the docs
    current_example = get_dynamic_example()
    
    # Update the Body example dynamically
    if current_example:
        # This is a bit of a hack, but we'll set the example in the response
        pass

    try:
        # Validate that all required features are present
        required_features = feature_info['feature_names']
        missing_features = [f for f in required_features if f not in payload]
        
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required features: {missing_features}. Use GET /example to see the correct format."
            )
        
        # Convert the payload to a DataFrame
        df = pd.DataFrame([payload])

        # Reorder columns to match training and select only required features
        df = df[required_features]

        # Make a prediction
        prediction = pipeline.predict(df)
        
        # Try to get prediction probabilities if available
        probabilities = None
        if hasattr(pipeline, 'predict_proba'):
            try:
                probabilities = pipeline.predict_proba(df)
                probabilities = probabilities.tolist()
            except:
                pass

        result = {
            "prediction": prediction.tolist(),
            "input_features": payload,
            "model_info": {
                "features_used": len(required_features),
                "prediction_type": "classification" if probabilities else "regression"
            }
        }
        
        if probabilities:
            result["prediction_probabilities"] = probabilities
            result["confidence"] = max(probabilities[0]) if probabilities else None

        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input values: {e}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Custom endpoint to get the current example for the OpenAPI schema
@app.get("/predict/example")
def get_predict_example():
    """Get the current example that would be used in the predict endpoint"""
    current_example = get_dynamic_example()
    
    if not current_example:
        raise HTTPException(status_code=503, detail="No example available - model not loaded")
    
    return {
        "example": current_example,
        "description": "This example is dynamically generated from your model's feature statistics",
        "copy_paste_ready": True
    }

@app.post("/reload-model")
def reload_model():
    """🔄 Reload the model and regenerate examples"""
    global pipeline, feature_info, PredictionPayload, sample_input_for_docs
    
    old_example = sample_input_for_docs.copy() if sample_input_for_docs else None
    
    pipeline = None
    feature_info = None
    PredictionPayload = None
    sample_input_for_docs = None
    
    startup_event()
    
    new_example = sample_input_for_docs
    
    return {
        "message": "Model reloaded successfully",
        "model_loaded": pipeline is not None,
        "example_updated": old_example != new_example,
        "new_example": new_example,
        "feature_count": len(new_example) if new_example else 0,
        "tip": "Visit /docs to see the updated example in the predict endpoint!"
    }

# Override the OpenAPI schema to include dynamic examples
@app.get("/openapi.json", include_in_schema=False)
def custom_openapi():
    """Custom OpenAPI schema with dynamic examples"""
    from fastapi.openapi.utils import get_openapi
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add dynamic example to the predict endpoint
    current_example = get_dynamic_example()
    if current_example and "paths" in openapi_schema and "/predict" in openapi_schema["paths"]:
        predict_post = openapi_schema["paths"]["/predict"]["post"]
        if "requestBody" in predict_post:
            predict_post["requestBody"]["content"]["application/json"]["example"] = current_example
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema