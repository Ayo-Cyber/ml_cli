import pytest
from fastapi.testclient import TestClient
from ml_cli.api.main import app
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def dummy_model():
    """Create a dummy model for testing"""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
    joblib.dump(model, temp_file.name)
    
    yield temp_file.name
    
    import os
    os.unlink(temp_file.name)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint(dummy_model, tmp_path):
    """Test the predict endpoint with a properly loaded model"""
    # Create proper model structure
    import json
    import shutil
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Copy model to output directory
    shutil.copy(dummy_model, output_dir / "fitted_pipeline.pkl")
    
    # Create feature info matching the model
    feature_info = {
        "feature_names": ["feature1", "feature2", "feature3", "feature4"],
        "feature_types": {
            "feature1": "float64",
            "feature2": "float64", 
            "feature3": "float64",
            "feature4": "float64"
        }
    }
    with open(output_dir / "feature_info.json", 'w') as f:
        json.dump(feature_info, f)
    
    # Load model into the app
    from ml_cli.api.main import load_model
    load_model(str(output_dir))
    
    # Create client AFTER loading model so PredictionPayload is defined
    from ml_cli.api.main import app as test_app
    client = TestClient(test_app)
    
    # Make prediction request
    response = client.post(
        "/predict",
        json={
            "feature1": 1.0,
            "feature2": 2.0,
            "feature3": 3.0,
            "feature4": 4.0
        }
    )
    
    # Verify response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.json()}"
    assert "prediction" in response.json()
