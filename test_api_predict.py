#!/usr/bin/env python3
"""
Quick API prediction test script.
Tests the /predict endpoint to ensure numpy types are properly converted.
"""

import requests
import json

# API endpoint
API_URL = "http://127.0.0.1:8000"

def test_predict():
    """Test single prediction endpoint"""
    # First, get the example input format
    try:
        response = requests.get(f"{API_URL}/predict/example")
        if response.status_code == 200:
            example = response.json()
            print("✅ Got example input:")
            print(json.dumps(example, indent=2))
            
            # Use the example to make a prediction
            print("\n🔮 Making prediction...")
            pred_response = requests.post(
                f"{API_URL}/predict",
                json=example,
                headers={"Content-Type": "application/json"}
            )
            
            if pred_response.status_code == 200:
                result = pred_response.json()
                print("✅ Prediction successful!")
                print(json.dumps(result, indent=2))
                return True
            else:
                print(f"❌ Prediction failed with status {pred_response.status_code}")
                print(pred_response.text)
                return False
        else:
            print(f"❌ Could not get example: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the server running?")
        print("   Run 'ml serve' first")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ API is healthy")
            print(f"   Model loaded: {health.get('model_loaded')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing ML-CLI API\n")
    print("=" * 50)
    
    # Test health first
    print("\n1. Testing /health endpoint...")
    if not test_health():
        print("\n⚠️  API is not healthy. Exiting.")
        exit(1)
    
    # Test prediction
    print("\n" + "=" * 50)
    print("\n2. Testing /predict endpoint...")
    if test_predict():
        print("\n" + "=" * 50)
        print("\n🎉 All tests passed!")
    else:
        print("\n" + "=" * 50)
        print("\n❌ Tests failed!")
        exit(1)
