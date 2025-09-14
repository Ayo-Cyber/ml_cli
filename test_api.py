#!/usr/bin/env python3
"""
Test script for the flexible ML CLI serving functionality.
This script demonstrates how the API works with different datasets.
"""

import requests
import json
import time

def test_api(base_url="http://localhost:8000"):
    """Test the ML API endpoints."""
    
    print("🚀 Testing ML CLI Flexible Serving API")
    print("="*50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ API is not running. Please start it with 'ml serve'")
        return
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test model info endpoint
    print("\n3. Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model loaded successfully!")
            print(f"Features: {model_info['feature_names']}")
            print(f"Task type: {model_info['task_type']}")
            print(f"Total features: {model_info['total_features']}")
        else:
            print(f"❌ No model loaded. Status: {response.status_code}")
            print(f"Response: {response.json()}")
            return
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return
    
    # Test sample input endpoint
    print("\n4. Testing sample input endpoint...")
    response = requests.get(f"{base_url}/sample-input")
    if response.status_code == 200:
        sample_data = response.json()
        print(f"✅ Sample input format:")
        print(json.dumps(sample_data['sample_input'], indent=2))
        
        # Test prediction with sample data
        print("\n5. Testing prediction with sample data...")
        pred_response = requests.post(
            f"{base_url}/predict",
            json=sample_data['sample_input']
        )
        if pred_response.status_code == 200:
            prediction = pred_response.json()
            print(f"✅ Prediction successful!")
            print(f"Result: {prediction}")
        else:
            print(f"❌ Prediction failed. Status: {pred_response.status_code}")
            print(f"Response: {pred_response.json()}")
    
    # Test with invalid data
    print("\n6. Testing with invalid data...")
    invalid_data = {"invalid_feature": 123}
    response = requests.post(f"{base_url}/predict", json=invalid_data)
    print(f"Status: {response.status_code} (should be 400)")
    print(f"Response: {response.json()}")
    
    print("\n" + "="*50)
    print("🎉 API testing completed!")

if __name__ == "__main__":
    test_api()
