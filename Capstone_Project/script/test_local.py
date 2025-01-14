import requests
import json
import time
from Capstone_Project.script.preprocessing import DataPreprocessor

def test_api():
    # Local server URL
    base_url = 'http://localhost:5000'
    
    # Test data
    preprocessor = DataPreprocessor()

    test_data = {
        'make': 'ACURA',
        'model': 'ILX',
        'vehicle_class': 'COMPACT',
        'engine_size': 2.0,
        'cylinders': 4,
        'transmission': 'AS5',
        'fuel_type': 'Z',
        'fuel_efficiency': 9.9
    }

    # Test prediction endpoint
    print("\nTesting prediction endpoint...")
    try:
        # Prepare request data
        processed_input = preprocessor.preprocess_input(test_data)

        # Prepare the request data
        request_data = {
            'features': processed_input.tolist()[0]  # Convert numpy array to list
        }
        
        # Make prediction request with increased timeout
        response = requests.post(
            f'{base_url}/predict',
            json=request_data,
            timeout=30
        )
        
        print("Status Code:", response.status_code)
        
        if response.status_code == 200:
            result = response.json()
            print("Prediction result:", result)
        else:
            print("Error response:", response.text)
            
    except requests.Timeout:
        print("Request timed out")
    except Exception as e:
        print("Prediction request failed:", str(e))

if __name__ == "__main__":
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    test_api()