import requests
from preprocessing import DataPreprocessor

base_url = 'https://rdtgeo65.pythonanywhere.com/'

def check_health():
    """Check the health status of the API"""
    try:
        response = requests.get(f'{base_url}/health')
        print("\n=== Health Check ===")
        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        try:
            print("JSON Response:", response.json())
        except Exception as e:
            print("Error parsing JSON:", e)

    except requests.RequestException as e:
        print(f"Error making health check request: {str(e)}")

def test_prediction():
    """Test the prediction endpoint with sample data"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Test data
    test_data = {
        'make': 'ACURA',
        'model': 'ILX',
        'vehicle_class': 'COMPACT',
        'engine_size(l)': 2.0,
        'cylinders': 4,
        'transmission': 'AS5',
        'fuel_type': 'Z',
        'fuel_efficiency': 8.3
    }

    try:
        # Preprocess the data
        processed_input = preprocessor.preprocess_input(test_data)

        # Prepare the request data
        request_data = {
            'features': processed_input.tolist()[0]  # Convert numpy array to list
        }

        print("\n=== Prediction Test ===")
        print("Sending data:", test_data)
        print("Processed features:", request_data['features'])

        # Make prediction request
        response = requests.post(
            f'{base_url}/predict',
            json=request_data,
            headers={'Content-Type': 'application/json'},
            timeout=200
        )

        print("\nPrediction Results:")
        print("Status Code:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Result:", result)
        else:
            print("\nError Response:", response.text)

    except Exception as e:
        print(f"Error making prediction: {str(e)}")

def main():
    print("Testing Model API Service with PythonAnywhere for Capstone Project")
    print("=" * 50)

    # Run health check
    check_health()

    # Run prediction test
    test_prediction()

if __name__ == "__main__":
    main()