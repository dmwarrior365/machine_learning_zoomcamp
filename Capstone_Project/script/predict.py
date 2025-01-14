import requests

# Test health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# Test predictions
data = {
    'features': [1.0, 2.0, 3.0, 4.0]  # Replace with your feature values
}
response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())