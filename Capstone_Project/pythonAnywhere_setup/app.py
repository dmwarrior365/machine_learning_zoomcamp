from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import xgboost as xgb
import os

app = Flask(__name__)
CORS(app)

# Global variable for model
model = None

def load_model():
    global model
    try:
        # Get absolute path to model file
        model_path = os.path.join(os.path.dirname(__file__), 'xgboost_best_model.pkl')
        print(f"Loading model from: {model_path}")

        # Load the model
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

# Initialize model when app starts
load_model()

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        print("Attempting to reload model...")
        success = load_model()
        if not success:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500

    try:
        # Get data from POST request
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                'status': 'error',
                'message': 'No features provided'
            }), 400

        # Convert input data to numpy array
        features = [
            data['make'],
            data['model'],
            data['vehicle_class'],
            data['engine_size(l)'],
            data['cylinders'],
            data['transmission'],
            data['fuel_type'],
            data['fuel_efficiency']
        ]

        input_features = np.array(features).reshape(1, -1)

        # Make prediction (with timeout protection)
        try:
            prediction = model.predict(input_features)

            return jsonify({
                'status': 'success',
                'prediction': float(prediction[0])
            })
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Prediction failed: {str(e)}'
            }), 500

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    global model
    model_status = model is not None
    return jsonify({
        'status': 'healthy',
        'message': 'ML service is running',
        'model_loaded': model_status
    })

if __name__ == '__main__':
    app.run(debug=True)