from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
CORS(app)

model = None
label_encoders = None

def load_model():
    """Load the ML model and label encoders"""
    global model, label_encoders
    try:
        # Get the directory containing app.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load XGBoost model
        model_path = os.path.join(base_dir, 'xgboost_best_model.pkl')
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # Load label encoders
        encoders_path = os.path.join(base_dir, 'label_encoders.pkl')
        print(f"Loading label encoders from: {encoders_path}")
        label_encoders = joblib.load(encoders_path)
        
        print("Model and encoders loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model or encoders: {str(e)}")
        return False

def preprocess_input(data):
    """Preprocess input data using loaded label encoders"""
    try:
        # Expected feature order
        categorical_columns = ['make', 'model', 'vehicle_class', 'transmission', 'fuel_type']
        numeric_columns = ['engine_size(l)', 'cylinders', 
                         'fuel_efficiency']
        
        # Process categorical features
        processed_features = []
        
        # Handle categorical features
        for i, column in enumerate(categorical_columns):
            value = data[column]
            if column in label_encoders:
                try:
                    encoded_value = label_encoders[column].transform([value])[0]
                    processed_features.append(encoded_value)
                except ValueError as e:
                    print(f"Error encoding {column}: {str(e)}")
                    # Handle unknown categories by using a default value
                    processed_features.append(0)
            else:
                processed_features.append(0)
        
        # Handle numeric features
        for column in numeric_columns:
            processed_features.append(float(data[column]))
        
        return np.array(processed_features)
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    if model is None:
        success = load_model()
        if not success:
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
    
    try:
        # Get data from POST request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No input data provided'
            }), 400
        
        # Preprocess the input data
        processed_input = preprocess_input(data)
        
        # Reshape for prediction
        processed_input = processed_input.reshape(1, -1)
        print(f"Processed input shape: {processed_input.shape}")
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        # Return prediction
        return jsonify({
            'status': 'success',
            'prediction': prediction.tolist()
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health_check():
    global model, label_encoders
    return jsonify({
        'status': 'healthy',
        'message': 'ML service is running',
        'model_loaded': model is not None,
        'encoders_loaded': label_encoders is not None
    })

# Load model when app starts
load_model()

if __name__ == '__main__':
    app.run(debug=True)