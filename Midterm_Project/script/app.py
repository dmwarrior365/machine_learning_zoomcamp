from flask import Flask, request, jsonify
import numpy as np
import pickle

with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
# App
app = Flask(__name__)

# Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.get_json()
    
    input_features = np.array([list(client_data.values())])
    
    predictions = model.predict(input_features)
    
    oil_rate, water_rate = predictions[0]
    
    return jsonify({"oil_rate": oil_rate, "water_rate": water_rate})

if __name__ == '__main__':
    app.run(debug=True)