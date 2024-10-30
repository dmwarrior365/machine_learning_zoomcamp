from flask import Flask, request, jsonify
import pickle

with open('/workspaces/machine_learning_zoomcamp/module_05/dv.bin', 'rb') as dv_file:
    dv = pickle.load(dv_file)

with open('/workspaces/machine_learning_zoomcamp/module_05/model1.bin', 'rb') as model_file:
    model = pickle.load(model_file)

# App
app = Flask(__name__)

# Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    client_data = request.get_json()
    client_vector = dv.transform([client_data])

    probability = model.predict_proba(client_vector)[0][1]

    return jsonify({"probability": probability})

if __name__ == '__main__':
    app.run(debug=True)