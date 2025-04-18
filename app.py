from flask import Flask, request, jsonify
import os
import requests
import pickle

app = Flask(__name__)

MODEL_URL = "https://huggingface.co/sanchitpahurkar/HomeScope/resolve/main/housing_india_model.pkl"
MODEL_PATH = "housing_india_model.pkl"

# download and load the model
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
    
    
@app.route("/predict", methods=["POST"])
def predict() :
    try :
        data = request.json
        features = data["features"]
        prediction = model.predict([features])
        return jsonify({"Prediction" : prediction[0]})
    except Exception as e:
        return jsonify({"error" : str(e)}), 500