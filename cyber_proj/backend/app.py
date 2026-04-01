from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

@app.route("/users", methods=["GET"])
def get_users():
    try:
        # Lists all trained subjects
        files = os.listdir(MODELS_DIR)
        users = [f.replace("_model.pkl", "") for f in files if f.endswith("_model.pkl")]
        return jsonify({"users": sorted(users)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    sample = data.get("sample")
    
    if not username or not sample:
        return jsonify({"error": "Username and 31-feature sample vector required."}), 400
        
    model_path = os.path.join(MODELS_DIR, f"{username}_model.pkl")
    if not os.path.exists(model_path):
        return jsonify({"error": f"User '{username}' point not found or model not trained."}), 404
        
    try:
        clf = joblib.load(model_path)
        
        prediction = clf.predict([sample])[0]
        
        if prediction == 1:
            return jsonify({"authenticated": True, "message": "Genuine user verified based on continuous cadence pattern."}), 200
        else:
            return jsonify({"authenticated": False, "message": "Imposter detected! Pattern does not match subject."}), 401
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
