import pickle
import numpy as np
import os
import sys

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    if not os.path.exists(model_path):
        import subprocess
        # Auto-train if missing
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "train_model.py")])
        
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

def predict_rul(features: list) -> int:
    """ Expects an array of 24 floats: 3 op settings + 21 sensors """
    features_array = np.array([features])
    rul = model.predict(features_array)[0]
    return max(0, int(rul))

def classify_status(rul: int) -> str:
    if rul > 100:
        return "Healthy"
    elif rul > 30:
        return "Warning"
    else:
        return "Danger"
