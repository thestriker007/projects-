from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from functools import wraps

app = Flask(__name__)
CORS(app)

USER_ID = "prince_admin" 
USER_PWD = "secure_password123"
MODEL_PATH = "cataract_classifier_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def check_auth(username, password):
    return username == USER_ID and password == USER_PWD

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return jsonify({"message": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated

def prepare_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

@app.route("/predict", methods=["POST"])
@requires_auth 
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    
    processed_image = prepare_image(image, target_size=(150, 150))
    prediction = model.predict(processed_image)
    
    result = "Cataract Detected" if prediction[0][0] > 0.5 else "Normal/Healthy"
    confidence = float(prediction[0][0]) if result == "Cataract Detected" else float(1 - prediction[0][0])

    return jsonify({
        "status": "success",
        "prediction": result,
        "confidence": f"{confidence:.2%}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)