import base64
import os
from io import BytesIO

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from mobilenet_utils import load_mobilenetv2


app = Flask(__name__)
CORS(app)

CLASSES = {0: "Leaf", 1: "Not a Leaf"}
STATUS = {0: "leaf", 1: "not_leaf"}
CONFIDENCE_THRESHOLD = 70
IMG_SIZE = (224, 224)


print("Loading MobileNetV2 ...")
mobilenet = load_mobilenetv2(include_top=False, pooling="avg", input_shape=(224, 224, 3))
mobilenet.trainable = False

print("Loading SVM and scaler ...")
svm = joblib.load(os.path.join("models", "svm_model.pkl"))
scaler = joblib.load(os.path.join("models", "scaler.pkl"))
print("Models ready.\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "message": "Backend is running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file is not a valid image"}), 400

    original_size = img.size
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    features = mobilenet.predict(arr, verbose=0)
    features_scaled = scaler.transform(features)
    prediction = svm.predict(features_scaled)[0]
    probabilities = svm.predict_proba(features_scaled)[0]
    confidence = round(float(probabilities[prediction]) * 100, 2)
    class_name = CLASSES[prediction]
    status = STATUS[prediction]

    if confidence < CONFIDENCE_THRESHOLD:
        class_name = "Uncertain - please upload a clearer image"
        status = "uncertain"

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify(
        {
            "class": class_name,
            "confidence": confidence,
            "status": status,
            "image_size": f"{original_size[0]} x {original_size[1]}",
            "features": 1280,
            "image_data": img_base64,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
