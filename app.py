import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import base64
from io import BytesIO

app = Flask(__name__)

CLASSES  = {0: 'Large Leaf', 1: 'Small Leaf'}
IMG_SIZE = (224, 224)

# ── Load models once at startup ───────────────────────────────────────────────
print('Loading MobileNetV2 ...')
mobilenet = MobileNetV2(weights='imagenet', include_top=False,
                        pooling='avg', input_shape=(224, 224, 3))
mobilenet.trainable = False

print('Loading SVM and scaler ...')
svm    = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')
print('Models ready.\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Step 1 - Load image
    img = Image.open(file.stream).convert('RGB')
    original_size = img.size

    # Step 2 - Preprocess
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # Step 3 - Extract features with MobileNetV2
    features = mobilenet.predict(arr, verbose=0)

    # Step 4 - Scale and predict with SVM
    features_scaled = scaler.transform(features)
    prediction      = svm.predict(features_scaled)[0]
    probabilities   = svm.predict_proba(features_scaled)[0]
    confidence      = round(float(probabilities[prediction]) * 100, 2)
    class_name      = CLASSES[prediction]

    # Convert image to base64 for display
    buffered = BytesIO()
    img.save(buffered, format='JPEG')
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        'class'      : class_name,
        'confidence' : confidence,
        'image_size' : f'{original_size[0]} x {original_size[1]}',
        'features'   : 1280,
        'image_data' : img_base64
    })


if __name__ == '__main__':
    app.run(debug=True)
