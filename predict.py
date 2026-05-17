import numpy as np
import joblib
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mobilenet_utils import load_mobilenetv2

# ── Config ───────────────────────────────────────────────────────────────────
TEST_DIR  = os.path.join('test', 'leaf')
IMG_SIZE  = (224, 224)
CLASSES   = {0: 'large_leaf', 1: 'small_leaf'}

# ── Load MobileNetV2 feature extractor ───────────────────────────────────────
print('Loading MobileNetV2 ...')
mobilenet = load_mobilenetv2(include_top=False,
                             pooling='avg', input_shape=(224, 224, 3))
mobilenet.trainable = False

# ── Load trained SVM and scaler ──────────────────────────────────────────────
print('Loading SVM model and scaler ...')
svm    = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ── Predict ──────────────────────────────────────────────────────────────────
print('\n=== Predictions on Test Images ===')
print(f'{"Image":<15} {"Predicted Class":<20} {"Confidence"}')
print('-' * 50)

test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith('.jpg')])
results = []

for fname in test_images:
    # Step 1 - Load and preprocess image
    img = Image.open(os.path.join(TEST_DIR, fname)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # Step 2 - Extract features with MobileNetV2
    features = mobilenet.predict(arr, verbose=0)

    # Step 3 - Scale features
    features_scaled = scaler.transform(features)

    # Step 4 - SVM prediction
    prediction   = svm.predict(features_scaled)[0]
    probabilities = svm.predict_proba(features_scaled)[0]
    confidence   = probabilities[prediction] * 100
    class_name   = CLASSES[prediction]

    print(f'{fname:<15} {class_name:<20} {confidence:.2f}%')
    results.append((fname, class_name, confidence))

# ── Save results ─────────────────────────────────────────────────────────────
with open('results/predictions.txt', 'w') as f:
    f.write('=== Test Image Predictions ===\n\n')
    f.write(f'{"Image":<15} {"Predicted Class":<20} {"Confidence"}\n')
    f.write('-' * 50 + '\n')
    for fname, class_name, confidence in results:
        f.write(f'{fname:<15} {class_name:<20} {confidence:.2f}%\n')

print('\nPredictions saved to results/predictions.txt')
