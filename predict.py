import os

import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from mobilenet_utils import load_mobilenetv2


TEST_DIR = os.path.join("test", "leaf")
RESULTS_DIR = "results"
IMG_SIZE = (224, 224)
CLASSES = {0: "Leaf", 1: "Not a Leaf"}
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading MobileNetV2 ...")
    mobilenet = load_mobilenetv2(include_top=False, pooling="avg", input_shape=(224, 224, 3))
    mobilenet.trainable = False

    print("Loading SVM model and scaler ...")
    svm = joblib.load(os.path.join("models", "svm_model.pkl"))
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))

    print("\n=== Predictions on Test Images ===")
    print(f'{"Image":<20} {"Predicted Class":<20} {"Confidence"}')
    print("-" * 60)

    test_images = sorted(
        f for f in os.listdir(TEST_DIR) if f.lower().endswith(VALID_EXTS)
    )
    results = []

    for fname in test_images:
        image_path = os.path.join(TEST_DIR, fname)
        with Image.open(image_path) as img:
            img = img.convert("RGB").resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32)

        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        features = mobilenet.predict(arr, verbose=0)
        features_scaled = scaler.transform(features)

        prediction = svm.predict(features_scaled)[0]
        probabilities = svm.predict_proba(features_scaled)[0]
        confidence = probabilities[prediction] * 100
        class_name = CLASSES[prediction]

        print(f"{fname:<20} {class_name:<20} {confidence:.2f}%")
        results.append((fname, class_name, confidence))

    output_path = os.path.join(RESULTS_DIR, "predictions.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Test Image Predictions ===\n\n")
        f.write(f'{"Image":<20} {"Predicted Class":<20} {"Confidence"}\n')
        f.write("-" * 60 + "\n")
        for fname, class_name, confidence in results:
            f.write(f"{fname:<20} {class_name:<20} {confidence:.2f}%\n")

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
