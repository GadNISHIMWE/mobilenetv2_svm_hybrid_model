import os
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mobilenet_utils import load_mobilenetv2

TEST_DIR    = "test"
RESULTS_DIR = "results"
IMG_SIZE    = (224, 224)
VALID_EXTS  = (".jpg", ".jpeg", ".png", ".webp")

CLASS_NAMES = [
    "Tea Algal Leaf Spot",
    "Brown Blight",
    "Gray Blight",
    "Helopeltis",
    "Red Spider",
    "Green Mirid Bug",
    "Healthy Leaf",
]


def predict_image(image_path, mobilenet, svm, scaler):
    with Image.open(image_path) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    features = mobilenet.predict(arr, verbose=0)
    features_scaled = scaler.transform(features)
    prediction = svm.predict(features_scaled)[0]
    confidence = svm.predict_proba(features_scaled)[0][prediction] * 100
    return CLASS_NAMES[prediction], confidence


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    print("Loading MobileNetV2 ...")
    mobilenet = load_mobilenetv2(include_top=False, pooling="avg", input_shape=(224, 224, 3))
    mobilenet.trainable = False

    print("Loading SVM model and scaler ...")
    svm    = joblib.load(os.path.join("models", "svm_model.pkl"))
    scaler = joblib.load(os.path.join("models", "scaler.pkl"))

    test_images = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith(VALID_EXTS))

    if not test_images:
        print(f"\nNo images found in '{TEST_DIR}'. Add images and run again.")
        return

    print(f"\n=== Predictions on {len(test_images)} Test Images ===")
    print(f"{'Image':<30} {'Predicted Class':<25} {'Confidence'}")
    print("-" * 70)

    results = []
    for fname in test_images:
        image_path = os.path.join(TEST_DIR, fname)
        try:
            class_name, confidence = predict_image(image_path, mobilenet, svm, scaler)
            print(f"{fname:<30} {class_name:<25} {confidence:.2f}%")
            results.append((fname, class_name, confidence))
        except Exception as e:
            print(f"{fname:<30} ERROR: {e}")

    output_path = os.path.join(RESULTS_DIR, "predictions.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Tea Leaf Disease Predictions ===\n\n")
        f.write(f"{'Image':<30} {'Predicted Class':<25} {'Confidence'}\n")
        f.write("-" * 70 + "\n")
        for fname, class_name, confidence in results:
            f.write(f"{fname:<30} {class_name:<25} {confidence:.2f}%\n")

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
