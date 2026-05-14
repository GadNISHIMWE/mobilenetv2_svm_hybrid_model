import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_DIR  = "train"
TEST_DIR   = os.path.join("test", "leaf")
IMG_SIZE   = (224, 224)   # MobileNetV2 expected input size
FEAT_DIR   = "features"
# ────────────────────────────────────────────────────────────────────────────

# Load MobileNetV2 without the top classification layer
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
model.trainable = False
print("MobileNetV2 loaded — output feature size:", model.output_shape)


def extract(image_dir):
    features, filenames = [], []

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
    total = len(image_files)
    print(f"\nExtracting features from {total} images in '{image_dir}' ...")

    for i, fname in enumerate(image_files):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)          # shape: (1, 224, 224, 3)

        feat = model.predict(arr, verbose=0)        # shape: (1, 1280)
        features.append(feat[0])
        filenames.append(fname)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total} done")

    return np.array(features), filenames


# Extract train features
train_features, train_files = extract(TRAIN_DIR)
np.save(os.path.join(FEAT_DIR, "train_features.npy"), train_features)
np.save(os.path.join(FEAT_DIR, "train_filenames.npy"), np.array(train_files))
print(f"\nTrain features saved — shape: {train_features.shape}")

# Extract test features
test_features, test_files = extract(TEST_DIR)
np.save(os.path.join(FEAT_DIR, "test_features.npy"), test_features)
np.save(os.path.join(FEAT_DIR, "test_filenames.npy"), np.array(test_files))
print(f"Test  features saved — shape: {test_features.shape}")

print("\nFeature extraction complete.")
