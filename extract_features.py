import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from mobilenet_utils import load_mobilenetv2

DATASET_DIR = "teaLeafBD"
FEAT_DIR = "features"
IMG_SIZE = (224, 224)
VALID_EXTS = (".jpg", ".jpeg", ".png")

CLASS_NAMES = [
    "1. Tea algal leaf spot",
    "2. Brown Blight",
    "3. Gray Blight",
    "4. Helopeltis",
    "5. Red spider",
    "6. Green mirid bug",
    "7. Healthy leaf",
]


def build_feature_extractor():
    model = load_mobilenetv2(include_top=False, pooling="avg", input_shape=(224, 224, 3))
    model.trainable = False
    print("MobileNetV2 loaded - output feature size:", model.output_shape)
    return model


def extract_all(model):
    all_features, all_labels, all_filenames = [], [], []

    for label, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(DATASET_DIR, class_name)
        images = sorted(f for f in os.listdir(class_path) if f.lower().endswith(VALID_EXTS))
        print(f"\n[{label}] {class_name} — {len(images)} images")

        for i, fname in enumerate(images, start=1):
            img_path = os.path.join(class_path, fname)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize(IMG_SIZE)
                    arr = np.array(img, dtype=np.float32)
                arr = preprocess_input(arr)
                arr = np.expand_dims(arr, axis=0)
                feat = model.predict(arr, verbose=0)
                all_features.append(feat[0])
                all_labels.append(label)
                all_filenames.append(fname)
            except Exception as e:
                print(f"  Skipped {fname}: {e}")

            if i % 100 == 0:
                print(f"  {i}/{len(images)} done")

    return np.array(all_features), np.array(all_labels), np.array(all_filenames)


def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    model = build_feature_extractor()

    features, labels, filenames = extract_all(model)

    np.save(os.path.join(FEAT_DIR, "features.npy"), features)
    np.save(os.path.join(FEAT_DIR, "labels.npy"), labels)
    np.save(os.path.join(FEAT_DIR, "filenames.npy"), filenames)

    print(f"\nFeature extraction complete.")
    print(f"  Features shape : {features.shape}")
    print(f"  Labels shape   : {labels.shape}")
    print(f"  Classes        : {len(CLASS_NAMES)}")
    for i, name in enumerate(CLASS_NAMES):
        count = np.sum(labels == i)
        print(f"    [{i}] {name}: {count} samples")


if __name__ == "__main__":
    main()
