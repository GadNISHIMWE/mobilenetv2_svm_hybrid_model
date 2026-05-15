import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── Config ───────────────────────────────────────────────────────────────────
NONLEAF_DIR = 'non-leaf'
IMG_SIZE    = (224, 224)
VALID_EXTS  = ('.jpg', '.jpeg', '.png')

# ── Load MobileNetV2 ─────────────────────────────────────────────────────────
print('Loading MobileNetV2 ...')
model = MobileNetV2(weights='imagenet', include_top=False,
                    pooling='avg', input_shape=(224, 224, 3))
model.trainable = False
print('MobileNetV2 ready.\n')

# ── Extract features ─────────────────────────────────────────────────────────
image_files = sorted([f for f in os.listdir(NONLEAF_DIR)
                      if f.lower().endswith(VALID_EXTS)])
total = len(image_files)
print(f'Found {total} non-leaf images in "{NONLEAF_DIR}"')
print('Extracting features ...\n')

features, filenames, skipped = [], [], []

for i, fname in enumerate(image_files):
    try:
        img = Image.open(os.path.join(NONLEAF_DIR, fname)).convert('RGB')
        img = img.resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        feat = model.predict(arr, verbose=0)
        features.append(feat[0])
        filenames.append(fname)

        if (i + 1) % 50 == 0:
            print(f'  {i + 1}/{total} done')

    except Exception as e:
        print(f'  Skipped {fname}: {e}')
        skipped.append(fname)

features  = np.array(features)
filenames = np.array(filenames)

# ── Save ─────────────────────────────────────────────────────────────────────
np.save('features/nonleaf_features.npy',  features)
np.save('features/nonleaf_filenames.npy', filenames)

print(f'\nExtracted : {len(features)} images')
print(f'Skipped   : {len(skipped)} images')
print(f'Shape     : {features.shape}')
print('\nSaved to features/nonleaf_features.npy')
