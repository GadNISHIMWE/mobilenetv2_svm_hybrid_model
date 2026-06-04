# MobileNet-SVM Hybrid Model

A local image classification pipeline that combines MobileNetV2 feature extraction with an SVM classifier.

## Project Overview

This repository implements a hybrid model where:

- `MobileNetV2` is used as a fixed feature extractor.
- Extracted deep features are scaled and passed into an SVM classifier.
- A Flask app provides image upload and online prediction.

## Key Scripts

- `extract_features.py` - extracts MobileNetV2 features for the training and test image sets.
- `extract_nonleaf_features.py` - extracts MobileNetV2 features for non-leaf images.
- `merge_and_relabel.py` - combines leaf and non-leaf feature arrays into one binary dataset.
- `train_svm.py` - trains the SVM and saves `models/svm_model.pkl` and `models/scaler.pkl`.
- `cross_validate.py` - validates the SVM model using stratified 5-fold cross validation.
- `predict.py` - runs batch predictions on images in `test/leaf`.
- `app.py` - starts the Flask web app for upload-based classification.
- `features_check.py` - inspects extracted feature arrays.
- `compare_models.py` - compares raw-pixel SVM vs MobileNetV2+SVM performance.

## Directory Structure

- `features/` - saved feature arrays, filenames, and labels.
- `models/` - saved SVM model and scaler artifacts.
- `results/` - saved evaluation, cross-validation, and prediction outputs.
- `templates/` - Flask HTML template for the web UI.
- `train/`, `test/`, `non-leaf/` - image datasets used by the pipeline.

## Requirements

Install dependencies from the included requirements file:

```bash
pip install -r requirements.txt
```

The requirements are:

```text
numpy
scipy
scikit-learn
tensorflow
flask
flask-cors
pillow
joblib
```

## Offline MobileNetV2 Weights

The project first checks for a local MobileNetV2 weights file. If it does not find one, it falls back to the standard Keras ImageNet weights, which may download the file if it is not already cached.

Default Windows path:

```text
C:\Users\<your-user>\.keras\models\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
```

You can also set a custom path:

```bat
set MOBILENETV2_WEIGHTS_PATH=C:\path\to\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
```

## How to Run

1. Extract MobileNetV2 features:

   ```bash
   python extract_features.py
   python extract_nonleaf_features.py
   ```

2. Merge leaf and non-leaf features:

   ```bash
   python merge_and_relabel.py
   ```

3. Train the SVM:

   ```bash
   python train_svm.py
   ```

4. Validate with cross validation:

   ```bash
   python cross_validate.py
   ```

5. Run predictions on the test folder:

   ```bash
   python predict.py
   ```

6. Start the Flask app:

   ```bash
   python app.py
   ```

Then open `http://127.0.0.1:5000/` in your browser.

## Notes

- `app.py` returns `Leaf`, `Not a Leaf`, or `Uncertain` for low-confidence predictions.
- `train_svm.py` expects `features/combined_features.npy` and `features/combined_labels.npy`, which are produced by `merge_and_relabel.py`.
- Keep `venv/`, `__pycache__/`, generated logs, and local environment files out of Git.
