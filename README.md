# MobileNet-SVM Hybrid Model

A local image classification pipeline that combines MobileNetV2 feature extraction with an SVM classifier.

## Project Overview

This repository implements a hybrid model where:
- `MobileNetV2` is used as a fixed feature extractor
- extracted deep features are fed into an SVM
- a Flask app provides image upload and online prediction

## Key Scripts

- `train_svm.py` - trains the SVM on precomputed features and saves `models/svm_model.pkl` and `models/scaler.pkl`
- `cross_validate.py` - validates the SVM model using stratified 5-fold cross validation
- `predict.py` - runs prediction on images in `test/leaf`
- `app.py` - Flask web application for image upload and classification
- `extract_features.py` - extracts MobileNetV2 features for the training and test sets
- `extract_nonleaf_features.py` - extracts MobileNetV2 features for non-leaf images
- `compare_models.py` - compares SVM-only vs MobileNetV2+SVM performance
- `features_check.py` - inspects extracted feature arrays

## Directory Structure

- `features/` - saved feature arrays and filenames
- `models/` - saved SVM model and scaler artifacts
- `results/` - saved evaluation, cross-validation, and prediction outputs
- `templates/` - Flask HTML template for the web UI
- `test/`, `train/`, `non-leaf/` - image datasets used by the pipeline

## Requirements

Install the required packages before running the scripts:

```bash
pip install numpy scipy scikit-learn tensorflow flask pillow joblib

Or install from the included requirements file:

```bash
pip install -r requirements.txt
```
```

> If you want to use TensorFlow offline, make sure the MobileNetV2 weights are already downloaded.

## Offline MobileNetV2 Weights

The project uses `MobileNetV2(weights='imagenet')` by default.

### Local weight file

The weights file is expected at:

- Windows: `C:\Users\<your-user>\.keras\models\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5`

### Alternative via environment variable

You can set a custom path with:

```bash
set MOBILENETV2_WEIGHTS_PATH=C:\path\to\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5
```

If the local file exists, the project will load it directly. Otherwise, it falls back to the standard Keras ImageNet download.

## How to Run

1. Extract MobileNetV2 features (if needed):
   ```bash
   python extract_features.py
   python extract_nonleaf_features.py
   ```

2. Train the SVM:
   ```bash
   python train_svm.py
   ```

3. Validate with cross validation:
   ```bash
   python cross_validate.py
   ```

4. Run predictions on a test folder:
   ```bash
   python predict.py
   ```

5. Start the Flask app:
   ```bash
   python app.py
   ```

Then open `http://127.0.0.1:5000/` in your browser.

## Notes

- `app.py` returns `Leaf` or `Not a Leaf` and handles uncertain predictions.
- `cross_validate.py` uses the same combined features as the training script to ensure a reliable model check.
- If you need a requirements file, you can create `requirements.txt` from the packages above.
