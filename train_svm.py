import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES_PATH = os.path.join("features", "features.npy")
LABELS_PATH   = os.path.join("features", "labels.npy")
MODEL_DIR     = "models"
RESULTS_DIR   = "results"

CLASS_NAMES = [
    "Tea Algal Leaf Spot",
    "Brown Blight",
    "Gray Blight",
    "Helopeltis",
    "Red Spider",
    "Green Mirid Bug",
    "Healthy Leaf",
]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X = np.load(FEATURES_PATH)
    y = np.load(LABELS_PATH)

    print("=== Data Loaded ===")
    print(f"Features shape : {X.shape}")
    print(f"Labels shape   : {y.shape}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  [{i}] {name}: {np.sum(y == i)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n=== Train/Test Split ===")
    print(f"Train samples : {X_train.shape[0]}")
    print(f"Test samples  : {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("Features scaled with StandardScaler")

    print("\n=== Training SVM ===")
    svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    print("SVM training complete")

    y_pred = svm.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall    = recall_score(y_test, y_pred, average="weighted")
    f1        = f1_score(y_test, y_pred, average="weighted")

    print("\n=== Evaluation Results ===")
    print(f"Accuracy  : {accuracy  * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall    * 100:.2f}%")
    print(f"F1 Score  : {f1        * 100:.2f}%")

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
    print("\n=== Classification Report ===")
    print(report)

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':25}", " ".join(f"{n[:6]:>7}" for n in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:25}", " ".join(f"{v:>7}" for v in row))

    joblib.dump(svm,    os.path.join(MODEL_DIR, "svm_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\nModel  saved to {MODEL_DIR}/svm_model.pkl")
    print(f"Scaler saved to {MODEL_DIR}/scaler.pkl")

    results_path = os.path.join(RESULTS_DIR, "evaluation.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== SVM Evaluation Results ===\n")
        f.write(f"Accuracy  : {accuracy  * 100:.2f}%\n")
        f.write(f"Precision : {precision * 100:.2f}%\n")
        f.write(f"Recall    : {recall    * 100:.2f}%\n")
        f.write(f"F1 Score  : {f1        * 100:.2f}%\n\n")
        f.write(report)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
