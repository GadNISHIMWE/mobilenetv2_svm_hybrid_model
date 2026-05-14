import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)

# ── Load features and labels ─────────────────────────────────────────────────
X = np.load('features/train_features.npy')
y = np.load('features/train_labels.npy')
filenames = np.load('features/train_filenames.npy')

print('=== Data Loaded ===')
print(f'Features shape : {X.shape}')
print(f'Labels shape   : {y.shape}')
print(f'Class 0 (large_leaf): {np.sum(y == 0)}')
print(f'Class 1 (small_leaf): {np.sum(y == 1)}')

# ── Split into train and test sets (80% / 20%) ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\n=== Train/Test Split ===')
print(f'Train samples : {X_train.shape[0]}')
print(f'Test  samples : {X_test.shape[0]}')

# ── Scale features ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print('\nFeatures scaled with StandardScaler')

# ── Train SVM ────────────────────────────────────────────────────────────────
print('\n=== Training SVM ===')
svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
svm.fit(X_train, y_train)
print('SVM training complete')

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = svm.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')

print('\n=== Evaluation Results ===')
print(f'Accuracy  : {accuracy  * 100:.2f}%')
print(f'Precision : {precision * 100:.2f}%')
print(f'Recall    : {recall    * 100:.2f}%')
print(f'F1 Score  : {f1        * 100:.2f}%')

print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred,
      target_names=['large_leaf', 'small_leaf']))

print('=== Confusion Matrix ===')
cm = confusion_matrix(y_test, y_pred)
print(f'                 Predicted')
print(f'                 large  small')
print(f'Actual large  :  {cm[0][0]:<5}  {cm[0][1]}')
print(f'Actual small  :  {cm[1][0]:<5}  {cm[1][1]}')

# ── Save model and scaler ────────────────────────────────────────────────────
joblib.dump(svm,    'models/svm_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print('\nModel  saved to models/svm_model.pkl')
print('Scaler saved to models/scaler.pkl')

# ── Save results ─────────────────────────────────────────────────────────────
with open('results/evaluation.txt', 'w') as f:
    f.write('=== SVM Evaluation Results ===\n')
    f.write(f'Accuracy  : {accuracy  * 100:.2f}%\n')
    f.write(f'Precision : {precision * 100:.2f}%\n')
    f.write(f'Recall    : {recall    * 100:.2f}%\n')
    f.write(f'F1 Score  : {f1        * 100:.2f}%\n\n')
    f.write(classification_report(y_test, y_pred,
            target_names=['large_leaf', 'small_leaf']))
print('Results saved to results/evaluation.txt')
