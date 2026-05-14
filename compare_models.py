import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from PIL import Image
import os

# ── Load MobileNetV2 features and labels ─────────────────────────────────────
X_hybrid  = np.load('features/train_features.npy')
y         = np.load('features/train_labels.npy')
filenames = np.load('features/train_filenames.npy')

print('=== Model Comparison ===')
print('Comparing SVM alone vs MobileNetV2 + SVM hybrid\n')

# ── Build raw pixel features for SVM alone ───────────────────────────────────
# SVM alone: resize images to small size and flatten pixels as features
print('Preparing raw pixel features for SVM alone ...')
TRAIN_DIR = 'train'
IMG_SIZE  = (64, 64)   # small size to keep it manageable
raw_features = []

for fname in filenames:
    img = Image.open(os.path.join(TRAIN_DIR, fname)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    raw_features.append(arr.flatten())

X_raw = np.array(raw_features)
print(f'Raw pixel features shape: {X_raw.shape}')

# ── Scale features ───────────────────────────────────────────────────────────
scaler_hybrid = StandardScaler()
scaler_raw    = StandardScaler()
X_hybrid_scaled = scaler_hybrid.fit_transform(X_hybrid)
X_raw_scaled    = scaler_raw.fit_transform(X_raw)

# ── Define models ────────────────────────────────────────────────────────────
svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)

# ── Define scoring ───────────────────────────────────────────────────────────
scoring = {
    'accuracy' : make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall'   : make_scorer(recall_score,    average='weighted'),
    'f1'       : make_scorer(f1_score,        average='weighted')
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Evaluate SVM alone (raw pixels) ─────────────────────────────────────────
print('\nEvaluating SVM alone (raw pixels) ...')
results_raw = cross_validate(svm, X_raw_scaled, y, cv=kfold, scoring=scoring)

# ── Evaluate MobileNetV2 + SVM hybrid ────────────────────────────────────────
print('Evaluating MobileNetV2 + SVM hybrid ...')
results_hybrid = cross_validate(svm, X_hybrid_scaled, y, cv=kfold, scoring=scoring)

# ── Print comparison table ───────────────────────────────────────────────────
print('\n=== Comparison Results ===')
print(f'{"Model":<25} {"Accuracy":>10} {"Precision":>10} {"Recall":>10} {"F1":>10}')
print('-' * 70)

raw_acc  = results_raw['test_accuracy'].mean()    * 100
raw_prec = results_raw['test_precision'].mean()   * 100
raw_rec  = results_raw['test_recall'].mean()      * 100
raw_f1   = results_raw['test_f1'].mean()          * 100

hyb_acc  = results_hybrid['test_accuracy'].mean()  * 100
hyb_prec = results_hybrid['test_precision'].mean() * 100
hyb_rec  = results_hybrid['test_recall'].mean()    * 100
hyb_f1   = results_hybrid['test_f1'].mean()        * 100

print(f'{"SVM alone":<25} {raw_acc:>9.2f}% {raw_prec:>9.2f}% {raw_rec:>9.2f}% {raw_f1:>9.2f}%')
print(f'{"MobileNetV2 + SVM":<25} {hyb_acc:>9.2f}% {hyb_prec:>9.2f}% {hyb_rec:>9.2f}% {hyb_f1:>9.2f}%')
print('-' * 70)

improvement = hyb_acc - raw_acc
print(f'\nAccuracy improvement : +{improvement:.2f}%')
print(f'Feature size SVM alone  : {X_raw.shape[1]} (raw pixels)')
print(f'Feature size Hybrid     : {X_hybrid.shape[1]} (MobileNetV2 deep features)')

# ── Conclusion ───────────────────────────────────────────────────────────────
print('\n=== Conclusion ===')
if hyb_acc > raw_acc:
    print(f'MobileNetV2 + SVM hybrid outperforms SVM alone by {improvement:.2f}%')
    print('Transfer learning significantly improves classification performance.')
else:
    print('Results are comparable — further tuning may be needed.')

# ── Save results ─────────────────────────────────────────────────────────────
with open('results/comparison.txt', 'w') as f:
    f.write('=== Model Comparison Results ===\n\n')
    f.write(f'{"Model":<25} {"Accuracy":>10} {"Precision":>10} {"Recall":>10} {"F1":>10}\n')
    f.write('-' * 70 + '\n')
    f.write(f'{"SVM alone":<25} {raw_acc:>9.2f}% {raw_prec:>9.2f}% {raw_rec:>9.2f}% {raw_f1:>9.2f}%\n')
    f.write(f'{"MobileNetV2 + SVM":<25} {hyb_acc:>9.2f}% {hyb_prec:>9.2f}% {hyb_rec:>9.2f}% {hyb_f1:>9.2f}%\n')
    f.write('-' * 70 + '\n')
    f.write(f'\nAccuracy improvement : +{improvement:.2f}%\n')
    f.write(f'Feature size SVM alone : {X_raw.shape[1]} (raw pixels)\n')
    f.write(f'Feature size Hybrid    : {X_hybrid.shape[1]} (MobileNetV2 deep features)\n')

print('\nResults saved to results/comparison.txt')
