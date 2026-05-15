import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# ── Load features and labels from the same combined dataset used by train_svm.py ─────────────────────────────────────────────────
X = np.load('features/combined_features.npy')
y = np.load('features/combined_labels.npy')

print('=== K-Fold Cross Validation ===')
print(f'Total samples  : {X.shape[0]}')
print(f'Feature size   : {X.shape[1]}')
print(f'Class 0 (leaf)    : {np.sum(y == 0)}')
print(f'Class 1 (not_leaf): {np.sum(y == 1)}')

# ── Scale features ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Define SVM and K-Fold ────────────────────────────────────────────────────
svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)

# StratifiedKFold ensures each fold has same class ratio
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Define scoring metrics ───────────────────────────────────────────────────
scoring = {
    'accuracy' : make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'recall'   : make_scorer(recall_score,    average='weighted'),
    'f1'       : make_scorer(f1_score,        average='weighted')
}

# ── Run cross validation ─────────────────────────────────────────────────────
print('\nRunning 5-Fold Cross Validation ...')
results = cross_validate(svm, X_scaled, y, cv=kfold, scoring=scoring)

# ── Print results per fold ───────────────────────────────────────────────────
print('\n=== Results Per Fold ===')
print(f'{"Fold":<6} {"Accuracy":>10} {"Precision":>10} {"Recall":>10} {"F1":>10}')
print('-' * 50)
for i in range(5):
    acc  = results['test_accuracy'][i]  * 100
    prec = results['test_precision'][i] * 100
    rec  = results['test_recall'][i]    * 100
    f1   = results['test_f1'][i]        * 100
    print(f'{i+1:<6} {acc:>9.2f}% {prec:>9.2f}% {rec:>9.2f}% {f1:>9.2f}%')

# ── Print average results ────────────────────────────────────────────────────
print('-' * 50)
print(f'{"Avg":<6} '
      f'{results["test_accuracy"].mean()*100:>9.2f}% '
      f'{results["test_precision"].mean()*100:>9.2f}% '
      f'{results["test_recall"].mean()*100:>9.2f}% '
      f'{results["test_f1"].mean()*100:>9.2f}%')

print(f'\n{"Std":<6} '
      f'{results["test_accuracy"].std()*100:>9.2f}% '
      f'{results["test_precision"].std()*100:>9.2f}% '
      f'{results["test_recall"].std()*100:>9.2f}% '
      f'{results["test_f1"].std()*100:>9.2f}%')

# ── Save results ─────────────────────────────────────────────────────────────
with open('results/cross_validation.txt', 'w') as f:
    f.write('=== 5-Fold Cross Validation Results ===\n\n')
    f.write(f'{"Fold":<6} {"Accuracy":>10} {"Precision":>10} {"Recall":>10} {"F1":>10}\n')
    f.write('-' * 50 + '\n')
    for i in range(5):
        acc  = results['test_accuracy'][i]  * 100
        prec = results['test_precision'][i] * 100
        rec  = results['test_recall'][i]    * 100
        f1   = results['test_f1'][i]        * 100
        f.write(f'{i+1:<6} {acc:>9.2f}% {prec:>9.2f}% {rec:>9.2f}% {f1:>9.2f}%\n')
    f.write('-' * 50 + '\n')
    f.write(f'{"Avg":<6} '
            f'{results["test_accuracy"].mean()*100:>9.2f}% '
            f'{results["test_precision"].mean()*100:>9.2f}% '
            f'{results["test_recall"].mean()*100:>9.2f}% '
            f'{results["test_f1"].mean()*100:>9.2f}%\n')
    f.write(f'{"Std":<6} '
            f'{results["test_accuracy"].std()*100:>9.2f}% '
            f'{results["test_precision"].std()*100:>9.2f}% '
            f'{results["test_recall"].std()*100:>9.2f}% '
            f'{results["test_f1"].std()*100:>9.2f}%\n')

print('\nResults saved to results/cross_validation.txt')
