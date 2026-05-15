import numpy as np

# ── Load leaf features ────────────────────────────────────────────────────────
leaf_features  = np.load('features/train_features.npy')
leaf_filenames = np.load('features/train_filenames.npy')

# ── Load non-leaf features ────────────────────────────────────────────────────
nonleaf_features  = np.load('features/nonleaf_features.npy')
nonleaf_filenames = np.load('features/nonleaf_filenames.npy')

print('=== Before Merge ===')
print(f'Leaf features     : {leaf_features.shape}')
print(f'Non-leaf features : {nonleaf_features.shape}')

# ── Merge features ────────────────────────────────────────────────────────────
X = np.vstack([leaf_features, nonleaf_features])
filenames = np.concatenate([leaf_filenames, nonleaf_filenames])

# ── Create labels ─────────────────────────────────────────────────────────────
# 0 = leaf, 1 = not_leaf
y = np.array([0] * len(leaf_features) + [1] * len(nonleaf_features))

print('\n=== After Merge ===')
print(f'Combined features : {X.shape}')
print(f'Combined labels   : {y.shape}')
print(f'Class 0 (leaf)    : {np.sum(y == 0)}')
print(f'Class 1 (not_leaf): {np.sum(y == 1)}')

# ── Save merged dataset ───────────────────────────────────────────────────────
np.save('features/combined_features.npy',  X)
np.save('features/combined_labels.npy',    y)
np.save('features/combined_filenames.npy', filenames)

print('\nSaved to:')
print('  features/combined_features.npy')
print('  features/combined_labels.npy')
print('  features/combined_filenames.npy')
