import numpy as np
import csv

# ── Load saved filenames ─────────────────────────────────────────────────────
filenames = np.load('features/train_filenames.npy')
print(f'Total images: {len(filenames)}')

# ── Assign labels based on image number ──────────────────────────────────────
# Class 0 → large images (1024x1024): LEAF_0009–0070, LEAF_1120–1178
# Class 1 → small images (416x416) : LEAF_0100–1115

labels = []
for fname in filenames:
    number = int(fname.replace('LEAF_', '').replace('.jpg', ''))
    if (9 <= number <= 70) or (1120 <= number <= 1178):
        labels.append(0)
    else:
        labels.append(1)

labels = np.array(labels)

# ── Save labels ───────────────────────────────────────────────────────────────
np.save('features/train_labels.npy', labels)

with open('features/train_labels.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label', 'class_name'])
    for fname, label in zip(filenames, labels):
        class_name = 'large_leaf' if label == 0 else 'small_leaf'
        writer.writerow([fname, label, class_name])

# ── Summary ───────────────────────────────────────────────────────────────────
class0 = np.sum(labels == 0)
class1 = np.sum(labels == 1)

print('\n=== Label Summary ===')
print(f'Class 0 (large_leaf) : {class0} images')
print(f'Class 1 (small_leaf) : {class1} images')
print(f'Total                : {len(labels)} images')

print('\n=== Sample Mapping (first 5 and last 5) ===')
print(f'{"Filename":<20} {"Label":<8} {"Class"}')
print('-' * 40)
for fname, label in zip(filenames[:5], labels[:5]):
    print(f'{fname:<20} {label:<8} {"large_leaf" if label == 0 else "small_leaf"}')
print('...')
for fname, label in zip(filenames[-5:], labels[-5:]):
    print(f'{fname:<20} {label:<8} {"large_leaf" if label == 0 else "small_leaf"}')

print('\nLabels saved to features/train_labels.npy and features/train_labels.csv')
