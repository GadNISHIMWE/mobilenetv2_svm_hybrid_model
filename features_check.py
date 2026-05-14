import numpy as np
from numpy.linalg import norm

features  = np.load('features/train_features.npy')
filenames = np.load('features/train_filenames.npy')

print('=== Feature Quality Analysis ===')
print(f'Shape: {features.shape}')
print(f'Total feature dimensions per image: {features.shape[1]}')

print('\n--- Value Distribution ---')
print(f'Min   : {features.min():.4f}')
print(f'Max   : {features.max():.4f}')
print(f'Mean  : {features.mean():.4f}')
print(f'Std   : {features.std():.4f}')

print('\n--- Active vs Dead Features ---')
zero_per_feature = np.sum(features == 0, axis=0)
dead   = np.sum(zero_per_feature == features.shape[0])
active = features.shape[1] - dead
print(f'Dead features (always 0) : {dead}')
print(f'Active features          : {active}')
print(f'Activation rate          : {active / features.shape[1] * 100:.1f}%')

print('\n--- Feature Variance (how much each feature varies across images) ---')
variances = np.var(features, axis=0)
print(f'High variance features (>0.1) : {np.sum(variances > 0.1)}')
print(f'Low  variance features (<0.01): {np.sum(variances < 0.01)}')
print(f'Mean variance                 : {variances.mean():.4f}')

print('\n--- Image Similarity Check ---')
f0   = features[0]
f1   = features[1]
f100 = features[100]
f500 = features[500]
sim_close = np.dot(f0, f1)   / (norm(f0) * norm(f1))
sim_far   = np.dot(f0, f100) / (norm(f0) * norm(f100))
sim_far2  = np.dot(f0, f500) / (norm(f0) * norm(f500))
print(f'Similarity image[0] vs image[1]  : {sim_close:.4f}')
print(f'Similarity image[0] vs image[100]: {sim_far:.4f}')
print(f'Similarity image[0] vs image[500]: {sim_far2:.4f}')
print('(1.0 = identical, 0.0 = completely different)')

print('\n--- Feature Sparsity per Image ---')
zeros_per_image = np.sum(features == 0, axis=1)
print(f'Avg zeros per image : {zeros_per_image.mean():.1f} / {features.shape[1]}')
print(f'Max zeros in one img: {zeros_per_image.max()}')
print(f'Min zeros in one img: {zeros_per_image.min()}')

print('\n--- Conclusion ---')
if active > 1000 and variances.mean() > 0.01:
    print('Features look GOOD — MobileNetV2 extracted rich meaningful representations.')
    print('Edges, textures, shapes and semantic patterns are encoded in the 1280 dimensions.')
else:
    print('Features may need further inspection.')
