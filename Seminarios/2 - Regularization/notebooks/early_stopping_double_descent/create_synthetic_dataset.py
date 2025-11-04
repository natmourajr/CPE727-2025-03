#!/usr/bin/env python3
import numpy as np

# Create a synthetic CIFAR-like dataset
# CIFAR-10 has 32x32 RGB images with 10 classes
# We'll create a smaller version for testing

# Parameters
n_train = 1000  # Small training set
n_test = 200   # Small test set
img_size = 32
n_channels = 3
n_classes = 10

# Generate random images (normalized to 0-255 range)
X_train = np.random.randint(0, 256, (n_train, img_size, img_size, n_channels), dtype=np.uint8)
X_test = np.random.randint(0, 256, (n_test, img_size, img_size, n_channels), dtype=np.uint8)

# Generate random labels
y_train = np.random.randint(0, n_classes, n_train, dtype=np.int64)
y_test = np.random.randint(0, n_classes, n_test, dtype=np.int64)

# Save as npz file
np.savez('datasets/cifar.npz', 
         X_train=X_train, 
         y_train=y_train,
         X_test=X_test, 
         y_test=y_test)

print(f"Created synthetic CIFAR dataset:")
print(f"  Training: {X_train.shape} images, {y_train.shape} labels")
print(f"  Test: {X_test.shape} images, {y_test.shape} labels")
print(f"  Classes: {n_classes}")
print(f"  Saved to: datasets/cifar.npz")
