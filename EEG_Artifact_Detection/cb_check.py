import numpy as np
y_val = np.load("data/val/Y.npy")
print(np.bincount(y_val.astype(int)))
y_train = np.load("data/train/Y.npy")
print(np.bincount(y_train.astype(int)))
print(np.unique(y_val, return_counts=True))