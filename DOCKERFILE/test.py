import numpy as np

# données
X = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

# "entraînement" simple
weight = np.sum(y) / np.sum(X)

print("Poids appris :", weight)