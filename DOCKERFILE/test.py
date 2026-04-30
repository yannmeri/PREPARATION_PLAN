import numpy as np

# simulation simple
weights = np.array([0.2, 0.5, 0.3])
inputs = np.array([10, 5, 8])

result = np.dot(weights, inputs)

print("Résultat du modèle :", result)