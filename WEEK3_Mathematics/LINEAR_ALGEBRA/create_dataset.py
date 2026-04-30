import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

data = {
    "ph": np.random.uniform(4.5, 8.5, n),           # pH entre acide et basique
    "humidite": np.random.uniform(5, 40, n),        # humidité %
    "type": np.random.choice(["argileux", "sableux", "limoneux"], n)
}

df = pd.DataFrame(data)

# Ajouter des valeurs manquantes (réalisme)
for col in ["ph", "humidite"]:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# Sauvegarde
df.to_csv("dataset.csv", index=False)

print("Dataset créé !")