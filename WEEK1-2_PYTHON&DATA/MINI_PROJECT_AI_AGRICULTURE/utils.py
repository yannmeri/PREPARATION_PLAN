import pandas as pd
import numpy as np

# Fix seed for reproducibility
np.random.seed(42)

# Number of rows
n = 500

# Possible soil types
soil_types = ["argileux", "sableux", "limoneux"]

# Generate data
data = {
    "type": np.random.choice(soil_types, n),
    "ph": np.round(np.random.uniform(5.0, 8.0, n), 2),
    "humidite": np.random.randint(10, 35, n)
}

df = pd.DataFrame(data)

# Introduce missing values randomly
for col in ["ph", "humidite"]:
    df.loc[df.sample(frac=0.1).index, col] = np.nan  # 10% missing

# Save to CSV
df.to_csv("soil_dataset.csv", index=False)

print("Dataset generated successfully!")