from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API IA active 🚀"}

@app.get("/predict")
def predict(x: float, y: float):

    weights = np.array([0.4, 0.6])
    inputs = np.array([x, y])

    result = np.dot(weights, inputs)

    return {"prediction": float(result)}