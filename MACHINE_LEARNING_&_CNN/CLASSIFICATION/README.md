# 📐 Mathematics – Classification in AI (Week 5)

## 📌 Overview

This chapter focuses on understanding classification problems in Artificial Intelligence, particularly Logistic Regression.

It builds on linear algebra and optimization concepts to explain how models predict probabilities and make decisions.

---

## 🎯 Learning Goal

The objective is to understand how an AI model performs classification using:

- Probability estimation
- Sigmoid function
- Log Loss (Cross-Entropy)
- Gradient descent

You must be able to:

- Understand classification vs regression
- Compute probabilities using sigmoid
- Interpret predictions as classes
- Compute log loss
- Derive gradients for classification
- Implement Logistic Regression from scratch
- Debug instability (NaN, overflow)

---

## 🧠 Core Concepts

---

## 🔹 1. Classification vs Regression

### ➤ Difference

Regression → Predict continuous values  
Classification → Predict categories (0 or 1)

---

### 🔹 Example

Soil data:

[ph, humidity] → fertile (1) or not (0)

---

## 🔹 2. Linear Model (Same as Regression)

### ➤ Formula

z = Xw + b

---

### ➤ Interpretation

- X = features
- w = weights
- b = bias
- z = linear score

---

## 🔹 3. Sigmoid Function (VERY IMPORTANT)

### ➤ Formula

σ(z) = 1 / (1 + e^(-z))

---

### ➤ Output

Range: (0, 1)

---

### 🔹 Interpretation

👉 Converts score into probability

---

## 🔹 4. Prediction

### ➤ Formula

y_pred = σ(Xw + b)

---

### ➤ Interpretation

- y_pred ≈ 1 → class 1
- y_pred ≈ 0 → class 0

---

## 🔹 5. Decision Rule

### ➤ Threshold

if y_pred ≥ 0.5 → class 1  
else → class 0  

---

## 🔹 6. Error (Vector)

### ➤ Formula

error = y_pred - y

---

### 🔹 Important

Each sample has its own error

---

## 🔹 7. Loss Function (Log Loss)

### ➤ Formula

J = -(1/n) Σ [ y log(y_pred) + (1 - y) log(1 - y_pred) ]

---

### ➤ Interpretation

- Penalizes wrong confident predictions
- Works with probabilities

---

### 🔹 Key Insight

error → vector  
loss → scalar  

---

## 🔹 8. Why Not MSE?

### ➤ Problem

- Not suitable for probabilities
- Slow learning
- Poor gradients

---

### ➤ Solution

👉 Use Log Loss

---

## 🔹 9. Gradient (VERY IMPORTANT)

### ➤ Formulas

dw = (1/n) Xᵀ (y_pred - y)

db = mean(y_pred - y)

---

### ➤ Interpretation

- Same structure as linear regression
- Comes from log loss derivative

---

## 🔹 10. Gradient Descent

### ➤ Update Rule

w = w - learning_rate × dw  
b = b - learning_rate × db  

---

### ➤ Learning Process

1. Compute z
2. Apply sigmoid
3. Compute error
4. Compute loss
5. Update parameters

---

## 🔹 11. Numerical Stability (IMPORTANT)

### ➤ Problem

log(0) → undefined → NaN

---

### ➤ Solution

Clip predictions:

y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

---

---

## 🔹 12. Shape Consistency

### ➤ Required Shapes

X → (n, features)  
w → (features, 1)  
y → (n, 1)  
y_pred → (n, 1)

---

---

## 🔹 13. Training Loop

### ➤ Structure

for i in range(1000):

    z = Xw + b  
    y_pred = sigmoid(z)  

    error = y_pred - y  

    loss = log_loss  

    dw = Xᵀ error / n  
    db = mean(error)  

    update w, b  

---

---

## 🧩 Exercises

---

## 🟢 Exercise 1: Sigmoid

Compute:

σ(0), σ(2), σ(-2)

---

## 🟢 Exercise 2: Prediction

Given:

z = [0, 2, -1]

👉 Compute y_pred

---

## 🟡 Exercise 3: Classification

Given:

y_pred = [0.7, 0.3, 0.8]

👉 Convert into classes

---

## 🟡 Exercise 4: Log Loss

Given:

y = [1, 0]  
y_pred = [0.9, 0.2]

👉 Compute loss

---

## 🔴 Exercise 5: Debugging

Explain why this gives NaN:

y_pred = [0, 1]

👉 How to fix it?

---

## 🔴 Exercise 6: Shapes

Given:

X.shape = (200, 2)

👉 What are shapes of:

- w ?
- y ?
- y_pred ?

---

---

## 🚀 Key Takeaways

- Classification predicts probabilities
- Sigmoid converts scores into probabilities
- Log Loss measures performance
- Gradient descent optimizes parameters
- Shapes must match
- Numerical stability is critical

---

## 🧠 Final Insight

Classification = Linear Model + Sigmoid + Log Loss

👉 Score → Probability  
👉 Probability → Decision  
👉 Learning → Gradient Descent  

---

## 🧑‍💻 Author

Yann-Méric SOSSOU  
Computer Science Graduate | Future AI Engineer