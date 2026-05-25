# 📐 Mathematics – Linear Algebra for AI (Week 4)

## 📌 Overview

This chapter focuses on understanding the mathematical foundations behind Linear Regression and Gradient Descent.

It builds on vector and matrix operations to explain how AI models learn from data.

---

## 🎯 Learning Goal

The objective is to understand how an AI model learns using:

- Matrix operations
- Predictions
- Error computation
- Cost function
- Gradient descent

You must be able to:

- Perform matrix-vector multiplication
- Understand prediction using linear models
- Compute error and loss
- Understand gradients mathematically
- Implement gradient descent
- Handle shape issues in NumPy
- Debug training instability (NaN, divergence)

---

## 🧠 Core Concepts

---

## 🔹 1. Matrix × Vector Multiplication (VERY IMPORTANT)

### ➤ Rule

Multiply each row of the matrix by the vector.

### ➤ Example

A = [[1, 2],
     [3, 4]]

x = [[5],
     [6]]

Result:

[1×5 + 2×6] = 17  
[3×5 + 4×6] = 39  

---

### 🔹 Key Insight

👉 Matrix × vector = dot product per row

---

## 🔹 2. Shapes in Machine Learning

### ➤ Important Shapes

X → (n, features)  
w → (features, 1)  
y → (n, 1)

---

### ➤ Rule

(number of columns of X) = (number of rows of w)

---

## 🔹 3. Prediction Model

### ➤ Formula

y_pred = Xw + b

---

### ➤ Interpretation

- X = data
- w = weights
- b = bias
- y_pred = prediction

---

## 🔹 4. Error Vector

### ➤ Formula

error = y_pred - y

---

### ➤ Important

👉 Error is a vector

Each data point has its own error.

---

## 🔹 5. Loss Function (MSE)

### ➤ Formula

loss = mean(error²)

---

### ➤ Interpretation

- Converts vector → scalar
- Measures global performance

---

### 🔹 Key Insight

error → vector  
loss → scalar  

---

## 🔹 6. Why Use Transpose?

### ➤ Expression

errorᵀ error

---

### ➤ Purpose

- Converts vector → scalar
- Computes sum of squared errors

---

## 🔹 7. Gradient (VERY IMPORTANT)

### ➤ Formulas

dw = (1/n) Xᵀ (y_pred - y)

db = mean(y_pred - y)

---

### ➤ Interpretation

- Gradient shows direction of error increase
- We move in opposite direction

---

## 🔹 8. Gradient Descent

### ➤ Update Rule

w = w - learning_rate × dw  
b = b - learning_rate × db  

---

### ➤ Learning Process

1. Predict
2. Compute error
3. Compute gradient
4. Update parameters

---

## 🔹 9. Learning Rate

### ➤ Role

Controls step size during learning

---

### ➤ Effects

Small → slow but stable  
Large → fast but unstable (NaN possible)

---

## 🔹 10. Why NaN Happens

### ➤ Causes

- Learning rate too high
- Values become too large
- Overflow → NaN

---

### ➤ Solution

- Reduce learning rate
- Normalize data

---

## 🔹 11. Feature Normalization

### ➤ Formula

X = (X - mean) / std

---

### ➤ Purpose

- Stabilize training
- Prevent large gradients

---

## 🔹 12. Reshape (NumPy)

### ➤ Example

y = y.reshape(-1, 1)

---

### ➤ Why?

- Ensure correct matrix operations
- Avoid shape errors

---

## 🔹 13. Vector vs Scalar

Vector → multiple values  
Scalar → single value  

---

### ➤ Example

error → vector  
loss → scalar  

---

## 🔹 14. Training Loop

### ➤ Structure

for i in range(1000):

    y_pred = Xw + b
    error = y_pred - y
    loss = mean(error²)

    dw = Xᵀ error / n
    db = mean(error)

    update w, b

---

## 🧩 Exercises

---

## 🟢 Exercise 1: Matrix Multiplication

Given:

A = [[2, 3],
     [4, 5]]

x = [[1],
     [2]]

👉 Compute Ax manually

---

## 🟢 Exercise 2: Prediction

Given:

X = [[1, 2],
     [3, 4]]

w = [[0.5],
     [1.0]]

b = 1

👉 Compute y_pred

---

## 🟡 Exercise 3: Error and Loss

Given:

y_pred = [10, 12, 8]  
y = [9, 11, 7]

👉 Compute:

- error
- squared error
- loss (MSE)

---

## 🟡 Exercise 4: Gradient

Given:

X = [[1, 2],
     [3, 4]]

error = [[1],
         [2]]

👉 Compute:

- Xᵀ
- dw = Xᵀ × error

---

## 🔴 Exercise 5: Debugging

Explain why this produces NaN:

learning_rate = 1

👉 How to fix it?

---

## 🔴 Exercise 6: Shapes

Given:

X.shape = (100, 2)

👉 What should be the shape of:

- w ?
- y ?
- y_pred ?

---

## 🚀 Key Takeaways

- Matrix operations are the foundation of AI
- Error is a vector, loss is a scalar
- Gradient guides learning
- Learning rate controls stability
- Normalization prevents divergence
- Shape consistency is critical

---

## 🧠 Final Insight

Machine Learning = Linear Algebra + Optimization

👉 Data → Matrix  
👉 Model → Equation  
👉 Learning → Gradient Descent  

---

## 🧑‍💻 Author

Yann-Méric SOSSOU  
Computer Science Graduate | Future AI Engineer