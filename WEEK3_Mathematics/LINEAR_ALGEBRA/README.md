# PREPARATION_PLAN

# 📐 Mathematics – Linear Algebra for AI (Week 3)

## 📌 Overview

This section introduces the fundamental concepts of linear algebra required for data science and artificial intelligence.

It is part of a preparation plan to strengthen skills in:

* Mathematical modeling
* Data representation
* Machine learning foundations
* Numerical computation

---

## 🎯 Learning Goal

The objective is to understand how data can be represented and manipulated using vectors and matrices.

You must be able to:

* Represent data using vectors
* Perform vector operations (addition, scalar multiplication, dot product)
* Understand vector norms (magnitude)
* Work with matrices
* Perform matrix operations (shape, indexing, transpose, multiplication)
* Understand linear regression
* Implement a simple AI learning model
* Apply gradient descent to optimize parameters
* Apply these concepts to real-world datasets (e.g., agriculture data)

---

## 🧠 Learning Objectives

This section helps you practice:

* Representing data mathematically
* Performing numerical computations using NumPy
* Understanding how AI models use vectors and matrices
* Manipulating datasets as matrices
* Understanding prediction models
* Implementing gradient descent
* Preparing for machine learning algorithms

---

## 🧩 Topics Covered

### 🔹 1. Vectors

* Definition:
  A vector is a list of numerical values.

* Example:
  Soil = [pH, humidity] → [6.5, 25]

---

### 🔹 2. Vector Operations

#### ➤ Addition

* Combine two vectors element-wise

Example:
[6, 25] + [7, 20] = [13, 45]

---

#### ➤ Scalar Multiplication

* Multiply a vector by a number

Example:
2 × [6, 25] = [12, 50]

---

#### ➤ Dot Product (VERY IMPORTANT)

* Formula:
  dot(A, B) = A₁×B₁ + A₂×B₂ + ...

* Example:
[6, 25] · [7, 20] = (6×7) + (25×20) = 42 + 500 = 542

---

### 🔹 Interpretation (AI)

👉 Dot product = weighted sum

Used to:

* Combine features
* Compute scores
* Make predictions

---

### 🔹 3. Vector Norm (Magnitude)

* Measures the size (length) of a vector

* Formula:
  ||v|| = √(x² + y²)

* Example:
||[6, 25]|| = √(6² + 25²)

---

### 🔹 4. Simple AI Model (Linear Combination)

* Formula:
  score = w₁×x₁ + w₂×x₂

* Example:
weights = [0.4, 0.6]
soil = [6.5, 25]

👉 score = dot(weights, soil)

---

### 🔹 Interpretation

* The model combines features using weights
* The result is a score used for decision-making

---

### 🔹 5. Matrices

* Definition:
  A matrix is a 2D array (table of numbers)

* Example:

[
 [6.5, 25],
 [7.2, 15],
 [5.8, 30]
]

---

### 🔹 6. Matrix Operations

#### ➤ Shape

* Gives number of rows and columns

Example:
(3, 2) → 3 rows, 2 columns

---

#### ➤ Indexing

* Access specific elements

Examples:
matrix[1, :] → second row  
matrix[:, 0] → first column  

---

#### ➤ Transpose

* Swap rows and columns

matrix.T

---

#### ➤ Matrix Multiplication

* Combine matrices mathematically

np.dot(A, B)

---

### 🔹 Practical Use (Data Science)

👉 Convert DataFrame into matrix:

df[["ph", "humidite"]].values

---

## 🔹 7. Linear Regression (VERY IMPORTANT)

### ➤ Definition

Linear regression is a machine learning method used to predict numerical values using input features.

---

### ➤ Goal

Find the best mathematical relationship between input variables and output values.

---

### ➤ Linear Regression Formula

y = w₁x₁ + w₂x₂ + b

Where:

* y = prediction
* x = input features
* w = weights
* b = bias

---

### 🔹 Example (Agriculture AI)

Inputs:

* pH
* humidity

Prediction:

* soil quality score
* crop yield
* fertility score

Example:

soil = [6.5, 25]
weights = [0.4, 0.6]
bias = 2

prediction = (0.4 × 6.5) + (0.6 × 25) + 2

---

### 🔹 Interpretation

👉 Weights determine feature importance.

* Large weight → strong influence
* Small weight → weak influence

---

## 🔹 8. Prediction Function

### ➤ Formula

prediction = np.dot(X, weights) + bias

---

### ➤ Purpose

Compute model predictions using:

* input data
* weights
* bias

---

## 🔹 9. Error (Loss)

### ➤ Definition

The error measures the difference between:

* predicted value
* real value

---

### ➤ Formula

error = y_pred - y

---

### 🔹 Interpretation

* error close to 0 → good prediction
* large error → poor prediction

---

## 🔹 10. Loss Function

### ➤ Mean Squared Error (MSE)

loss = mean(error²)

---

### 🔹 Purpose

Measure overall model performance.

---

### 🔹 Interpretation

* low loss → model performs well
* high loss → model performs poorly

---

## 🔹 11. Gradient Descent (VERY IMPORTANT)

### ➤ Definition

Gradient descent is an optimization algorithm used to reduce model error.

---

### ➤ Main Idea

The model learns by:

1. Predicting
2. Measuring error
3. Computing corrections
4. Updating parameters

---

### 🔹 Weight Update Formula

weights = weights - learning_rate × gradient

bias = bias - learning_rate × gradient

---

### 🔹 Interpretation

* gradient → direction of increasing error
* subtraction → move toward lower error

---

## 🔹 12. Learning Rate

### ➤ Definition

Controls the speed of learning.

---

### 🔹 Interpretation

* small learning rate → slow but stable learning
* large learning rate → fast but unstable learning

---

## 🔹 13. Gradients

### ➤ Definition

Gradients indicate how much weights and bias should change to reduce error.

---

### ➤ Formulas

dw = np.dot(X.T, error) / len(X)

db = error.mean()

---

### 🔹 Interpretation

👉 Gradients guide parameter corrections.

---

## 🔹 14. Feature Normalization

### ➤ Purpose

Normalize data to avoid instability during training.

---

### ➤ Formula

X = (X - mean) / std

---

### 🔹 Why Important?

Without normalization:

* gradients may become too large
* training may diverge

---

## 🔹 15. Training Loop

### ➤ Main Learning Process

for i in range(1000):

    prediction
    error
    gradients
    parameter updates

---

### 🔹 Interpretation

👉 The model improves progressively through iterations.

---

## 🔹 16. Practical AI Pipeline

### ➤ Complete Workflow

1. Load dataset
2. Clean missing values
3. Analyze correlations
4. Create target
5. Normalize data
6. Initialize weights
7. Predict
8. Compute error
9. Update parameters
10. Repeat learning process

---

### 🔹 Practical Use (Agriculture AI)

Using linear regression, you can:

* Predict crop yield
* Predict soil fertility
* Estimate irrigation needs
* Analyze soil quality
* Build simple AI systems

---

### 🔹 Why Linear Regression is Important in AI

Linear regression introduces:

* machine learning logic
* optimization
* gradient descent
* parameter learning
* prediction systems

👉 It is one of the first real AI models.

---

## 🧠 Key Insight

Linear Algebra + Statistics + Optimization = Foundation of Machine Learning

👉 In AI:

Data → Vectors  
Dataset → Matrix  
Prediction → Linear combination  
Learning → Gradient descent  

---

## 🚀 How to Practice

* Use NumPy:

  * np.array()
  * np.dot()
  * np.sqrt()
  * matrix.T
  * np.mean()

* Use Pandas:

  * df.values
  * df[["col1", "col2"]]
  * df.corr()

---

## 🧑‍💻 Author

Yann-Méric SOSSOU  
Computer Science Graduate | Future AI Engineer