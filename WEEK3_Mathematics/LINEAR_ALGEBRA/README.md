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
* Apply these concepts to real-world datasets (e.g., agriculture data)

---

## 🧠 Learning Objectives

This section helps you practice:

* Representing data mathematically
* Performing numerical computations using NumPy
* Understanding how AI models use vectors and matrices
* Manipulating datasets as matrices
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

### 🔹 Why Linear Algebra is Important in AI

* Data is represented as vectors
* Datasets are matrices
* Models use dot products
* Neural networks rely heavily on matrix operations

---

## 🧪 Practical Applications (Agriculture AI)

Using these concepts, you can:

* Represent soil data as vectors
* Build simple prediction models
* Analyze multiple soil samples as matrices
* Compute relationships between variables
* Prepare for machine learning algorithms

---

## 🧠 Key Insight

Linear Algebra = Language of AI

👉 In AI:
Data → Vectors  
Dataset → Matrix  
Model → Mathematical operations  

---

## 🚀 How to Practice

* Use NumPy:

  * np.array()
  * np.dot()
  * np.sqrt()
  * matrix.T

* Use Pandas:

  * df.values
  * df[["col1", "col2"]]

---

## 🧑‍💻 Author

Yann-Méric SOSSOU  
Computer Science Graduate | Future AI Engineer