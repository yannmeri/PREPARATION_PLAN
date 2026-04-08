# 📊 NumPy & Pandas Practice Exercises

## 📌 Overview

This project contains practical exercises to build a strong foundation in **NumPy** and **Pandas** for data analysis.

---

## 🎯 Learning Objectives

- Manipulate NumPy arrays  
- Perform statistical operations (mean, variance)  
- Work with 2D data  
- Load and explore datasets using Pandas  
- Filter and transform data  
- Apply functions to data  
- Analyze and summarize results  

---

## 🧠 Topics Covered

### 🔹 NumPy
- Array creation (1D, 2D)  
- Mathematical operations  
- Statistical functions  
- Indexing  

### 🔹 Pandas
- Reading CSV files  
- DataFrame basics  
- `head()`, `describe()`  
- Data filtering  
- `apply()` function  
- Creating new columns  
- Counting and aggregating data  

---

## 🏗️ Exercises

### 🟢 Exercise 1: NumPy 1D Array
- Create an array:  
  ```python
  [5, 10, 15, 20]
- Compute:
    - Mean
    - Variance
    - Multiply all elements by 2

---

### 🟢 Exercise 2: NumPy 2D Array

#### Goal: Understand how to work with matrices and indexing.

#### Tasks:
- Create a matrix:
    - [[1, 2, 3],
        [4, 5, 6]]
- Display:
    - The shape of the array
    - The value 5 using indexing

---

### 🟡 Exercise 3: Data Exploration with Pandas

#### Goal: Explore and understand a dataset.

#### Tasks:
- Load a CSV file into a DataFrame
- Display:
    - First rows using head()
    - Dataset statistics using describe()
    - Filter data where pH > 6

---

### 🟡 Exercise 4: Data Transformation

#### Goal: Transform and enrich the dataset.

#### Tasks:
- Create a function to classify pH:
    - if pH < 6:
        return "acid"
      elif pH > 7:
        return "basic"
      else:
        return "neutral"
- Apply the function using apply()
- Add a new column: ph_type

---

### 🔴 Exercise 5: Fertility Analysis

#### Goal: Analyze soil fertility based on given conditions.

#### Tasks:
- Create a function to determine fertility:
    - return 6 <= pH <= 7 and humidity >= 20
- Add a new column: fertility
- Count:
    - Number of fertile soils
    - Number of infertile soils

---

## ⚙️ Technologies
- Python
- NumPy
- Pandas

---

## 🚀 How to Run
- pip install numpy pandas
- python your_script.py

---

## 🧑‍💻 Author

Yann-Méric SOSSOU
Computer Science Graduate | Future AI Engineer