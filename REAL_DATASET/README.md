# 📊 Real Dataset Manipulation Exercises

## 📌 Overview

This project contains practical exercises to learn how to work with a real dataset using Python.

It focuses on loading, exploring, cleaning, transforming, and analyzing data.

---

## 🎯 Learning Objectives

- Load a real dataset from a CSV file  
- Explore data using Pandas functions  
- Detect and handle missing values  
- Transform data by creating new columns  
- Filter and analyze data efficiently  
- Compute statistics from real data  
- Visualize relationships between variables  

---

## 🧠 Topics Covered

### 🔹 Data Handling (Pandas)
- Reading CSV files  
- Data exploration: `head()`, `info()`, `describe()`  
- Handling missing values (`isnull()`, `dropna()`)  
- Filtering data  

### 🔹 Data Transformation
- Creating new columns  
- Using `apply()` function  
- Logical conditions on data  

### 🔹 Data Analysis
- Mean calculation  
- Counting values  
- Boolean operations  

### 🔹 Data Visualization (Matplotlib)
- Scatter plot  
- Graph labeling (title, axes)  

---

## 🏗️ Exercises

### 🟢 Exercise 1: Data Exploration

#### Goal: Understand the dataset structure.

#### Tasks:
- Load the CSV file into a DataFrame  
- Display:
  - First rows using `head()`  
  - Dataset structure using `info()`  
  - Statistics using `describe()`  

---

### 🟢 Exercise 2: Data Cleaning

#### Goal: Handle missing data.

#### Tasks:
- Check missing values using:
  - `df.isnull().sum()`  
- Remove missing values using:
  - `dropna()`  

---

### 🟡 Exercise 3: Data Transformation

#### Goal: Enrich the dataset with new information.

#### Tasks:
- Create a function to classify pH:
  - if pH < 6 → "acid"  
  - if pH > 7 → "basic"  
  - else → "neutral"  
- Apply the function using `apply()`  
- Create a new column: `ph_type`  

---

### 🟡 Exercise 4: Fertility Analysis

#### Goal: Identify fertile soils.

#### Tasks:
- Define fertility condition:
  - 6 ≤ pH ≤ 7 and humidity ≥ 20  
- Create a new column: `fertility`  
- Filter:
  - Fertile soils  
  - Infertile soils  

---

### 🔴 Exercise 5: Statistical Analysis

#### Goal: Extract useful insights from the dataset.

#### Tasks:
- Compute:
  - Mean pH  
  - Mean humidity  
- Count:
  - Number of neutral soils  

---

### 🔴 Exercise 6: Data Visualization

#### Goal: Visualize relationships between variables.

#### Tasks:
- Create a scatter plot:
  - X-axis: pH  
  - Y-axis: humidity  
- Add:
  - Title  
  - Axis labels  
  - Grid  

---

## ⚙️ Technologies

- Python  
- Pandas  
- NumPy  
- Matplotlib  

---

## 🚀 How to Run

- pip install pandas numpy matplotlib  
- python your_script.py  

---

## 🧑‍💻 Author

Yann-Méric SOSSOU  
Computer Science Graduate | Future AI Engineer

