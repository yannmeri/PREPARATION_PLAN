# PREPARATION_PLAN

# 🌱 Soil Analysis & Smart Recommendation Project (Python)

## 📌 Overview

This project is a Python application designed to analyze soil data, evaluate soil quality, and provide intelligent recommendations for agriculture.

It is part of a preparation plan to strengthen skills in:
- Python programming  
- Data analysis (NumPy, Pandas)  
- Data visualization (Matplotlib)  
- Real dataset manipulation  
- Problem-solving and logical thinking  

---

## 🎯 Project Goal

The program must be able to:

- Read a CSV file containing soil data  
- Clean the dataset by handling missing values  
- Analyze and transform the dataset  
- For each soil:
  - Classify the pH (**acidic, neutral, basic**)  
  - Determine fertility (based on pH and humidity)  
  - Recommend suitable crops  
  - Suggest improvements if the soil is not optimal  
- Visualize data using graphs  
- Compute statistics and correlations  
- Save results into a text file (`results.txt`)  

---

## 🧠 Learning Objectives

This project helps you practice:

- Data loading and cleaning  
- Data transformation using Pandas  
- Statistical analysis using NumPy  
- Data visualization with Matplotlib  
- Writing clean and structured Python code  
- Applying logic to real-world agricultural problems  

---

## 🏗️ Project Structure

soil_project/
│
├── soil_dataset.csv
├── analysis.py
├── visualization.py
├── main.py
├── results.txt
└── README.txt

---

## ⚙️ Instructions

### 1. Load and Explore the Dataset

- Use Pandas to read the CSV file  
- Display:
  - First rows (`head()`)  
  - Dataset info (`info()`)  
  - Statistics (`describe()`)  

---

### 2. Clean the Dataset

- Detect missing values using `isnull()`  
- Remove or handle missing values using `dropna()`  

---

### 3. Transform the Dataset

- Create a new column `ph_type`:
  - pH < 6 → "acid"  
  - 6 ≤ pH ≤ 7 → "neutral"  
  - pH > 7 → "basic"  

- Create a new column `fertility`:
  - "fertile" if 6 ≤ pH ≤ 7 and humidity ≥ 20  
  - "infertile" otherwise  

---

### 4. Data Analysis

- Compute:
  - Mean pH  
  - Mean humidity  
  - Number of fertile soils  
  - Number of each soil type  

- Analyze correlation between pH and humidity using:
  df.corr()

---

### 5. Data Visualization

Create the following graphs:

- Histogram of pH values  
- Scatter plot (pH vs humidity)  
- Line plot (pH evolution)  
- (Optional) Boxplot for distribution analysis  

---

### 6. Crop Recommendation

Based on soil type:

- Clay soil → rice, cabbage, beans  
- Sandy soil → peanut, cassava  
- Loamy soil → maize, tomatoes  

---

### 7. Soil Improvement Suggestions (AI Logic)

If soil conditions are not optimal:

- pH < 6 → Add lime (reduce acidity)  
- pH > 7 → Add organic matter  
- Humidity < 20 → Increase irrigation  
- Humidity > 30 → Improve drainage  

---

### 8. Save Results

Write results into `results.txt`.

Each entry should include:

- Soil type  
- pH value  
- Humidity  
- Fertility status  
- Recommended crops  
- Suggested improvements  

---

## 🚀 How to Run the Project

1. Install dependencies:

pip install pandas numpy matplotlib

2. Run the program:

python main.py

---

## 🧑‍💻 Author

Yann-Méric SOSSOU  
Computer Science Graduate | Future AI Engineer