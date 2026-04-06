# PREPARATION_PLAN

# 🌱 Soil Analysis Mini Project (Python)

## 📌 Overview

This project is a simple Python application designed to analyze soil data and recommend suitable crops.

It is part of a preparation plan to strengthen skills in:
- Python programming  
- Data manipulation  
- Object-Oriented Programming (OOP)  
- File handling  

---

## 🎯 Project Goal

The program must be able to:

- Read a CSV file containing information about different soils  
- For each soil:
  - Classify the pH (**acidic, neutral, basic**)  
  - Check if the soil is fertile (based on pH and humidity)  
  - Recommend suitable crops depending on the soil type  
- Save all results into a text file (`resultats.txt`)  
- Display a summary of fertile and non-fertile soils  

---

## 🧠 Learning Objectives

This mini project helps you practice:

- Functions  
- Classes and Object-Oriented Programming  
- File handling (CSV and TXT)  
- Python logic applied to real-world data  

---

## 🏗️ Project Structure
├── mini_projet_sols_csv.csv
├── read_csv_file.py
├── main.py
├── resultats.txt
└── README.md


---

## ⚙️ Instructions

### 1. Create a Class `Sol`

#### 🔹 Attributes:
- `type` (soil type)  
- `pH`  
- `humidity`  

#### 🔹 Methods:
- `classifier_ph()` → returns `"acidic"`, `"neutral", or "basic"`  
- `est_fertile()` → returns `True` if `6 ≤ pH ≤ 7` and `humidity ≥ 20`  
- `recommander_culture()` → returns suitable crops based on soil type  

---

### 2. Read the CSV File

- Load the CSV file using Pandas  
- Create a `Sol` object for each row  

---

### 3. Process Each Soil

- Apply all methods  
- Store results  

---

### 4. Save Results

Write the results into a file called `resultats.txt`.

#### ✅ Example Output:

Sol argileux (ph 6.5, humidite 25.0) -> neutre, fertile soil, culture proposed: ['rice', 'cabbage', 'spinash', 'beans', 'fruits trees (orange tree, mango tree, apple tree)']
Sol sableux (ph 7.5, humidite 15.0) -> basique, infertile soil, culture proposed: ['peanut', 'cassava', 'watermelon', 'sweet potatoes', 'carrot']


---

### 5. Display Summary

- Number of fertile soils  
- Number of non-fertile soils  

#### ✅ Example Output:
There are 3 fertile soils and 2 infertile soils.

---

## 🚀 How to Run the Project

1. Install dependencies:
pip install pandas

2. Run your program:
python type_of_soil.py

---

## 🧑‍💻 Author

Yann-Méric SOSSOU  
Computer Science Graduate | Future AI Engineer  