# 🌱 Soil Data Visualization (Python & Matplotlib)

## 📌 Overview

This project focuses on visualizing soil data using Python and Matplotlib.
It is part of your preparation plan to strengthen skills in:

- Python programming
- Data visualization
- Data analysis
- Working with CSV files using Pandas

---

## 🎯 Project Goal

The program enables you to:

- Read soil data from a CSV file (mini_projet_sols_csv.csv)
- Visualize relationships between soil properties, such as pH and humidity
- Plot line graphs, scatter plots, and histograms
- Analyze and interpret soil characteristics based on graphs

---

## 🧠 Learning Objectives

After completing this project, you will be able to:

- Create line graphs, scatter plots, and histograms using Matplotlib
- Label axes, add titles, and include grids for clarity
- Read CSV data using Pandas
- Analyze trends and patterns in soil data
- Write clear observations based on visualizations

---

## 🏗️ Project Structure

├── mini_projet_sols_csv.csv
├── read_csv_file.py
├── line_plot.py
├── scatter_plot.py
├── histogram.py
└── README.txt

---

## ⚙️ Instructions & Exercises

### 🟢 Exercise 1: Line Graph

#### Goal: Visualize the trend of data as a line graph.

- Create a line graph with sample X and Y values.
- Add title, axis labels, and grid.

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
create_graph(x, y)
```


---

### 🟢 Exercise 2: Scatter Plot (pH vs Humidity)

#### Goal: Identify trends between pH and humidity in soils.

- Show the relationship between soil pH and humidity using a scatter plot.
- Label axes and add a title.

```python
ph = [5.5, 6.2, 6.8, 7.5, 6.0]
humidity = [15, 22, 30, 10, 25]
relation_ph_humidity(ph, humidity)
```


Observation:
- Soils with pH between 6 and 7 tend to have higher humidity.
- When pH exceeds 7, humidity tends to decrease.

---

### 🟢 Exercise 3: Histogram of pH Values

#### Goal: Understand how soil pH values are distributed.

- Create a histogram to display the distribution of pH values.

```python
ph_values = [5.5, 6.2, 6.8, 7.5, 6.0, 6.3, 5.8]
create_histogram(ph_values)
```


Observation:
- Most soils are neutral.
- Some soils are acidic, and a few may be slightly basic.

---

### 🟢 Exercise 4: Combining Visualizations

#### Goal: Practice combining multiple visualization techniques for better insights.

- Read soil data from CSV using Pandas.
- Visualize the data using:
  - Histogram
  - Scatter plot
  - Line plot

```python
df = read_csv_file()
create_histogram(df["ph"])
relation_ph_humidity(df["ph"], df["humidite"])
line_plot_ph(df["ph"])
```

---

## 🚀 How to Run the Project

1. Install dependencies:

```bash
pip install pandas matplotlib
```

2. Run the visualization scripts:

```bash
python read_csv_file.py
python line_plot.py
python scatter_plot.py
python histogram.py
```

---

## 🧑‍💻 Author

Yann-Méric SOSSOU
Computer Science Graduate | Future AI Engineer

