===============================================================================
HOUSE PRICES PREDICTION PROJECT
Linear Regression from Scratch with Docker
===============================================================================

Author: Yann-Méric SOSSOU
Goal: Predict house prices using a Linear Regression model implemented
from scratch with NumPy and Pandas.

===============================================================================
1. PROJECT OVERVIEW
===============================================================================

This project implements a complete Machine Learning pipeline without using
Scikit-Learn's LinearRegression model.

The model is trained using Gradient Descent and is designed to solve the
House Prices prediction problem.

The workflow includes:

- Loading datasets;
- Encoding categorical variables;
- Handling missing values;
- Feature selection using correlation;
- Splitting data into training and validation sets;
- Feature normalization;
- Training Linear Regression from scratch;
- Evaluating the model using R² Score;
- Saving learned parameters;
- Generating predictions for Kaggle submission.

The project is fully containerized using Docker and Docker Compose.

===============================================================================
2. PROJECT STRUCTURE
===============================================================================

project/

│
├── house_prices_prediction.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── save_weights.csv
├── save_bias.csv
├── sample_submission.csv
│
└── house-prices-advanced-regression-techniques/
    ├── train.csv
    └── test.csv

===============================================================================
3. REQUIRED LIBRARIES
===============================================================================

The project uses the following Python libraries:

- numpy
- pandas
- scikit-learn

requirements.txt:

numpy
pandas
scikit-learn

===============================================================================
4. DOCKER CONFIGURATION
===============================================================================

Dockerfile:

FROM python:3.12.7

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "house_prices_prediction.py"]

-------------------------------------------------------------------------------

docker-compose.yml:

services:
  house_prices_prediction:
    build: .
    volumes:
      - .:/app
    environment:
      - MODE_PROJECT=predict

===============================================================================
5. EXECUTION MODES
===============================================================================

The program supports two execution modes controlled through the variable:

MODE_PROJECT

Possible values:

1. train
2. predict

-------------------------------------------------------------------------------
TRAIN MODE
-------------------------------------------------------------------------------

MODE_PROJECT=train

Purpose:

- Train the Linear Regression model;
- Learn optimal weights and bias;
- Evaluate the model;
- Save parameters into CSV files.

Generated files:

save_weights.csv
save_bias.csv

-------------------------------------------------------------------------------
PREDICT MODE
-------------------------------------------------------------------------------

MODE_PROJECT=predict

Purpose:

- Load previously saved parameters;
- Predict prices on the Kaggle test dataset;
- Generate submission file.

Generated file:

sample_submission.csv

===============================================================================
6. HOW TO RUN THE PROJECT
===============================================================================

STEP 1: Clone or download the project.

STEP 2: Place Kaggle datasets inside:

house-prices-advanced-regression-techniques/

Required files:

train.csv
test.csv

STEP 3: Choose execution mode.

Inside docker-compose.yml:

environment:
  - MODE_PROJECT=train

or

environment:
  - MODE_PROJECT=predict

STEP 4: Build and run.

Command:

docker-compose up --build

===============================================================================
7. MACHINE LEARNING PIPELINE
===============================================================================

The program follows the following sequence.

-------------------------------------------------------------------------------
STEP 1: Load datasets
-------------------------------------------------------------------------------

Function:

load_datasets()

Purpose:

Load:

- train.csv
- test.csv

Output:

df_train
df_test

-------------------------------------------------------------------------------
STEP 2: Encode categorical variables
-------------------------------------------------------------------------------

Function:

encode_categorical_features()

Purpose:

Convert categorical variables into numerical values.

Methods used:

A) Binary Encoding

Examples:

Street
Alley
CentralAir

B) Ordinal Encoding

Examples:

ExterQual
KitchenQual
GarageQual

C) One-Hot Encoding

Examples:

Neighborhood
HouseStyle
RoofStyle

-------------------------------------------------------------------------------
STEP 3: Handle missing values
-------------------------------------------------------------------------------

Function:

clean_dataframe()

Purpose:

Remove remaining NaN values.

Method:

dropna()

-------------------------------------------------------------------------------
STEP 4: Feature Selection
-------------------------------------------------------------------------------

Function:

compute_top_correlations()

Purpose:

Identify features strongly related to SalePrice.

Method:

Pearson Correlation.

Formula:

corr(feature, SalePrice)

Selection criterion:

Absolute correlation ≥ 0.30

-------------------------------------------------------------------------------
STEP 5: Remove weak features
-------------------------------------------------------------------------------

Function:

drop_low_correlation_features()

Purpose:

Keep only useful predictors.

Result:

Reduced dimensionality.

-------------------------------------------------------------------------------
STEP 6: Train/Validation Split
-------------------------------------------------------------------------------

Function:

train_val_split()

Purpose:

Evaluate model performance on unseen data.

Split ratio:

80% → Training

20% → Validation

-------------------------------------------------------------------------------
STEP 7: Separate Features and Target
-------------------------------------------------------------------------------

Function:

separate_features_and_target()

Purpose:

Extract:

X → Features

y → SalePrice

Shapes:

X_train → (n_samples, n_features)

y_train → (n_samples, 1)

-------------------------------------------------------------------------------
STEP 8: Feature Normalization
-------------------------------------------------------------------------------

Function:

fit_transform_scaling()

Purpose:

Standardize training data.

Formula:

X_norm = (X − mean) / std

Why?

- Stabilize Gradient Descent;
- Avoid very large gradients;
- Improve convergence speed.

Important:

Mean and standard deviation are computed ONLY on training data.

-------------------------------------------------------------------------------
STEP 9: Normalize Validation and Test Data
-------------------------------------------------------------------------------

Function:

transform_scaling_val_test()

Purpose:

Apply the SAME transformation.

Formula:

X_test_norm = (X_test − train_mean) / train_std

Important:

Never recompute statistics on validation or test datasets.

===============================================================================
8. LINEAR REGRESSION MODEL
===============================================================================

Prediction formula:

y_pred = Xw + b

Where:

X → feature matrix

w → weights

b → bias

y_pred → predicted prices

===============================================================================
9. LOSS FUNCTION
===============================================================================

Mean Squared Error (MSE):

Loss = mean((y_pred − y)²)

Purpose:

Measure prediction quality.

Interpretation:

Small loss → good predictions.

Large loss → poor predictions.

===============================================================================
10. GRADIENT DESCENT
===============================================================================

Weight gradient:

dw = (2 / n) Xᵀ (y_pred − y)

Bias gradient:

db = mean(2 × (y_pred − y))

Parameter updates:

w = w − learning_rate × dw

b = b − learning_rate × db

Learning rate used:

0.001

===============================================================================
11. MODEL EVALUATION
===============================================================================

Metric used:

R² Score

Computed using:

r2_score()

Interpretation:

R² ≥ 0.80

Excellent predictions.

R² ≥ 0.50

Good predictions.

R² ≈ 0

Poor predictions.

R² < 0

Very poor predictions.

===============================================================================
12. SAVING TRAINED PARAMETERS
===============================================================================

Weights:

save_weights.csv

Bias:

save_bias.csv

Purpose:

Reuse trained parameters without retraining.

===============================================================================
13. FINAL PREDICTIONS
===============================================================================

During prediction mode:

The model:

1. Loads saved weights;
2. Loads saved bias;
3. Normalizes Kaggle test data;
4. Computes predictions.

Formula:

prediction = Xw + b

===============================================================================
14. KAGGLE SUBMISSION
===============================================================================

Generated file:

sample_submission.csv

Format:

Id,SalePrice
1461,169277.05
1462,187758.39
...

Submission file structure:

id → house identifier

SalePrice → predicted house price

===============================================================================
15. IMPORTANT NOTES
===============================================================================

- Always train before predict mode.
- Do not delete save_weights.csv.
- Do not delete save_bias.csv.
- Do not recompute normalization statistics on test data.
- Correlation filtering helps reduce noise.
- Docker guarantees reproducibility across environments.

===============================================================================
16. COMPLETE WORKFLOW SUMMARY
===============================================================================

1. Load train and test datasets;

2. Encode categorical variables;

3. Remove missing values;

4. Compute correlations;

5. Keep highly correlated features;

6. Split train and validation datasets;

7. Separate X and y;

8. Normalize training features;

9. Normalize validation and test features;

10. Initialize weights and bias;

11. Train using Gradient Descent;

12. Evaluate with R² Score;

13. Save learned parameters;

14. Load parameters during inference;

15. Predict SalePrice values;

16. Generate Kaggle submission file.

===============================================================================
END OF DOCUMENT
===============================================================================