================================================================================
TECHNICAL DOCUMENTATION: BANK CUSTOMER CHURN PREDICTION (MULTI-STAGE CLASSIFICATION)
================================================================================

1. PROJECT DESCRIPTION
----------------------
This project implements a complete Machine Learning pipeline to classify and
predict bank customer attrition (churn). The primary goal is to determine 
whether a customer will leave the bank (churn = 1) or stay (churn = 0) based on
demographic and financial features.

The classification algorithm (Logistic Regression) was built entirely from scratch
using NumPy for matrix operations and optimized using batch gradient descent.

2. TECHNICAL CODE ARCHITECTURE
------------------------------
The script is structured into modular components adhering to industry production standards:

   * Global Constants: Centralized configuration for categorical variables to encode
     (gender, country) and mapping dictionaries for seamless maintenance.
   * Encoding Pipeline (ETL): Applies One-Hot Encoding to the country feature, 
     utilizing "drop_first=True" to eliminate the first column and prevent 
     multicollinearity. It also performs strict binary conversion for gender 
     (Female=1, Male=0).
   * Missing Value Management: Implements robust median imputation across all 
     numerical columns to prevent data loss caused by brute-force row deletion.
   * Feature Selection: A statistical filter that retains only the features with 
     an absolute linear correlation score >= 0.02 with the target variable (churn).
   * Robust Validation Strategy: Partitions the dataset into 3 distinct subsets:
       - Training set (50%): To learn the model weights and bias.
       - Validation set (30%): To tune hyperparameters and monitor performance.
       - Testing set (20%): For final evaluation of the model's generalization capability.
   * Normalization: Applies Z-score standardization (centering and scaling) based 
     strictly on Training set statistics to prevent any data leakage.

3. MATHEMATICAL FOUNDATIONS OF THE MODEL
----------------------------------------
The model is built upon the combination of three core mathematical pillars:

   A. The Sigmoid Activation Function:
      Maps raw linear predictions (z) strictly between 0 and 1 to transform them 
      into logical probability scores:
      g(z) = 1 / (1 + e^-z)

   B. Binary Cross-Entropy Loss (Log Loss):
      A convex cost function that measures the distance between predicted 
      probabilities and actual binary labels:
      Loss = -1/N * sum( y*log(y_pred) + (1-y)*log(1-y_pred) )

   C. Gradient Descent:
      Computes partial derivatives (dw and db) to iteratively update weights and 
      bias values in order to minimize the Loss:
      dw = (1/N) * X.T * (y_pred - y)
      db = (1/N) * sum(y_pred - y)

4. DEPLOYMENT ARCHITECTURE (ENVIRONMENT-BASED EXECUTION)
-------------------------------------------------------
The main block integrates a production routing logic driven by the system 
environment variable "MODE_PROJECT":

   * "train" MODE:
     The script initializes weights to zero, runs 10,000 gradient descent iterations, 
     prints the cost function progression, calculates the training F1-Score, and 
     saves the optimized parameters into "save_weights.csv" and "save_bias.csv".

   * "predict" MODE:
     The script instantly loads the pre-trained weights and bias matrices, applies 
     the historical training normalization parameters to the validation and test sets, 
     and generates a final predictions file ("save_prediction.csv").

5. EVALUATION METRIC
--------------------
The model performance is evaluated using the F1-Score (from scikit-learn). 
Unlike simple global Accuracy, the F1-Score combines Precision and Recall (harmonic 
mean). This metric is critical here because the bank churn dataset is imbalanced 
(the number of customers leaving is significantly lower than those who stay).
================================================================================