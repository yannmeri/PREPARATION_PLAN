import os #access to my laptop environment
import pandas as pd #create dataframe
import numpy as np #apply mathematical operation
from sklearn.metrics import f1_score #compute score

mode = os.getenv("MODE_PROJECT")

NOMINAL_VARIABLES = ['country']

GENDER_VALUES = {
    'Female': 1,
    'Male': 0
}

def load_data(): #Load the raw bank customer dataset from CSV.

    df = pd.read_csv("Bank Customer Churn Prediction.csv")

    return (df)


def encoding_unnumeric_features(df):#Encode categorical variables and impute missing numerical fields.

    # 1. Binary conversion for gender
    df['gender'] = df['gender'].map(GENDER_VALUES)

    # 2. One-Hot encoding for nominal features (country)
    df = pd.get_dummies(df, columns=NOMINAL_VARIABLES, drop_first=True, dtype=int)

    # 3. Security check: Impute remaining NaN values with column median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if (col != "churn"):
            df[col] = df[col].fillna(df[col].median())
    
    return (df)


def clean_dataframe(df_treated):#Drop any remaining rows containing NaN values.

    df_clean = df_treated.dropna()

    return (df_clean)


def compute_top_correlations(df_clean):#Filter features based on their absolute correlation score with the target.

    correlation = df_clean.corr()['churn'].abs().sort_values(ascending=False)
    best_corr = correlation[correlation >= 0.02]

    return (best_corr)


def keep_top_features(df_clean, best_corr):#Drop columns that do not meet the minimum correlation threshold.

    check = False

    for col in df_clean:
        for feature, value in best_corr.items():
            if (col == feature):
                check = True
                break
        if (check == False):
            df_clean = df_clean.drop(columns=col)
        check = False

    return (df_clean)


def split_dataframe(df_clean):#Split the dataset into training (50%), validation (30%), and testing (20%) sets.
    
    split_train = int(df_clean.shape[0] * 0.5)
    split_val = int(df_clean.shape[0] * 0.8)

    train = df_clean.iloc[:split_train, :]
    validation = df_clean.iloc[split_train: split_val, :]
    test = df_clean.iloc[split_val:, :]

    return (train, validation, test)


def extract_features_target(train, validation, test):#Extract features matrices (X) and target vectors (y) for all subsets.

    y_train = train[['churn']]
    X_train = train.drop(columns='churn')

    y_validation = validation[['churn']]
    X_validation = validation.drop(columns='churn')

    y_test = test[['churn']]
    X_test = test.drop(columns='churn')

    datas = {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test
    }

    return (datas)


def normalized_X_train_values(df_clean):#Compute mean and standard deviation from training set to scale features.

    moy = df_clean.mean(axis=0)
    stand_d = df_clean.std(axis=0)
    
    df_norm = (df_clean - moy) / stand_d

    return (df_norm, moy, stand_d)


def sigmoid_function(z):# Compute the sigmoid activation function

    sig = 1 / (1 + np.exp(-z))

    return (sig)


def train_classification_model(X_norm, y_train, w, b, learning_rate): #Train weights and bias using Binary Cross-Entropy Gradient Descent

    y_pred = 0
    error = 0
    epsilon = 1e-8
    n_features = X_norm.shape[0]

    for i in range (10000):

        #Classification Model
        z = np.dot(X_norm, w) + b
        y_pred = sigmoid_function(z)

        #error
        error = y_pred - y_train

        #Gradients
        dw = (1 / n_features) * np.dot(X_norm.T, error)
        db = (1 / n_features) * np.sum(error)

        #Gradients descente
        w = w - learning_rate * dw
        b = b - learning_rate * db

        #loss
        loss = (-1 / n_features) * (np.dot(y_train.T, np.log(y_pred + epsilon)) + np.dot((1 - y_train.T), np.log(1 - y_pred + epsilon)))

        if (i % 100 == 0):
            print(f"loss: {loss}")

    params = {
        "weights": w,
        "bias": b,
        "train_pred": y_pred
    }

    return (params)


def normalized_X_val_X_test_values(X, moy, stand_d):#Scale any feature matrix using training mean and standard deviation.
    
    X_norm = (X - moy) / stand_d

    return (X_norm)


def classify_churn(prediction):#Convert probability scores into binary classes (0 or 1) using vectorization.

    prediction = np.where(prediction < 0.5, 0, 1)

    return (prediction)


def make_prediction(X, w, b):#Predict a customer tenure

    pred = np.dot(X, w) + b
    activ = sigmoid_function(pred)

    is_churn = classify_churn(activ)

    return (is_churn)


if __name__=="__main__":#Main test

    #1- Load the raw bank customer dataset from CSV.
    df = load_data() #shape (10000, 12)

    #2- Encode categorical variables and impute missing numerical fields.
    df_treated = encoding_unnumeric_features(df)

    #3- Drop any remaining rows containing NaN values.
    df_clean = clean_dataframe(df_treated)

    #4- Filter features based on their absolute correlation score with the target.
    best_corr = compute_top_correlations(df_clean)

    #5- Drop columns that do not meet the minimum correlation threshold.
    df_clean = keep_top_features(df_clean, best_corr)

    #6- Split the dataset into training (50%), validation (30%), and testing (20%) sets.
    train, validation, test = split_dataframe(df_clean)

    #7- Extract features matrices (X) and target vectors (y) for all subsets.
    datas = extract_features_target(train, validation, test)

    X_train = datas['X_train'] #shape(5000, 8)
    y_train = datas['y_train'] #shape(5000, 1)

    X_validation = datas['X_validation'] #shape(3000, 8)
    y_validation = datas['y_validation'] #shape(3000, 1)

    X_test = datas['X_test'] #shape(2000, 8)
    y_test = datas['y_test'] #shape(2000, 1)

    #8- Compute mean and standard deviation from training set to scale features.
    X_norm, moy, stand_d = normalized_X_train_values(X_train)

    
    if (mode == "train"):

        print(f"-----MODE TRAINING-----\n")
        n_features = X_norm.shape[1]
        w = np.zeros((n_features, 1))
        b = 0

        #9- Train weights and bias using Binary Cross-Entropy Gradient Descent
        params = train_classification_model(X_norm, y_train, w, b, learning_rate=0.001)

        w = params['weights']
        b = params['bias']
        y_pred = params['train_pred']

        save_weights = pd.DataFrame({
                "weights": w.flatten()
        })
        save_weights = save_weights.to_csv("save_weights.csv", index=False)

        save_bias = pd.DataFrame({
                "bias": b.flatten()
        })
        save_bias = save_bias.to_csv("save_bias.csv", index=False)
        
        #10- Convert probability scores into binary classes (0 or 1) using vectorization.
        is_churn_train = classify_churn(y_pred)
        print(f"Training_prediction:\n{is_churn_train}\n")

        score_train = f1_score(is_churn_train, y_train)
        print(f"F1 Score training: {score_train}\n")
    
    elif (mode == "predict"):

        print(f"-----MODE PREDICTION-----\n")
        save_weights = pd.read_csv("save_weights.csv")
        save_bias = pd.read_csv("save_bias.csv")
        w = save_weights[['weights']].to_numpy()
        b = save_bias[['bias']].to_numpy()

        #Scale any feature matrix using training mean and standard deviation.
        X_val_norm = normalized_X_val_X_test_values(X_validation, moy, stand_d)

        #Predict a customer tenure
        is_churn_val = make_prediction(X_val_norm, w, b)

        score_validation = f1_score(is_churn_val, y_validation)
        print(f"F1 Score validation: {score_validation}\n")

        #Scale any feature matrix using training mean and standard deviation.
        X_test_norm = normalized_X_val_X_test_values(X_test, moy, stand_d)

        #Predict a customer tenure
        is_churn_test = make_prediction(X_test_norm, w, b)
        
        score_test = f1_score(is_churn_test, y_test)
        print(f"F1 Score test: {score_test}\n")

        save_prediction = pd.DataFrame({
                "prediction": is_churn_test.flatten()
        })
        save_prediction = save_prediction.to_csv("save_prediction.csv", index=False)