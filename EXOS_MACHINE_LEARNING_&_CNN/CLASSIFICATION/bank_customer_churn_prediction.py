import os #access to my laptop environment
import pandas as pd #create dataframe
import numpy as np #apply mathematical operation
from sklearn.metrics import f1_score #compute score

mode = os.getenv("MODE_PROJECT")

def load_data(): #Load Dataframe 
    df = pd.read_csv("Bank Customer Churn Prediction.csv")

    return (df)


def encoding_unnumeric_features(df):#encode non_numerical values with One-Hot Encoding and binary method

    Binary_variable = ['gender']

    Nominale_variable = ['country']

    gender_value = {
        'Female': 1,
        'Male': 0
    }

    df['gender'] = df['gender'].map(gender_value)

    df = pd.get_dummies(df, columns=Nominale_variable, drop_first=True, dtype=int)

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if (col != "churn"):
            df[col] = df[col].fillna(df[col].median())
    
    return (df)


def clean_dataframe(df_treated):#delete NaN values
    df_clean = df_treated.dropna()

    return (df_clean)


def compute_top_correlations(df_clean):#detect features with strongest correlations with churn
    correlation = df_clean.corr()['churn'].abs().sort_values(ascending=False)
    best_corr = correlation[correlation >= 0.02]

    return (best_corr)


def conserv_top_features(df_clean, best_corr):#keep top features

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


def split_df_into_train_val_test(df_clean):#split dataframe into 3 parts: train, validation and test
    
    split_train = int(df_clean.shape[0] * 0.5)
    split_val = int(df_clean.shape[0] * 0.8)

    train = df_clean.iloc[:split_train, :]
    validation = df_clean.iloc[split_train: split_val, :]
    test = df_clean.iloc[split_val:, :]

    return (train, validation, test)


def get_X_y_values(train, validation, test):#split sub-dataframes(train, validation, test) into X and y values

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


def normalized_X_train_values(df_clean):#standardize X_train values

    moy = df_clean.mean(axis=0)
    stand_d = df_clean.std(axis=0)

    df_norm = (df_clean - moy) / stand_d

    return (df_norm, moy, stand_d)


def sigmoid_function(z): #Sigmoid function for classification

    sig = 1 / (1 + np.exp(-z))

    return (sig)


def train_classification_model(X_norm, y_train, w, b, learning_rate): #Model training

    y_pred = 0
    error = 0
    epsilon = 1e-8

    for i in range (10000):

        #Classification Model
        z = np.dot(X_norm, w) + b
        y_pred = sigmoid_function(z)

        #error
        error = y_pred - y_train

        #Gradients
        dw = (1 / X_norm.shape[0]) * np.dot(X_norm.T, error)
        db = (1 / X_norm.shape[0]) * np.sum(error)

        #Gradients descente
        w = w - learning_rate * dw
        b = b - learning_rate * db

        #loss
        loss = (-1 / X_norm.shape[0]) * (np.dot(y_train.T, np.log(y_pred)) + np.dot((1 - y_train.T), np.log(1 - y_pred + epsilon)))

        if (i % 100 == 0):
            print(f"loss: {loss}")

    params = {
        "weights": w,
        "bias": b,
        "train_pred": y_pred
    }

    return (params)


def normalized_X_val_X_test_values(X, moy, stand_d):#standardize X_val or X_test values
    
    X_norm = (X - moy) / stand_d

    return (X_norm)


def classify_churn(prediction):#classify prediction values into 2 categories: 1 or 0

    index = 0

    for val in prediction:
        if (val < 0.5):
            prediction[index] = 0
        else:
            prediction[index] = 1

        index += 1

    return (prediction)


def make_prediction(X, w, b):#Predict a customer tenure

    pred = np.dot(X, w) + b
    activ = sigmoid_function(pred)

    is_churn = classify_churn(activ)

    return (is_churn)


if __name__=="__main__":#Main test

    df = load_data() #shape (10000, 12)

    df_treated = encoding_unnumeric_features(df)

    df_clean = clean_dataframe(df_treated)

    best_corr = compute_top_correlations(df_clean)

    df_clean = conserv_top_features(df_clean, best_corr)

    train, validation, test = split_df_into_train_val_test(df_clean)

    datas = get_X_y_values(train, validation, test)

    X_train = datas['X_train'] #shape(5000, 8)
    y_train = datas['y_train'] #shape(5000, 1)

    X_validation = datas['X_validation'] #shape(3000, 8)
    y_validation = datas['y_validation'] #shape(3000, 1)

    X_test = datas['X_test'] #shape(2000, 8)
    y_test = datas['y_test'] #shape(2000, 1)

    X_norm, moy, stand_d = normalized_X_train_values(X_train)

    n_features = X_norm.shape[1]
    w = np.zeros((n_features, 1))
    b = 0
    
    if (mode == "train"):

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
        
        is_churn_train = classify_churn(y_pred)
        print(f"Training_prediction:\n{is_churn_train}\n")

        score_train = f1_score(is_churn_train, y_train)
        print(f"Score training: {score_train}\n")
    
    elif (mode == "predict"):

        save_weights = pd.read_csv("save_weights.csv")
        save_bias = pd.read_csv("save_bias.csv")
        w = save_weights['weights'].to_numpy()
        b = save_bias['bias'].to_numpy()

        X_val_norm = normalized_X_val_X_test_values(X_validation, moy, stand_d)
        is_churn_val = make_prediction(X_val_norm, w, b)
        score_validation = f1_score(is_churn_val, y_validation)
        print(f"Score valscore_validation: {score_validation}\n")

        X_test_norm = normalized_X_val_X_test_values(X_test, moy, stand_d)
        is_churn_test = make_prediction(X_test_norm, w, b)
        score_test = f1_score(is_churn_test, y_test)
        print(f"Score test: {score_test}\n")

        save_prediction = pd.DataFrame({
                "prediction": is_churn_test.flatten()
        })
        save_prediction = save_prediction.to_csv("save_prediction.csv", index=False)