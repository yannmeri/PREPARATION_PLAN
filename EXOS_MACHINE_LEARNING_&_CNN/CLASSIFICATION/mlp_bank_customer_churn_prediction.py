import os #access to my laptop environment
import pandas as pd #create dataframe
import numpy as np #apply mathematical operation
from sklearn.metrics import f1_score #compute score
from tqdm import tqdm #loading bar

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

    #copying my dataframe
    df_copy = df.copy()
    
    # 1. Binary conversion for gender

    df_copy['gender'] = df_copy['gender'].map(GENDER_VALUES)

    # 2. One-Hot encoding for nominal features (country)
    df_copy = pd.get_dummies(df_copy, columns=NOMINAL_VARIABLES, drop_first=True, dtype=int)

    # 3. Security check: Impute remaining NaN values with column median
    numerical_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if (col != "churn"):
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    return (df_copy)


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

    y_train = train[['churn']].to_numpy()
    X_train = train.drop(columns='churn').to_numpy()

    y_validation = validation[['churn']].to_numpy()
    X_validation = validation.drop(columns='churn').to_numpy()

    y_test = test[['churn']].to_numpy()
    X_test = test.drop(columns='churn').to_numpy()

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


def initialisation(X_norm, nb_layers):#Initialize weights and bias and create layers of neural network 
    n_features = X_norm.shape[1]
    params = {}

    for i in range(nb_layers):

        if (i == 0):
            params[f'w{i}'] = np.random.randn(n_features, 32)
            params[f'b{i}'] = np.zeros((1, 32))

        elif (i + 1 == nb_layers):
            params[f'w{i}'] = np.random.randn(32, 1)
            params[f'b{i}'] = np.zeros((1, 1))

        else:
            params[f'w{i}'] = np.random.randn(32, 32)
            params[f'b{i}'] = np.zeros((1, 32))


    return (params)


def sigmoid_function(X, params):# Compute the sigmoid activation function

    len_params = int(len(params) / 2)
    
    activation = {}

    #activation 
    for i in range(len_params):
        if (i == 0):
            Z = np.dot(X, params[f'w{i}']) + params[f'b{i}']
            activation[f'A{i}'] = 1 / (1 + np.exp(-Z))
        else:
            Z = np.dot(activation[f'A{i - 1}'], params[f'w{i}']) + params[f'b{i}']
            activation[f'A{i}'] = 1 / (1 + np.exp(-Z))

    return (activation)

def gradients_back_propagation(X_norm, y_train, activation, params, learning_rate):

    n_data = X_norm.shape[0]
    len_act = int(len(params) / 2) - 1
    count = int(len(params) / 2) - 1
    dZ = y_train - activation[f'A{len_act}']

    while(count >= 0):
        #Gradients

        if (count == 0):
            dw = (1 / n_data) * np.dot(X_norm.T, dZ)
            db = (1 / n_data) * np.sum(dZ, axis=0, keepdims=True)

            #Gradients descente
            params[f'w{count}'] = params[f'w{count}'] - learning_rate * dw
            params[f'b{count}'] = params[f'b{count}'] - learning_rate * db

            break

        dw = (1 / n_data) * np.dot(activation[f'A{count - 1}'].T, dZ)
        db = (1 / n_data) * np.sum(dZ, axis=0, keepdims=True)


        #Gradients descente
        params[f'w{count}'] = params[f'w{count}'] - learning_rate * dw
        params[f'b{count}'] = params[f'b{count}'] - learning_rate * db

        #update dZ
        dZ = np.dot(dZ, params[f'w{count}'].T) * (activation[f'A{count - 1}'] * (1 - activation[f'A{count - 1}']))

        count -= 1
    
    return (params)

def train_classification_model(X_norm, y_train, nb_layers, learning_rate): #Train weights and bias using Binary Cross-Entropy Gradient Descent

    y_pred = 0
    error = 0
    epsilon = 1e-8
    n_data = X_norm.shape[0]

    params = initialisation(X_norm, nb_layers)

    for i in tqdm(range (10000)):

        #Classification Model
        activation = sigmoid_function(X_norm, params)        

        #error
        A_final = activation[f'A{len(activation) - 1}']
        error = A_final - y_train

        params = gradients_back_propagation(X_norm, y_train, activation, params, learning_rate)

        #loss
        loss = (-1 / n_data) * (np.dot(y_train.T, np.log(A_final + epsilon)) + np.dot((1 - y_train.T), np.log(1 - A_final + epsilon)))

        if (i % 100 == 0):
            print(f"loss: {loss}")


    return (params, A_final)


def normalized_X_val_X_test_values(X, moy, stand_d):#Scale any feature matrix using training mean and standard deviation.
    
    X_norm = (X - moy) / stand_d

    return (X_norm)


def classify_churn(prediction):#Convert probability scores into binary classes (0 or 1) using vectorization.

    prediction = np.where(prediction < 0.5, 0, 1)

    return (prediction)


def make_prediction(X, params):#Predict a customer tenure

    activation = sigmoid_function(X, params)
    A_final = activation[f'A{len(activation) - 1}']


    is_churn = classify_churn(A_final)

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
    nb_layers = 15
    
    if (mode == "train"):

        print(f"-----MODE TRAINING-----\n")

        #9- Train weights and bias using Binary Cross-Entropy Gradient Descent
        params, y_pred = train_classification_model(X_norm, y_train, nb_layers, learning_rate=0.001)

        np.savez_compressed('save_params.npz', **params)
        print("Parameters saved into save_params.npz file.")
        
        #10- Convert probability scores into binary classes (0 or 1) using vectorization.
        is_churn_train = classify_churn(y_pred)
        print(f"Training_prediction:\n{is_churn_train}\n")

        score_train = f1_score(is_churn_train, y_train)
        print(f"F1 Score training: {score_train}\n")
    
    elif (mode == "predict"):

        print(f"-----MODE PREDICTION-----\n")
        save_params = np.load('save_params.npz')
        params = {}
        for i in range(int(len(save_params) / 2)):
            
            params[f'w{i}'] = save_params[f'w{i}']
            params[f'b{i}'] = save_params[f'b{i}']
        
        print("Parameters loaded!")

        #Scale any feature matrix using training mean and standard deviation.
        X_val_norm = normalized_X_val_X_test_values(X_validation, moy, stand_d)

        #Predict a customer tenure
        is_churn_val = make_prediction(X_val_norm, params)

        score_validation = f1_score(is_churn_val, y_validation)
        print(f"F1 Score validation: {score_validation}\n")

        #Scale any feature matrix using training mean and standard deviation.
        X_test_norm = normalized_X_val_X_test_values(X_test, moy, stand_d)

        #Predict a customer tenure
        is_churn_test = make_prediction(X_test_norm, params)
        
        score_test = f1_score(is_churn_test, y_test)
        print(f"F1 Score test: {score_test}\n")

        np.savez_compressed('save_prediction.npz', prediction=is_churn_test)
