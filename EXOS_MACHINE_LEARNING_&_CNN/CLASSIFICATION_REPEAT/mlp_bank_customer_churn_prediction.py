import os #get environment Variable
import pandas as pd #import datas from csv file
import numpy as np #numpy array and mathematical operation
from tqdm import tqdm #loading bar
from sklearn.metrics import f1_score #getting F1_score

mode = os.getenv("MODE_PROJECT") #getting mode project from environment in docker-compose.yml

Gender = {
        "Female": 0,
        "Male": 1
    }

NOMINAL_VARIABLES = ['country']

def load_data(): #load data from csv file
    df = pd.read_csv("Bank Customer Churn Prediction.csv")

    return (df)

def encode_values(df): #encoding values
    df_copy = df.copy()

    df_copy['gender'] = df_copy['gender'].map(Gender)

    df_copy = pd.get_dummies(df_copy, columns=NOMINAL_VARIABLES, drop_first=True, dtype=int)

    #pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)

    #print(df_copy)
    return (df_copy)

def clean_dataframe(df): #cleaning dataframe obtained

    return (df.dropna())


def strongest_corr(df): #filtering features with strongest correlation with churn

    strong_corr = df.corr()['churn'].abs().sort_values(ascending=False)
    top_score = strong_corr[strong_corr >= 0.02]

    return (top_score)

def keep_top_features(df, top_score): #keeping best features

    check = False

    for c in df:
        for feature, value in top_score.items():
            if (feature == c):
                check = True
                break
        if (check == False):
            df = df.drop(columns=c)
        check = False

    return (df)


def split_train_test(df): #splitting my dataframe into three part(train, validation, test)

    split_train = int(df.shape[0] * 0.7)
    split_val = int(df.shape[0] * 0.9)

    train = df.iloc[:split_train, :]
    validation = df.iloc[split_train:split_val, :]
    test = df.iloc[split_val:, :]

    return (train, validation, test)


def split_X_train_X_test_X_val(train, validation, test): #splitting each part into X and y
    y_train = train[['churn']].to_numpy()
    y_validation = validation[['churn']].to_numpy()
    y_test = test[['churn']].to_numpy()


    X_train = train.drop(columns='churn').to_numpy()
    X_validation = validation.drop(columns='churn').to_numpy()
    X_test = test.drop(columns='churn').to_numpy()

    datas = {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test
    }

    return (datas)

def normalized_X_train_values(X_train): #normalizing train values
    moy = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_normal = (X_train - moy) / std

    return (X_normal, moy, std)


def normalized_X_val_X_test_values(X, moy, std): #normalizing test values

    return ((X - moy) / std)


def initialization(X, nb_layers): #initializing parameters(weights and bias)

    nb_features = X.shape[1]
    params = {}
    
    params['W0'] = np.random.randn(nb_features, 32) * np.sqrt(2.0 / nb_features)

    params['b0'] = np.zeros((1, 32))

    for i in range(1, nb_layers):

        if (i == nb_layers - 1):
            params['W' + str(i)] = np.random.randn(32, 1) * np.sqrt(2.0 / 32)
            params['b' + str(i)] = np.zeros((1, 1))
            break

        params['W' + str(i)] = np.random.randn(32, 32) * np.sqrt(2.0 / 32)
        params['b' + str(i)] = np.zeros((1, 32))

    return (params)

#linear regression formula
def linear_regression(X, W, b):
    Z = np.dot(X, W) + b

    return(Z)

def sigmoid_function(Z): #establish sigmoid function for my AI model
    # We limit Z between [-500, 500] to avoid huge values that make my model crash.
    Z_clipped = np.clip(Z, -500, 500)
    A = 1 / (1 + np.exp(-Z_clipped))

    return (A)

# ReLU for hidden layers
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def classification_model(params, X): #getting different activation values
    len_params = len(params) // 2

    activation = {}
    Z = linear_regression(X, params['W0'], params['b0'])
    activation['A0'] = relu(Z)

    for i in range (1, len_params):
        if (i == len_params - 1):
            Z = linear_regression(activation['A' + str(i - 1)],  params['W' + str(i)], params['b' + str(i)])
            activation['A' + str(i)] = sigmoid_function(Z)
        
        Z = linear_regression(activation['A' + str(i - 1)],  params['W' + str(i)], params['b' + str(i)])
        activation['A' + str(i)] = relu(Z)

    return (activation)

def gradients_back_propagation(X_train_norm, params, activation, y_train): #back-ward propagation

    nb_datas = X_train_norm.shape[0]
    len_params = len(params) // 2
    len_activ = len(activation) - 1
    first_term = 1

    dZ2 = activation['A' + str(len_activ)] - y_train
    gradients = {}


    while(len_activ >= 0):

        if (len_activ == 0):
            gradients['dW' + str(len_activ)] = (1 / nb_datas) * np.dot(X_train_norm.T, dZ2)
            gradients['db' + str(len_activ)] = (1 / nb_datas) * (np.sum(dZ2, axis=0, keepdims=True))
            break

        gradients['dW' + str(len_activ)] = (1 / nb_datas) * np.dot(activation['A' + str(len_activ - 1)].T, dZ2)
        gradients['db' + str(len_activ)] = (1 / nb_datas) * (np.sum(dZ2, axis=0, keepdims=True))
        
        
        dZ2 = np.dot(dZ2, params['W' + str(len_activ)].T) * relu_derivative(activation['A' + str(len_activ - 1)])
        
        len_activ -= 1


    return (gradients)

def train_classification_model(X_train_norm, y_train, params, learning_rate): #training model
    
    epsi = 1e-8
    nb_datas = X_train_norm.shape[0]
    len_activ = len(params) // 2
    Loss = []
    
    for i in tqdm(range(10000)):
        #Model and Forward Propagation
        activation = classification_model(params, X_train_norm)

        #Gradients
        gradients = gradients_back_propagation(X_train_norm, params, activation, y_train)

        #Gradients Descente
        for count in range(len_activ):
            params['W' + str(count)] = params['W' + str(count)] - (learning_rate * gradients['dW' + str(count)])
            params['b' + str(count)] = params['b' + str(count)] - (learning_rate * gradients['db' + str(count)])

        #loss function
        A_last = np.clip(activation['A' + str(len_activ - 1)], 1e-15, 1 - 1e-15)
        terms = (y_train * np.log(A_last + epsi)) + ((1 - y_train) * np.log(1 - A_last + epsi))
        loss = (-1 / nb_datas) * np.sum(terms, axis=0, keepdims=True)
        if (i % 1000 == 0):
            Loss.append(loss)

    print(f"loss: {Loss}")

    return (params)


def predict(X, params): #predicting part for final test
    len_params = len(params) // 2

    activation = classification_model(params, X)
    pred = activation['A' + str(len_params - 1)]

    return (pred >= 0.5).astype(int)


if __name__=="__main__":
    
    ##----Part1 (Preprocessing): loading datas and make them exploitable for my AI model----

    #1- Loading datas from csv file
    df = load_data()
    
    #2- encoding values
    df = encode_values(df)

    #3- cleaning dataframe from NA datas
    df = clean_dataframe(df)

    #4- filtering features with strongest correlation with churn
    top_score = strongest_corr(df)

    #5- keeping best features
    df = keep_top_features(df, top_score)

    #6- splitting our dataframe into three parts: train, validation and test
    train, validation, test = split_train_test(df)

    #7- splitting our three parts into X, y
    datas = split_X_train_X_test_X_val(train, validation, test)

    X_train = datas['X_train'] #(7000, 8)
    X_validation = datas['X_validation'] #(2000, 8)
    X_test = datas['X_test'] #(1000, 8)
    y_train = datas['y_train'] #(7000, 1)
    y_validation = datas['y_validation'] #(2000, 1)
    y_test = datas['y_test'] #(1000, 1)

    ##----------------------End of part1--------------------------


    ##----Part2(AI model and Training/Params Updating): Params initialization, model, forward and backward propagation, training----
    
    #8- normalizing X_train values
    X_train_norm, moy, std = normalized_X_train_values(X_train)

    if (mode == "train"):

        print("-----------------Mode train activated-----------------------")
        
        #9- parameters initialization
        params = initialization(X_train_norm, 32)

        #10- training model
        params = train_classification_model(X_train_norm, y_train, params, learning_rate=0.05)

        #11- saving params
        np.savez_compressed('save_params.npz', **params)
        print("Parameters saved into save_params.npz file.")

    ##----------------------------------------End of Part2------------------------------------


    ##----Part3 (prediction and accuracy): make prediction and check the model accuracy----------------

        #12- using validation part for the first test
        X_validation_norm = normalized_X_val_X_test_values(X_validation, moy, std)
        y_predict = predict(X_validation_norm, params)

        val_score = f1_score(y_validation, y_pred=y_predict)
        print(f"F1_score: {val_score}")

    elif (mode == "predict"):

        print("-----------------Mode predict activated-----------------------")


        #13- final test
        X_test_norm = normalized_X_val_X_test_values(X_test, moy, std)
        save_params = np.load('save_params.npz')
        params = {}
        
        for i in range(len(save_params) // 2):
            params['W' + str(i)] = save_params['W' + str(i)]
            params['b' + str(i)] = save_params['b' + str(i)]
        
        print("Parameters loaded from save_params.npz file.")


        y_predict = predict(X_test_norm, params)
        print(f"Test_prediction: {y_predict}")

        test_score = f1_score(y_test, y_pred=y_predict)
        print(f"F1_score: {test_score}")
        
    ##------------------------------------End of Part3------------------------------------------




