import pandas as pd#Pandas is used for creating Dataframe
import numpy as np#NumPy is used for mathematical computing

def create_dataframe():#Create a dataframe
    data = {
        "ph": [6.5, 7.0, 5.8, 6.2, 7.1, 5.5],
        "humidite": [25, 30, 15, 20, 35, 18],
        "fertile": [1, 1, 0, 1, 1, 0]
    }

    df = pd.DataFrame(data)

    return (df)


def clean_dataframe(df):#Clean dataframe created
    df_clean = df.dropna()

    return (df_clean)


def split_dataframe(df_clean):#split dataframe into test and train
    data_train = df_clean.iloc[:4, :]
    data_test = df_clean.iloc[4:, :]

    X_train = data_train.iloc[:, :2].to_numpy()
    y_train = data_train.iloc[:, 2:].to_numpy()

    X_test = data_test.iloc[:, :2].to_numpy()
    y_test = data_test.iloc[:, 2:].to_numpy()

    dict_data = {
        "X_train" : X_train, #shape(4,2)
        "y_train" : y_train, #shape(4,1)
        "X_test" : X_test, #shape(2,2)
        "y_test" : y_test, #shape(2,1)
        "data_train" : data_train, #shape(4,3)
        "data_test" : data_test #shape(2,3)
    }

    return (dict_data)


def normalize_data(X_train):#Normalized data train

    moy = X_train.mean(axis=0)
    stand_d = X_train.std(axis=0)

    X_norm = (X_train - moy) / stand_d
    
    #print(f"moy: \n{X_norm.mean(axis=0)}\nstand_d : \n{X_norm.std(axis=0)}\n")
    return (X_norm, moy, stand_d)

def normalize_data_test(X_test, moy, stand_d):#normalize data test
    
    X_test_norm = (X_test - moy) / stand_d

    return (X_test_norm)

def sigmoid_function(z):#implementing sigmoid function

    sig = 1 / (1 + np.exp(-z))

    return (sig)


def train_model(w, X_norm, y_train, learning_rate, b):#implementing training model

    y_pred = 0
    error = 0
    epsilon = 1e-8

    for i in range(10000):

        #AI Model
        z = np.dot(X_norm, w) + b
        y_pred = sigmoid_function(z)

        #computing error
        error = y_pred - y_train
        #print(error.shape)

        #loss
        loss = -np.mean(y_train * np.log(y_pred + epsilon) + (1 - y_train) * np.log(1 - y_pred + epsilon))

        #gradients
        dw = np.dot(X_norm.T, (y_pred - y_train)) / len(X_norm)
        db = np.mean(error)

        #updates
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if (i % 100 == 0):
            print(f"loss: \n{loss}")
    
    params = {
        "weights" : w, #shape(2, 1)
        "bias" : b, #shape(1,)
        "loss" : loss, #shape(1,)
        "prediction" : y_pred #shape(4, 1)
    }

    return (params)


def prediction(X_test_norm, w, b):#implementing prediction model
    z = np.dot(X_test_norm, w) + b
    sig = sigmoid_function(z)

    return (sig)


if __name__=="__main__":

    df = create_dataframe()
    df_clean = clean_dataframe(df)

    dict_data = split_dataframe(df_clean)

    X_train = dict_data['X_train']
    y_train = dict_data['y_train']
    X_test = dict_data['X_test']
    y_test = dict_data['y_test']
    data_train = dict_data['data_train']
    data_test = dict_data['data_test']
    

    # print(f"dataframe : \n{df_clean}\n")
    # print(f"X_train : \n{X_train}\nX_train_shape : {X_train.shape}\n")
    # print(f"y_train : \n{y_train}\ny_train_shape : {y_train.shape}\n")
    # print(f"X_test : \n{X_test}\nX_test_shape : {X_test.shape}\n")
    # print(f"y_test : \n{y_test}\ny_test_shape : {y_test.shape}\n")
    # print(f"data_train : \n{data_train}\ndata_train_shape : {data_train.shape}\n")
    # print(f"data_test : \n{data_test}\ndata_test_shape : {data_test.shape}\n")

    (X_norm, moy, stand_d) = normalize_data(X_train)

    w = np.array([[0.0],
                [0.0]]) #shape(2, 1)
    
    params = train_model(w, X_norm, y_train, learning_rate=0.001, b=0)

    w = params['weights']
    b = params['bias']
    predic = params['prediction']
    loss = params['loss']

    print(f"Weights :\n{w}\n")
    print(f"bias :\n{b}\n")
    print(f"training_prediction :\n{predic}\n")
    print(f"loss :\n{loss}\n")

    X_test_norm = normalize_data_test(X_test, moy, stand_d)
    #print(f"X_test_norm : \n{X_test_norm}\nX_test_norm_shape : {X_test_norm.shape}\n")

    y_test_pred = prediction(X_test_norm, w, b)
    print(f"Final prediction :\n{y_test_pred}\n")

    predict_class = (y_test_pred >= 0.5).astype(int)
    print(f"Class: \n{predict_class}")





        