import pandas as pd#this library helps creating a datafrram
import numpy as np#NumPy is a library specialized in mathematical computing

def create_dataframe():#create a dataframe here
    data = {
    "ph": [6.5, 7.0, 5.8, 6.2, 7.1, 6.8, 5.5, 6.0],
    "humidite": [25, 30, 15, 20, 35, 28, 18, 22],
    "temperature": [30, 32, 35, 28, 40, 33, 36, 29],
    "yield": [90, 100, 60, 80, 110, 95, 55, 85]
    }

    df = pd.DataFrame(data)

    return (df)

def clear_dataframe(df):#clear dataframe here
    df_clean = df.dropna()

    return (df_clean)

def split_dataframe(df_clean):#split dataframe into train_data and test_data and get their X and y values respectively

    #I split the dataframe into two categories train and test dataframe
    #Ps: the mark ':' means all.
    data_train = df.iloc[:5, :]
    data_test = df.iloc[5:, :]

    X_train = data_train.iloc[:, :3]#before the comma (',') means all the lines and and after the comma (',') means all columns until index 3
    y_train = data_train.iloc[:, 3:]#before the comma (',') means all the lines and and after the comma (',') means all columns from index 3

    X_test = data_test.iloc[:, :3]#before the comma (',') means all the lines and and after the comma (',') means all columns until index 3
    y_test = data_test.iloc[:, 3:]#before the comma (',') means all the lines and and after the comma (',') means all columns from index 3

    dict_data = {
        "X_train" : X_train, #shape(5, 3)
        "y_train" : y_train, #shape(5, 1)
        "X_test" : X_test, #shape(3, 3)
        "y_test" : y_test, #shape(3, 1)
        "data_train" : data_train, #shape(5, 4)
        "data_test" : data_test #shape(3, 4)
    }

    return (dict_data)

def normalize_train(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std

    norm = {
        "X_norm" : X_norm,
        "moy" : mean,
        "std" : std
    }
    return norm

def normalize_test(X_test, norm):

    mean = norm['moy']
    std = norm['std']

    return (X_test - mean) / std
    

def train_model(norm, w, dict_data, learning_rate, b):#AI model to find appropriate parameters

    y_pred = 0
    error = 0
    loss = 0
    X = dict_data["X_train"]
    X_norm = norm["X_norm"]
    y = dict_data["y_train"]

    for i in range (10000):

        #predict y values
        y_pred = np.dot(X_norm, w) + b

        #error
        error = y_pred - y

        #loss function
        loss = np.mean(error ** 2)

        #Gradients
        dw = 2 * np.dot(X_norm.T, error)
        db = 2 * np.mean(error)

        #Update
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if (i % 100 == 0):
            print(f"loss : {loss}")
    
    params = {
        "weights": w, #shape(3, 1)
        "bias" : b, #shape(1,)
        "prediction" : y_pred, #shape(5,1)
        "loss" : loss #shape(1,)
    }

    return (params)

def predict(dict_data, w, b, norm):#predict of agricultural yield from ph, humidite and temperature
    X_test = dict_data["X_test"]
    X_test_norm = normalize_test(X_test, norm)

    predi_test = np.dot(X_test_norm, w) + b

    return (predi_test)


def correlation(df):#check link between each feature
    result = df[["ph", "humidite", "temperature"]].corr()

    print(f"\ncorrelation : \n{result}")


def standard_deviation(df):#check convergence's values

    moy = df[["ph", "humidite", "temperature"]].mean()

    conv = df[["ph", "humidite", "temperature"]].std()

    print(f"\nmean : \n{moy}")
    print(f"\nconvergence : \n{conv}")


if __name__=="__main__":
    df = create_dataframe()
    df_clean = clear_dataframe(df)
    dict_data = split_dataframe(df_clean)

    print(df)

    print(f"\nX_train : \n{dict_data['X_train']}\nX_train_shape : {dict_data['X_train'].shape}\n")
    print(f"y_train : \n{dict_data['y_train']}\ny_train_shape : {dict_data['y_train'].shape}\n")
    print(f"X_test : \n{dict_data['X_test']}\nX_test_shape : {dict_data['X_test'].shape}\n")
    print(f"y_test : \n{dict_data['y_test']}\ny_test_shape : {dict_data['y_test'].shape}\n")
    print(f"data_train : \n{dict_data['data_train']}\ndata_train_shape : {dict_data['data_train'].shape}\n")
    print(f"data_test : \n{dict_data['data_test']}\ndata_test_shape : {dict_data['data_test'].shape}\n")

    w = [[0.0],
        [0.0],
        [0.0]]

    b = 0

    norm = normalize_train(dict_data['X_train'])
    params = train_model(norm, w, dict_data, learning_rate=0.001, b=0)

    print(f"\nweights : \n{params['weights']}\n")
    print(f"bias : \n{params['bias']}\n")
    print(f"prediction : \n{params['prediction']}\n")
    print(f"loss : {params['loss']}\n")

    predi_test = predict(dict_data, w, params['bias'], norm)

    print(f"final_prediction : \n{predi_test}")

    correlation(dict_data["data_train"])

    #PH is the feature that influences more the yield.

    standard_deviation(dict_data["data_train"])
    #Convergence analysis:
    #if ph convergence is between [0, 1] it means ph values are close to each others.
    #if humidite convergence is average, it means majority of humidite values are close to each others.
    #if temperature convergence is average, it means majority of temperature values are close to each others.

    #Prediction analysis
    #For training, I get a good score but it is not good for testing data. It is maybe because of overfitting.

