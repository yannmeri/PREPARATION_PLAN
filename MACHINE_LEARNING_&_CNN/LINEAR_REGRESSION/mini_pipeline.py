import pandas as pd#this library helps creating a datafrram
import numpy as np#NumPy is a library specialized in mathematical computing

def read_data():#Creating df dataframe here
    data = {
        "ph": [6.5, 7.0, 5.8, 6.2, 7.1],
        "humidite": [25, 30, 15, 20, 35],
        "yield": [90, 100, 60, 80, 110]
    }

    df = pd.DataFrame(data)
    print(df)

    return (df)


def clean_data(df):
    df_new = df.dropna()
    return (df_new)

def split_data(df_new):


    #I split the dataframe into two categories train and test dataframe
    #Ps: the mark ':' means all.
    df_train = df_new.iloc[:3,:]#before the comma (',') means all the lines until index 3 and after the comma (',') means all columns
    df_test = df_new.iloc[3:,:]#before the comma (',') means all the lines from index 3 and after the comma (',') means all columns

    #This is my method to split X and y
    #Ps: the mark ':' means all.
    df_train_x = df_train.iloc[:,:2]#before the comma (',') means all the lines and after the comma (',') means all columns until index 2
    df_train_y = df_train.iloc[:, 2:]#before the comma (',') means all the lines and after the comma (',') means all columns from index 2
    

    #This is my method to split X and y
    #Ps: the mark ':' means all.
    df_test_x = df_test.iloc[:,:2]#before the comma (',') means all the lines and after the comma (',') means all columns until index 2
    df_test_y = df_test.iloc[:, 2:]#before the comma (',') means all the lines and after the comma (',') means all columns from index 2
    

    # print(df_x)
    # print(df_y)
    # print(train_dataframe)
    # print(test_dataframe)

    split_dict = {
        "df_train_x": df_train_x, #shape(3,2)
        "df_train_y": df_train_y, #shape(3,1)
        "df_test_x": df_test_x, #shape(2,2)
        "df_test_y": df_test_y, #shape(2,1)
        "df_train": df_train, #shape(3,3)
        "df_test": df_test #shape(2,3)
    }

    return (split_dict)


def train_model(W, split_dict, learning_rate, b):

    y_pred = 0
    error = 0
    X = np.array(split_dict["df_train_x"])#shape(3,2)
    y = np.array(split_dict["df_train_y"])#shape(3,1)
    
    for i in range(1000):

        #predict y values
        y_pred = np.dot(X, W) + b

        #error
        error = y_pred - y

        #loss function
        loss = np.mean(error ** 2)

        #Gradients
        dW = (2 / len(X)) * np.dot(X.T, error)
        db = 2 * np.mean(y_pred - y)

        #Update
        W = W - learning_rate * dW
        b = b - learning_rate * db
    
    dict_train = {
        "W": W, #shape(2,1)
        "b": b, #shape(1,)
        "y_pred": y_pred, #shape(3,1)
        "loss": loss #shape(1,)
    }

    return (dict_train)


def predict(dict_train, split_dict):
    W = dict_train['W'] #shape(2, 1)
    b = dict_train['b'] #shape(1)

    X_test = split_dict['df_test_x'] #shape(2,2)

    y_test_pred = np.dot(X_test, W) + b

    return (y_test_pred) #shape(2,1)

if __name__=="__main__":
    df = read_data()
    df_new = clean_data(df)
    split_dict = split_data(df_new)

    print(f"\nShape_X : \n{split_dict['df_train_x'].shape}\nShape_y : \n{split_dict['df_train_y'].shape}\nShape_train_dataframe : \n{split_dict['df_train'].shape}\n")

    W = np.array([[0.0],
                [0.0]])
    b = 0
    
    dict_train = train_model(W, split_dict, learning_rate=0.001, b=0)

    print(f"Train_prediction : \n{dict_train['y_pred']}\nLoss_function : {dict_train['loss']}\n W : \n {dict_train['W']}\n")

    y_test_pred = predict(dict_train, split_dict)

    print(f"test_prediction: \n{y_test_pred}")