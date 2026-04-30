import pandas as pd#Pandas is used to load and manipulate datasets
import numpy as np#NumPy is used for numerical computations

def create_dataframe():#Load the CSV file into a DataFrame
    df = pd.read_csv("dataset.csv")

    return (df)

def create_target(df):#Create the target variable for regression
    df["target"] = df["ph"] * 10 + df["humidite"]

    return (df)

def check_missing_value(df):
    #Interpretation
    #Verify whether the dataset contains missing values before removing themm
    #df = df.isnull().sum()
    #print(df)

    df = df.dropna()
    return (df)

def correlation(df):
    #Interpretation
    #Analyze the relationship between pH and humidity
    print(df[["ph", "humidite"]].corr())

def predict(X, weights, bias):#Compute predictions using the linear regression model
    return (np.dot(X, weights) + bias)

def train_model(df, weights, bias, learning_rate):# Update weights and bias using gradient descent to reduce prediction error
    #Prediction

    X = df[["ph", "humidite"]].to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)


    y = df["target"].to_numpy().reshape(-1, 1)#Convert target column to numpy array

    for i in range(1000):
        y_pred = predict(X, weights, bias)
        error = y_pred - y

        dw = np.dot(X.T, error) / len(X)
        db = error.mean()

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        loss = (error ** 2).mean()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    print("Weights:", weights)
    print("Bias:", bias)

    return (y_pred)


if __name__=="__main__":#I test my program
    df = create_dataframe()
    df = create_target(df)

    #Display all rows of the DataFrame
    #pd.set_option('display.max_rows', None)
    #print(f"The dataframe is:\n{df}\n")

    df = check_missing_value(df)
    #pd.set_option('display.max_rows', None)
    print(f"The new dataframe is:\n{df}\n")

    correlation(df)

    weights = np.zeros((2,1))

    bias = 0.0

    y_pred = train_model(df, weights, bias, learning_rate=0.1)

    print(f"Final prediction:\n{y_pred}\nThe shape is: {y_pred.shape}")