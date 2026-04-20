import pandas as pd#Pandas is used to manipulate datasets
import numpy as np#NumPy is used for numerical computations

def read_csv_file():#Load the CSV file into a DataFrame.
    df = pd.read_csv("mini_projet_sols_csv.csv")

    return (df)

def mean_dataset(df):#Compute the mean humidity.
    return(np.mean(df["humidite"]))

def var_dataset(df):#Compute the variance of humidity.
    return(np.var(df["humidite"]))

if __name__=="__main__":#I test my program here
    df = read_csv_file()
    moy_humidity = mean_dataset(df)
    var_humidity = var_dataset(df)

    print(f"Mean humidity: {moy_humidity}")
    print(f"Variance of humidity: {var_humidity}")

    print("\nInterpretation:")
    print("- High variance → humidity varies a lot between soils")
    print("- Low variance → humidity is stable across soils")