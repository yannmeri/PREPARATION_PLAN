import pandas as pd

def read_csv_file():
    df = pd.read_csv("mini_projet_sols_csv.csv")
    print(df)
    return (df)

