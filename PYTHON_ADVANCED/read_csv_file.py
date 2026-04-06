import pandas as pd #pandas library is useful for reading inside a csv file and create a dataframe(df) from it.

def read_csv_file():
    df = pd.read_csv("mini_projet_sols_csv.csv")
    print(df)
    return (df)

