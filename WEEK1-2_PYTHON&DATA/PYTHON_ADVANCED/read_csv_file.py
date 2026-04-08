import pandas as pd#Pandas is used to read and manipulate data from CSV files

def read_csv_file():#This function reads a CSV file and returns a DataFrame
    df = pd.read_csv("mini_projet_sols_csv.csv")
    print(df)#Display the content of the DataFrame (for checking)
    return df