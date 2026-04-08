from read_csv_file import read_csv_file#we load the dataset and read it
import numpy as np#numpy is used for mathematics operation
import matplotlib.pyplot as plt #matplotlib is used for interpretating data with graph
import pandas as pd #pandas is used to manipulate dataset(read, write, etc)

def explore_dataset(df):#we explore the data of the datasets

    print(df.head())#we check the first five rows
    df.info()#we check number of rows, columns, the type of values
    print(df.describe())#we obtain some values from the dataset such as: mean, minimum value, maximum value, etc


def clean_dataset(df):#we clean the dataset if there is a missed data
    print(f"\n{df.isnull().sum()}\n")#isnull() give the number of missed data per column
    df = df.dropna()#dropna() delete missed data existing

    return (df)


def classify_ph(ph):# Classify the soil based on its pH value (acidic, neutral, or basic)
    if (ph < 6):
        return ("acid")
    elif(ph > 7):
        return ("basic")
    else:
        return ("neutral")

def transform_dataset(df):
    
    df["ph_type"] = df["ph"].apply(classify_ph)#apply is used to apply a function to each value of a column
    print(f"{df}\n")
    return (df)

def stats_dataset(df, count_n):#we compute stats here: ph and humidity mean.
    ph_moy = np.mean(df["ph"])
    humidity_moy = np.mean(df["humidite"])


    print(f"The mean of ph: {ph_moy}\nThe mean of humidity: {humidity_moy}\n")
    print(f"{count_n} soils are neutral!")


def filter_dataset(df):

    # Determine soil fertility:
    # Fertile if 6 <= pH <= 7 and humidity >= 20

    df["fertility"] = ((df["ph"].between(6, 7)) & (df["humidite"] >= 20)).replace({True: "fertile", False: "infertile"})
    
    #we check the condition for each ph and humidity value
    #we replace booleen True obtained by fertile
    #we replace booleen False obtained by infertile
    print(df["fertility"])
    
    count_n = (df["ph_type"] == "neutral").sum()
    return (count_n, df)

if __name__=="__main__":
    df = read_csv_file()
    explore_dataset(df)
    #There are 5 lines.
    #THere are 3 columns.

    df = clean_dataset(df)
    df = transform_dataset(df)
    (count_n, df) = filter_dataset(df)
    stats_dataset(df, count_n)

    plt.scatter(df["ph"], df["humidite"])#we create a scatter graph to visualize the relationship between ph and humidity
    plt.title("Relationship between ph and humidity")#we label the graph
    plt.xlabel("Ph values")#we label the X axis
    plt.ylabel("Humidite values")#we label the Y axis
    plt.show()#we print the graph