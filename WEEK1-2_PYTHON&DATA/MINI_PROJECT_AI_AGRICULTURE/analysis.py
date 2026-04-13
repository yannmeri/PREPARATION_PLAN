import pandas as pd
import numpy as np

def read_csv():
    df = pd.read_csv("soil_dataset.csv")
    print(df.head())
    df.info()
    print(df.describe())

    return (df)

def clean_missed_value(df):
    #df = df.isnull().sum()
    df_clean = df.dropna()
    
    return (df_clean)

def classify_ph(ph):

    if (ph < 6):
        return "acid"
    elif (6 <= ph <= 7):
        return "neutral"
    else:
        return "basic"

def check_fertility(df):

    df["fertility"] = ((df["ph"].between(6, 7)) & (df["humidite"] >= 20)).replace({True: "fertile", False: "infertile"})

    return (df)


def compute_stats(df):
    moy_ph = np.mean(df["ph"])
    moy_humidity = np.mean(df["humidite"])

    count_f = (df["fertility"] == "fertile").sum()
    count_clay_s = (df["type"] == "argileux").sum()
    count_sandy_s = (df["type"] == "sableux").sum()
    count_loamy_s = (df["type"] == "limoneux").sum()

    dict_analysis = {
    "mean ph" : moy_ph,
    "mean humidity" : moy_humidity,
    "number fertile soil" : count_f,
    "number clay soil" : count_clay_s,
    "number sandy soil" : count_sandy_s,
    "number loamy soil" : count_loamy_s
    }
    
    return (dict_analysis)


def correlation_ph_humidity(df):
    print(df[["ph", "humidite"]].corr())

    #I have noticed that the correlation between the ph and the humidity is negative.
    #Indeed, more the ph is high, more the humidity is low.