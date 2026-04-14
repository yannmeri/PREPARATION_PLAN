import pandas as pd# Pandas is used to load and manipulate datasets
import numpy as np# NumPy is used for numerical computations

def read_csv():# Load dataset from CSV file and display basic information
    df = pd.read_csv("soil_dataset.csv")
    print(df.head())#we print the first five rows of the dataframe
    df.info()#we print the different information about the dataframe such as: number row, number column, ...
    print(df.describe())#we print the stats of values in the dataframe such as: mean, std, etc

    return (df)

def clean_missed_value(df):
    #df = df.isnull().sum()
    df_clean = df.dropna()#we delete rows with NaN values
    
    return (df_clean)

def classify_ph(ph):#we check the soil state

    if (ph < 6):
        return "acid"
    elif (6 <= ph <= 7):
        return "neutral"
    else:
        return "basic"

def check_fertility(df):#we check the fertility of the soil based on ph and humidity

    df["fertility"] = ((df["ph"].between(6, 7)) & (df["humidite"] >= 20)).replace({True: "fertile", False: "infertile"})

    return (df)


def compute_stats(df):#we compute stats of dataframe such as mean of ph and humidity, number of: clay soil, sandy soil, and loamy soil.
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


def correlation_ph_humidity(df):#we check the correlation between ph and humidity
    correlation = df[["ph", "humidite"]].corr()
    print("\nCorrelation matrix:\n", correlation)

    # Interpretation
    print("\nInterpretation:")
    print("If the value is close to 1 → strong positive correlation")
    print("If close to -1 → strong negative correlation")
    print("If close to 0 → weak or no correlation")

    #I have noticed that the correlation between the ph and the humidity is negative.
    #More the ph is high, more the humidity is low.

    


def recommend_crop(type):#we recommend the appropriate crop depending on the type of soil

    cultures = {
        "clay_s" : ["rice", "cabbage", "spinash", "beans", "fruits trees (orange tree, mango tree, apple tree)"],
        "sandy_s": ["peanut", "cassava", "watermelon", "sweet potatoes", "carrot"],
        "loamy_s" : ["maize", "tomatoes", "beans", "lettuce", "rice (with irrigation)"]
    }
    if (type == "argileux"):
        return (cultures["clay_s"])
    elif (type == "sableux"):
        return (cultures["sandy_s"])
    else:
        return (cultures["loamy_s"])

def suggest_improvement(ph, humidite):#We give suggestion based on ph and humidity
    suggestions = []

    if ph < 6:
        suggestions.append("Add lime to reduce acidity")
    elif ph > 7:
        suggestions.append("Add organic matter to reduce alkalinity")

    if humidite < 20:
        suggestions.append("Increase irrigation")
    elif humidite > 30:
        suggestions.append("Improve drainage")

    return suggestions