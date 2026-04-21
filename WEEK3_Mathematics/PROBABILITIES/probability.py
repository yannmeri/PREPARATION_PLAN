import numpy as np#NumPy is used for numerical computations
from dataset_mean_var import read_csv_file
import pandas as pd#Pandas is used to manipulate datasets

def check_fertility(df):#Create a new column indicating whether the soil is fertile
    df["fertility"] = (df["ph"].between(6,7) & (df["humidite"] >= 20))
    return (df)


def compute_probability_fertile(fertility):#Compute the probability of a soil being fertile
    return (fertility.sum() / len(fertility))

def conditional_probability(df_new):#Compute P(fertile | humidity >= 20)
    
    return (df_new[df_new["humidite"] >= 20]["fertility"].mean())


def check_independance_fertility_ph(prob_f, df):#Check whether fertility and ph[6,7] are independent or not

    P_fertility_given_ph = df[df["ph"].between(6,7)]["fertility"].mean()

    if (P_fertility_given_ph == prob_f):
        print(f"P(fertile | ph[6,7]): {P_fertility_given_ph}")
        print ("Fertility and ph are independent.\n")
    else:
        print(f"P(fertile | ph[6,7]): {P_fertility_given_ph}")
        print ("Fertility and ph are dependent.\n")

def check_independance_fertility_humidity(prob_f, df):#Check whether fertility and humidity >=20 are independent or not

    P_fertility_given_humidity = df[df["humidite"] >= 20]["fertility"].mean()

    if (P_fertility_given_humidity == prob_f):
        print(f"P(fertile | humidity >= 20): {P_fertility_given_humidity}")
        print ("Fertility and humidity are independent.")
    else:
        print(f"P(fertile | humidity >= 20): {P_fertility_given_humidity}")
        print ("Fertility and humidity are dependent.")


if __name__=="__main__":#I test my program here
    df = read_csv_file()
    df_new = check_fertility(df)

    print(f"New dataset:\n{df_new}")

    prob_f = compute_probability_fertile(df_new["fertility"])
    condi_prob = conditional_probability(df_new)

    print(f"P(fertile): {prob_f}")
    print(f"P(fertile | humidite > 20): {condi_prob}\n")
    
    check_independance_fertility_ph(prob_f, df_new)
    check_independance_fertility_humidity(prob_f, df_new)
