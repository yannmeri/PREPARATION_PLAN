from probability import check_fertility, compute_probability_fertile
import numpy as np#NumPy is used for numerical computations
from dataset_mean_var import read_csv_file


def compute_probability_humidity(df):#Compute P(humidity >= 20)
    return ((df["humidite"] >= 20).mean())

def compute_probability_humidity_given_fertile(df):#Compute P(humidity >= 20 | fertile)
    return ((df[df["fertility"] == True]["humidite"] >= 20).mean())

if __name__ == "__main__":
    df = read_csv_file()
    df = check_fertility(df)

    prob_f = compute_probability_fertile(df["fertility"])
    prob_h = compute_probability_humidity(df)
    prob_h_given_f = compute_probability_humidity_given_fertile(df)

    #Bayes formula
    prob_f_given_h = (prob_h_given_f * prob_f) / prob_h

    print(f"P(fertile): {prob_f}")
    print(f"P(humidity >= 20): {prob_h}")
    print(f"P(humidity >= 20 | fertile): {prob_h_given_f}")

    print("\nUsing Bayes theorem:")
    print(f"P(fertile | humidity >= 20): {prob_f_given_h}")

    #Direct computation (verification)
    prob_direct = df[df["humidite"] >= 20]["fertility"].mean()

    print("\nDirect computation:")
    print(f"P(fertile | humidity >= 20): {prob_direct}\n")

    print("If P(fertile | humidity >= 20) is high, it means humidity is a strong indicator of soil fertility.")