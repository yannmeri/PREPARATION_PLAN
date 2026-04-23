from probability import check_fertility, compute_probability_fertile
import numpy as np#NumPy is used for numerical computations
from dataset_mean_var import read_csv_file


def expected_value(outcome_f, outcome_inf, prob_f, prob_inf):#Compute the expected value (average outcome) based on probabilities.
    return (outcome_f * prob_f + outcome_inf * prob_inf)


if __name__=="__main__":#I test my program here
    df = read_csv_file()
    df_new = check_fertility(df)

    print(f"New dataset:\n{df_new}\n")

    prob_f = compute_probability_fertile(df_new["fertility"])
    prob_inf = 1 - prob_f
    
    exp_value = expected_value(120, 50, prob_f, prob_inf)

    print(f"P(fertile): {prob_f}\nP(infertile): {prob_inf}\n")
    print(f"Expected yield: {exp_value}\n")
    print("This expected value represents the average yield we can expect based on soil conditions.")