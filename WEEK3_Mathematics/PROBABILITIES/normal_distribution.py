import numpy as np#NumPy is used for numerical computations
from dataset_mean_var import read_csv_file


def normal_distribution(ph_values):#Compute the mean and standard deviation of pH values.
    moy_ph = np.mean(ph_values)
    std_ph = np.std(ph_values)

    return (moy_ph, std_ph)

if __name__=="__main__":
    df = read_csv_file()
    (moy_ph, std_ph) = normal_distribution(df["ph"])

    print(f"Mean ph: {moy_ph}\nStandar Deviation ph: {std_ph}\n")

    # Count how many values fall within [mean - std, mean + std]
    count = df["ph"].between(moy_ph - std_ph, moy_ph + std_ph).sum()

    ratio = count / len(df["ph"])

    print(f"Proportion: {ratio:.2f}")

    # Interpretation
    if ratio >= 0.68:
        print("The data roughly follows a normal distribution.")
    else:
        print("The data may not follow a normal distribution.")