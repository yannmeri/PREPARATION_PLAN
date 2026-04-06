import pandas as pd #pandas library is useful for reading inside a csv file and create a dataframe(df) from it.

def read_csv_file():
    df = pd.read_csv("mini_projet_sols_csv.csv")
    print(df)

    return (df)


def classify_ph(ph):
    if (ph < 6):
        return ("acid")
    elif(ph > 7):
        return ("basic")
    else:
        return ("neutral")


def check_fertility(df):

    if (6 <= df["ph"] <= 7 and df["humidite"] >= 20):
        return (True)
    else:
        return (False)


def count_number_of_fertile_soil(df):

    count_f = df["fertility"].sum()
    count_inf = len(df) - count_f

    print(f"\nThere are {count_f} fertile soils and {count_inf} infertile soils.")


if __name__=="__main__":

    df = read_csv_file()
    fertile = df[df["ph"] > 6]

    print(f"\n{df.head(3)}\n")
    print(f"{df.describe()}\n\nSoils with ph equal to 6 are:\n{fertile["type"]}\n\n")

    df["soil_concentration"] = df["ph"].apply(classify_ph)

    df["fertility"] = df.apply(check_fertility, axis=1)
    print(df)

    count_number_of_fertile_soil(df)
    