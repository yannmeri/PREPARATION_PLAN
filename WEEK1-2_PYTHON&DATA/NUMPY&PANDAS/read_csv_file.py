import pandas as pd #Pandas is used to read CSV files and create a DataFrame (df) from the data.

def read_csv_file(): #open a .csv file and read it with pandas library
    df = pd.read_csv("mini_projet_sols_csv.csv")
    print(df)

    return (df)


def classify_ph(ph):# Classify the soil based on its pH value (acidic, neutral, or basic)
    if (ph < 6):
        return ("acid")
    elif(ph > 7):
        return ("basic")
    else:
        return ("neutral")


def check_fertility(df):     # Determine if a soil is fertile based on its pH and humidity values

    if (6 <= df["ph"] <= 7 and df["humidite"] >= 20):
        return (True)
    else:
        return (False)


def count_number_of_fertile_soil(df): #count the number of soil fertiles

    count_f = df["fertility"].sum()
    # In the "fertility" column:
    # True = 1 (fertile), False = 0 (infertile)
    # The sum() function adds all values to count fertile soils

    count_inf = len(df) - count_f # Total rows minus fertile soils

    print(f"\nThere are {count_f} fertile soils and {count_inf} infertile soils.")


if __name__=="__main__":

    df = read_csv_file()
    fertile = df[df["ph"] > 6]# Create a new DataFrame containing only rows where pH > 6

    print(f"\n{df.head(3)}\n") # Display the first 3 rows of the dataset

    print(f"{df.describe()}\n\nSoils with ph equal to 6 are:\n{fertile["type"]}\n\n")
    # Display statistical summary and filtered soil types

    df["soil_concentration"] = df["ph"].apply(classify_ph) # Apply the classify_ph function to each value in the "ph" column

    df["fertility"] = df.apply(check_fertility, axis=1)
    # Apply the check_fertility function row by row (axis=1)
    # This allows access to both "ph" and "humidite" columns

    print(df)

    count_number_of_fertile_soil(df)
    