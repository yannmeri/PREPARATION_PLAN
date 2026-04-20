import numpy as np#NumPy is used for numerical computations
from dataset_mean_var import read_csv_file

def simple_correlation(ph, humidity):#Compute correlation between ph and humidity (simple lists)
    return(np.corrcoef(ph, humidity))

def dataset_correlation(df):#Compute correlation between ph and humidity in the dataset
    return(df[["ph", "humidite"]].corr())

def new_dataset_correlation(df):#Add a temperature column and compute correlation matrix.
    df["temperature"] = [30, 32, 35, 40, 38]
    return (df[["ph", "humidite", "temperature"]].corr())

if __name__=="__main__":#I test my program here

    df = read_csv_file()

    ph = [5.5, 6.0, 6.5, 7.0]
    humidity = [15, 20, 25, 30]

    #Simple correlation
    s_corr = simple_correlation(ph, humidity)
    print(f"Simple correlation (pH vs humidity): {s_corr}")

    print("\nInterpretation:")
    print("- Close to 1 → strong positive correlation")
    print("- Close to -1 → strong negative correlation")
    print("- Close to 0 → weak or no correlation")

    #Dataset correlation
    d_corr = dataset_correlation(df)
    print(f"\nDataset correlation:\n{d_corr}")

    #Extended dataset correlation
    new_d_corr = new_dataset_correlation(df)
    print(f"\nCorrelation with temperature:\n{new_d_corr}")

    print("\nFinal interpretation:")
    print("- Check which variables are strongly or weakly related based on the matrix values.\n")

    print(f"If correlation > 0.7 → strong\nIf 0.3 <= correlation < 0.7 → moderate\nIf correlation < 0.3 → weak\n")