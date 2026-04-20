import numpy as np#NumPy is used for numerical computations

def data_mean(data):#Compute the mean of a dataset.
    return (np.mean(data))

def data_variance(data):#Compute the variance of a dataset.
    return (np.var(data))

if __name__=="__main__":#I test my program here
    data = [10, 12, 14, 16]
    moy_data = data_mean(data)
    var_data = data_variance(data)

    print(f"Mean: {moy_data}")
    print(f"Variance: {var_data}")

    print("\nInterpretation:")
    print("- A high variance means the values are spread out.")
    print("- A low variance means the values are close to the mean.")