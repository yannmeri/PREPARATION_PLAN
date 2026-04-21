import numpy as np#NumPy is used for numerical computations

def data_mean(data):#Compute the mean of a dataset.
    return (np.mean(data))

def data_variance(data):#Compute the variance of a dataset.
    return (np.var(data))

def ecart_type(variance):#Compute the standard deviation
    return (np.sqrt(variance))

if __name__=="__main__":#I test my program here
    data = [10, 12, 14, 18, 20]
    moy_data = data_mean(data)
    var_data = data_variance(data)
    ecart_t = ecart_type(var_data)

    print(f"Mean: {moy_data}")
    print(f"Variance: {var_data}")
    print(f"Standard Deviation: {ecart_t}")

    print("\nInterpretation:")
    print("- A high variance means the values are spread out.")
    print("- A low variance means the values are close to the mean.\n")

    print(f"- A high standard deviation means the values are spread out.")
    print("- A low standard deviation means the values are close to the mean.\n")