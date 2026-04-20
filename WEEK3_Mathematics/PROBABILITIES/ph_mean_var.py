import numpy as np# NumPy is used for numerical computations

def ph_mean(ph_values):#Compute the mean ph values.
    return (np.mean(ph_values))

def ph_variance(ph_values):#Compute the variance of ph values.
    return (np.var(ph_values))

if __name__=="__main__":#I test my program here
    ph_values = [5.5, 6.2, 6.8, 7.5, 6.0]
    moy_ph = ph_mean(ph_values)
    var_ph = ph_variance(ph_values)

    print(f"Mean pH: {moy_ph}")
    print(f"Variance of pH: {var_ph}")

    print("\nInterpretation:")
    print("- High variance → pH values are very different (heterogeneous soils)")
    print("- Low variance → pH values are similar (homogeneous soils)")