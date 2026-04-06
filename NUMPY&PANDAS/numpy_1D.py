import numpy as np

def create_1d():
    arr = np.array([5, 10, 15, 20])

    return (arr)

def compute_mean(arr):
    moy = np.mean(arr)

    return (moy)

def compute_var(arr):
    variance = np.var(arr)

    return (variance)

def multiplicate_arr_by_two(arr):
    mult_arr = arr * 2

    return (mult_arr)


if __name__=="__main__":

    arr = create_1d()
    moy = compute_mean(arr)
    variance = compute_var(arr)
    mult_arr = multiplicate_arr_by_two(arr)

    print("Array:", arr)
    print("Mean:", moy)
    print("Variance:", variance)
    print("Array * 2:", mult_arr)