import numpy as np #NumPy is used for performing mathematical operations on 1D, 2D, and higher-dimensional arrays

def create_1d():
    arr = np.array([5, 10, 15, 20])#we create a 1D array here

    return (arr)

def compute_mean(arr):
    moy = np.mean(arr) #we compute the mean of this array here

    return (moy)


def compute_var(arr):
    variance = np.var(arr)  #Compute the variance of the array
    return variance

def multiplicate_arr_by_two(arr):
    mult_arr = arr * 2  #Multiply each element of the array by 2
    return mult_arr


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