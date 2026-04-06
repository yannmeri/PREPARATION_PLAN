import numpy as np

def create_2D():
    arr = np.array([[1,2,3],
                    [4,5,6]]) # Create a 2D NumPy array (matrix)


    print("Shape (rows, columns):", arr.shape) #Display the dimensions of the array
    print("Value 5:", arr[1][1]) #Access the element at row 1, column 1

    return (arr)


if __name__=="__main__":

    create_2D()