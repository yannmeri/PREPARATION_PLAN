import numpy as np

def create_2D():
    arr = np.array([[1,2,3],
                    [4,5,6]])

    print("Shape (rows, columns):", arr.shape)
    print("Value 5:", arr[1][1])

    return (arr)


if __name__=="__main__":

    create_2D()