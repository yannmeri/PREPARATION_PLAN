import numpy as np

def exercise_1():

    image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    kernel = np.array([
        [1, 0],
        [0, 1]
    ])


    i = 0
    j = 0
    nb_row_kernel = kernel.shape[0]
    nb_col_kernel = kernel.shape[1]
    nb_row_image = image.shape[0]
    nb_col_image = image.shape[1]

    list = []
    while (i + nb_row_kernel <= nb_row_image):
        while (j + nb_col_kernel <= nb_col_image):
            region = image[i:i+nb_row_kernel, j:j+nb_col_kernel]
            list.append(np.sum(region * kernel))
            j += 1
        j = 0
        i += 1
    
    list = np.array(list)
    feature_map = list.reshape(2, 2)

    return (feature_map)


def exercise_2(n, f, p, s):
    
    output_size = int(((n - f + 2 * p) / s ) + 1)

    return (output_size)


def exercise_3():
    image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    kernel = np.array([
        [1, 0],
        [0, 1]
    ])


    i = 0
    j = 0
    nb_row_kernel = kernel.shape[0]
    nb_col_kernel = kernel.shape[1]
    nb_row_image = image.shape[0]
    nb_col_image = image.shape[1]

    list = []
    while (i + nb_row_kernel <= nb_row_image):
        while (j + nb_col_kernel <= nb_col_image):
            region = image[i:i+nb_row_kernel, j:j+nb_col_kernel]
            list.append(np.sum(region * kernel))
            j += 2
        j = 0
        i += 2
    
    list = np.array(list)
    feature_map = list.reshape(2, 2)

    return (feature_map)


def exercise_4():

    image = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    list = []
    for row in range(5):
        if (row == 0 or row == 4):
            for col in range(5):
                list.append(0)
        else:
            for col in range(5):
                if (col == 0 or col == 4):
                    list.append(0)
                else:
                    list.append(int(image[row - 1, col-1]))
    
    new_l = np.array(list)
    new_image = new_l.reshape(5,5)

    return (new_image)

def exercise_6():
    leaf_image = np.array([
        [10, 20, 15, 5],
        [30, 80, 90, 10],
        [20, 70, 100, 15],
        [5, 10, 20, 5]
    ])

    spot_detector = np.array([
        [-1, -1],
        [-1,  4]
    ])


    i = 0
    j = 0
    nb_row_spot_detector = spot_detector.shape[0]
    nb_col_spot_detector = spot_detector.shape[1]
    nb_row_leaf_image = leaf_image.shape[0]
    nb_col_leaf_image = leaf_image.shape[1]

    list = []
    while (i + nb_row_spot_detector <= nb_row_leaf_image):
        while (j + nb_col_spot_detector <= nb_col_leaf_image):
            region = leaf_image[i:i+nb_row_spot_detector, j:j+nb_col_spot_detector]
            list.append(np.sum(region * spot_detector))
            j += 1
        j = 0
        i += 1
    
    list = np.array(list)
    feature_map = list.reshape(3, 3)

    return (feature_map)


if __name__=="__main__":

    #Exercise 1
    feature_map = exercise_1()
    print(f"feature_map:\n{feature_map}")
    print(f"feature_map_shape: {feature_map.shape}")

    #Exercise 2
    ##case A
    n = 7
    f = 3
    s = 1
    p = 0
    output_size = exercise_2(n, f, p, s)
    print(f"In A case, the output_size is: {output_size}")

    ##case B
    n = 7
    f = 3
    s = 1
    p = 1
    output_size = exercise_2(n, f, p, s)
    print(f"In B case, the output_size is: {output_size}")

    ##case C
    n = 8
    f = 3
    s = 2
    p = 1
    output_size = exercise_2(n, f, p, s)
    print(f"In C case, the output_size is: {output_size}")


    final_feature_map = exercise_3()
    print(f"The final feature map is: \n{final_feature_map}")

    n = 5
    f = 2
    s = 1
    p = 1
    output_size = exercise_2(n, f, p, s)
    new_image = exercise_4()
    print(f"the new_image size is: {new_image.shape}")
    print(f"the output size is: {output_size} ")

    leaf_feature_map = exercise_6()
    print(f"leaf_feature_map is: \n{leaf_feature_map}")
    print("The feature map wants to detect leaves' forms.")