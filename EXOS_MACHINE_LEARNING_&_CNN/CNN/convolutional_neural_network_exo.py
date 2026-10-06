import numpy as np
import os
from PIL import Image
from sklearn.metrics import f1_score
import pandas as pd

folder_path = "./dataset_v3/train/Clay/"

def initialize_kernel(size):


    list_pixel = []
    pixel_row = []

    for row in range(size):
        for col in range(size):
            W = np.random.randn(1, 3) * np.sqrt(2.0 / 1)
            pixel_row.append(W)

        list_pixel.append(pixel_row)
        pixel_row = []

    list_kernel = np.array(list_pixel)
    b = np.zeros((1,1))
    print(list_kernel)
    params = {
        "kernel": list_kernel,
        "bias": b
    }

    return (params)


def load_images():

    list_img = []

    for f in os.listdir(folder_path):
        if (os.path.isfile(os.path.join(folder_path, f))):
            img = Image.open(os.path.join(folder_path, f))
            img_arr = np.array(img) / 255.0
            #print(img_arr)
            list_img.append(img_arr)
            #break

    clay_arr = np.array(list_img)
    #print(clay_arr.shape)
    return (clay_arr)


def feature_maps(images, params):

    row = 0
    col = 0
    kernel = params['kernel']
    b = params['bias']
    list_map = []

    for img in images:
        while (row + kernel.shape[0] <= img.shape[0]):
            while (col + kernel.shape[1] <= img.shape[1]):
                portion = img[row:row + kernel.shape[0], col:col + kernel.shape[1]]
                #print(portion)
                feature_map = np.sum((portion * kernel), axis=1) + b
                list_map.append(feature_map)
                col += 1
            row +=1

    #print(np.array(list_map))


if __name__=="__main__":

    params = initialize_kernel(3)
    images = load_images()
    feature_maps(images, params)
    #print(params)