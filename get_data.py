import os
import cv2
import numpy as np

cats_dir = "./cats"
dogs_dir = "./dogs"

def get_cats_image():
    images = []
    # get filename from folder
    for filename in os.listdir(cats_dir):
        try:
            # open and read image with cv2
            image = cv2.imread(f"{cats_dir}/{filename}")
            # normalize image to (64x64)
            image = cv2.resize(image, (64, 64)) # here is matrix of image
            images.append(image)
        except: # skip error images
            continue
    return np.stack(images)

def get_dogs_image():
    images = []
    # get filename from folder
    for filename in os.listdir(dogs_dir):
        try:
            # open and read image with cv2
            image = cv2.imread(f"{dogs_dir}/{filename}")
            # normalize image to (64x64)
            image = cv2.resize(image, (64, 64)) # here is matrix of image
            images.append(image)
        except:
            continue
    return np.stack(images)