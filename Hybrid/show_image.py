from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

path = os.getcwd()


def resize_112_112(image, ideal_shape = (112, 112)):
    img_new2 = resize(image, ideal_shape, anti_aliasing=True)
    return img_new2


def shw_img_normal(data,label):
    img = np.asarray(np.loadtxt(data))
    plt.figure(num=label)
    plt.imshow(img, cmap='gray')
    plt.show()


def shw_img_resize(data,label):
    img = np.asarray(np.loadtxt(data))
    img  = resize_112_112(img)
    plt.figure(num=label)
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == "__main__":
    plt.rcParams['toolbar'] = 'None'
    home = str(Path.home())
    data_health =  path+"/CSE6389_project2/CSE6389_project2/Training/Health_raw/average_fmri_feature_matrix1.txt"
    data_patient =  path+"/CSE6389_project2/CSE6389_project2/Training/Patient_raw/average_fmri_feature_matrix1.txt"
    shw_img_normal(data_patient,"Patient Sample Original(150*150)")
    shw_img_normal(data_health,"Healthy Sample Original(150*150)")
    shw_img_resize(data_patient, "Patient Sample Resized(112*112)")
    shw_img_resize(data_health, "Healthy Sample Resized(112*112)")

