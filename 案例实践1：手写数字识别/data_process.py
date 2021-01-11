import cv2
import os, re
import numpy as np

def get_img_names(path):
    filnames = os.listdir(path)
    imgnames = []
    for i in filnames:
        if re.findall('^\d_\d+\.png$', i)!=[]:
            imgnames.append(i)
    return imgnames

def get_img_data(path):
    imgnames = get_img_names(path)
    n = len(imgnames)
    shape = (28, 28)
    data = np.zeros([n, shape[0]*shape[1]])
    labels = np.zeros([n])
    for i in range(n):
        img = cv2.imread(path+imgnames[i])
        img = cv2.resize(img, shape)[:,:,0]/255
        data[i, :] = img.reshape(shape[0]*shape[1])
        labels[i] = imgnames[i][0]

    return data, labels