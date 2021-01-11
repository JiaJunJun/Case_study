from PIL import Image
import numpy as np
import os, re
# 读取图片文件

def var(rd):
    mid = np.mean((rd - rd.mean()) ** 3)
    return(np.sign(mid) * abs(mid) ** (1 / 3))

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
    for i in range(n):
        img = Image.open(path+imgnames[i])
        # 提取特征
        # r,g,b通道
        r, g, b = img.split()
        rd = np.ndarray(r)
        gd = np.ndarray(g)
        bd = np.ndarray(b)

        data = np.zeros([n, 9])
        labels = np.zeros([n])

        data[i, 0] = rd.mean()
        data[i, 1] = rd.std()
        data[i, 2] = var(rd)

        data[i, 3] = gd.mean()
        data[i, 4] = gd.std()
        data[i, 5] = var(gd)

        data[i, 6] = bd.mean()
        data[i, 7] = bd.std()
        data[i, 8] = var(bd)

        labels[i] = imgnames[0]

    return data, labels


