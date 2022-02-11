import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


img = cv.imread("D:/code/NRVQA/imgs/CLAHE/0025_CLAHE.jpg", 1)
img_temp = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
grayHist = calcGrayHist(img_temp)
x = np.arange(256)
# 绘制灰度直方图
plt.plot(x, grayHist, 'r', linewidth=2, c='black')
plt.xlabel("gray Label")
plt.ylabel("number of pixels")
plt.show()
# cv.imshow("img", img)
# cv.waitKey()

