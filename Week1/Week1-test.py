import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def reduce_gray_level(img,level):
    num = 256 / level
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            img[i, j] = np.fix((img[i, j] / num)) * 255 / (level - 1)
    return img


level = 256
img = cv.imread("../rose.jpg", cv.IMREAD_GRAYSCALE)
rows, cols = img.shape
size = img.size
print("image shape: ", rows, cols)
print("image size: ", size)
print("image type: ", img.dtype)
#plt.imshow(img, cmap='gray', interpolation='bicubic')
#plt.show()

while(level >= 2):
     img = reduce_gray_level(img, level)
     plt.imshow(img, cmap='gray', interpolation='bicubic')
     plt.show()
     level = level/2
print((255 / 128.0) * 255 / 1)


img2 = cv.imread("../rose.jpg", cv.IMREAD_GRAYSCALE)
#plt.imshow(img2, cmap='gray', interpolation='bicubic')
#plt.show()
